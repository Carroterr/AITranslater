import asyncio
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from io import BytesIO

import httpx
import numpy as np
import webrtcvad

from audio import ASR_SAMPLE_RATE, Transcript


ASR_URL = "https://api.deepseek.com/v1/tools/asr"
FRAME_MS = 20
FRAME_BYTES = ASR_SAMPLE_RATE * FRAME_MS // 1000 * 2
END_SILENCE = 27  # 540 ms
MAX_FRAMES = 650  # 13 seconds
FIRST_PREVIEW_FRAMES = 55  # 1.1 seconds
PREVIEW_INTERVAL_FRAMES = 110  # 2.2 seconds


class AudioFormat(Enum):
    OGG = "OGG (Opus)"
    WAV = "WAV (PCM)"


@dataclass(frozen=True)
class AudioSnapshot:
    segment: int
    pcm: bytes
    final: bool
    recorded_at: float
    hard_cut: bool = False
    gap_before_ms: int | None = None


class AudioSegments:
    def __init__(self, emit):
        self.emit = emit
        self.vad = webrtcvad.Vad(2)
        self.buffer = bytearray()
        self.leading = deque(maxlen=10)
        self.frames = []
        self.segment = 1
        self.recorded_at = None
        self.voiced = 0
        self.quiet = 0
        self.next_preview = FIRST_PREVIEW_FRAMES
        self.frame_number = 0
        self.previous_voice_end = None
        self.gap_before_ms = None

    def accept(self, pcm, captured_at):
        self.buffer.extend(pcm)
        size = len(self.buffer) // FRAME_BYTES * FRAME_BYTES
        for start in range(0, size, FRAME_BYTES):
            frame_start = captured_at - (len(self.buffer) - start) / (2 * ASR_SAMPLE_RATE)
            self.add_frame(bytes(self.buffer[start:start + FRAME_BYTES]), frame_start)
        del self.buffer[:size]

    def add_frame(self, frame, frame_start):
        self.frame_number += 1
        speech = self.vad.is_speech(frame, ASR_SAMPLE_RATE)
        if not self.frames:
            if not speech:
                self.leading.append(frame)
                return
            if self.previous_voice_end is not None:
                self.gap_before_ms = (self.frame_number - self.previous_voice_end - 1) * FRAME_MS
            self.recorded_at = frame_start
            self.frames.extend(self.leading)
            self.leading.clear()

        self.frames.append(frame)
        self.voiced += int(speech)
        self.quiet = 0 if speech else self.quiet + 1
        count = len(self.frames)
        if self.quiet >= END_SILENCE or count >= MAX_FRAMES:
            if self.voiced >= 4:
                end = count - max(0, self.quiet - 5)
                self.emit(AudioSnapshot(
                    self.segment, b"".join(self.frames[:end]), final=True, recorded_at=self.recorded_at,
                    hard_cut=self.quiet < END_SILENCE, gap_before_ms=self.gap_before_ms,
                ))
                self.previous_voice_end = self.frame_number - self.quiet
                self.segment += 1
            if self.quiet:
                self.leading.extend(self.frames[-min(self.quiet, self.leading.maxlen):])
            self.frames.clear()
            self.voiced = 0
            self.quiet = 0
            self.next_preview = FIRST_PREVIEW_FRAMES
        elif speech and self.voiced >= 12 and count >= self.next_preview:
            self.emit(AudioSnapshot(
                self.segment, b"".join(self.frames), final=False, recorded_at=self.recorded_at,
            ))
            self.next_preview = count + PREVIEW_INTERVAL_FRAMES


def encode_audio(pcm, audio_format):
    audio = BytesIO()
    if audio_format is AudioFormat.WAV:
        with wave.open(audio, "wb") as wav:
            wav.setparams((1, 2, ASR_SAMPLE_RATE, 0, "NONE", "not compressed"))
            wav.writeframes(pcm)
        return "speech.wav", audio.getvalue(), "audio/wav"

    try:
        import av
    except ImportError as exc:
        raise RuntimeError("无法加载 OGG 编码器。请安装 requirements.txt 中的依赖，或在设置中选择 WAV。") from exc

    try:
        samples = np.frombuffer(pcm, dtype="<i2").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = ASR_SAMPLE_RATE
        frame.pts = 0
        frame.time_base = Fraction(1, ASR_SAMPLE_RATE)
        with av.open(audio, mode="w", format="ogg") as container:
            stream = container.add_stream("libopus", rate=ASR_SAMPLE_RATE)
            stream.layout = "mono"
            stream.bit_rate = 24_000
            stream.options = {"application": "voip"}
            container.mux(stream.encode(frame))
            container.mux(stream.encode(None))
    except (av.FFmpegError, ValueError) as exc:
        raise RuntimeError("无法编码 OGG 音频。请检查 PyAV 安装，或在设置中选择 WAV。") from exc
    return "speech.ogg", audio.getvalue(), "audio/ogg"


async def transcribe(client, pcm, audio_format=AudioFormat.OGG):
    payload = await asyncio.to_thread(encode_audio, pcm, audio_format)
    async with asyncio.timeout(12):
        response = await client.post(ASR_URL, files={"file": payload})
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ValueError("DeepSeek 语音识别返回格式异常")
    return data["text"].strip()


class SpeechQueue:
    def __init__(self, submit, status, audio_format=AudioFormat.OGG):
        self.submit = submit
        self.status = status
        self.audio_format = audio_format
        self.finals = asyncio.Queue(maxsize=3)
        self.preview_ready = asyncio.Event()
        self.preview = None
        self.recent_previews = {}
        self.latest_segment = 0
        self.finished_through = 0
        self.capture_message = ""
        self.unavailable = False

    def capture_status(self, message):
        self.capture_message = message
        if not self.unavailable:
            self.status(message)

    def enqueue(self, sample):
        self.latest_segment = sample.segment
        for segment in self.recent_previews.keys() - {sample.segment, sample.segment - 1}:
            self.recent_previews.pop(segment)
        if sample.final:
            if self.preview is not None and self.preview.segment == sample.segment:
                self.preview = None
                self.preview_ready.clear()
            if self.finals.full():
                self.finals.get_nowait()
            self.finals.put_nowait(sample)
        else:
            self.preview = sample
            self.preview_ready.set()

    async def run(self, client, *, final):
        while True:
            if final:
                sample = await self.finals.get()
            else:
                await self.preview_ready.wait()
                sample, self.preview = self.preview, None
                self.preview_ready.clear()
                if sample is None:
                    continue

            if sample.segment < self.latest_segment - 1:
                continue
            previous = self.recent_previews.get(sample.segment)
            try:
                if final and previous is not None and previous[0] == sample.pcm:
                    text = previous[1]
                else:
                    text = await transcribe(client, sample.pcm, self.audio_format)
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code != 429 and 400 <= code < 500 and code != 408:
                    raise RuntimeError(f"DeepSeek 语音识别请求失败（{code}），请检查 API Key、余额或接口权限。") from exc
                self.recognition_unavailable()
                continue
            except (httpx.RequestError, TimeoutError, ValueError):
                self.recognition_unavailable()
                continue
            if self.unavailable:
                self.unavailable = False
                self.status(self.capture_message)
            if sample.segment <= self.finished_through or sample.segment < self.latest_segment - 1:
                continue

            if final:
                self.finished_through = sample.segment
                previous = self.recent_previews.pop(sample.segment, None)
                if not text and previous is not None:
                    text = previous[1]
            elif text:
                self.recent_previews[sample.segment] = sample.pcm, text
                if previous is not None and text == previous[1]:
                    continue

            if text:
                self.submit(Transcript(
                    sample.segment, text, final=final,
                    hard_cut=sample.hard_cut, gap_before_ms=sample.gap_before_ms, recorded_at=sample.recorded_at,
                ))

    def recognition_unavailable(self):
        if not self.unavailable:
            self.unavailable = True
            self.status("DeepSeek 语音识别暂时不可用，将继续尝试识别后续音频。")


async def recognize_audio(api_key, stop, submit, status, *, capture_audio, audio_format=AudioFormat.OGG):
    loop = asyncio.get_running_loop()
    queue = SpeechQueue(submit, status, audio_format)
    segments = AudioSegments(lambda sample: loop.call_soon_threadsafe(queue.enqueue, sample))

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(connect=4, read=10, write=4, pool=3),
        limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
    ) as client:
        capture = asyncio.create_task(asyncio.to_thread(
            capture_audio, stop, segments.accept,
            lambda message: loop.call_soon_threadsafe(queue.capture_status, message),
        ))
        workers = [asyncio.create_task(queue.run(client, final=final)) for final in (True, False)]
        try:
            completed, _ = await asyncio.wait([capture, *workers], return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                task.result()
        finally:
            stop.set()
            for task in workers:
                task.cancel()
            await asyncio.gather(capture, *workers, return_exceptions=True)
