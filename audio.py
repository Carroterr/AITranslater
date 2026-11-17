import audioop
from collections import Counter
from dataclasses import dataclass
from queue import Empty, Full, Queue
from time import monotonic, time

import numpy as np
import pyaudiowpatch as pyaudio


ASR_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AudioDevice:
    name: str
    duplicate_index: int | None = None

    @property
    def label(self):
        if self.duplicate_index is not None:
            return f"{self.name}（设备 {self.duplicate_index}）"
        return self.name


def _named_devices(devices, *, loopback=False):
    devices = [device for device in devices if device["maxInputChannels"] > 0]
    names = [device["name"].removesuffix(" [Loopback]") if loopback else device["name"] for device in devices]
    counts = Counter(names)
    # PortAudio indices can change between enumerations, so only use them for duplicate names.
    return {
        AudioDevice(name, device["index"] if counts[name] > 1 else None): device
        for name, device in zip(names, devices)
    }


def _playback_loopbacks(audio):
    return _named_devices(audio.get_loopback_device_info_generator(), loopback=True)


def _microphones(audio):
    return _named_devices(
        device for device in audio.get_device_info_generator_by_host_api(host_api_type=pyaudio.paWASAPI)
        if not device["isLoopbackDevice"]
    )


def list_playback_devices():
    with pyaudio.PyAudio() as audio:
        return list(_playback_loopbacks(audio))


def list_microphones():
    with pyaudio.PyAudio() as audio:
        return list(_microphones(audio))


def _resolve_loopback(audio, playback_device):
    if playback_device is None:
        try:
            return audio.get_default_wasapi_loopback()
        except (OSError, LookupError) as exc:
            raise RuntimeError("找不到 Windows 默认播放设备，请连接扬声器或耳机后重试。") from exc

    try:
        return _playback_loopbacks(audio)[playback_device]
    except (OSError, KeyError) as exc:
        raise RuntimeError(f"播放设备「{playback_device.label}」已不可用，请重新选择。") from exc


def _resolve_microphone(audio, microphone_device):
    if microphone_device is None:
        try:
            return audio.get_default_wasapi_device(d_in=True)
        except (OSError, LookupError) as exc:
            raise RuntimeError("找不到 Windows 默认麦克风。请在 Windows 声音设置中选定输入设备。") from exc

    try:
        return _microphones(audio)[microphone_device]
    except (OSError, KeyError) as exc:
        raise RuntimeError(f"麦克风「{microphone_device.label}」已不可用，请重新选择。") from exc


@dataclass(frozen=True)
class Transcript:
    segment: int
    text: str
    final: bool
    hard_cut: bool = False
    gap_before_ms: int | None = None
    recorded_at: float | None = None
    speaker: str = ""


def capture_system_audio(stop, send_chunk, status, *, playback_device=None):
    with pyaudio.PyAudio() as audio:
        device = _resolve_loopback(audio, playback_device)
        name = playback_device.label if playback_device is not None else device["name"].removesuffix(" [Loopback]")
        _capture_audio(audio, device, name, "播放设备", stop, send_chunk, status,
                       "未检测到播放声音 · {name}。请确认播放器的输出设备。")


def capture_microphone(stop, send_chunk, status, *, microphone_device=None):
    with pyaudio.PyAudio() as audio:
        device = _resolve_microphone(audio, microphone_device)
        name = microphone_device.label if microphone_device is not None else device["name"]
        _capture_audio(audio, device, name, "麦克风", stop, send_chunk, status,
                       "等待麦克风语音 · {name}")


def _capture_audio(audio, device, name, device_type, stop, send_chunk, status, idle_message):
    capturing = f"正在采集 · {name}"
    rate = int(device["defaultSampleRate"])
    channels = int(device["maxInputChannels"])
    frames = rate * 80 // 1000
    silence = bytes(ASR_SAMPLE_RATE * 120 // 1000 * 2)
    blocks = Queue(maxsize=12)

    def receive(data, _frames, _time, _flags):
        block = data, time()
        try:
            blocks.put_nowait(block)
        except Full:
            try:
                blocks.get_nowait()
            except Empty:
                pass
            blocks.put_nowait(block)
        return None, pyaudio.paContinue

    last_signal = monotonic()
    waiting_for_audio = False
    resample_state = None
    try:
        with audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=frames,
            stream_callback=receive,
        ) as stream:
            status(capturing)
            while not stop.is_set():
                try:
                    data, captured_at = blocks.get(timeout=0.12)
                except Empty:
                    if not stream.is_active():
                        raise RuntimeError(f"{device_type}「{name}」已停止采集，请检查连接后重新开始。")
                    # WASAPI may stop delivering packets when playback goes idle.
                    mono = silence
                    captured_at = time()
                    has_sound = False
                else:
                    if not data:
                        continue
                    if channels == 1:
                        mono = data
                    else:
                        pcm = np.frombuffer(data, dtype="<i2").reshape(-1, channels)
                        mono = pcm.mean(axis=1, dtype=np.float32).astype("<i2").tobytes()
                    has_sound = audioop.max(mono, 2) >= 33
                    if rate != ASR_SAMPLE_RATE:
                        mono, resample_state = audioop.ratecv(mono, 2, 1, rate, ASR_SAMPLE_RATE, resample_state)

                now = monotonic()
                if has_sound:
                    last_signal = now
                    if waiting_for_audio:
                        status(capturing)
                        waiting_for_audio = False
                elif not waiting_for_audio and now - last_signal >= 3.0:
                    status(idle_message.format(name=name))
                    waiting_for_audio = True

                if mono and not stop.is_set():
                    send_chunk(mono, captured_at)
    except OSError as exc:
        raise RuntimeError(f"无法采集{device_type}「{name}」，请检查 Windows 声音设置和设备权限后重试。") from exc
