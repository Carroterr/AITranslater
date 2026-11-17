import json
from dataclasses import replace
from threading import Event
from time import monotonic

import ctranslate2
from opencc import OpenCC
import sentencepiece as spm

from model_assets import ModelAssets


TRANSLATION_MODEL = ModelAssets(
    "本地翻译模型",
    "ishiki-emo/mojicast-m2m100-ct2",
    "0ea75b31e8fe07a27ba6c2b9b1b1a279e7f44b93",
    {
        "config.json": (233, "5875dd832fcb028f0f3b82c552832c5187d45d2a"),
        "shared_vocabulary.json": (2924622, "cdebc788bf6329b2c86d59ca82b57d9c77bc6912"),
        "sentencepiece.model": (2423393, "d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a"),
        "LICENSE": (1327, "8e4180d52dbc0b286457bb83eea59635f2833c00"),
        "model.bin": (490667752, "a1826980fc5c037e69c7ac94fcb56c03001a66f380eb71863cc0a3879e71421b"),
    },
)
TOKENIZER_ASSETS = replace(
    TRANSLATION_MODEL,
    label="翻译词表",
    files={name: spec for name, spec in TRANSLATION_MODEL.files.items() if name != "model.bin"},
)


class LocalTranslator:
    def __init__(self, source, target, stop, status, gpu_first=True):
        self.source = source
        self.target = target.split("-", 1)[0]
        self.stop = stop
        self.status = status
        self.gpu = None
        self.backend = "本地转写"
        conversion = {"zh-Hans": "t2s", "zh-Hant": "s2t"}.get(target)
        self.chinese = OpenCC(conversion) if conversion else None
        if self.source == self.target:
            return

        TOKENIZER_ASSETS.prepare(stop, status)
        if stop.is_set():
            return
        model_dir = TRANSLATION_MODEL.directory
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(model_dir / "sentencepiece.model"))
        if gpu_first:
            try:
                from gpu_translation import DirectMLTranslator, find_gpu

                adapter = find_gpu()
                if adapter is not None:
                    vocabulary = json.loads((model_dir / "shared_vocabulary.json").read_text(encoding="utf-8"))
                    gpu = DirectMLTranslator(adapter, vocabulary, stop, status)
                    if stop.is_set():
                        return
                    sample = ["__en__", *self.tokenizer.encode("Hello world.", out_type=str), "</s>"]
                    check = self.tokenizer.decode(gpu.translate(sample, "__zh__", 16))
                    if "世界" not in check:
                        raise RuntimeError("GPU 预热结果异常")
                    self.gpu = gpu
                    self.backend = f"GPU · {gpu.name}"
            except Exception as exc:
                status(f"GPU 翻译不可用，切换至 CPU。{str(exc).splitlines()[0][:120] if str(exc) else ''}")
        if self.gpu is None:
            self._load_cpu()

    def _load_cpu(self):
        TRANSLATION_MODEL.prepare(self.stop, self.status)
        if self.stop.is_set():
            return
        self.backend = "CPU · INT8"
        self.status("正在加载 CPU 翻译模型…")
        self.model = ctranslate2.Translator(
            str(TRANSLATION_MODEL.directory), device="cpu", compute_type="int8", intra_threads=4,
        )

    def translate(self, text, progress=None, *, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            return ""
        if self.source != self.target:
            if self.source == "en" and text.isupper():
                text = text.lower()
            tokens = [f"__{self.source}__", *self.tokenizer.encode(text, out_type=str), "</s>"]
            target = f"__{self.target}__"
            max_length = min(320, max(32, len(tokens) * 3 + 12))
            pieces = []
            next_refresh = 0.0

            def on_token(piece):
                nonlocal next_refresh
                if self.stop.is_set() or (cancelled is not None and cancelled.is_set()):
                    return True
                if progress is not None and piece not in (target, "</s>"):
                    pieces.append(piece)
                    now = monotonic()
                    if now >= next_refresh:
                        partial = self.tokenizer.decode(pieces).strip()
                        if len(partial) >= 4:
                            progress(self._script(partial))
                            next_refresh = now + 0.12
                return False

            if self.gpu is not None:
                try:
                    output = self.gpu.translate(tokens, target, max_length, on_token)
                    return self._script(self.tokenizer.decode(output).strip())
                except Exception:
                    self.gpu = None
                    pieces.clear()
                    next_refresh = 0.0
                    self.status("GPU 推理中断，切换至 CPU…")
                    self._load_cpu()
            if self.stop.is_set() or (cancelled is not None and cancelled.is_set()):
                return ""
            result = self.model.translate_batch(
                [tokens], target_prefix=[[target]], beam_size=1, max_decoding_length=max_length,
                callback=lambda step: on_token(step.token),
            )
            text = self.tokenizer.decode(result[0].hypotheses[0][1:]).strip()
        return self._script(text)

    def _script(self, text):
        return self.chinese.convert(text) if self.chinese else text


if __name__ == "__main__":
    TRANSLATION_MODEL.prepare(Event(), print)
    print(f"本地翻译模型已就绪 · {TRANSLATION_MODEL.directory}")
