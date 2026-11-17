import ctypes as ct
from itertools import count
from uuid import UUID

import numpy as np
import onnxruntime as ort

from model_assets import ModelAssets


ENCODER = "onnx/encoder_model.onnx"
DECODER = "onnx/decoder_model_merged.onnx"
GPU_MODEL = ModelAssets(
    "GPU 翻译模型",
    "Xenova/m2m100_418M",
    "9c374f0b7aca709787cea97b047bfbbd1559d177",
    {
        ENCODER: (1133698852, "2aaf1612c2738dec7de0a12d3f5ebb1168a38b735a33db5be7719dd669750544"),
        DECODER: (1340271309, "22acc8d593ed1dfa9c45645c1f36ec4b9ed8ce721764509f465d7160eea78453"),
    },
)


class AdapterDescription(ct.Structure):
    _fields_ = [
        ("name", ct.c_wchar * 128),
        ("vendor_id", ct.c_uint32),
        ("device_id", ct.c_uint32),
        ("sub_sys_id", ct.c_uint32),
        ("revision", ct.c_uint32),
        ("video_memory", ct.c_size_t),
        ("system_memory", ct.c_size_t),
        ("shared_memory", ct.c_size_t),
        ("luid", ct.c_uint64),
        ("flags", ct.c_uint32),
    ]


def com_method(instance, slot, result, *args):
    table = ct.cast(instance, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *args)(table[slot])


def find_gpu():
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return None

    factory = ct.c_void_p()
    factory_id = (ct.c_ubyte * 16).from_buffer_copy(UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    create_factory = ct.WinDLL("dxgi").CreateDXGIFactory1
    create_factory.argtypes = (ct.c_void_p, ct.POINTER(ct.c_void_p))
    create_factory.restype = ct.c_long
    if create_factory(factory_id, ct.byref(factory)) < 0:
        return None

    candidates = []
    try:
        # IDXGIFactory1::EnumAdapters1 uses the same indices as DirectML's device_id.
        enumerate_adapter = com_method(factory, 12, ct.c_long, ct.c_uint, ct.POINTER(ct.c_void_p))
        for index in count():
            adapter = ct.c_void_p()
            if enumerate_adapter(factory, index, ct.byref(adapter)) < 0:
                break
            try:
                desc = AdapterDescription()
                describe = com_method(adapter, 10, ct.c_long, ct.POINTER(AdapterDescription))
                if describe(adapter, ct.byref(desc)) >= 0 and desc.vendor_id in (0x1002, 0x10DE) and not desc.flags & 2:
                    candidates.append((desc.video_memory, index, desc.name))
            finally:
                com_method(adapter, 2, ct.c_ulong)(adapter)
    finally:
        com_method(factory, 2, ct.c_ulong)(factory)
    return max(candidates, default=None)


class DirectMLTranslator:
    def __init__(self, adapter, vocab, stop, status):
        _, self.device_id, self.name = adapter
        GPU_MODEL.prepare(stop, status)
        if stop.is_set():
            return
        status(f"正在加载 GPU 翻译模型 · {self.name}")
        self.vocab = vocab
        self.token_ids = {token: i for i, token in enumerate(vocab)}
        self.end = self.token_ids["</s>"]
        self.unknown = self.token_ids["<unk>"]

        options = ort.SessionOptions()
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 2
        providers = [("DmlExecutionProvider", {"device_id": self.device_id}), "CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(GPU_MODEL.directory / ENCODER), options, providers=providers)
        self.decoder = ort.InferenceSession(str(GPU_MODEL.directory / DECODER), options, providers=providers)
        if any("DmlExecutionProvider" not in session.get_providers() for session in (self.encoder, self.decoder)):
            raise RuntimeError("DirectML 加载失败")
        self.cache_inputs = {}
        for node in self.decoder.get_inputs():
            if node.name.startswith("past_key_values."):
                dtype = np.float16 if node.type == "tensor(float16)" else np.float32
                _, heads, _, head_size = node.shape
                self.cache_inputs[node.name] = np.empty((1, heads, 0, head_size), dtype=dtype)
        self.cache_outputs = {name: "present." + name.removeprefix("past_key_values.") for name in self.cache_inputs}
        self.decoder_cache = {name: output for name, output in self.cache_outputs.items() if ".decoder." in name}

    def translate(self, tokens, target, max_length, on_token=None):
        inputs = np.array([[self.token_ids.get(token, self.unknown) for token in tokens]], dtype=np.int64)
        mask = np.ones_like(inputs)
        [hidden] = self.encoder.run(["last_hidden_state"], {"input_ids": inputs, "attention_mask": mask})
        values = {
            "input_ids": np.array([[self.end]], dtype=np.int64),
            "encoder_attention_mask": mask,
            "encoder_hidden_states": hidden,
            "use_cache_branch": np.array([False]),
            **self.cache_inputs,
        }
        result = []
        for step in range(max_length + 1):
            # Prime the cache with the decoder start token before feeding the target language.
            cache_names = self.cache_outputs if step == 0 else self.decoder_cache
            logits, *cache = self.decoder.run(["logits", *cache_names.values()], values)
            token_id = self.token_ids[target] if step == 0 else int(logits[0, -1].argmax())
            if token_id == self.end:
                break
            if step:
                token = self.vocab[token_id]
                result.append(token)
                if on_token is not None and on_token(token):
                    break
            values["input_ids"] = np.array([[token_id]], dtype=np.int64)
            values["use_cache_branch"][0] = True
            values.update(zip(cache_names, cache))
        return result
