"""Read tensors out of a .gturbo `model_weights.bin`.

Exists for parity work: to check the runtime's architecture against a
reference implementation, the reference has to run on the *same* weights.
Loading them from the install rather than re-fetching the checkpoint keeps
quantization out of the comparison, so a mismatch is a bug in the forward
pass and not in the repack.

Layout, from `NVMAIFormat/GTurboResidentIndexV1.swift`:
  header   24 B  = indexSize, residentSize, entryCount (all u64 LE)
  entries  72 B each, then a UTF-8 string table, then the payload
  entry    = nameOffset u32, nameLen u16, dtype u8, reserved u8,
             fileOffset u64, sizeBytes u64, shape 4 x u32,
             scaleOffset u64, scaleSize u64, biasOffset u64, biasSize u64

INT4 tensors are affine over groups of 64 along the row: a byte holds two
values, low nibble first, and `value = q * scale + bias` with bf16 scale and
bias per group.
"""
import json
import struct
from pathlib import Path

import numpy as np

HEADER_BYTES = 24
ENTRY_BYTES = 72
GROUP_SIZE = 64
DTYPE_U32, DTYPE_BF16, DTYPE_FP16, DTYPE_FP32 = 0, 1, 2, 3


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    wide = np.zeros(raw.shape, dtype=np.uint32)
    wide |= raw.astype(np.uint32) << 16
    return wide.view(np.float32)


class GTurboWeights:
    def __init__(self, directory):
        self.root = Path(directory)
        self.path = self.root / "model_weights.bin"
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        with self.path.open("rb") as handle:
            index_size, resident_size, count = struct.unpack(
                "<QQQ", handle.read(HEADER_BYTES))
            handle.seek(0)
            index = handle.read(index_size)
        self.index_size = index_size
        self.resident_size = resident_size
        self.entries = {}
        for i in range(count):
            base = HEADER_BYTES + i * ENTRY_BYTES
            (name_off, name_len, dtype, _res, file_off, size,
             s0, s1, s2, s3,
             scale_off, scale_size, bias_off, bias_size) = struct.unpack_from(
                "<IHBBQQIIIIQQQQ", index, base)
            name = index[name_off:name_off + name_len].decode("utf-8")
            shape = tuple(d for d in (s0, s1, s2, s3) if d != 0)
            self.entries[name] = dict(
                dtype=dtype, offset=file_off, size=size, shape=shape,
                scale_offset=scale_off, scale_size=scale_size,
                bias_offset=bias_off, bias_size=bias_size)
        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r")

    def names(self, contains=""):
        return sorted(n for n in self.entries if contains in n)

    def _raw(self, offset, size):
        return self._mmap[offset:offset + size]

    def get(self, name) -> np.ndarray:
        """Dequantized float32 tensor in its declared shape."""
        e = self.entries[name]
        raw = self._raw(e["offset"], e["size"])
        if e["dtype"] == DTYPE_BF16:
            values = _bf16_to_f32(raw.view(np.uint16))
            return values.reshape(e["shape"]) if e["shape"] else values
        if e["dtype"] == DTYPE_FP16:
            return raw.view(np.float16).astype(np.float32).reshape(e["shape"])
        if e["dtype"] == DTYPE_FP32:
            return raw.view(np.float32).reshape(e["shape"])
        if e["dtype"] != DTYPE_U32:
            raise ValueError(f"{name}: unhandled dtype {e['dtype']}")
        if e["scale_size"] == 0:
            return raw.view(np.uint32).reshape(e["shape"])
        rows, cols = e["shape"][0], e["shape"][1]
        groups = cols // GROUP_SIZE
        # Bit width is not recorded per tensor; the payload size states it.
        # 8-bit slots (embedding, router) store one byte per value, 4-bit
        # slots pack two.
        row_bytes = e["size"] // rows
        if row_bytes == cols:
            q = raw.reshape(rows, cols)
        elif row_bytes == cols // 2:
            packed = raw.reshape(rows, cols // 2)
            q = np.empty((rows, cols), dtype=np.uint8)
            q[:, 0::2] = packed & 0x0F
            q[:, 1::2] = packed >> 4
        else:
            raise ValueError(
                f"{name}: {row_bytes} bytes for {cols} values is neither "
                "4-bit nor 8-bit")
        scales = _bf16_to_f32(
            self._raw(e["scale_offset"], e["scale_size"]).view(np.uint16)
        ).reshape(rows, groups)
        biases = _bf16_to_f32(
            self._raw(e["bias_offset"], e["bias_size"]).view(np.uint16)
        ).reshape(rows, groups)
        out = q.astype(np.float32).reshape(rows, groups, GROUP_SIZE)
        out = out * scales[:, :, None] + biases[:, :, None]
        return out.reshape(rows, cols)


class PackedExperts:
    """Routed-expert weights, read out of `packed_experts/layer_NN.bin`.

    The experts are not in the resident index -- they are the streamed part of
    the model -- so parity work reads them the same way the runtime does:
    through the layout's per-expert offsets.
    """

    def __init__(self, directory):
        self.root = Path(directory) / "packed_experts"
        self.layout = json.loads((self.root / "layout.json").read_text())
        self._maps = {}

    def _map(self, layer):
        if layer not in self._maps:
            path = self.root / f"layer_{layer:02d}.bin"
            self._maps[layer] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._maps[layer]

    def tensor(self, layer: int, expert: int, name: str) -> np.ndarray:
        record = self.layout["layers"][layer]["experts"][expert]
        base = record["offset"]
        t = record["tensors"][name]
        blob = self._map(layer)
        rows, cols = t["shape"]

        def take(key):
            spec = record["tensors"][key]
            start = base + spec["offset"]
            return blob[start:start + spec["size"]]

        packed = take(name).reshape(rows, cols // 2)
        q = np.empty((rows, cols), dtype=np.uint8)
        q[:, 0::2] = packed & 0x0F
        q[:, 1::2] = packed >> 4
        groups = cols // GROUP_SIZE
        scales = _bf16_to_f32(take(f"{name}_scales").view(np.uint16)).reshape(rows, groups)
        biases = _bf16_to_f32(take(f"{name}_biases").view(np.uint16)).reshape(rows, groups)
        out = q.astype(np.float32).reshape(rows, groups, GROUP_SIZE)
        return (out * scales[:, :, None] + biases[:, :, None]).reshape(rows, cols)
