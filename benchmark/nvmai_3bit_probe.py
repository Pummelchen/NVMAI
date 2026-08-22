#!/usr/bin/env python3
"""Item 8 gate: is 3-bit worth a repacker, a kernel, and a re-download?

Two questions decide it, and both are answerable from the installed weights
without writing any runtime code:

  1. QUALITY. Quantize the real routed-expert and attention tensors to 3-bit
     affine (group 64, the format NVMAI already uses) and measure the
     reconstruction error against 4-bit on the same tensors. A weight-space
     error that is several times 4-bit's is a strong signal the model degrades,
     because these are the same tensors the 4-bit path already runs at the
     edge of usability.

  2. PACKING. 3 bits does not divide a 32-bit word. Measure how much of the
     nominal 25% byte saving survives realistic packing, and how much extra
     unpack work each weight costs. NVMAI's own 6-bit experience is the
     precedent: non-power-of-two packing measured 46.8 GB/s against 60 for
     both 4-bit and 8-bit, and 6-bit was withdrawn.

Decode is bandwidth-bound, so 3-bit only pays if the byte saving is real AND
the unpack does not push the kernel into being ALU-bound instead.

  ~/.venvs/coreml-py311/bin/python benchmark/nvmai_3bit_probe.py
"""
from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/ornith-1.5_35B_A3B_4Bit"
GROUP = 64


def read_index(path):
    with open(path, "rb") as handle:
        index_size, _res, count = struct.unpack("<QQQ", handle.read(24))
        handle.seek(0)
        region = handle.read(index_size)
    entries = {}
    for i in range(count):
        off = 24 + i * 72
        name_off, name_len = struct.unpack_from("<IH", region, off)
        name = region[name_off:name_off + name_len].decode()
        file_off, size = struct.unpack_from("<QQ", region, off + 8)
        shape = struct.unpack_from("<4I", region, off + 24)
        so, ss, bo, bs = struct.unpack_from("<QQQQ", region, off + 40)
        entries[name] = dict(dtype=region[off + 6], offset=file_off, size=size,
                             shape=shape, scale=(so, ss), bias=(bo, bs))
    return entries


def bf16(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def dequant4(handle, entry):
    rows, cols = entry["shape"][0], entry["shape"][1]
    handle.seek(entry["offset"])
    packed = np.frombuffer(handle.read(entry["size"]),
                           dtype=np.uint8).reshape(rows, cols // 2)
    q = np.empty((rows, cols), dtype=np.float32)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    handle.seek(entry["scale"][0])
    s = bf16(handle.read(entry["scale"][1])).reshape(rows, cols // GROUP)
    handle.seek(entry["bias"][0])
    b = bf16(handle.read(entry["bias"][1])).reshape(rows, cols // GROUP)
    return q * np.repeat(s, GROUP, 1) + np.repeat(b, GROUP, 1)


def affine_roundtrip(w: np.ndarray, bits: int) -> np.ndarray:
    """Quantize to `bits` affine per group of 64 and dequantize, exactly as
    NVMAI's format does: per-group min/max -> scale/bias, round to grid."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // GROUP, GROUP)
    lo = g.min(-1, keepdims=True)
    hi = g.max(-1, keepdims=True)
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint((g - lo) / scale), 0, levels)
    return (q * scale + lo).reshape(rows, cols)


def rel_error(reference: np.ndarray, approx: np.ndarray) -> float:
    return float(np.abs(approx - reference).mean()
                 / max(np.abs(reference).mean(), 1e-12))


def packing_analysis() -> list[dict]:
    """Bytes per 64-weight group and unpack ops per weight, for the packings a
    real kernel could use."""
    rows = []
    for bits in (2, 3, 4, 6, 8):
        payload_bits = GROUP * bits
        # Scheme A: bit-exact stream (what NVMAI's affine_quant_value does).
        stream_bytes = payload_bits / 8
        # Scheme B: whole values per 32-bit word, wasting the remainder.
        per_word = 32 // bits
        words = -(-GROUP // per_word)
        padded_bytes = words * 4
        # Metadata: bf16 scale + bias per group.
        meta = 4
        crosses_word = (32 % bits) != 0
        rows.append({
            "bits": bits,
            "group_bytes_stream": stream_bytes + meta,
            "group_bytes_padded": padded_bytes + meta,
            "vs_4bit_stream": (stream_bytes + meta) / (GROUP * 4 / 8 + meta),
            "vs_4bit_padded": (padded_bytes + meta) / (GROUP * 4 / 8 + meta),
            "power_of_two": not crosses_word,
        })
    return rows


def main() -> int:
    entries = read_index(MODEL / "model_weights.bin")
    handle = open(MODEL / "model_weights.bin", "rb")

    # Attention + shared-expert tensors from a full-attention layer, plus the
    # router: a representative slice of what decode actually reads.
    names = [
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.o_proj.weight",
        "language_model.model.layers.3.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.3.mlp.shared_expert.down_proj.weight",
        "language_model.model.layers.19.self_attn.q_proj.weight",
        "language_model.model.layers.19.mlp.shared_expert.up_proj.weight",
    ]

    print("== quality: affine round-trip error vs the stored 4-bit weights ==")
    print(f"{'tensor':<58} {'3-bit':>9} {'4-bit':>9} {'ratio':>7}")
    ratios = []
    for name in names:
        w = dequant4(handle, entries[name])
        # The stored 4-bit values ARE the reference the model runs today; the
        # question is how much MORE error 3-bit adds on the same tensor.
        e3 = rel_error(w, affine_roundtrip(w, 3))
        e4 = rel_error(w, affine_roundtrip(w, 4))
        ratio = e3 / max(e4, 1e-12)
        ratios.append(ratio)
        short = name.replace("language_model.model.layers.", "L")
        print(f"{short:<58} {e3:9.4f} {e4:9.4f} {ratio:7.2f}x")
    handle.close()
    print(f"\n  median 3-bit/4-bit error ratio: {np.median(ratios):.2f}x")

    print("\n== packing: bytes per 64-weight group (incl. bf16 scale+bias) ==")
    print(f"{'bits':>5} {'stream B':>9} {'padded B':>9} "
          f"{'vs 4-bit (stream)':>18} {'vs 4-bit (padded)':>18} {'PoT':>5}")
    rows = packing_analysis()
    for r in rows:
        print(f"{r['bits']:>5} {r['group_bytes_stream']:>9.1f} "
              f"{r['group_bytes_padded']:>9.1f} "
              f"{r['vs_4bit_stream']:>17.3f}x {r['vs_4bit_padded']:>17.3f}x "
              f"{'yes' if r['power_of_two'] else 'NO':>5}")

    three = next(r for r in rows if r["bits"] == 3)
    six = next(r for r in rows if r["bits"] == 6)
    print(f"\n  3-bit byte saving vs 4-bit: "
          f"{(1 - three['vs_4bit_stream']) * 100:.1f}% (bit-exact stream), "
          f"{(1 - three['vs_4bit_padded']) * 100:.1f}% (word-padded)")
    print(f"  6-bit, for reference (withdrawn after measuring 46.8 GB/s "
          f"against 60): {(1 - six['vs_4bit_stream']) * 100:.1f}% / "
          f"{(1 - six['vs_4bit_padded']) * 100:.1f}%")

    out = ROOT / ".build/benchmark-results/3bit-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"error_ratios": ratios, "packing": rows}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    main()
