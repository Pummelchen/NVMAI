#!/usr/bin/env python3
"""Track A rehearsal: the 10 real full-attention layers on the ANE.

Reads the actual int4 affine weights (group-64, bf16 scales/biases) of every
full-attention layer out of the installed 4-bit `.gturbo`, dequantizes them to
fp16, bakes each layer into the probe's Core ML block (decomposed attention —
the fused SDPA op NaNs on this ANE from sequence 2048), and replays the exact
layer-chunk sequence of a 6,103-token prefill: chunk 4096 with no history,
then chunk 2007 against 4096 tokens of history, for each of the 10 layers.

Reports per-chunk latency, the 20-chunk total against the measured GPU
reference (84.3 s on this machine, same prompt shape), and per-layer numerics
against a float32 NumPy reference of the same math with the same real weights.

  ~/.venvs/coreml-py311/bin/python benchmark/nvmai_ane_realweight_rehearsal.py
"""
from __future__ import annotations

import json
import pathlib
import struct
import time

import numpy as np

import nvmai_ane_attention_probe as probe

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_BIN = str(ROOT / "models/ornith-1.5_35B_A3B_4Bit/model_weights.bin")
FULL_LAYERS = list(range(3, 40, 4))          # (i + 1) % 4 == 0
CHUNKS = [(4096, 0), (2007, 4096)]           # the 6,103-token prefill
GPU_REFERENCE_S = 84.3                       # measured, same machine, same shape


def read_index(path: str) -> dict[str, dict]:
    with open(path, "rb") as handle:
        index_size, _resident, entry_count = struct.unpack("<QQQ", handle.read(24))
        handle.seek(0)
        region = handle.read(index_size)
    entries: dict[str, dict] = {}
    for i in range(entry_count):
        off = 24 + i * 72
        name_off, name_len = struct.unpack_from("<IH", region, off)
        name = region[name_off:name_off + name_len].decode()
        file_off, size = struct.unpack_from("<QQ", region, off + 8)
        shape = struct.unpack_from("<4I", region, off + 24)
        scale_off, scale_size, bias_off, bias_size = struct.unpack_from(
            "<QQQQ", region, off + 40)
        entries[name] = dict(dtype=region[off + 6], offset=file_off, size=size,
                             shape=shape, scale=(scale_off, scale_size),
                             bias=(bias_off, bias_size))
    return entries


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def load_tensor(handle, entry) -> np.ndarray:
    rows, cols = entry["shape"][0], entry["shape"][1]
    if entry["dtype"] == 1:                                   # bf16
        handle.seek(entry["offset"])
        flat = bf16_to_f32(handle.read(entry["size"]))
        return flat.reshape([d for d in entry["shape"] if d] or [flat.size])
    # int4 affine, group 64, per-row nibble packing (element e of a row is
    # bits e*4.. of that row's little-endian u32 word stream: even elements
    # take the low nibble of byte e//2, odd elements the high nibble).
    handle.seek(entry["offset"])
    packed = np.frombuffer(handle.read(entry["size"]), dtype=np.uint8)
    packed = packed.reshape(rows, cols // 2)
    q = np.empty((rows, cols), dtype=np.float32)
    q[:, 0::2] = (packed & 0x0F).astype(np.float32)
    q[:, 1::2] = (packed >> 4).astype(np.float32)
    handle.seek(entry["scale"][0])
    scales = bf16_to_f32(handle.read(entry["scale"][1])).reshape(rows, cols // 64)
    handle.seek(entry["bias"][0])
    biases = bf16_to_f32(handle.read(entry["bias"][1])).reshape(rows, cols // 64)
    scale_full = np.repeat(scales, 64, axis=1)
    bias_full = np.repeat(biases, 64, axis=1)
    return q * scale_full + bias_full


def load_layer_weights(handle, entries, layer: int) -> dict[str, np.ndarray]:
    prefix = f"language_model.model.layers.{layer}.self_attn."
    def get(name):
        return load_tensor(handle, entries[prefix + name])
    return {
        "wq": get("q_proj.weight").astype(np.float16),
        "wk": get("k_proj.weight").astype(np.float16),
        "wv": get("v_proj.weight").astype(np.float16),
        "wo": get("o_proj.weight").astype(np.float16),
        "q_norm": get("q_norm.weight").astype(np.float16),
        "k_norm": get("k_norm.weight").astype(np.float16),
    }


def main() -> int:
    entries = read_index(MODEL_BIN)
    rng = np.random.default_rng(41)
    results = []
    total_predict_s = 0.0
    handle = open(MODEL_BIN, "rb")
    for layer in FULL_LAYERS:
        weights = load_layer_weights(handle, entries, layer)
        wstd = float(weights["wq"].astype(np.float32).std())
        layer_row = {"layer": layer, "wq_std": round(wstd, 5), "chunks": []}
        for t, hist in CHUNKS:
            model = probe.build_block(t, hist, weights)
            hidden = (rng.standard_normal((t, probe.D)) * 0.5).astype(np.float16)
            k_hist = (rng.standard_normal(
                (1, probe.N_KV_HEADS, hist, probe.HEAD_DIM)) * 0.5).astype(np.float16)
            v_hist = (rng.standard_normal(
                (1, probe.N_KV_HEADS, hist, probe.HEAD_DIM)) * 0.5).astype(np.float16)
            cos_t, sin_t = probe.rope_tables(hist, t)
            mask = probe.causal_mask(t, hist)
            feed = {"hidden": hidden, "cos_t": cos_t, "sin_t": sin_t, "mask": mask}
            if hist:
                feed.update(k_hist=k_hist, v_hist=v_hist)
            spec = model.get_spec()
            names = [o.name for o in spec.description.output]
            out_name = next(n for n in names if tuple(
                spec.description.output[names.index(n)]
                .type.multiArrayType.shape) == (t, probe.D))
            # Warmup once (first predict pays one-time setup), then time.
            model.predict(feed)
            times = []
            for _ in range(3):
                start = time.perf_counter()
                out = model.predict(feed)
                times.append(time.perf_counter() - start)
            elapsed = sorted(times)[1]
            total_predict_s += elapsed
            ref = probe.reference(hidden, k_hist, v_hist, cos_t, sin_t,
                                  mask, weights)
            got = np.asarray(out[out_name], dtype=np.float32)
            rel = float(np.abs(got - ref).mean() / max(np.abs(ref).mean(), 1e-9))
            bad = int(np.isnan(got).sum() + np.isinf(got).sum())
            layer_row["chunks"].append(
                {"chunk": t, "history": hist,
                 "ane_ms": round(elapsed * 1000, 1),
                 "rel_err": round(rel, 5), "nan_inf": bad})
            print(f"layer {layer:2d} chunk {t}:{hist}  "
                  f"{elapsed * 1000:8.1f} ms  rel {rel:.4f}  nan/inf {bad}",
                  flush=True)
            del model
        results.append(layer_row)
    handle.close()

    print("\n" + "=" * 66)
    print("REAL-WEIGHT ANE REHEARSAL — 10 full-attention layers, "
          "6,103-token prefill shape")
    print("=" * 66)
    print(f"  ANE, 20 layer-chunks:  {total_predict_s:8.2f} s "
          f"(prediction wall, marshaling included)")
    print(f"  GPU, same layer-chunks: {GPU_REFERENCE_S:7.1f} s (measured)")
    print(f"  speedup on the offloadable block: "
          f"{GPU_REFERENCE_S / total_predict_s:.1f}x")
    remainder = 133.2 - GPU_REFERENCE_S
    projected = remainder + total_predict_s
    print(f"  projected end-to-end prefill: 133.2 s -> "
          f"{projected:.1f} s ({133.2 / projected:.2f}x)")
    worst = max(c["rel_err"] for r in results for c in r["chunks"])
    bad = sum(c["nan_inf"] for r in results for c in r["chunks"])
    print(f"  worst per-layer rel err vs fp32 reference: {worst:.4f}   "
          f"total nan/inf: {bad}")
    with open(ROOT / ".build/benchmark-results/ane-realweight-rehearsal.json",
              "w") as fh:
        json.dump({"results": results,
                   "ane_total_s": round(total_predict_s, 3),
                   "gpu_reference_s": GPU_REFERENCE_S}, fh, indent=2)
    print("wrote .build/benchmark-results/ane-realweight-rehearsal.json")
    return 0


if __name__ == "__main__":
    main()
