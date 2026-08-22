#!/usr/bin/env python3
"""Track A probe: the full-attention prefill block on the Neural Engine.

Builds the complete Qwen3.5-MoE full-attention block — packed QKV projection
with output gate, per-head q/k RMS norms, NeoX-subdim RoPE (64 of 256), GQA
SDPA (16 query heads over 2 KV heads) against a KV history, sigmoid output
gate, and the O projection — as one Core ML MIL program at the real shapes,
runs it with CPU_AND_NE and CPU_ONLY, and reports per-chunk latency.

Reference GPU numbers (measured this session, 6,103-token prefill, 4-bit,
NVMAI_KERNEL_STATS): the 10 full-attention layers cost 84.3 s of the 133.2 s
prefill — 4.21 s per layer-chunk. The go/no-go: the ANE must beat that per
layer-chunk by enough to survive integration overheads.

Weights are random fp16 at the real shapes (throughput does not depend on
values). Numerics are sanity-checked against a float32 NumPy reference of the
same math; exact parity with the Metal kernels' conventions is integration
work, not probe work.

  ~/.venvs/coreml-py311/bin/python benchmark/nvmai_ane_attention_probe.py
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# Qwen3.5-MoE 35B-A3B full-attention geometry (ArchConfig.qwen36_35B_A3B).
D = 2048
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = N_Q_HEADS * HEAD_DIM          # 4096
Q_PROJ_ROWS = 2 * Q_DIM               # packed query+gate
KV_DIM = N_KV_HEADS * HEAD_DIM        # 512
ROTARY = 64                           # headDim * partialRotaryFactor(0.25)
THETA = 10_000_000.0
SCALE = 0.0625                        # 256^-0.5
EPS = 1e-6


def rope_tables(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    half = ROTARY // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / ROTARY)
    pos = np.arange(start, start + count, dtype=np.float64)[:, None] * inv[None, :]
    return (np.cos(pos).astype(np.float16), np.sin(pos).astype(np.float16))


def make_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    def w(rows, cols):
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float16)
    return {
        "wq": w(Q_PROJ_ROWS, D),
        "wk": w(KV_DIM, D),
        "wv": w(KV_DIM, D),
        "wo": w(D, Q_DIM),
        "q_norm": np.abs(rng.standard_normal(HEAD_DIM) * 0.1 + 1).astype(np.float16),
        "k_norm": np.abs(rng.standard_normal(HEAD_DIM) * 0.1 + 1).astype(np.float16),
    }


def build_block(t: int, history: int, weights: dict[str, np.ndarray]):
    """One full-attention block: hidden [t, D] + K/V history -> output [t, D]
    plus the chunk's rotated K and raw V for the cache write.

    history == 0 builds a variant without history inputs: Core ML rejects
    zero-length tensor dimensions on model inputs."""
    total = history + t
    fp16 = ct.converters.mil.mil.types.fp16
    specs = [mb.TensorSpec(shape=(t, D), dtype=fp16)]
    if history > 0:
        specs += [mb.TensorSpec(shape=(1, N_KV_HEADS, history, HEAD_DIM), dtype=fp16),
                  mb.TensorSpec(shape=(1, N_KV_HEADS, history, HEAD_DIM), dtype=fp16)]
    specs += [mb.TensorSpec(shape=(t, ROTARY // 2), dtype=fp16),
              mb.TensorSpec(shape=(t, ROTARY // 2), dtype=fp16),
              mb.TensorSpec(shape=(1, 1, t, total), dtype=fp16)]

    def body(hidden, k_hist, v_hist, cos_t, sin_t, mask):
        def rms_head(x, weight_name, heads):
            # x: [heads, seq, HEAD_DIM] per-head RMS norm with weight.
            sq = mb.mul(x=x, y=x)
            mean = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            denom = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=x, y=denom),
                          y=weights[weight_name].reshape(1, 1, HEAD_DIM))

        def rope(x, heads, seq):
            # NeoX half-split on the first ROTARY dims; passthrough beyond.
            r1 = mb.slice_by_index(x=x, begin=[0, 0, 0], end=[heads, seq, ROTARY // 2],
                                   begin_mask=[True, True, False],
                                   end_mask=[True, True, False])
            r2 = mb.slice_by_index(x=x, begin=[0, 0, ROTARY // 2],
                                   end=[heads, seq, ROTARY],
                                   begin_mask=[True, True, False],
                                   end_mask=[True, True, False])
            rest = mb.slice_by_index(x=x, begin=[0, 0, ROTARY],
                                     end=[heads, seq, HEAD_DIM],
                                     begin_mask=[True, True, False],
                                     end_mask=[True, True, True])
            cos_b = mb.reshape(x=cos_t, shape=[1, seq, ROTARY // 2])
            sin_b = mb.reshape(x=sin_t, shape=[1, seq, ROTARY // 2])
            o1 = mb.sub(x=mb.mul(x=r1, y=cos_b), y=mb.mul(x=r2, y=sin_b))
            o2 = mb.add(x=mb.mul(x=r2, y=cos_b), y=mb.mul(x=r1, y=sin_b))
            return mb.concat(values=[o1, o2, rest], axis=-1)

        # Projections: one matmul each, weights transposed at build time.
        packed = mb.matmul(x=hidden, y=weights["wq"].T)          # [t, 8192]
        k = mb.matmul(x=hidden, y=weights["wk"].T)               # [t, 512]
        v = mb.matmul(x=hidden, y=weights["wv"].T)               # [t, 512]

        # Split packed query+gate: per head, first half query, second gate.
        packed_h = mb.reshape(x=packed, shape=[t, N_Q_HEADS, 2 * HEAD_DIM])
        q = mb.slice_by_index(x=packed_h, begin=[0, 0, 0],
                              end=[t, N_Q_HEADS, HEAD_DIM],
                              begin_mask=[True, True, False],
                              end_mask=[True, True, False])
        gate = mb.slice_by_index(x=packed_h, begin=[0, 0, HEAD_DIM],
                                 end=[t, N_Q_HEADS, 2 * HEAD_DIM],
                                 begin_mask=[True, True, False],
                                 end_mask=[True, True, True])

        q = mb.transpose(x=q, perm=[1, 0, 2])                    # [16, t, 256]
        k_h = mb.transpose(x=mb.reshape(x=k, shape=[t, N_KV_HEADS, HEAD_DIM]),
                           perm=[1, 0, 2])                       # [2, t, 256]
        v_h = mb.transpose(x=mb.reshape(x=v, shape=[t, N_KV_HEADS, HEAD_DIM]),
                           perm=[1, 0, 2])

        q = rms_head(q, "q_norm", N_Q_HEADS)
        k_h = rms_head(k_h, "k_norm", N_KV_HEADS)
        q = rope(q, N_Q_HEADS, t)
        k_h = rope(k_h, N_KV_HEADS, t)

        k_new = mb.reshape(x=k_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        v_new = mb.reshape(x=v_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        if history > 0:
            k_all = mb.concat(values=[k_hist, k_new], axis=2)    # [1,2,total,256]
            v_all = mb.concat(values=[v_hist, v_new], axis=2)
        else:
            k_all, v_all = k_new, v_new

        # GQA: query head h reads KV head h // 8 — expand each KV head into
        # a contiguous block of 8, matching np.repeat on the head axis.
        rep = N_Q_HEADS // N_KV_HEADS

        def gqa_expand(x):
            x5 = mb.reshape(x=x, shape=[1, N_KV_HEADS, 1, total, HEAD_DIM])
            x5 = mb.concat(values=[x5] * rep, axis=2)
            return mb.reshape(x=x5, shape=[1, N_Q_HEADS, total, HEAD_DIM])

        k_g = gqa_expand(k_all)
        v_g = gqa_expand(v_all)

        # Decomposed attention, deliberately NOT the fused
        # scaled_dot_product_attention op: on this M3/macOS the fused op
        # produces NaN/inf on the ANE from sequence length 2048 even at tame
        # score scales (std 0.25), while matmul+softmax+matmul is clean and
        # slightly faster (isolated A/B: rel err inf vs 0.007 at 2048,
        # 58.4 vs 50.0 ms). The explicit scale matches attentionScale.
        q4 = mb.reshape(x=q, shape=[1, N_Q_HEADS, t, HEAD_DIM])
        scores = mb.matmul(x=q4, y=k_g, transpose_y=True)
        scores = mb.mul(x=scores, y=np.float16(SCALE))
        scores = mb.add(x=scores, y=mask)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=v_g)                         # [1,16,t,256]

        gated = mb.mul(x=mb.transpose(x=attn, perm=[0, 2, 1, 3]),
                       y=mb.sigmoid(x=mb.reshape(
                           x=gate, shape=[1, t, N_Q_HEADS, HEAD_DIM])))
        merged = mb.reshape(x=gated, shape=[t, Q_DIM])
        out = mb.matmul(x=merged, y=weights["wo"].T)             # [t, D]
        return out, k_new, v_new

    if history > 0:
        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(hidden, k_hist, v_hist, cos_t, sin_t, mask):
            return body(hidden, k_hist, v_hist, cos_t, sin_t, mask)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(hidden, cos_t, sin_t, mask):
            return body(hidden, None, None, cos_t, sin_t, mask)

    return ct.convert(prog, convert_to="mlprogram",
                      minimum_deployment_target=ct.target.iOS18,
                      compute_precision=ct.precision.FLOAT16,
                      compute_units=ct.ComputeUnit.CPU_AND_NE)


def causal_mask(t: int, history: int) -> np.ndarray:
    total = history + t
    mask = np.zeros((1, 1, t, total), dtype=np.float16)
    for i in range(t):
        mask[0, 0, i, history + i + 1:] = np.float16(-np.inf)
    return mask


def reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, w):
    """Float32 NumPy of the same math, for a numerics sanity check."""
    h = hidden.astype(np.float32)
    packed = h @ w["wq"].T.astype(np.float32)
    k = h @ w["wk"].T.astype(np.float32)
    v = h @ w["wv"].T.astype(np.float32)
    t = h.shape[0]
    packed = packed.reshape(t, N_Q_HEADS, 2 * HEAD_DIM)
    q, gate = packed[..., :HEAD_DIM], packed[..., HEAD_DIM:]

    def rms(x, weight):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) \
            * weight.astype(np.float32)

    def rope(x, cos_t, sin_t):
        r1, r2 = x[..., :ROTARY // 2], x[..., ROTARY // 2:ROTARY]
        c = cos_t.astype(np.float32)[:, None, :]
        s = sin_t.astype(np.float32)[:, None, :]
        return np.concatenate([r1 * c - r2 * s, r2 * c + r1 * s,
                               x[..., ROTARY:]], -1)

    q = rope(rms(q, w["q_norm"]), cos_t, sin_t).transpose(1, 0, 2)
    kh = k.reshape(t, N_KV_HEADS, HEAD_DIM)
    kh = rope(rms(kh, w["k_norm"]), cos_t, sin_t).transpose(1, 0, 2)
    vh = v.reshape(t, N_KV_HEADS, HEAD_DIM).transpose(1, 0, 2)
    k_all = np.concatenate([k_hist[0].astype(np.float32), kh], 1)
    v_all = np.concatenate([v_hist[0].astype(np.float32), vh], 1)
    rep = N_Q_HEADS // N_KV_HEADS
    k_g = np.repeat(k_all, rep, axis=0)
    v_g = np.repeat(v_all, rep, axis=0)
    scores = q * SCALE @ k_g.transpose(0, 2, 1) + mask[0, 0].astype(np.float32)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    attn = (p @ v_g).transpose(1, 0, 2)
    out = (attn * (1 / (1 + np.exp(-gate)))).reshape(t, Q_DIM)
    return out @ w["wo"].T.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="1024:0,2048:0,4096:0,2048:4096",
                        help="comma list of chunk:history")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(41)
    weights = make_weights(rng)
    results = []
    for spec in args.configs.split(","):
        t, hist = (int(x) for x in spec.split(":"))
        print(f"== chunk {t}, history {hist} ==", flush=True)
        model = build_block(t, hist, weights)
        cpu_model = None
        hidden = (rng.standard_normal((t, D)) * 0.5).astype(np.float16)
        k_hist = (rng.standard_normal((1, N_KV_HEADS, hist, HEAD_DIM)) * 0.5
                  ).astype(np.float16)
        v_hist = (rng.standard_normal((1, N_KV_HEADS, hist, HEAD_DIM)) * 0.5
                  ).astype(np.float16)
        cos_t, sin_t = rope_tables(hist, t)
        mask = causal_mask(t, hist)
        feed = {"hidden": hidden, "cos_t": cos_t, "sin_t": sin_t, "mask": mask}
        if hist > 0:
            feed.update(k_hist=k_hist, v_hist=v_hist)

        # Outputs are (block output [t, D], k_new, v_new) with generated
        # names; identify the block output by shape, not position.
        out_name = next(
            name for name in model.output_description
            if tuple(model.get_spec().description.output[
                [o.name for o in model.get_spec().description.output].index(name)
            ].type.multiArrayType.shape) == (t, D))
        ane_times = []
        for i in range(args.repeats + 2):
            start = time.perf_counter()
            out = model.predict(feed)
            elapsed = time.perf_counter() - start
            if i >= 2:
                ane_times.append(elapsed)
        ane_ms = sorted(ane_times)[len(ane_times) // 2] * 1000

        ref = reference(hidden, k_hist, v_hist, cos_t, sin_t, mask, weights)
        got = np.asarray(out[out_name], dtype=np.float32)
        denom = np.abs(ref).mean()
        rel = np.abs(got - ref).mean() / max(denom, 1e-9)

        cpu_ms = None
        if t <= 2048:
            cpu_model = ct.models.MLModel(
                model.get_spec(), weights_dir=model.weights_dir,
                compute_units=ct.ComputeUnit.CPU_ONLY)
            times = []
            for i in range(3 + 1):
                start = time.perf_counter()
                cpu_model.predict(feed)
                elapsed = time.perf_counter() - start
                if i >= 1:
                    times.append(elapsed)
            cpu_ms = sorted(times)[len(times) // 2] * 1000

        row = {"chunk": t, "history": hist, "cpu_and_ne_ms": round(ane_ms, 2),
               "cpu_only_ms": round(cpu_ms, 2) if cpu_ms else None,
               "mean_rel_error": float(f"{rel:.5f}")}
        results.append(row)
        print(f"  CPU_AND_NE {ane_ms:9.2f} ms/chunk-layer"
              + (f"   CPU_ONLY {cpu_ms:9.2f} ms  (ratio {cpu_ms / ane_ms:.2f}x)"
                 if cpu_ms else "")
              + f"   mean rel err {rel:.4f}", flush=True)

    print("\nGPU reference (measured, 4-bit, this machine): full-attention "
          "block = 4,215 ms per layer-chunk averaged over a 6,103-token "
          "prefill (84.3 s / 20 layer-chunks).")
    with open(".build/benchmark-results/ane-attention-probe.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote .build/benchmark-results/ane-attention-probe.json")
    return 0


if __name__ == "__main__":
    main()
