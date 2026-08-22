#!/usr/bin/env python3
"""Export the ANE prefill attention sidecar for a `.gturbo` model.

Produces `<model>/ane_prefill/layer_<L>.mlpackage` for every full-attention
layer: a multifunction Core ML program whose functions `h0, h4096, ...` share
one set of fp16 weights (dequantized from the model's int4 affine tensors)
and differ only in how much KV history they attend to. Chunk width is fixed
at 4096 — the production prefill chunk — with the causal mask and NeoX RoPE
tables built in-graph, so the Swift runtime feeds only the normed hidden
chunk and the token-major fp16 K/V history.

Why these choices (all measured, see docs/v4.4-decode-width-plan.md Track A):
- decomposed attention, never the fused SDPA op — the fused op produces
  NaN/inf on this M3's ANE from sequence length 2048;
- fp16 weights — they amortize over 4,096-token chunks, so quantized palettes
  buy nothing at prefill; the sidecar is ~52 MB per layer;
- fixed enumerated shapes — chunk boundaries in this runtime are always
  multiples of 4096, so history is too, and fixed shapes keep the ANE
  scheduler on the fast path;
- additive -30000 mask instead of -inf — exp() underflows identically and
  fp16 infinity arithmetic stays out of the graph.

  ~/.venvs/coreml-py311/bin/python tools/export_ane_prefill.py \
      --model models/ornith-1.5_35B_A3B_4Bit --max-history 12288
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import struct
import sys

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

# Qwen3.5-MoE 35B-A3B full-attention geometry.
D = 2048
N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = N_Q_HEADS * HEAD_DIM
KV_DIM = N_KV_HEADS * HEAD_DIM
ROTARY = 64
THETA = 10_000_000.0
SCALE = 0.0625
EPS = 1e-6
CHUNK = 4096
NEG = -30000.0
FULL_LAYERS = list(range(3, 40, 4))
EXPORT_VERSION = 1


def read_index(path: pathlib.Path) -> dict[str, dict]:
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


def load_tensor(handle, entry, weight_bits: int = 4) -> np.ndarray:
    rows, cols = entry["shape"][0], entry["shape"][1]
    if entry["dtype"] == 1:                                   # bf16
        handle.seek(entry["offset"])
        flat = bf16_to_f32(handle.read(entry["size"]))
        return flat.reshape([d for d in entry["shape"] if d] or [flat.size])
    handle.seek(entry["offset"])
    packed = np.frombuffer(handle.read(entry["size"]), dtype=np.uint8)
    if weight_bits == 8:
        q = packed.reshape(rows, cols).astype(np.float32)
    elif weight_bits == 4:
        packed = packed.reshape(rows, cols // 2)
        q = np.empty((rows, cols), dtype=np.float32)
        q[:, 0::2] = (packed & 0x0F).astype(np.float32)
        q[:, 1::2] = (packed >> 4).astype(np.float32)
    else:
        raise SystemExit(f"unsupported attention weightBits {weight_bits}")
    handle.seek(entry["scale"][0])
    scales = bf16_to_f32(handle.read(entry["scale"][1])).reshape(rows, cols // 64)
    handle.seek(entry["bias"][0])
    biases = bf16_to_f32(handle.read(entry["bias"][1])).reshape(rows, cols // 64)
    return q * np.repeat(scales, 64, axis=1) + np.repeat(biases, 64, axis=1)


def load_layer_weights(handle, entries, layer: int,
                       weight_bits: int = 4) -> dict[str, np.ndarray]:
    prefix = f"language_model.model.layers.{layer}.self_attn."
    def get(name):
        return load_tensor(handle, entries[prefix + name],
                           weight_bits=weight_bits)
    return {
        "wq": get("q_proj.weight").astype(np.float16),
        "wk": get("k_proj.weight").astype(np.float16),
        "wv": get("v_proj.weight").astype(np.float16),
        "wo": get("o_proj.weight").astype(np.float16),
        "q_norm": get("q_norm.weight").astype(np.float16),
        "k_norm": get("k_norm.weight").astype(np.float16),
    }


def rope_tables(start: int) -> tuple[np.ndarray, np.ndarray]:
    half = ROTARY // 2
    inv = THETA ** (-np.arange(half, dtype=np.float64) * 2 / ROTARY)
    pos = np.arange(start, start + CHUNK, dtype=np.float64)[:, None] * inv[None, :]
    return (np.cos(pos).astype(np.float16), np.sin(pos).astype(np.float16))


def build_variant(history: int, weights: dict[str, np.ndarray]):
    """One (chunk=4096, history) function. Inputs are token-major so the
    runtime can wrap its staging buffers zero-copy:
      normed  [4096, 2048]  post-input-norm hidden
      k_hist  [H, 512]      rotated+normed K rows already in the cache
      v_hist  [H, 512]
    Outputs, token-major for the same reason:
      out     [4096, 2048]  attention branch output (pre-residual)
      k_new   [4096, 512]   rotated+normed K of this chunk (cache layout)
      v_new   [4096, 512]
    """
    t = CHUNK
    total = history + t
    cos_np, sin_np = rope_tables(history)
    fp16 = ct.converters.mil.mil.types.fp16
    specs = [mb.TensorSpec(shape=(t, D), dtype=fp16)]
    if history > 0:
        specs += [mb.TensorSpec(shape=(history, KV_DIM), dtype=fp16),
                  mb.TensorSpec(shape=(history, KV_DIM), dtype=fp16)]
    # The causal mask is an input, not a baked or generated constant: MIL
    # const-folds any constant-shaped fill/band_part chain, and a folded
    # [4096, 8192] fp16 mask is 64 MB per function — it tripled the package.
    # The runtime allocates each variant's mask once and wraps it zero-copy.
    specs += [mb.TensorSpec(shape=(1, 1, t, total), dtype=fp16)]

    def body(normed, k_hist, v_hist, mask):
        def rms_head(x, weight_name):
            sq = mb.mul(x=x, y=x)
            mean = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
            denom = mb.rsqrt(x=mb.add(x=mean, y=np.float16(EPS)))
            return mb.mul(x=mb.mul(x=x, y=denom),
                          y=weights[weight_name].reshape(1, 1, HEAD_DIM))

        def rope(x, heads):
            r1 = mb.slice_by_index(x=x, begin=[0, 0, 0],
                                   end=[heads, t, ROTARY // 2],
                                   begin_mask=[True, True, False],
                                   end_mask=[True, True, False])
            r2 = mb.slice_by_index(x=x, begin=[0, 0, ROTARY // 2],
                                   end=[heads, t, ROTARY],
                                   begin_mask=[True, True, False],
                                   end_mask=[True, True, False])
            rest = mb.slice_by_index(x=x, begin=[0, 0, ROTARY],
                                     end=[heads, t, HEAD_DIM],
                                     begin_mask=[True, True, False],
                                     end_mask=[True, True, True])
            cos_b = cos_np.reshape(1, t, ROTARY // 2)
            sin_b = sin_np.reshape(1, t, ROTARY // 2)
            o1 = mb.sub(x=mb.mul(x=r1, y=cos_b), y=mb.mul(x=r2, y=sin_b))
            o2 = mb.add(x=mb.mul(x=r2, y=cos_b), y=mb.mul(x=r1, y=sin_b))
            return mb.concat(values=[o1, o2, rest], axis=-1)

        packed = mb.matmul(x=normed, y=weights["wq"].T)
        k = mb.matmul(x=normed, y=weights["wk"].T)
        v = mb.matmul(x=normed, y=weights["wv"].T)

        packed_h = mb.reshape(x=packed, shape=[t, N_Q_HEADS, 2 * HEAD_DIM])
        q = mb.slice_by_index(x=packed_h, begin=[0, 0, 0],
                              end=[t, N_Q_HEADS, HEAD_DIM],
                              begin_mask=[True, True, False],
                              end_mask=[True, True, False])
        gate = mb.slice_by_index(x=packed_h, begin=[0, 0, HEAD_DIM],
                                 end=[t, N_Q_HEADS, 2 * HEAD_DIM],
                                 begin_mask=[True, True, False],
                                 end_mask=[True, True, True])

        q = mb.transpose(x=q, perm=[1, 0, 2])
        k_h = mb.transpose(x=mb.reshape(x=k, shape=[t, N_KV_HEADS, HEAD_DIM]),
                           perm=[1, 0, 2])
        v_h = mb.transpose(x=mb.reshape(x=v, shape=[t, N_KV_HEADS, HEAD_DIM]),
                           perm=[1, 0, 2])

        q = rope(rms_head(q, "q_norm"), N_Q_HEADS)
        k_h = rope(rms_head(k_h, "k_norm"), N_KV_HEADS)

        # Cache-layout outputs: token-major [t, 512].
        k_new = mb.reshape(x=mb.transpose(x=k_h, perm=[1, 0, 2]),
                           shape=[t, KV_DIM])
        v_new = mb.reshape(x=mb.transpose(x=v_h, perm=[1, 0, 2]),
                           shape=[t, KV_DIM])

        k_cur = mb.reshape(x=k_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        v_cur = mb.reshape(x=v_h, shape=[1, N_KV_HEADS, t, HEAD_DIM])
        if history > 0:
            k_hh = mb.reshape(x=k_hist, shape=[history, N_KV_HEADS, HEAD_DIM])
            k_hh = mb.reshape(x=mb.transpose(x=k_hh, perm=[1, 0, 2]),
                              shape=[1, N_KV_HEADS, history, HEAD_DIM])
            v_hh = mb.reshape(x=v_hist, shape=[history, N_KV_HEADS, HEAD_DIM])
            v_hh = mb.reshape(x=mb.transpose(x=v_hh, perm=[1, 0, 2]),
                              shape=[1, N_KV_HEADS, history, HEAD_DIM])
            k_all = mb.concat(values=[k_hh, k_cur], axis=2)
            v_all = mb.concat(values=[v_hh, v_cur], axis=2)
        else:
            k_all, v_all = k_cur, v_cur

        rep = N_Q_HEADS // N_KV_HEADS

        def gqa_expand(x):
            x5 = mb.reshape(x=x, shape=[1, N_KV_HEADS, 1, total, HEAD_DIM])
            x5 = mb.concat(values=[x5] * rep, axis=2)
            return mb.reshape(x=x5, shape=[1, N_Q_HEADS, total, HEAD_DIM])

        k_g = gqa_expand(k_all)
        v_g = gqa_expand(v_all)

        q4 = mb.reshape(x=q, shape=[1, N_Q_HEADS, t, HEAD_DIM])
        scores = mb.matmul(x=q4, y=k_g, transpose_y=True)
        scores = mb.mul(x=scores, y=np.float16(SCALE))
        scores = mb.add(x=scores, y=mask)
        probs = mb.softmax(x=scores, axis=-1)
        attn = mb.matmul(x=probs, y=v_g)

        gated = mb.mul(x=mb.transpose(x=attn, perm=[0, 2, 1, 3]),
                       y=mb.sigmoid(x=mb.reshape(
                           x=gate, shape=[1, t, N_Q_HEADS, HEAD_DIM])))
        merged = mb.reshape(x=gated, shape=[t, Q_DIM])
        out = mb.matmul(x=merged, y=weights["wo"].T)
        return out, k_new, v_new

    if history > 0:
        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(normed, k_hist, v_hist, mask):
            return body(normed, k_hist, v_hist, mask)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
        def prog(normed, mask):
            return body(normed, None, None, mask)

    model = ct.convert(prog, convert_to="mlprogram",
                       minimum_deployment_target=ct.target.iOS18,
                       compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
    # Stable I/O names for the Swift runtime.
    spec = model.get_spec()
    rename = {}
    for out_obj, want in zip(spec.description.output, ("out", "k_new", "v_new")):
        rename[out_obj.name] = want
    for old, new in rename.items():
        ct.utils.rename_feature(spec, old, new)
    return ct.models.MLModel(spec, weights_dir=model.weights_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="path to the installed .gturbo directory")
    parser.add_argument("--max-history", type=int, default=12288,
                        help="largest KV history variant (multiple of 4096); "
                             "prompts beyond max-history+4096 tokens fall "
                             "back to the GPU path")
    parser.add_argument("--layers", default=None,
                        help="comma list of layer indices (default: all full-"
                             "attention layers)")
    args = parser.parse_args()

    model_dir = pathlib.Path(args.model)
    weights_bin = model_dir / "model_weights.bin"
    if not weights_bin.exists():
        raise SystemExit(f"not a .gturbo directory: {model_dir}")
    if args.max_history % CHUNK != 0:
        raise SystemExit("--max-history must be a multiple of 4096")
    histories = list(range(0, args.max_history + 1, CHUNK))
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else FULL_LAYERS)

    manifest = json.load(open(model_dir / "manifest.json"))
    if manifest["arch"]["family"] != "qwen36":
        raise SystemExit("ANE prefill export supports the qwen36 family only")
    # Attention weights are 4-bit in the 4-bit build and 8-bit in the 8-bit
    # build; both dequantize to the same fp16 graph, so only the unpack
    # differs. The sidecar itself is fp16 either way.
    weight_bits = manifest["quant"]["attention"]["weightBits"]

    out_dir = model_dir / "ane_prefill"
    out_dir.mkdir(exist_ok=True)
    entries = read_index(weights_bin)
    handle = open(weights_bin, "rb")
    for layer in layers:
        weights = load_layer_weights(handle, entries, layer,
                                     weight_bits=weight_bits)
        stage = out_dir / f".stage_layer_{layer}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        desc = ct.utils.MultiFunctionDescriptor()
        for history in histories:
            variant = build_variant(history, weights)
            variant_path = stage / f"h{history}.mlpackage"
            variant.save(str(variant_path))
            desc.add_function(str(variant_path),
                              src_function_name="main",
                              target_function_name=f"h{history}")
            print(f"layer {layer}: built h{history}", flush=True)
        desc.default_function_name = "h0"
        final = out_dir / f"layer_{layer}.mlpackage"
        if final.exists():
            shutil.rmtree(final)
        ct.utils.save_multifunction(desc, str(final))
        shutil.rmtree(stage)
        print(f"layer {layer}: wrote {final}", flush=True)
    handle.close()

    meta = {
        "version": EXPORT_VERSION,
        "family": "qwen36",
        "sourceWeightBits": weight_bits,
        "chunkTokens": CHUNK,
        "histories": histories,
        "layers": layers,
        # Binds the sidecar to the exact weights it was built from. The
        # runtime refuses a mismatch: a sidecar from different weights would
        # compute plausible-looking but wrong attention.
        "weightsSha256": manifest["files"]["model_weights.bin"]["sha256"],
    }
    with open(out_dir / "ane_prefill.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out_dir / 'ane_prefill.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
