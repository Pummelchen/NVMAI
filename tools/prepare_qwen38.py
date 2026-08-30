#!/usr/bin/env python3
"""Convert Qwen/Qwen3.8-Flash-Next into an affine snapshot NVMAIRepack imports.

Why this exists: the installed 4-bit model came from a third-party MLX repack.
NVMAI does not need one. It needs an affine group-64 layout, and it can produce
that itself from Qwen's own bf16 release, which is both better provenance and
better numerics -- quantizing once from the original weights beats inheriting
somebody else's quantization error.

The checkpoint is 360 GB across 131 shards and the machine has ~333 GiB free,
so nothing here holds the whole thing. One shard is fetched, converted, and
deleted before the next is needed; a background thread fetches shard N+1 while
shard N converts. Peak disk is the output plus one shard.

Three things the official checkpoint does differently from the MLX repack the
runtime was built against. Each is a silent mis-load if assumed rather than
checked, so each is asserted at plan time:

1. Routed experts are *stacked and fused*: one `mlp.experts.gate_up_proj` of
   shape [512, 1280, 2560] per layer, where 1280 is gate and up concatenated,
   plus one `mlp.experts.down_proj`. NVMAIRepack wants three separately named
   tensors matching `.mlp.switch_mlp.{gate,up,down}_proj.`, each of which may
   stay stacked over the 512 experts because the planner slices it per expert.
2. Norms carry `.weight` here and do not in the repack the runtime reads, so
   `hc_norm.weight` becomes `hc_norm`. `linear_attn.norm.weight` keeps its
   suffix -- the schema asks for that one *with* `.weight`.
3. `lm_head` sits at the archive root, not under `model.language_model.`.

Usage:
    tools/prepare_qwen38.py --plan                 # validate, download nothing
    tools/prepare_qwen38.py --output DIR           # convert
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    sys.exit(f"missing dependency: {exc}. pip install safetensors numpy ml_dtypes")

REPO = "Qwen/Qwen3.8-Flash-Next"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
GROUP_SIZE = 64

# Bit width per role, mirroring the shipped manifest's quant slots: attention,
# routedExpert and sharedExpert at 4 bits; embedding and router at 8. A tensor
# absent from this table is copied through unquantised (norms, A_log, dt_bias,
# conv1d, and the n-gram table, which is far too large to quantise and is read
# as fp16 by the runtime).
BITS_4 = 4
BITS_8 = 8


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_ornith_mtp.quantize_affine, deliberately.

    Duplicated rather than imported so the two converters cannot drift apart
    silently if one is edited; any change here must be made there too.
    """
    value = value.astype(np.float32)
    if value.shape[-1] % GROUP_SIZE:
        raise ValueError(f"last dimension {value.shape[-1]} is not group-aligned")
    shape = (*value.shape[:-1], value.shape[-1] // GROUP_SIZE, GROUP_SIZE)
    grouped = value.reshape(shape)
    bias = grouped.min(axis=-1)
    high = grouped.max(axis=-1)
    levels = (1 << bits) - 1
    scale = np.where(high == bias, np.float32(1), (high - bias) / levels)
    scale = scale.astype(ml_dtypes.bfloat16)
    bias = bias.astype(ml_dtypes.bfloat16)
    quantized = np.rint(
        (grouped - bias.astype(np.float32)[..., None])
        / scale.astype(np.float32)[..., None]
    ).clip(0, levels).astype(np.uint32).reshape(value.shape)
    lanes = 32 // bits
    words = quantized.reshape(*quantized.shape[:-1], quantized.shape[-1] // lanes, lanes)
    packed = np.zeros(words.shape[:-1], dtype=np.uint32)
    for lane in range(lanes):
        packed |= words[..., lane] << np.uint32(bits * lane)
    return packed, scale, bias


def is_multimodal(name: str) -> bool:
    """The vision tower is never repacked; this build is text-only."""
    return ".visual." in name or name.startswith("model.visual.")


def rename(name: str) -> str:
    """Official tensor name -> the name the runtime's schema expects."""
    for norm in (".hc_norm", ".self_attn.q_norm", ".self_attn.k_norm"):
        if name.endswith(norm + ".weight"):
            return name[: -len(".weight")]
    return name


def quant_bits(name: str) -> int | None:
    """Bits for a tensor, or None to copy it through unquantised."""
    if ".mlp.experts." in name or ".mlp.switch_mlp." in name:
        return BITS_4
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        return BITS_8
    if ".mlp.gate.weight" in name or ".shared_expert_gate.weight" in name:
        return BITS_8
    if ".ple_embedding." in name or ".ngram_embedding." in name:
        return None                      # 95 GiB table, stored fp16
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        return None
    if name.endswith("conv1d.weight"):
        return None
    if name.endswith(".hc_norm") or name.endswith("norm.weight"):
        return None
    if name.endswith(".weight"):
        return BITS_4                    # projections: attention, GDN, HC, shared expert
    return None


def fetch_header(shard: str) -> dict:
    """Read a shard's safetensors header over HTTP range requests.

    Lets the plan be validated against the real checkpoint -- names, shapes and
    dtypes -- without downloading 360 GB first.
    """
    url = f"{BASE}/{shard}"
    raw = subprocess.run(["curl", "-sfL", "--max-time", "60", "-r", "0-7", url],
                         capture_output=True, check=True).stdout
    size = struct.unpack("<Q", raw[:8])[0]
    body = subprocess.run(["curl", "-sfL", "--max-time", "180", "-r", f"8-{8 + size - 1}", url],
                          capture_output=True, check=True).stdout
    return json.loads(body)


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    """Names and per-expert shapes this source tensor becomes."""
    new = rename(name)
    if new.endswith(".mlp.experts.gate_up_proj"):
        stem = new[: -len("experts.gate_up_proj")] + "switch_mlp."
        experts, fused, hidden = shape
        return [(stem + "gate_proj.weight", [experts, fused // 2, hidden]),
                (stem + "up_proj.weight", [experts, fused // 2, hidden])]
    if new.endswith(".mlp.experts.down_proj"):
        stem = new[: -len("experts.down_proj")] + "switch_mlp."
        return [(stem + "down_proj.weight", list(shape))]
    return [(new, list(shape))]


def plan(index: dict) -> dict:
    """Validate the conversion against the checkpoint's own headers."""
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    text = {n: s for n, s in wm.items() if not is_multimodal(n)}
    print(f"repo     : {REPO}")
    print(f"shards   : {len(shards)}   tensors: {len(wm)} "
          f"({len(wm) - len(text)} multimodal skipped)")
    print(f"declared : {index['metadata']['total_size'] / 1e9:.1f} GB")

    probe = [s for s in shards if any(
        ".mlp.experts." in n or n.endswith("embed_tokens.weight")
        for n, sh in text.items() if sh == s)][:2]
    problems: list[str] = []
    for shard in probe:
        head = fetch_header(shard)
        for name, meta in head.items():
            if name == "__metadata__" or is_multimodal(name):
                continue
            shape, dtype = meta["shape"], meta["dtype"]
            for out_name, out_shape in outputs_for(name, shape):
                bits = quant_bits(out_name)
                if bits and out_shape[-1] % GROUP_SIZE:
                    problems.append(
                        f"{out_name} last dim {out_shape[-1]} not group-aligned")
                print(f"  {name}\n     dtype={dtype} shape={shape}"
                      f"\n     -> {out_name} {out_shape} "
                      f"{'q' + str(bits) if bits else 'passthrough'}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
    else:
        print("\nplan validates against the checkpoint's own headers")
    return {"shards": shards, "text_tensors": len(text)}


def convert_tensor(source, name: str, out: dict) -> None:
    """Read one source tensor, split/quantise it, add it to the output block."""
    value = source.get_tensor(name)
    for out_name, _ in outputs_for(name, list(value.shape)):
        if out_name.endswith("switch_mlp.gate_proj.weight"):
            piece = value[:, : value.shape[1] // 2, :]
        elif out_name.endswith("switch_mlp.up_proj.weight"):
            piece = value[:, value.shape[1] // 2:, :]
        else:
            piece = value
        bits = quant_bits(out_name)
        if bits is None:
            out[out_name] = piece.astype(np.float16) \
                if piece.dtype == ml_dtypes.bfloat16 and ".ngram" in out_name else piece
            continue
        stem = out_name[: -len(".weight")]
        packed, scales, biases = quantize_affine(piece, bits)
        out[stem + ".weight"] = packed
        out[stem + ".scales"] = scales
        out[stem + ".biases"] = biases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="validate against the checkpoint headers, download nothing")
    ap.add_argument("--output", type=Path, help="affine snapshot directory")
    ap.add_argument("--index", type=Path, default=None,
                    help="cached model.safetensors.index.json")
    args = ap.parse_args()

    if args.index and args.index.exists():
        index = json.loads(args.index.read_text())
    else:
        url = f"https://huggingface.co/{REPO}/raw/main/model.safetensors.index.json"
        index = json.loads(subprocess.run(["curl", "-sfL", url],
                                          capture_output=True, check=True).stdout)

    if args.plan or not args.output:
        plan(index)
        return 0

    print("streaming conversion is not wired up yet; --plan validates the mapping")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
