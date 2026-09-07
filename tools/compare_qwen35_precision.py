#!/usr/bin/env python3.13
"""How far apart are the 4-bit and 8-bit Qwen3.5-2B snapshots?

Raw weight error is the number people quote and the least useful one: a
quantizer is judged by what its output does, not by how far a weight moved.
So this reports three things per tensor kind, weakest evidence first:

  * **weight error** -- relative reconstruction error against the other
    width, and against the bf16 source for a sampled anchor set.
  * **output error** -- feed the same random activation through the real
    GEMV and compare the results. This is what the next layer actually sees,
    and errors that cancel within a row never reach it.
  * **direction** -- cosine of the two output vectors. Magnitude can be
    rescaled by a downstream norm; direction cannot, so a cosine that stays
    at 1.0 while the relative error grows means the widths still agree about
    what the layer is *saying*.

Anchoring against the source costs a few HTTP range requests, not whole
shards, so it does not need the 4.5 GB checkpoint on disk.

    python3.13 tools/compare_qwen35_precision.py
    python3.13 tools/compare_qwen35_precision.py --anchor 0      # local only
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
EIGHT = ROOT / ".build/qwen35-2b-affine-8bit"
FOUR = ROOT / ".build/qwen35-2b-affine-4bit"
REPO = "Qwen/Qwen3.5-2B"
COMMIT = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SHARD = "model.safetensors-00001-of-00001.safetensors"
URL = f"https://huggingface.co/{REPO}/resolve/{COMMIT}/{SHARD}"
GROUP = 64


def dequantize(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
               bits: int) -> np.ndarray:
    """Unpack the affine layout the converter writes: `bits`-wide unsigned
    lanes low-first inside each uint32, one BF16 scale and bias per group of
    64. Deliberately the arithmetic the kernels use, not a shortcut."""
    lanes = 32 // bits
    mask = (1 << bits) - 1
    rows = packed.shape[0]
    out = np.zeros((rows, packed.shape[1] * lanes), dtype=np.float32)
    for lane in range(lanes):
        out[:, lane::lanes] = ((packed >> (bits * lane)) & mask).astype(np.float32)
    grouped = out.reshape(rows, -1, GROUP)
    scaled = grouped * scales.astype(np.float32)[..., None] \
        + biases.astype(np.float32)[..., None]
    return scaled.reshape(rows, -1)


def kind_of(name: str) -> str:
    """Group tensors by the role they play, since that is the level at which
    a precision decision is actually taken."""
    stem = name.removeprefix("language_model.model.")
    stem = re.sub(r"^layers\.\d+\.", "", stem)
    return stem.removesuffix(".weight")


class Source:
    """The bf16 checkpoint, read by HTTP range request."""

    def __init__(self) -> None:
        size = struct.unpack("<Q", self._range(0, 7))[0]
        self.header = json.loads(self._range(8, 8 + size - 1))
        self.base = 8 + size

    @staticmethod
    def _range(start: int, end: int) -> bytes:
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(request, timeout=300).read()

    def rows(self, checkpoint_name: str, count: int) -> np.ndarray | None:
        meta = self.header.get(checkpoint_name)
        if meta is None:
            return None
        start, end = meta["data_offsets"]
        shape = meta["shape"]
        per_row = (end - start) // shape[0]
        end = start + per_row * min(count, shape[0])
        raw = self._range(self.base + start, self.base + end - 1)
        dtype = np.float32 if meta["dtype"] == "F32" else ml_dtypes.bfloat16
        return np.frombuffer(raw, dtype=dtype).astype(np.float32) \
            .reshape(-1, shape[1] if len(shape) > 1 else 1)


def snapshot_name_to_checkpoint(name: str) -> str:
    return "model.language_model." + name.removeprefix("language_model.model.")


def relative(a: np.ndarray, b: np.ndarray) -> float:
    """Error as a fraction of the reference's own scale, so tensors with
    different magnitudes can sit in one table."""
    spread = float(np.abs(b).max())
    return float(np.abs(a - b).max() / spread) if spread else 0.0


def _widths(snapshot: Path) -> tuple[int, dict[str, int]]:
    quantization = json.loads((snapshot / "config.json").read_text())["quantization"]
    base = quantization["bits"]
    overrides = {stem: entry["bits"] for stem, entry in quantization.items()
                 if isinstance(entry, dict)}
    return base, overrides


def bits_for(stem: str, widths: tuple[int, dict[str, int]]) -> int:
    base, overrides = widths
    return overrides.get(stem, base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=6,
                    help="tensors to also compare against the bf16 source (0 = skip)")
    ap.add_argument("--rows", type=int, default=64,
                    help="rows per tensor; the comparison is per row, so a sample "
                         "is as informative as the whole and far cheaper")
    args = ap.parse_args()
    for path in (EIGHT, FOUR):
        if not (path / "model.safetensors.index.json").exists():
            raise SystemExit(f"missing snapshot: {path}")

    eight = safe_open(str(next(EIGHT.glob("model-*.safetensors"))), framework="np")
    four = safe_open(str(next(FOUR.glob("model-*.safetensors"))), framework="np")
    # Width is per tensor, not per snapshot: --head-bits keeps the tied
    # embedding at 8 even in the 4-bit build, because it is the output
    # projection as well and 508M parameters of it decide every token.
    widths = {name: _widths(path) for name, path in (("8", EIGHT), ("4", FOUR))}
    shared = sorted({n for n in eight.keys() if n.endswith(".weight")}
                    & set(four.keys()))
    quantized = [n for n in shared
                 if n.removesuffix(".weight") + ".scales" in set(eight.keys())]

    rng = np.random.default_rng(0)
    per_kind: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for name in quantized:
        stem = name.removesuffix(".weight")
        b8, b4 = bits_for(stem, widths["8"]), bits_for(stem, widths["4"])
        if b8 == b4:
            continue          # same width in both builds: nothing to compare
        w8 = dequantize(eight.get_tensor(name)[: args.rows],
                        eight.get_tensor(stem + ".scales")[: args.rows],
                        eight.get_tensor(stem + ".biases")[: args.rows], b8)
        w4 = dequantize(four.get_tensor(name)[: args.rows],
                        four.get_tensor(stem + ".scales")[: args.rows],
                        four.get_tensor(stem + ".biases")[: args.rows], b4)
        x = rng.standard_normal(w8.shape[1]).astype(np.float32)
        y8, y4 = w8 @ x, w4 @ x
        cosine = float(y8 @ y4 / (np.linalg.norm(y8) * np.linalg.norm(y4) + 1e-30))
        per_kind[kind_of(name)].append((relative(w4, w8), relative(y4, y8), cosine))

    print(f"4-bit against 8-bit, {args.rows} rows per tensor, "
          f"{len(quantized)} quantized tensors\n")
    print(f"  {'tensor kind':34s} {'n':>3s} {'weight err':>11s} "
          f"{'output err':>11s} {'cosine':>9s}")
    worst_cosine = 1.0
    for kind in sorted(per_kind):
        rows = per_kind[kind]
        weight = max(r[0] for r in rows)
        output = max(r[1] for r in rows)
        cosine = min(r[2] for r in rows)
        worst_cosine = min(worst_cosine, cosine)
        print(f"  {kind:34s} {len(rows):3d} {weight:10.4f} {output:10.4f} {cosine:9.6f}")
    print(f"\n  worst direction agreement across every kind: {worst_cosine:.6f}")

    if args.anchor:
        print(f"\nanchored against the bf16 source ({args.anchor} tensors):")
        print(f"  {'tensor':44s} {'8-bit':>9s} {'4-bit':>9s}")
        source = Source()
        step = max(1, len(quantized) // args.anchor)
        for name in quantized[::step][: args.anchor]:
            stem = name.removesuffix(".weight")
            reference = source.rows(snapshot_name_to_checkpoint(name), args.rows)
            if reference is None:
                continue
            w8 = dequantize(eight.get_tensor(name)[: args.rows],
                            eight.get_tensor(stem + ".scales")[: args.rows],
                            eight.get_tensor(stem + ".biases")[: args.rows],
                            bits_for(stem, widths["8"]))
            w4 = dequantize(four.get_tensor(name)[: args.rows],
                            four.get_tensor(stem + ".scales")[: args.rows],
                            four.get_tensor(stem + ".biases")[: args.rows],
                            bits_for(stem, widths["4"]))
            label = name.removeprefix("language_model.model.").removesuffix(".weight")
            print(f"  {label:44s} {relative(w8, reference):8.4f} "
                  f"{relative(w4, reference):8.4f}")
        print("\n  8-bit allows 1/255 = 0.0039 of range, 4-bit 1/15 = 0.0667.")
    print("\nThis measures the weights. Whether the two widths choose the same\n"
          "tokens is a question for the engine, and is not answered here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
