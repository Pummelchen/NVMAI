#!/usr/bin/env python3.13
"""Where should Qwen3.5-2B keep more precision, and what does it buy?

A uniform width is a guess. Some tensors quantise almost perfectly and some
lose real signal, and the difference is worth knowing before spending
megabytes: the small model has to stay small enough to sit beside a 35B.

The measurement is per tensor, against the bf16 source, using the
converter's own quantiser so the arithmetic is what the snapshot will
actually contain:

  * **output error** -- `W·x` at the quantised width against `W·x` at bf16,
    for random unit-variance activations. Weight error is the number people
    quote; this is the one the next layer sees, and errors that cancel
    within a row never reach it.
  * **cosine** -- direction agreement. A downstream norm can rescale
    magnitude but not direction, so this is the part that cannot be undone.
  * **outlier ratio** -- max|w| over the 99th percentile within each group
    of 64. Affine quantisation spends its levels on the range it is given;
    one outlier in a group stretches the scale and every other weight in
    that group loses resolution. This is the *explanation* for a bad tensor,
    and it is what says whether promoting it will help.

Cost is reported per tensor so a promotion can be judged by error bought
per megabyte, which is the only sensible way to choose.

    python3.13 tools/precision_plan_qwen35.py                 # 8-bit and 4-bit
    python3.13 tools/precision_plan_qwen35.py --rows 128
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path

import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / ".build/qwen35-2b-affine-8bit"
REPO = "Qwen/Qwen3.5-2B"
COMMIT = "15852e8c16360a2fea060d615a32b45270f8a8fc"
URL = (f"https://huggingface.co/{REPO}/resolve/{COMMIT}"
       "/model.safetensors-00001-of-00001.safetensors")
GROUP = 64

# The converter's own quantiser, so this measures what will be shipped
# rather than a re-implementation that might round differently.
_spec = importlib.util.spec_from_file_location(
    "prep", ROOT / "tools/prepare_qwen35_2b.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def dequantize(packed, scales, biases, bits: int) -> np.ndarray:
    lanes = 32 // bits
    mask = (1 << bits) - 1
    rows = packed.shape[0]
    out = np.zeros((rows, packed.shape[1] * lanes), dtype=np.float32)
    for lane in range(lanes):
        out[:, lane::lanes] = ((packed >> (bits * lane)) & mask).astype(np.float32)
    grouped = out.reshape(rows, -1, GROUP)
    return (grouped * scales.astype(np.float32)[..., None]
            + biases.astype(np.float32)[..., None]).reshape(rows, -1)


def roundtrip(value: np.ndarray, bits: int) -> np.ndarray:
    packed, scales, biases = prep.quantize_affine(value, bits)
    return dequantize(packed, scales, biases, bits)


class Source:
    def __init__(self) -> None:
        size = struct.unpack("<Q", self._range(0, 7))[0]
        self.header = json.loads(self._range(8, 8 + size - 1))
        self.base = 8 + size

    @staticmethod
    def _range(start: int, end: int) -> bytes:
        request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        return urllib.request.urlopen(request, timeout=300).read()

    def rows(self, name: str, count: int):
        meta = self.header.get(name)
        if meta is None or len(meta["shape"]) != 2:
            return None, 0
        start, end = meta["data_offsets"]
        rows, columns = meta["shape"]
        per_row = (end - start) // rows
        taken = min(count, rows)
        raw = self._range(self.base + start, self.base + start + per_row * taken - 1)
        dtype = np.float32 if meta["dtype"] == "F32" else ml_dtypes.bfloat16
        block = np.frombuffer(raw, dtype=dtype).astype(np.float32).reshape(taken, columns)
        return block, rows * columns


def kind_of(name: str) -> str:
    stem = name.removeprefix("model.language_model.")
    return re.sub(r"^layers\.\d+\.", "", stem).removesuffix(".weight")


def outlier_ratio(block: np.ndarray) -> float:
    """max|w| over the 99th percentile, within each group of 64, averaged.

    One large weight in a group forces the affine scale to cover it, and the
    other 63 weights then share the levels that are left. A ratio near 1
    means the group is flat and quantises well; a large ratio is a warning.
    """
    groups = block.reshape(block.shape[0], -1, GROUP)
    magnitude = np.abs(groups)
    peak = magnitude.max(axis=-1)
    typical = np.percentile(magnitude, 99, axis=-1)
    return float(np.mean(peak / np.maximum(typical, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=64,
                    help="rows sampled per tensor (the statistic is per row)")
    ap.add_argument("--samples", type=int, default=4,
                    help="random activations averaged per tensor")
    args = ap.parse_args()

    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())
    stems = sorted({n.removesuffix(".weight") for n in index["weight_map"]
                    if n.endswith(".weight")
                    and n.removesuffix(".weight") + ".scales" in index["weight_map"]})
    source = Source()
    rng = np.random.default_rng(0)

    per_kind: dict[str, dict[str, list]] = defaultdict(
        lambda: {"e4": [], "e8": [], "c4": [], "c8": [], "outlier": [], "params": 0,
                 "count": 0})
    for stem in stems:
        checkpoint = ("model.language_model."
                      + stem.removeprefix("language_model.model.") + ".weight")
        block, params = source.rows(checkpoint, args.rows)
        if block is None:
            continue
        w4, w8 = roundtrip(block, 4), roundtrip(block, 8)
        errors = {"e4": [], "e8": [], "c4": [], "c8": []}
        for _ in range(args.samples):
            x = rng.standard_normal(block.shape[1]).astype(np.float32)
            reference = block @ x
            scale = np.abs(reference).max() or 1.0
            for tag, weights in (("4", w4), ("8", w8)):
                y = weights @ x
                errors[f"e{tag}"].append(float(np.abs(y - reference).max() / scale))
                errors[f"c{tag}"].append(float(
                    reference @ y / (np.linalg.norm(reference) * np.linalg.norm(y) + 1e-30)))
        entry = per_kind[kind_of(checkpoint)]
        for key in errors:
            entry[key].append(float(np.mean(errors[key])))
        entry["outlier"].append(outlier_ratio(block))
        entry["params"] += params
        entry["count"] += 1

    print(f"Qwen3.5-2B, {args.rows} rows and {args.samples} activations per tensor, "
          f"against the bf16 source\n")
    print(f"  {'tensor kind':28s} {'n':>3s} {'MB@4':>6s} {'MB@8':>6s} "
          f"{'err@4':>7s} {'err@8':>7s} {'cos@4':>8s} {'cos@8':>8s} {'outlier':>8s}")
    rows = []
    for kind, entry in sorted(per_kind.items()):
        mb4 = entry["params"] * 0.5 / 1e6
        mb8 = entry["params"] * 1.0 / 1e6
        rows.append((kind, entry["count"], mb4, mb8,
                     float(np.mean(entry["e4"])), float(np.mean(entry["e8"])),
                     float(np.mean(entry["c4"])), float(np.mean(entry["c8"])),
                     float(np.mean(entry["outlier"]))))
    for row in sorted(rows, key=lambda r: -r[4]):
        print(f"  {row[0]:28s} {row[1]:3d} {row[2]:6.0f} {row[3]:6.0f} "
              f"{row[4]:7.4f} {row[5]:7.4f} {row[6]:8.5f} {row[7]:8.5f} {row[8]:8.2f}")

    print("\npromotions ranked by error removed per megabyte added:")
    print(f"  {'promotion':40s} {'+MB':>7s} {'err drop':>9s} {'per MB':>9s}")
    plan = []
    for kind, count, mb4, mb8, e4, e8, c4, c8, _ in rows:
        # 4-bit -> 8-bit costs the same megabytes again; 8-bit -> bf16 costs
        # another two bytes per weight over one.
        plan.append((f"{kind}: 4-bit -> 8-bit", mb8 - mb4, e4 - e8))
        plan.append((f"{kind}: 8-bit -> bf16", mb8 * 2 - mb8, e8))
    for label, cost, drop in sorted(plan, key=lambda p: -(p[2] / max(p[1], 1e-6))):
        if cost <= 0:
            continue
        print(f"  {label:40s} {cost:7.0f} {drop:9.4f} {drop / cost:9.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
