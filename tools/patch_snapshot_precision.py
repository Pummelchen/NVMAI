#!/usr/bin/env python3
"""Re-quantize the precision-sensitive tensors of an existing affine snapshot.

The 4-bit snapshot produced before the precision fix carries the QSA indexer,
the hyper-connection write gate and the GDN gating projections at 4 bits. Those
are about 10 MB of a 157 GiB install, so re-running a five-hour conversion to
change them would be absurd. This re-fetches just those tensors from the
checkpoint by HTTP range request, re-quantizes them at 8 bits, and rewrites
only the output shards that contain them.

Why those tensors: every tensor measures roughly 10% relative error at 4 bits
and 0.6% at 8, bulk projections included, so the error is not what
distinguishes them. What distinguishes them is that their error does not
average away -- a ranking either matches or it does not, and a gate with four
outputs has nothing to average over. See `quant_bits` in prepare_qwen38.py.

    tools/patch_snapshot_precision.py --snapshot DIR [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}\n"
             "  python3.13 -m pip install safetensors numpy ml_dtypes")

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "prepare_qwen38", Path(__file__).parent / "prepare_qwen38.py")
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)

BASE = pq.BASE


def wants_eight_bits(name: str) -> bool:
    """Names the fixed policy puts at 8 bits but a 4-bit build once wrote at 4."""
    if not name.endswith(".weight"):
        return False
    stem = name[: -len(".weight")]
    return (".indexer." in name
            or name.endswith("block_inject_weight.weight")
            or name.endswith("in_proj_a.weight")
            or name.endswith("in_proj_b.weight")) and ".scales" not in stem


def source_name(out_name: str) -> str:
    """Map a snapshot tensor name back to the checkpoint's own name."""
    # None of the affected tensors are renamed or split by the converter, so
    # the mapping is the identity. Asserted rather than assumed: a split tensor
    # could not be patched this way.
    assert ".switch_mlp." not in out_name, f"{out_name} is a split tensor"
    return out_name


_headers: dict[str, tuple[dict, int]] = {}


def header(shard: str, index: dict) -> tuple[dict, int]:
    if shard not in _headers:
        url = f"{BASE}/{shard}"
        raw = subprocess.run(["curl", "-sfL", "--max-time", "60", "-r", "0-7", url],
                             capture_output=True, check=True).stdout
        n = struct.unpack("<Q", raw[:8])[0]
        body = subprocess.run(["curl", "-sfL", "--max-time", "180",
                               "-r", f"8-{8 + n - 1}", url],
                              capture_output=True, check=True).stdout
        _headers[shard] = (json.loads(body), 8 + n)
    return _headers[shard]


def fetch_bf16(name: str, index: dict) -> np.ndarray:
    shard = index["weight_map"][name]
    head, data_start = header(shard, index)
    meta = head[name]
    lo, hi = meta["data_offsets"]
    raw = subprocess.run(["curl", "-sfL", "--max-time", "600", "-r",
                          f"{data_start + lo}-{data_start + hi - 1}",
                          f"{BASE}/{shard}"],
                         capture_output=True, check=True).stdout
    if len(raw) != hi - lo:
        raise RuntimeError(f"{name}: got {len(raw)} bytes, expected {hi - lo}")
    return np.frombuffer(raw, dtype=ml_dtypes.bfloat16).astype(np.float32) \
             .reshape(meta["shape"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--index", type=Path, help="cached checkpoint index")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index_path = args.snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        sys.exit(f"no index at {index_path}\n"
                 "  the converter writes it last, so the snapshot is either "
                 "still being built or did not finish")
    snap_index = json.loads(index_path.read_text())
    if args.index and args.index.exists():
        ck_index = json.loads(args.index.read_text())
    else:
        url = f"https://huggingface.co/{pq.REPO}/raw/main/model.safetensors.index.json"
        ck_index = json.loads(subprocess.run(["curl", "-sfL", url],
                                             capture_output=True, check=True).stdout)

    targets = sorted({n for n in snap_index["weight_map"] if wants_eight_bits(n)})
    by_shard: dict[str, list[str]] = {}
    for name in targets:
        by_shard.setdefault(snap_index["weight_map"][name], []).append(name)

    print(f"snapshot : {args.snapshot}")
    print(f"tensors to re-quantize at 8 bits : {len(targets)}")
    print(f"output shards to rewrite         : {len(by_shard)} of "
          f"{len(set(snap_index['weight_map'].values()))}")
    for shard, names in sorted(by_shard.items()):
        print(f"  {shard}: {len(names)} tensors")
    if args.dry_run:
        print("\ndry run; nothing written")
        return 0
    if not targets:
        print("nothing to do")
        return 0

    for shard, names in sorted(by_shard.items()):
        path = args.snapshot / shard
        print(f"\nrewriting {shard} ({len(names)} tensors)", flush=True)
        with safe_open(path, framework="np") as src:
            block = {k: src.get_tensor(k) for k in src.keys()}
        for name in names:
            original = fetch_bf16(source_name(name), ck_index)
            stem = name[: -len(".weight")]
            packed, scales, biases = pq.quantize_affine(original, pq.BITS_8)
            before = block[name].nbytes
            block[name] = packed
            block[stem + ".scales"] = scales
            block[stem + ".biases"] = biases
            print(f"    {name.split('.language_model.')[-1]}  "
                  f"{before / 1e6:.2f} MB -> {packed.nbytes / 1e6:.2f} MB", flush=True)
        tmp = path.with_suffix(".safetensors.new")
        save_file(block, str(tmp))
        tmp.replace(path)

    # Sizes changed, so the index's total must be recomputed or a later reader
    # that trusts it will be wrong about the payload.
    total = 0
    for shard in sorted(set(snap_index["weight_map"].values())):
        with safe_open(args.snapshot / shard, framework="np") as src:
            total += sum(src.get_tensor(k).nbytes for k in src.keys())
    snap_index["metadata"]["total_size"] = total
    (args.snapshot / "model.safetensors.index.json").write_text(
        json.dumps(snap_index, indent=1))
    print(f"\nindex total_size updated to {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
