#!/usr/bin/env python3
"""Build an MTP-only affine snapshot for Qwen3.8-Flash-Next from the original.

The draft head is 31 tensors of the checkpoint's `mtp.*` namespace, scattered
across 28 of its 131 shards. `prepare_qwen38.py` streams whole shards, so
producing the draft that way means re-reading the entire 360 GB release for
~5.5 GB of tensors. This range-fetches exactly those 31 and writes a snapshot
`NVMAIRepack --input-snapshot ... --draft-head` can import.

Every policy decision is imported from `prepare_qwen38.py` rather than
restated -- the name mapping, the expert split, the width per slot, the
zero-centred-norm fold and the quantiser itself. Four of five defects in this
port were a constant quietly copied between families; this file exists to move
bytes, not to decide anything.

    tools/prepare_qwen38_mtp.py --output .build/qwen38-mtp-affine
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

try:
    import ml_dtypes
    import numpy as np
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}\n"
             "  python3.13 -m pip install safetensors numpy ml_dtypes")

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("prepare_qwen38",
                                               _HERE / "prepare_qwen38.py")
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)
_pspec = importlib.util.spec_from_file_location("patch_snapshot_precision",
                                                _HERE / "patch_snapshot_precision.py")
patcher = importlib.util.module_from_spec(_pspec)
_pspec.loader.exec_module(patcher)

# One expert slab at a time: the fused gate_up tensor is 3.35 GB in bf16 and
# 6.7 GB once widened to float32, which does not belong in a 24 GiB machine's
# memory all at once. The packed output is an eighth of that, so only the
# working slice is large.
EXPERT_CHUNK = 64


def fetch_index() -> dict:
    url = f"https://huggingface.co/{pq.REPO}/raw/main/model.safetensors.index.json"
    return json.loads(subprocess.run(["curl", "-sfL", "--max-time", "120", url],
                                     capture_output=True, check=True).stdout)


def fetch_slice(name: str, index: dict, lo: int, count: int) -> np.ndarray:
    """Rows [lo, lo+count) of a stacked tensor, by HTTP range."""
    shard = index["weight_map"][name]
    head, data_start = patcher.header(shard, index)
    meta = head[name]
    begin, _ = meta["data_offsets"]
    shape = meta["shape"]
    per = int(np.prod(shape[1:]))
    itemsize = 2
    start = data_start + begin + lo * per * itemsize
    want = count * per * itemsize
    for attempt in range(8):
        result = subprocess.run(
            ["curl", "-sfL", "--max-time", "900", "--retry", "5",
             "--retry-delay", "2", "--retry-all-errors",
             "-r", f"{start}-{start + want - 1}",
             f"{patcher.BASE}/{shard}"], capture_output=True)
        if result.returncode == 0 and len(result.stdout) == want:
            return (np.frombuffer(result.stdout, dtype=ml_dtypes.bfloat16)
                    .astype(np.float32).reshape([count] + list(shape[1:])))
        import time
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{name}: range fetch failed after retries")


def slice_for(out_name: str, value: np.ndarray) -> np.ndarray:
    """The same slicing `convert_shard` applies, kept in one place."""
    if out_name.endswith("switch_mlp.gate_proj.weight"):
        return value[:, : value.shape[1] // 2, :]
    if out_name.endswith("switch_mlp.up_proj.weight"):
        return value[:, value.shape[1] // 2:, :]
    if out_name.endswith("indexer.index_q_proj.weight"):
        return value[:pq.INDEXER_QUERY_ROWS]
    if out_name.endswith("indexer.index_k_proj.weight"):
        return value[pq.INDEXER_QUERY_ROWS:]
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    index = fetch_index()
    sources = sorted(n for n in index["weight_map"] if n.startswith("mtp."))
    print(f"{len(sources)} mtp tensors across "
          f"{len({index['weight_map'][n] for n in sources})} checkpoint shards")

    block: dict[str, np.ndarray] = {}
    for name in sources:
        head, _ = patcher.header(index["weight_map"][name], index)
        shape = head[name]["shape"]
        stacked = name.endswith((".mlp.experts.gate_up_proj",
                                 ".mlp.experts.down_proj"))
        for out_name, out_shape in pq.outputs_for(name, list(shape)):
            bits = pq.quant_bits(out_name, args.bits)
            stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
            if stacked:
                packed, scales, biases = [], [], []
                for lo in range(0, shape[0], EXPERT_CHUNK):
                    count = min(EXPERT_CHUNK, shape[0] - lo)
                    piece = slice_for(out_name, fetch_slice(name, index, lo, count))
                    p, s, b = pq.quantize_affine(np.ascontiguousarray(piece), bits)
                    packed.append(p); scales.append(s); biases.append(b)
                block[out_name] = np.concatenate(packed, axis=0)
                block[stem + ".scales"] = np.concatenate(scales, axis=0)
                block[stem + ".biases"] = np.concatenate(biases, axis=0)
                print(f"  {out_name}  {out_shape}  q{bits}", flush=True)
                continue
            value = slice_for(out_name, fetch_slice(name, index, 0, shape[0]))
            if bits is None:
                folded = pq.fold_unit_offset(out_name, value)
                block[out_name] = np.ascontiguousarray(folded).astype(ml_dtypes.bfloat16)
                print(f"  {out_name}  {out_shape}  passthrough"
                      f"{'  (+1 folded)' if folded is not value else ''}", flush=True)
                continue
            p, s, b = pq.quantize_affine(np.ascontiguousarray(value), bits)
            block[out_name] = p
            block[stem + ".scales"] = s
            block[stem + ".biases"] = b
            print(f"  {out_name}  {out_shape}  q{bits}", flush=True)

    shard_name = "model-00001-of-00001.safetensors"
    save_file(block, str(out / shard_name))
    total = sum(v.nbytes for v in block.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total},
         "weight_map": {k: shard_name for k in block}}, indent=1))

    config = json.loads(subprocess.run(
        ["curl", "-sfL", "--max-time", "120",
         f"https://huggingface.co/{pq.REPO}/raw/main/config.json"],
        capture_output=True, check=True).stdout)
    pq.write_config(config, out, list(block), args.bits)
    print(f"\nsnapshot written: {out}  ({total / 1e9:.2f} GB, {len(block)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
