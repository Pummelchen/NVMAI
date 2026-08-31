#!/usr/bin/env python3
"""Bring an affine snapshot into line with the converter's current policy.

A five-hour conversion should not be repeated because a name was wrong or a
tensor needed splitting. This compares what `prepare_qwen38.py` would produce
today against what a snapshot actually holds, and repairs the difference by
re-fetching only the affected tensors and rewriting only the shards that carry
them.

It is also the check worth running before a repack: `--check` reports drift
without touching anything, and a clean report means the snapshot matches the
policy tensor for tensor, which is a stronger statement than "the repack did
not error".

    tools/reconcile_snapshot.py --snapshot DIR --check
    tools/reconcile_snapshot.py --snapshot DIR --bits 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

try:
    import ml_dtypes  # noqa: F401  - registers bfloat16 with numpy
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}\n"
             "  python3.13 -m pip install safetensors numpy ml_dtypes")

_spec = importlib.util.spec_from_file_location(
    "prepare_qwen38", Path(__file__).parent / "prepare_qwen38.py")
pq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pq)

_patch_spec = importlib.util.spec_from_file_location(
    "patch_snapshot_precision", Path(__file__).parent / "patch_snapshot_precision.py")
patcher = importlib.util.module_from_spec(_patch_spec)
_patch_spec.loader.exec_module(patcher)


def expected(ck_header_for, ck_index: dict, width: int) -> dict[str, tuple[str, int | None, list]]:
    """Map each output tensor name -> (source name, bits, shape)."""
    out: dict[str, tuple[str, int | None, list]] = {}
    for name, shard in ck_index["weight_map"].items():
        if pq.is_multimodal(name) or pq.is_ngram(name) or pq.is_ple_buffer(name):
            continue
        shape = ck_header_for(name)
        for out_name, out_shape in pq.outputs_for(name, shape):
            out[out_name] = (name, pq.quant_bits(out_name, width), out_shape)
    return out


def actual_bits(index_names: set[str], name: str, cols: int, want_cols: int) -> int | None:
    """Infer the width a stored tensor was written at, or None if unquantised.

    Membership is checked against the whole index, not the shard the weight
    happens to sit in: the writer flushes at a size threshold, so a tensor's
    `.scales` routinely lands in the next shard along. Looking only in the
    weight's own shard reports every such tensor as unquantised.
    """
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    if stem + ".scales" not in index_names:
        return None
    return 32 // (want_cols // cols) if cols else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--bits", type=int, choices=(4, 8), default=4)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    snap_index = json.loads(
        (args.snapshot / "model.safetensors.index.json").read_text())
    if args.index and args.index.exists():
        ck_index = json.loads(args.index.read_text())
    else:
        import subprocess
        url = f"https://huggingface.co/{pq.REPO}/raw/main/model.safetensors.index.json"
        ck_index = json.loads(subprocess.run(["curl", "-sfL", url],
                                             capture_output=True, check=True).stdout)

    shape_cache: dict[str, list] = {}

    def ck_shape(name: str) -> list:
        if name not in shape_cache:
            head, _ = patcher.header(ck_index["weight_map"][name], ck_index)
            for k, v in head.items():
                if k != "__metadata__":
                    shape_cache[k] = v["shape"]
        return shape_cache[name]

    want = expected(ck_shape, ck_index, args.bits)
    have = set(snap_index["weight_map"])
    have_weights = {n for n in have if not n.endswith((".scales", ".biases"))}

    missing = sorted(set(want) - have)
    extra = sorted(n for n in have_weights if n not in want)

    wrong: list[tuple[str, int | None, int | None]] = []
    dtypes: list[tuple[str, str, str]] = []
    for name, (_src, bits, shape) in want.items():
        if name not in have:
            continue
        shard = snap_index["weight_map"][name]
        with safe_open(args.snapshot / shard, framework="np") as f:
            stored = f.get_slice(name).get_shape()
            stored_dtype = f.get_slice(name).get_dtype()
        got = actual_bits(have, name, stored[-1], shape[-1])
        if got != bits:
            wrong.append((name, bits, got))
            continue
        # Width is not the whole contract: the resident index records a dtype
        # per entry, so a passthrough tensor written as F32 loads far enough to
        # look fine and is then refused as a corrupt index.
        want_dtype = "U32" if bits else "BF16"
        if stored_dtype != want_dtype:
            dtypes.append((name, want_dtype, stored_dtype))

    # A zero-centred RMSNorm stores the offset from one, and the runtime
    # multiplies by the stored value, so the converter folds the +1 in. Nothing
    # about a name, a width or a dtype records whether that happened: an
    # unfolded snapshot is structurally perfect and produces fluent nonsense.
    # Sample the affected tensors against the checkpoint and say so.
    unfolded: list[str] = []
    fold_names = [n for n in want
                  if (n[:-len(".weight")] if n.endswith(".weight") else n)
                  .endswith(pq.UNIT_OFFSET_NORM_SUFFIXES) and n in have]
    for name in fold_names[:6]:
        src = want[name][0]
        shard = snap_index["weight_map"][name]
        with safe_open(args.snapshot / shard, framework="np") as f:
            stored = np.asarray(f.get_tensor(name), np.float32).ravel()
        original = patcher.fetch_bf16(src, ck_index).astype(np.float32).ravel()
        if stored.size != original.size:
            continue
        if np.allclose(stored - original, 0.0, atol=1e-6):
            unfolded.append(name)

    print(f"snapshot : {args.snapshot}")
    print(f"policy   : {args.bits}-bit build, {len(want)} tensors expected")
    print(f"norm fold: {len(fold_names)} tensors need the +1; "
          f"{'UNFOLDED -- rebuild or repair' if unfolded else 'folded'} "
          f"(sampled {min(len(fold_names), 6)})")
    print(f"missing  : {len(missing)}")
    print(f"extra    : {len(extra)}")
    print(f"wrong bits: {len(wrong)}")
    print(f"wrong dtype: {len(dtypes)}")
    for group, items in (("missing", missing), ("extra", extra)):
        if items:
            import collections, re
            counts = collections.Counter(
                re.sub(r"layers\.\d+\.", "layers.N.", i) for i in items)
            for k, v in sorted(counts.items())[:8]:
                print(f"   {group:8} x{v:<3} {k}")
    for name, wantb, gotb in wrong[:8]:
        print(f"   bits     {name.split('language_model.')[-1]}: have {gotb}, want {wantb}")
    for name, wantd, gotd in dtypes[:8]:
        print(f"   dtype    {name.split('language_model.')[-1]}: have {gotd}, want {wantd}")

    if args.check:
        ok = not (missing or extra or wrong or dtypes or unfolded)
        print("\nsnapshot matches the converter policy" if ok
              else "\nsnapshot has drifted from the converter policy")
        return 0 if ok else 1
    if not (missing or extra or wrong or dtypes or unfolded):
        print("\nnothing to do")
        return 0

    # Group work by the shard that must be rewritten. A missing tensor is
    # written into the shard that holds its source sibling where possible, so
    # the index keeps its shape.
    touched: dict[str, list[str]] = {}
    for name in missing:
        src = want[name][0]
        sibling = next((n for n in want if want[n][0] == src and n in have), None)
        shard = snap_index["weight_map"][sibling] if sibling else \
            sorted(set(snap_index["weight_map"].values()))[0]
        touched.setdefault(shard, []).append(name)
    for name, _w, _g in wrong:
        touched.setdefault(snap_index["weight_map"][name], []).append(name)
    for name, _w, _g in dtypes:
        touched.setdefault(snap_index["weight_map"][name], []).append(name)
    for name in extra:
        touched.setdefault(snap_index["weight_map"][name], []).append(name)
    for name in fold_names if unfolded else []:
        touched.setdefault(snap_index["weight_map"][name], []).append(name)

    for shard, names in sorted(touched.items()):
        path = args.snapshot / shard
        print(f"\nrewriting {shard} ({len(names)} changes)", flush=True)
        with safe_open(path, framework="np") as f:
            block = {k: f.get_tensor(k) for k in f.keys()}
        for name in names:
            stem = name[: -len(".weight")] if name.endswith(".weight") else name
            if name in extra:
                for k in (name, stem + ".scales", stem + ".biases"):
                    block.pop(k, None)
                    snap_index["weight_map"].pop(k, None)
                print(f"    drop {name.split('language_model.')[-1]}", flush=True)
                continue
            src, bits, _shape = want[name]
            value = patcher.fetch_bf16(src, ck_index)
            piece = _slice(value, name)
            if bits is None:
                # fetch_bf16 upcasts to float32 so the quantiser can work in
                # it. A passthrough tensor must go back to bf16: the resident
                # index stores a dtype per entry and the runtime rejects the
                # install outright if a norm arrives as F32.
                block[name] = np.ascontiguousarray(
                    pq.fold_unit_offset(name, piece)).astype(ml_dtypes.bfloat16)
                for k in (stem + ".scales", stem + ".biases"):
                    block.pop(k, None); snap_index["weight_map"].pop(k, None)
                snap_index["weight_map"][name] = shard
            else:
                packed, scales, biases = pq.quantize_affine(
                    np.ascontiguousarray(piece), bits)
                block[name] = packed
                block[stem + ".scales"] = scales
                block[stem + ".biases"] = biases
                for k in (name, stem + ".scales", stem + ".biases"):
                    snap_index["weight_map"][k] = shard
            print(f"    {name.split('language_model.')[-1]} "
                  f"-> {bits or 'passthrough'}", flush=True)
        tmp = path.with_suffix(".safetensors.new")
        save_file(block, str(tmp))
        tmp.replace(path)

    total = 0
    for shard in sorted(set(snap_index["weight_map"].values())):
        with safe_open(args.snapshot / shard, framework="np") as f:
            total += sum(f.get_tensor(k).nbytes for k in f.keys())
    snap_index["metadata"]["total_size"] = total
    (args.snapshot / "model.safetensors.index.json").write_text(
        json.dumps(snap_index, indent=1))
    print(f"\nindex rewritten: {len(snap_index['weight_map'])} tensors, "
          f"{total / 1e9:.1f} GB")
    return 0


def _slice(value: np.ndarray, out_name: str) -> np.ndarray:
    """The same slicing convert_shard applies, kept in one place."""
    if out_name.endswith("switch_mlp.gate_proj.weight"):
        return value[:, : value.shape[1] // 2, :]
    if out_name.endswith("switch_mlp.up_proj.weight"):
        return value[:, value.shape[1] // 2:, :]
    if out_name.endswith("indexer.index_q_proj.weight"):
        return value[:pq.INDEXER_QUERY_ROWS]
    if out_name.endswith("indexer.index_k_proj.weight"):
        return value[pq.INDEXER_QUERY_ROWS:]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
