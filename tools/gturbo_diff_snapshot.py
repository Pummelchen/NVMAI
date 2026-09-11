#!/usr/bin/env python3
"""Diff a repacked `.gturbo` resident payload against the snapshot it came from.

A repack is a byte copy -- NVMAI's quantizer and the repacker both use affine
group-64, and `RepackPlanner` copies a source `u32 .weight` into the resident
file unchanged -- so a correct repack is exactly equal to its source. That
makes this comparison exact rather than approximate, which is the point: it
catches a misread `ArchInfo` field or a wrong destination name as a byte
difference here, instead of as fluent nonsense after the CPU engine starts
reading `.gturbo`.

    python3 tools/gturbo_diff_snapshot.py <model.gturbo> <snapshot-dir>

Reads the `.gturbo` resident index directly (layout: `GTurboEncoders.swift`,
24-byte header and 72-byte entries) and the snapshot's safetensors shards, so
it needs neither the Swift runtime nor a model in memory.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

INDEX_HEADER_BYTES = 24
INDEX_ENTRY_BYTES = 72


def read_resident(gturbo: Path) -> dict[str, list[tuple[str, bytes]]]:
    """name -> [(part, bytes)] for weight/scales/biases of every entry."""
    raw = (gturbo / "model_weights.bin").read_bytes()
    index_size, resident_size, entry_count = struct.unpack_from("<QQQ", raw, 0)
    entries_base = INDEX_HEADER_BYTES

    out: dict[str, list[tuple[str, bytes]]] = {}
    for i in range(entry_count):
        off = entries_base + i * INDEX_ENTRY_BYTES
        # Layout per GTurboEncoders.writeIndexEntry: nameOffset u32, nameLen u16,
        # dtype u8, reserved u8, fileOffset u64, sizeBytes u64, shape[4] u32,
        # scaleOffset/scaleSize/biasOffset/biasSize u64. Reads as fixed offsets
        # rather than one format string because the mixed widths are easy to
        # miscount -- which they were.
        name_off = struct.unpack_from("<I", raw, off)[0]
        name_len = struct.unpack_from("<H", raw, off + 4)[0]
        (_dtype, _res) = struct.unpack_from("<BB", raw, off + 6)
        file_off, size = struct.unpack_from("<QQ", raw, off + 8)
        (s0, s1, s2, s3) = struct.unpack_from("<IIII", raw, off + 24)
        scale_off, scale_size = struct.unpack_from("<QQ", raw, off + 40)
        bias_off, bias_size = struct.unpack_from("<QQ", raw, off + 56)
        # `nameOffset` is file-absolute (ResidentWriter adds the string-table
        # base itself), not relative to the table.
        name = raw[name_off:name_off + name_len].decode()
        parts: list[tuple[str, bytes]] = [("weight", raw[file_off:file_off + size])]
        if scale_size:
            parts.append(("scales", raw[scale_off:scale_off + scale_size]))
        if bias_size:
            parts.append(("biases", raw[bias_off:bias_off + bias_size]))
        out[name] = parts
        _ = (resident_size, s0, s1, s2, s3)
    return out


def read_safetensors(path: Path) -> dict[str, tuple[int, bytes]]:
    """name -> (data_offset, data_bytes) for one shard."""
    raw = path.read_bytes()
    header_len = struct.unpack_from("<Q", raw, 0)[0]
    header = json.loads(raw[8:8 + header_len])
    data_base = 8 + header_len
    out: dict[str, tuple[int, bytes]] = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        out[name] = (data_base + begin, raw[data_base + begin:data_base + end])
        _ = end
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    gturbo, snapshot = Path(argv[1]), Path(argv[2])

    resident = read_resident(gturbo)

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards: dict[str, dict[str, bytes]] = {}
    for tensor_name, shard in index["weight_map"].items():
        if shard not in shards:
            shards[shard] = {n: b for n, (_o, b) in
                             read_safetensors(snapshot / shard).items()}
    source = {name: data for shard in shards.values() for name, data in shard.items()}

    # A destination name is the source name with the family's prefixing applied;
    # compare by the tensor's own suffix so the check does not depend on which
    # spelling the repacker chose for the root.
    by_suffix: dict[str, tuple[str, bytes]] = {}
    for name, data in source.items():
        by_suffix[name] = (name, data)

    checked = mismatched = missing = 0
    problems: list[str] = []
    for name, parts in sorted(resident.items()):
        weight = dict(parts)["weight"]
        # `language_model.model.layers.N...` in the install matches the source's
        # own name for this family; try exact then suffix.
        candidate = by_suffix.get(name)
        if candidate is None:
            tail = name.split("language_model.model.")[-1]
            hits = [(n, d) for n, d in source.items() if n.endswith(tail)]
            if len(hits) != 1:
                missing += 1
                problems.append(f"{name}: no source tensor (suffix {tail!r}, "
                                f"{len(hits)} candidates)")
                continue
            candidate = hits[0]
        source_name, data = candidate
        checked += 1
        if weight != data:
            mismatched += 1
            problems.append(
                f"{name}: {len(weight)} bytes != source {source_name} "
                f"{len(data)} bytes")

    for line in problems[:40]:
        print(f"  {line}")
    print(f"\nresidents={len(resident)} compared={checked} "
          f"mismatched={mismatched} unmatched={missing}")
    if mismatched or missing:
        print("FAIL: the repack is not byte-identical to its source snapshot")
        return 1
    print("PASS: every resident tensor is byte-identical to its source snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
