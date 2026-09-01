# Sharing `ngram_table.bin` between Qwen3.8-Flash-Next installs

> **Implemented 2026-09-01.** `prepare_qwen38.py --reuse-ngram-table
> <dir-or-file>` hardlinks an existing table and skips the 31 checkpoint
> shards that carry nothing else; `NVMAIRepack --share-ngram-table` links it
> into the install rather than copying it. Linking the two copies that already
> existed on this machine recovered **95 GiB** (138 -> 233 GiB free), verified
> byte-identical with `cmp` beforehand.

## Why

The PLE n-gram table is 95.4 GiB and it is **the same bytes in every
quantization**. It is fp16 regardless of the routed-expert width, it is not in
the manifest's quant slots at all, and its contents are fully determined by the
source checkpoint. A 4-bit and an 8-bit install of the same checkpoint carry
two byte-identical copies of it.

That is most of what an install costs:

| | 4-bit | 8-bit |
| --- | ---: | ---: |
| n-gram table | 95.4 GiB | 95.4 GiB |
| routed experts | 59 GiB | 117 GiB |
| dense | 3 GiB | 6 GiB |
| **install** | **157 GiB** | **219 GiB** |

Both installs unshared is 376 GiB. The table is 60% of the 4-bit install and
44% of the 8-bit one, and half of the pair's total is one file stored twice.

## What this buys

| | unshared | shared |
| --- | ---: | ---: |
| 4-bit + 8-bit installs | 376 GiB | **281 GiB** |
| peak while building the 8-bit | 595 GiB | **405 GiB** |
| download for the second quantization | 360 GB | **258 GB** |

The download saving is the one that is easy to miss: 102 GB of the 360 GB
checkpoint is the n-gram shards. If the second conversion reuses the table it
already has, it never fetches them — about 1.5 hours off a 5-hour run at the
~19 MB/s this connection sustains.

## Why it is safe

The table is identical **by construction**, not by coincidence: it is the
concatenation of the checkpoint's 128 `ngram_embedding.shard_N` tensors, in
numeric order, narrowed bf16 -> fp16. Nothing about the routed-expert width
enters that path.

The gate should therefore be provenance, not content. `manifest.json` already
records `sourceSnapshotHash`; two installs that agree on it were built from the
same checkpoint and their tables cannot differ. Comparing the hash is O(1) where
comparing 95.4 GiB of content is minutes of I/O for a weaker guarantee — a
content match on a table built from a *different* checkpoint would be a
coincidence worth nothing, and a mismatch on the same checkpoint would mean one
of the two is already corrupt.

Belt and braces, cheaply: require the byte size to match exactly as well, since
a truncated table would otherwise link happily.

## Implementation

Two independent changes; either is useful without the other.

### 1. `NVMAIRepack --share-ngram-table <install>`

`RepackPlanner.passthroughRequirements` declares `ngram_table.bin`;
`RemoteStreamingRepacker` preallocates each passthrough destination with
`ftruncate` in the partial directory and the transfer fills it. The change is to
resolve the requirement to a *link* rather than an allocation:

- if `--share-ngram-table` names an install whose `manifest.json` has the same
  `sourceSnapshotHash` and whose `ngram_table.bin` is exactly the expected size,
  `link()` it into the partial directory and drop it from the transfer plan;
- otherwise fall back to the existing copy path, so the flag can only make a
  build cheaper, never different.

Hardlink rather than symlink. Both installs then hold a reference to one inode:
deleting either leaves the other intact, which a symlink would not, and the
runtime's `F_NOCACHE` reads are indifferent to the extra link. Requires both
installs on one filesystem, which they are.

`verified-install.json` hashes content, so a hardlinked table verifies exactly
as a copied one does. The receipt binds to an absolute path; sharing does not
change that, and `--verify-install` still works per install.

### 2. `prepare_qwen38.py --reuse-ngram-table <path>`

The converter writes the snapshot's table from the checkpoint's shards. Given
this flag it instead hardlinks the existing table and skips fetching the 128
n-gram shards entirely — the 102 GB and ~1.5 h saving above. Same gate: the
config's PLE parameters must match those the existing table was built from, and
its size must equal the computed `padded_rows * ple_head_dim * 2`.

## Order of operations

1. Finish the 4-bit install from the current run. Nothing here blocks it.
2. Free space. Both installs plus the 8-bit's snapshot peak at 405 GiB shared;
   this machine has 926 GiB with ~617 GiB spoken for.
3. Implement (2), then (1) — the converter change is smaller and is what saves
   the download.
4. Build the 8-bit snapshot reusing the 4-bit table, repack with
   `--share-ngram-table`, and confirm with `ls -li` that both tables report one
   inode and a link count of 2.
5. Run the parity harness against the 8-bit install. Sharing is only sound if
   the table is genuinely interchangeable, and parity is what proves it rather
   than assumes it.

## Both widths come from the bf16 original

Neither install is derived from the other, and neither is derived from a
quantized release. `prepare_qwen38.py --bits {4,8}` re-runs the whole
conversion against Qwen's bf16 checkpoint; the width only changes
`levels = (1 << bits) - 1` inside `quantize_affine`. That is why the 8-bit
build costs a second 360 GB fetch rather than being a cheap local transform --
if it could be derived from the 4-bit it would take minutes and be worthless.

Only `ngram_table.bin` is shared, and it is never quantized in either width:
the format stores it as fp16 (`rowDim * MemoryLayout<Float16>.stride`), so the
two installs' tables are the same bytes by construction rather than merely
close enough to reuse.

Qwen's FP8 release is not a shortcut worth taking. Its config excludes 943
modules, leaving everything except the routed experts in bf16 -- so the only
tensors it quantizes are the ones NVMAI streams -- and its [128, 128] blocks do
not map onto affine group-64. Building from it would store 8 bits per weight
carrying E4M3's 3-bit mantissa: the full bandwidth cost of 8-bit on an
I/O-bound decode, without the quality that is the only reason to pay it.

## Is the 8-bit install worth building at all

Worth deciding before spending the disk. Decode on this machine is expert-I/O
bound: 8-bit streams 117 GiB of experts against 4-bit's 59 GiB, so expect
roughly half the throughput — about **3.4 tok/s** against the measured 6.82.
That is slower than every other model in the catalogue, including 8-bit Ornith
at 11.89 tok/s, in exchange for quality headroom a 125B model already has at
4-bit. The sharing work is worth doing regardless, because it also makes the
4-bit install cheaper to rebuild; the 8-bit install is a separate question.
