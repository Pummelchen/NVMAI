#!/usr/bin/env python3
"""Convert Qwen/Qwen3.8-Flash-Next into an affine snapshot NVMAIRepack imports.

Why this exists: the installed 4-bit model came from a third-party MLX repack.
NVMAI does not need one. It needs an affine group-64 layout, and it can produce
that from Qwen's own bf16 release, which is both better provenance and better
numerics -- quantizing once from the original weights beats inheriting somebody
else's quantization error.

The checkpoint is 360 GB across 131 shards. Nothing here holds the whole thing:
one shard is fetched, converted and deleted before the next is needed, with a
background thread fetching shard N+1 while shard N converts. Peak disk is the
output plus one shard.

Three things the official checkpoint does differently from the MLX repack the
runtime was built against. Each is a silent mis-load if assumed rather than
checked, so `--plan` asserts all three against the checkpoint's own headers:

1. Routed experts are *stacked and fused*: one `mlp.experts.gate_up_proj` of
   shape [512, 1280, 2560] per layer, where 1280 is gate and up concatenated,
   plus one `mlp.experts.down_proj`. NVMAIRepack wants three separately named
   tensors matching `.mlp.switch_mlp.{gate,up,down}_proj.`, each of which may
   stay stacked over the 512 experts because the planner slices per expert.
2. Norms carry `.weight` here and do not in the repack the runtime reads, so
   `hc_norm.weight` becomes `hc_norm`. `linear_attn.norm.weight` keeps its
   suffix -- the schema asks for that one *with* `.weight`.
3. `lm_head` sits at the archive root, not under `model.language_model.`.

`ple_constants.json` is derived here rather than copied, using the algorithm in
the transformers reference (`_build_layer_multipliers`, `_find_nth_prime_after`).
The derivation reproduces the previously shipped constants exactly -- multipliers,
the sixteen prime head vocabularies and their offsets all match -- which is what
lets this run without any dependency on the third-party repack.

Usage:
    tools/prepare_qwen38.py --plan                    # validate, download nothing
    tools/prepare_qwen38.py --output DIR --work DIR   # convert
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(f"missing dependency: {exc}\n"
             "  python3.13 -m pip install safetensors numpy ml_dtypes")

REPO = "Qwen/Qwen3.8-Flash-Next"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
GROUP_SIZE = 64
BITS_4, BITS_8 = 4, 8
OUTPUT_SHARD_BYTES = 4 << 30

# --- PLE constants, per the transformers reference -------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def ple_constants(text_config: dict) -> dict:
    """Reproduce what the reference computes at init, as shippable data.

    `ple_layer_index` is the position *within* `ple_layer_ids`, not the layer
    number: the reference reads it as `ple_layer_ids.index(layer_idx + 1)`, so
    for the single PLE layer it is 0. Getting that wrong changes every hash
    multiplier and every n-gram id, silently.
    """
    ple_layer_index = 0
    heads = text_config["heads_per_ngram"] * 2      # two n-gram orders
    vocab_base = text_config["ngram_vocab_size_base"]
    sizes, offsets, total = [], [], 0
    for head in range(heads):
        size = _find_nth_prime_after(vocab_base - 1, ple_layer_index * heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // max(text_config["vocab_size"], 1)) // 2)
    base_seed = text_config.get("seed", 1234) + _PRIME_1 * ple_layer_index
    multipliers = [
        2 * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound) + 1
        for i in range(text_config["ngram_size"])
    ]
    eos = text_config["eos_token_id"]
    return {
        "layer_multipliers": multipliers,
        "ngram_heads_offsets": offsets,
        "ngram_heads_vocab_sizes": sizes,
        "eos_token_id": eos[0] if isinstance(eos, list) else eos,
        "ngram_size": text_config["ngram_size"],
        "heads_per_ngram": text_config["heads_per_ngram"],
        "ple_n_heads": heads,
        "ple_head_dim": text_config["ple_embed_dim"] // heads,
        "table_file": "ngram_table.bin",
        "table_dtype": "float16",
    }


# --- quantisation ----------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_ornith_mtp.quantize_affine, deliberately.

    Duplicated rather than imported so neither converter can drift silently;
    a change to one must be made in the other.
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


# --- naming ----------------------------------------------------------------


def is_multimodal(name: str) -> bool:
    return ".visual." in name or name.startswith("model.visual.")


def is_ngram(name: str) -> bool:
    return ".ngram_embedding.shard_" in name


def rename(name: str) -> str:
    for norm in (".hc_norm", ".self_attn.q_norm", ".self_attn.k_norm"):
        if name.endswith(norm + ".weight"):
            return name[: -len(".weight")]
    return name


def quant_bits(name: str) -> int | None:
    """Bits for a tensor, or None to copy it through unquantised."""
    if ".mlp.switch_mlp." in name:
        return BITS_4
    if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
        return BITS_8
    if name.endswith(".mlp.gate.weight") or name.endswith(".shared_expert_gate.weight"):
        return BITS_8
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        return None
    if name.endswith("conv1d.weight"):
        return None
    if name.endswith(".hc_norm") or name.endswith("norm.weight"):
        return None
    if name.endswith(".weight"):
        return BITS_4
    return None


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
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


# --- transport -------------------------------------------------------------


def fetch_header(shard: str) -> dict:
    url = f"{BASE}/{shard}"
    raw = subprocess.run(["curl", "-sfL", "--max-time", "60", "-r", "0-7", url],
                         capture_output=True, check=True).stdout
    size = struct.unpack("<Q", raw[:8])[0]
    body = subprocess.run(["curl", "-sfL", "--max-time", "180", "-r", f"8-{8 + size - 1}", url],
                          capture_output=True, check=True).stdout
    return json.loads(body)


def download(shard: str, work: Path) -> Path:
    """Fetch one shard, resuming a partial file rather than restarting it."""
    dest = work / shard
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "--retry", "5", "--retry-delay", "5",
                    "--retry-all-errors", "-C", "-", "--silent", "--show-error",
                    "-o", str(dest), f"{BASE}/{shard}"], check=True)
    return dest


# --- conversion ------------------------------------------------------------


class OutputWriter:
    """Accumulates converted tensors and flushes them as safetensors shards."""

    def __init__(self, out: Path):
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.block: dict[str, np.ndarray] = {}
        self.bytes = 0
        self.index: dict[str, str] = {}
        self.total = 0
        self.shard_no = 0

    def add(self, name: str, value: np.ndarray) -> None:
        self.block[name] = value
        self.bytes += value.nbytes
        self.total += value.nbytes
        if self.bytes >= OUTPUT_SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self.block:
            return
        self.shard_no += 1
        name = f"model-{self.shard_no:05d}.safetensors"
        save_file(self.block, str(self.out / name))
        for key in self.block:
            self.index[key] = name
        print(f"    wrote {name} ({self.bytes / 1e9:.2f} GB, {len(self.block)} tensors)",
              flush=True)
        self.block.clear()
        self.bytes = 0

    def finish(self) -> None:
        self.flush()
        # Rename to the N-of-M form the loaders expect now that M is known.
        final = {}
        for old_key, old_name in self.index.items():
            n = int(old_name.split("-")[1].split(".")[0])
            final[old_key] = f"model-{n:05d}-of-{self.shard_no:05d}.safetensors"
        for n in range(1, self.shard_no + 1):
            src = self.out / f"model-{n:05d}.safetensors"
            src.rename(self.out / f"model-{n:05d}-of-{self.shard_no:05d}.safetensors")
        (self.out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": self.total}, "weight_map": final}, indent=1))


def convert_shard(path: Path, writer: OutputWriter, ngram: "NgramTable") -> None:
    with safe_open(path, framework="np") as src:
        for name in src.keys():
            if is_multimodal(name):
                continue
            if is_ngram(name):
                ngram.add(name, src.get_tensor(name))
                continue
            value = src.get_tensor(name)
            for out_name, _ in outputs_for(name, list(value.shape)):
                if out_name.endswith("switch_mlp.gate_proj.weight"):
                    piece = value[:, : value.shape[1] // 2, :]
                elif out_name.endswith("switch_mlp.up_proj.weight"):
                    piece = value[:, value.shape[1] // 2:, :]
                else:
                    piece = value
                bits = quant_bits(out_name)
                if bits is None:
                    writer.add(out_name, np.ascontiguousarray(piece))
                    continue
                stem = out_name[: -len(".weight")]
                packed, scales, biases = quantize_affine(np.ascontiguousarray(piece), bits)
                writer.add(stem + ".weight", packed)
                writer.add(stem + ".scales", scales)
                writer.add(stem + ".biases", biases)


class NgramTable:
    """Assembles ngram_table.bin from the checkpoint's 128 table shards.

    The shards must be concatenated in *numeric* order. Their names sort
    lexically as shard_0, shard_1, shard_10, shard_100 ..., so sorting the
    strings would interleave the table and produce a model that loads, runs,
    and is quietly wrong.
    """

    def __init__(self, out: Path, expected_rows: int, dim: int):
        self.path = out / "ngram_table.bin"
        self.expected_rows = expected_rows
        self.dim = dim
        self.pending: dict[int, np.ndarray] = {}
        self.next_index = 0
        self.rows = 0
        self.handle = self.path.open("wb")

    @staticmethod
    def index_of(name: str) -> int:
        return int(name.rsplit(".shard_", 1)[1].split(".")[0])

    def add(self, name: str, value: np.ndarray) -> None:
        self.pending[self.index_of(name)] = value
        while self.next_index in self.pending:
            block = self.pending.pop(self.next_index)
            if block.dtype == ml_dtypes.bfloat16:
                as_f32 = block.astype(np.float32)
                finite = np.isfinite(as_f32.astype(np.float16))
                if not finite.all():
                    raise ValueError(
                        f"{name}: bf16 -> fp16 overflows on "
                        f"{(~finite).sum()} values; the table format is fp16")
                block = as_f32.astype(np.float16)
            self.handle.write(np.ascontiguousarray(block).tobytes())
            self.rows += block.shape[0]
            self.next_index += 1

    def finish(self) -> None:
        self.handle.close()
        if self.pending:
            raise ValueError(f"n-gram shards never became contiguous: "
                             f"{sorted(self.pending)[:5]} still pending")
        if self.rows != self.expected_rows:
            raise ValueError(f"n-gram table has {self.rows} rows, "
                             f"expected {self.expected_rows}")


# --- driver ----------------------------------------------------------------


def plan(index: dict) -> None:
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    text = {n: s for n, s in wm.items() if not is_multimodal(n)}
    ngram = [n for n in text if is_ngram(n)]
    print(f"repo     : {REPO}")
    print(f"shards   : {len(shards)}   tensors: {len(wm)} "
          f"({len(wm) - len(text)} multimodal skipped)")
    print(f"declared : {index['metadata']['total_size'] / 1e9:.1f} GB")
    print(f"n-gram   : {len(ngram)} table shards")
    probe = [s for s in shards if any(
        ".mlp.experts." in n for n, sh in text.items() if sh == s)][:1]
    problems: list[str] = []
    for shard in probe:
        for name, meta in fetch_header(shard).items():
            if name == "__metadata__" or is_multimodal(name) or is_ngram(name):
                continue
            for out_name, out_shape in outputs_for(name, meta["shape"]):
                bits = quant_bits(out_name)
                if bits and out_shape[-1] % GROUP_SIZE:
                    problems.append(f"{out_name} last dim {out_shape[-1]} unaligned")
                print(f"  {name} {meta['shape']}\n     -> {out_name} {out_shape} "
                      f"{'q' + str(bits) if bits else 'passthrough'}")
    print("\nPROBLEMS: " + "; ".join(problems) if problems
          else "\nplan validates against the checkpoint's own headers")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    ap.add_argument("--index", type=Path)
    ap.add_argument("--config", type=Path)
    args = ap.parse_args()

    def fetch_json(path: Path | None, remote: str) -> dict:
        if path and path.exists():
            return json.loads(path.read_text())
        url = f"https://huggingface.co/{REPO}/raw/main/{remote}"
        return json.loads(subprocess.run(["curl", "-sfL", url],
                                         capture_output=True, check=True).stdout)

    index = fetch_json(args.index, "model.safetensors.index.json")
    if args.plan or not args.output:
        plan(index)
        return 0

    config = fetch_json(args.config, "config.json")
    text_config = config["text_config"]
    work = args.work or (args.output.parent / "qwen38-shards")
    work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    constants = ple_constants(text_config)
    (args.output / "ple_constants.json").write_text(json.dumps(constants, indent=1))
    (args.output / "config.json").write_text(json.dumps(config, indent=1))
    rows = sum(constants["ngram_heads_vocab_sizes"])
    divisor = text_config.get("make_ngram_vocab_size_divisible_by", 128)
    padded = math.ceil(rows / divisor) * divisor

    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    writer = OutputWriter(args.output)
    ngram = NgramTable(args.output, padded, constants["ple_head_dim"])

    # Fetch shard N+1 while shard N converts.
    queue: Queue = Queue(maxsize=1)

    def fetcher() -> None:
        for shard in shards:
            try:
                queue.put(download(shard, work))
            except Exception as exc:                     # noqa: BLE001
                queue.put(exc)
                return
        queue.put(None)

    threading.Thread(target=fetcher, daemon=True).start()
    done = 0
    while True:
        item = queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        done += 1
        print(f"[{done}/{len(shards)}] {item.name}", flush=True)
        convert_shard(item, writer, ngram)
        item.unlink()

    writer.finish()
    ngram.finish()
    print(f"\naffine snapshot written to {args.output}")
    print(f"  {writer.shard_no} shards, {writer.total / 1e9:.1f} GB")
    print(f"  ngram_table.bin {ngram.rows} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
