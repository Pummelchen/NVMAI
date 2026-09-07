#!/usr/bin/env python3.13
"""Convert Qwen3.5-2B from its bf16 checkpoint into an affine snapshot.

This is the small resident model the memory shadow runs on the CPU, and it
is the same architecture family the 35B installs already use: 2048 hidden,
gated-DeltaNet with gated full attention every fourth layer, a fused output
gate in `q_proj`, and rotary confined to the first 64 of 256 head elements.
Three things differ from `prepare_agentworld.py`, and they are the whole of
this file's reason to exist:

  * **Dense feed-forward.** 24 layers of `mlp.{gate,up,down}_proj` at 6144,
    where the 35B has a router, 256 routed experts and a shared expert.
    Nothing here fuses or splits an expert tensor.
  * **Tied output.** The checkpoint ships no `lm_head`; the card says the LM
    output is tied to the token embedding. The snapshot records that rather
    than duplicating 508M parameters, so the reader must honour the tie.
  * **One shard.** 4.5 GB, so the pipelined fetch-while-converting of the
    35B converters is kept but rarely exercised.

Verified against the checkpoint before this was written, not assumed:
`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` and the
final `norm` are stored zero-centred (measured means +0.096, +0.046, +0.423,
+2.536) and the +1 is folded in here, exactly as the 35B converters do.
`linear_attn.norm` is the gated norm, stored around one (measured mean
+0.916, minimum +0.488) and is left alone. Getting that backwards was the
only bug in the AgentWorld port.

    python3.13 tools/prepare_qwen35_2b.py --plan
    python3.13 tools/prepare_qwen35_2b.py --bits 8 \\
        --output .build/qwen35-2b-affine-8bit --work .build/qwen35-2b-shards
    python3.13 tools/prepare_qwen35_2b.py --bits 4 8 \\
        --output .build/qwen35-2b-affine --work .build/qwen35-2b-shards

Disk while running: one source shard (4.5 GB) plus the output (about 1.3 GB
at 4-bit, 2.4 GB at 8-bit).
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
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

# Pinned: the install receipt records the source, and a moved `main` must not
# silently change what "Qwen3.5-2B 8-bit" means.
REPO = "Qwen/Qwen3.5-2B"
COMMIT = "15852e8c16360a2fea060d615a32b45270f8a8fc"
BASE = f"https://huggingface.co/{REPO}/resolve/{COMMIT}"

GROUP_SIZE = 64
BITS_4, BITS_8 = 4, 8
OUTPUT_SHARD_BYTES = 4 << 30

TOKENIZER_FILES = (
    ("tokenizer.json", True),
    ("tokenizer_config.json", True),
    ("special_tokens_map.json", False),
    ("chat_template.jinja", False),
    ("chat_template.json", False),
    ("generation_config.json", False),
    ("vocab.json", False),
    ("merges.txt", False),
)


# --- quantization ------------------------------------------------------------


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
    """Identical to prepare_agentworld.quantize_affine, deliberately.

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


# --- naming ------------------------------------------------------------------


HEAD_BITS = BITS_8      # --head-bits: the tied embedding slot
WITH_MTP = False        # --mtp: also convert the one-layer draft head

# Tensors carried a width above the build's, because measurement says they
# are worth it (tools/precision_plan_qwen35.py, against the bf16 source):
#
#   kind        err@4    err@8    MB@4   error removed per MB by 4 -> 8
#   v_proj     0.1231   0.0067       3   0.037
#   k_proj     0.1173   0.0067       3   0.035
#   o_proj     0.0967   0.0062      13   0.007
#   q_proj     0.0903   0.0057      25   0.003
#   mlp.*      0.094-0.099          151   0.0006
#
# K and V are the worst at 4-bit and the smallest, so promoting them is an
# order of magnitude the best trade on the board -- six megabytes removes
# 94% of their error. They are also the only projections whose output is
# cached and reused for every later token: with two KV heads shared by eight
# query heads, an error there is in the context for the rest of the session,
# where an error in a feed-forward is spent on one token.
#
# Nothing is promoted above 8-bit. At 8-bit every kind measures 0.006 error
# and 0.99998 direction agreement; bf16 would double the bytes to remove
# what is already three orders below the 4-bit case.
PROMOTE_TO_8BIT = (
    ".self_attn.k_proj",
    ".self_attn.v_proj",
)
PROMOTE = True          # --no-promote: build a uniform-width snapshot


def skipped(name: str) -> bool:
    """The vision tower is 297 of the checkpoint's 632 tensors and this
    snapshot is the text model. The MTP draft head rides along too; it is
    only converted when asked for, because nothing reads it yet."""
    if name.startswith("model.visual."):
        return True
    if name.startswith("mtp."):
        return not WITH_MTP
    return False


def rename(name: str) -> str:
    """Checkpoint name -> the MLX spelling this family is repacked from."""
    if name.startswith("mtp."):
        return "mtp." + name[len("mtp."):]
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "language_model.model." + name[len(prefix):]
    raise ValueError(f"unexpected tensor outside the language model: {name}")


# Zero-centred RMSNorm: transformers stores gamma - 1 and applies
# (1 + weight); the runtime applies the stored weight as is, so the +1 is
# folded here. Measured on this checkpoint (means +0.096, +0.046, +0.423,
# +2.536). The gated linear-attention norm is initialised at one and applied
# plainly -- measured mean +0.916, minimum +0.488 -- so it is not in this
# list. Reversing these two is the bug that broke the AgentWorld port.
UNIT_OFFSET_NORM_SUFFIXES = (
    ".input_layernorm",
    ".post_attention_layernorm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    "language_model.model.norm",
)


def is_mtp_norm(name: str) -> bool:
    """Every norm in the draft head is zero-centred, including
    `pre_fc_norm_embedding` and `pre_fc_norm_hidden`, whose names do not end
    in `norm`."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return name.startswith("mtp.") and "norm" in stem.rsplit(".", 1)[-1]


def fold_unit_offset(out_name: str, value: np.ndarray) -> np.ndarray:
    stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
    if is_mtp_norm(out_name) or stem.endswith(UNIT_OFFSET_NORM_SUFFIXES):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    return value


# Kept at bf16 in both widths. The 35B also keeps its router and shared-expert
# scalar gate; a dense model has neither. What remains is the pair of
# 16-row gated-DeltaNet projections, where a 64-wide group has almost nothing
# to amortise over. Together under 1 MB.
KEEP_BF16 = (
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
)


def kept_bf16(name: str) -> bool:
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return stem.endswith(KEEP_BF16)


def quant_bits(name: str, width: int) -> int | None:
    """Bits for a renamed tensor, or None to copy it through at bf16."""
    if not name.endswith(".weight"):
        return None                                   # A_log, dt_bias
    if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
        return None
    if is_mtp_norm(name):
        return None
    if kept_bf16(name):
        return None
    if name.endswith("embed_tokens.weight"):
        return HEAD_BITS                              # tied: this is the head too
    if PROMOTE and width < BITS_8:
        stem = name[: -len(".weight")]
        if stem.endswith(PROMOTE_TO_8BIT):
            return BITS_8
    return width


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    """One in, one out. The 35B converters split a fused expert tensor here;
    a dense feed-forward has nothing to split."""
    return [(rename(name), list(shape))]


def write_config(config: dict, out: Path, tensor_names, width: int) -> dict:
    """config.json with the `quantization` block the reader wants: a base
    width plus every tensor whose width differs, keyed by stem.

    The text config is lifted to the top level, because the checkpoint nests
    it under a vision-language wrapper this snapshot does not carry. The tie
    is recorded rather than resolved: the reader uses the embedding as the
    output projection, and a reader that cannot must be told, not silently
    handed 508M duplicated parameters.
    """
    text = dict(config.get("text_config", config))
    text["model_type"] = "qwen3_5_dense"
    text["architectures"] = ["Qwen3_5DenseForCausalLM"]
    text["tie_word_embeddings"] = True
    text["source_repo"] = REPO
    text["source_commit"] = COMMIT
    # The checkpoint states neither of these: transformers supplies them from
    # the architecture's own defaults, and a snapshot that inherits a default
    # it never wrote down is a snapshot whose rotation depends on which
    # library version reads it. Written explicitly, with the source of the
    # values named, so a reader here and a reader in a year agree.
    #
    # Both match this project's Qwen 3.6 install manifest (`arch.ropeTheta`
    # 10000000, `arch.partialRotaryFactor` 0.25), which is the same
    # architecture family, and the Qwen3.8-Flash-Next reference in
    # `tools/qwen38_reference.py`. They cannot be checked at position 0,
    # where the rotation is the identity whatever they are, so they are
    # checked by sequence parity instead.
    text.setdefault("rope_theta", 10_000_000.0)
    text.setdefault("partial_rotary_factor", 0.25)
    text["rope_constants_source"] = ("architecture default; absent from "
                                     f"{REPO}@{COMMIT[:7]}/config.json")
    overrides = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        bits = quant_bits(name, width)
        if bits is None or bits == width:
            continue
        overrides[name[: -len(".weight")]] = {"bits": bits, "group_size": GROUP_SIZE}
    text["quantization"] = {
        "bits": width, "group_size": GROUP_SIZE, "mode": "affine", **overrides,
    }
    (out / "config.json").write_text(json.dumps(text, indent=1))
    return text


# --- transport ---------------------------------------------------------------


# Small fetches retry like the shard download does; a single TLS hiccup on
# the index fetch ended one 70 GB build before it started.
RETRY = ["--retry", "5", "--retry-delay", "5", "--retry-all-errors"]


def fetch_json(remote: str) -> dict:
    raw = subprocess.run(["curl", "-sfL", "--max-time", "120", *RETRY, f"{BASE}/{remote}"],
                         capture_output=True, check=True).stdout
    return json.loads(raw)


_download: subprocess.Popen | None = None


def download(shard: str, work: Path) -> Path:
    global _download
    target = work / shard
    if target.exists():
        print(f"  have {shard}", flush=True)
        return target
    print(f"  fetching {shard}", flush=True)
    started = time.time()
    _download = subprocess.Popen(
        ["curl", "-sfL", *RETRY, "-o", str(target), f"{BASE}/{shard}"])
    if _download.wait() != 0:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"download failed: {shard}")
    _download = None
    size = target.stat().st_size / 1e9
    print(f"    {size:.2f} GB in {time.time() - started:.0f}s", flush=True)
    return target


def stop_download() -> None:
    if _download and _download.poll() is None:
        _download.terminate()


def fetch_tokenizer(out: Path) -> None:
    for name, required in TOKENIZER_FILES:
        done = subprocess.run(
            ["curl", "-sfL", "--max-time", "300", *RETRY, "-o", str(out / name),
             f"{BASE}/{name}"])
        if done.returncode != 0:
            (out / name).unlink(missing_ok=True)
            if required:
                raise RuntimeError(f"could not fetch {name}")


# --- output ------------------------------------------------------------------


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
        final = {}
        for old_key, old_name in self.index.items():
            n = int(old_name.split("-")[1].split(".")[0])
            final[old_key] = f"model-{n:05d}-of-{self.shard_no:05d}.safetensors"
        for n in range(1, self.shard_no + 1):
            src = self.out / f"model-{n:05d}.safetensors"
            src.rename(self.out / f"model-{n:05d}-of-{self.shard_no:05d}.safetensors")
        (self.out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": self.total}, "weight_map": final}, indent=1))


def convert_shard(path: Path, writers: dict[int, OutputWriter]) -> None:
    """One source shard into every requested width; the tensor is read once."""
    with safe_open(path, framework="np") as src:
        for name in src.keys():
            if skipped(name):
                continue
            value = src.get_tensor(name)
            for out_name, _ in outputs_for(name, list(value.shape)):
                piece = np.ascontiguousarray(value)
                for width, writer in writers.items():
                    bits = quant_bits(out_name, width)
                    if bits is None:
                        writer.add(out_name, fold_unit_offset(out_name, piece))
                        continue
                    stem = out_name[: -len(".weight")]
                    packed, scales, biases = quantize_affine(piece, bits)
                    writer.add(stem + ".weight", packed)
                    writer.add(stem + ".scales", scales)
                    writer.add(stem + ".biases", biases)


def plan(index: dict, width: int) -> None:
    """Classify every tensor from the index alone: no download, no memory."""
    kinds: dict[str, list[int]] = {}
    total = 0
    for name in sorted(index["weight_map"]):
        if skipped(name):
            kinds.setdefault("skipped", [0, 0])[0] += 1
            continue
        out_name = rename(name)
        bits = quant_bits(out_name, width)
        kind = "bf16" if bits is None else f"{bits}-bit"
        kinds.setdefault(kind, [0, 0])[0] += 1
        total += 1
    print(f"\n{width}-bit plan for {REPO} @ {COMMIT[:8]}")
    for kind, (count, _) in sorted(kinds.items()):
        print(f"  {kind:9s} {count:4d} tensors")
    print(f"  {'converted':9s} {total:4d} tensors")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-bits", type=int, choices=(4, 8), default=8,
                    help="tied embedding width (default 8; it is 508M parameters "
                         "and serves as both the embedding and the output head)")
    ap.add_argument("--no-promote", action="store_true",
                    help="uniform build width; by default a 4-bit snapshot keeps "
                         "k_proj and v_proj at 8-bit, which costs 6 MB and removes "
                         "94%% of their error (tools/precision_plan_qwen35.py)")
    ap.add_argument("--mtp", action="store_true",
                    help="also convert the one-layer mtp.* draft head")
    ap.add_argument("--plan", action="store_true",
                    help="classify from the index, download nothing")
    ap.add_argument("--bits", type=int, choices=(4, 8), nargs="+", default=[8],
                    help="one width, or both to write two snapshots from one download")
    ap.add_argument("--output", type=Path,
                    help="snapshot directory; with two widths, a prefix that gets -4bit/-8bit")
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    args = ap.parse_args()
    global HEAD_BITS, WITH_MTP, PROMOTE
    HEAD_BITS = args.head_bits
    WITH_MTP = args.mtp
    PROMOTE = not args.no_promote

    config = fetch_json("config.json")
    if config.get("model_type") != "qwen3_5":
        raise SystemExit(f"unexpected model_type {config.get('model_type')!r}")
    text = config.get("text_config", {})
    if text.get("num_hidden_layers") != 24 or text.get("hidden_size") != 2048:
        raise SystemExit(
            "unexpected geometry: "
            f"{text.get('num_hidden_layers')} layers of {text.get('hidden_size')}")
    index = fetch_json("model.safetensors.index.json")
    widths = sorted(set(args.bits))
    if args.plan:
        for width in widths:
            plan(index, width)
        return 0
    if not args.output or not args.work:
        ap.error("--output and --work are required unless --plan")
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    shards = sorted({s for n, s in index["weight_map"].items() if not skipped(n)})
    if len(widths) == 1:
        outputs = {widths[0]: args.output}
    else:
        outputs = {w: Path(f"{args.output}-{w}bit") for w in widths}
    writers = {w: OutputWriter(out) for w, out in outputs.items()}

    queue: Queue = Queue(maxsize=1)

    def fetcher() -> None:
        for shard in shards:
            try:
                queue.put(download(shard, work))
            except Exception as exc:                     # noqa: BLE001
                queue.put(exc)
                return
        queue.put(None)

    def interrupted(*_: object) -> None:
        stop_download()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupted)
    thread = threading.Thread(target=fetcher, daemon=True)
    thread.start()
    started = time.time()
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            print(f"  converting {item.name}", flush=True)
            convert_shard(item, writers)
            item.unlink(missing_ok=True)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    for width, writer in writers.items():
        writer.finish()
        out = outputs[width]
        write_config(config, out, writer.index.keys(), width)
        fetch_tokenizer(out)
        print(f"  {out}: {writer.total / 1e9:.2f} GB, {len(writer.index)} tensors")
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
