#!/usr/bin/env python3.13
"""Convert a Qwen3.5-MoE 35B-A3B release (Qwen-AgentWorld, Qwen3.6) from its
bf16 checkpoint into the affine 4-bit or 8-bit snapshot NVMAIRepack installs.

AgentWorld's text model is the Qwen3.5-MoE 35B-A3B geometry this runtime
already runs as the `qwen36` family (2048 hidden, 40 layers, 256 experts at
top-8, gated-DeltaNet with full attention every fourth layer). What differs
is the source: Qwen ships it as bf16 under the vision wrapper's tensor names,
with the routed experts fused per layer. So this is prepare_qwen38.py's job
-- fetch one shard at a time, quantize, write MLX-shaped output, delete the
shard -- with this family's renames and slot policy, and without Qwen3.8's
n-gram table, PLE constants and indexer.

Disk footprint while running: two source shards (~7 GB) plus the output
(about 19.5 GB at 4-bit, 37.8 GB at 8-bit). The install NVMAIRepack writes
afterwards is a second copy of the output.

    python3.13 tools/prepare_agentworld.py --plan            # no download
    python3.13 tools/prepare_agentworld.py --bits 4 \\
        --output .build/agentworld-affine-4bit --work .build/agentworld-shards
    python3.13 tools/prepare_agentworld.py --bits 4 8 \\
        --output .build/agentworld-affine --work .build/agentworld-shards
        # one download, two snapshots: <output>-4bit and <output>-8bit

Kept at the checkpoint's own bf16 in both widths -- "low-cost data at
16-bit where it helps": the router and the scalar shared-expert gate (they
produce decisions, and a decision either matches the reference or it does
not), the GDN a/b projections (the smallest projections, where a 64-wide
group has the least to amortise over), and every norm, conv tap, A_log and
dt_bias. Together about 45 MB of resident memory. The runtime reads these
tensors' width from their own dtype, which is what the Qwen3.8 8-bit build
established.
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
except ImportError as exc:  # pragma: no cover - environment, not logic
    sys.exit(f"missing dependency: {exc}\n"
             "  python3.13 -m pip install safetensors numpy ml_dtypes")

# Every Qwen3.5-MoE 35B-A3B release this converter builds. Pinned commits:
# the install receipt records the source, and a moved `main` must not
# silently change what "AgentWorld 4-bit" means.
MODELS = {
    "agentworld": ("Qwen/Qwen-AgentWorld-35B-A3B", "60d2b0434a53d2e62a7c00a489586815d94ebffb"),
    "qwen36":     ("Qwen/Qwen3.6-35B-A3B",         "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    "ornith15":   ("ornith-ai/Ornith-1.5-35B-A3B",  "10fbf86fed7ecee4a061f8b499a618f46001cac1"),
}
REPO = MODELS["agentworld"][0]
COMMIT = MODELS["agentworld"][1]
BASE = f"https://huggingface.co/{REPO}/resolve/{COMMIT}"


def select_model(name: str) -> None:
    global REPO, COMMIT, BASE
    REPO, COMMIT = MODELS[name]
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
    """Identical to prepare_qwen38.quantize_affine, deliberately.

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


DRAFT_HEAD = False   # --draft-head: convert only the `mtp.*` namespace
HEAD_BITS = BITS_8   # --head-bits: the embedding and lm_head slot


def is_draft_norm(name: str) -> bool:
    """Every norm in the draft head (the layer norms, q/k norms, `norm`, and
    `pre_fc_norm_embedding` / `pre_fc_norm_hidden`, whose names do not end
    in `norm`) is zero-centred and kept at bf16."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return "norm" in stem.rsplit(".", 1)[-1]


def skipped(name: str) -> bool:
    """The vision tower and the MTP draft head ride in Qwen3.6's index; the
    text model is what this snapshot is -- unless the draft head is."""
    if DRAFT_HEAD:
        return not name.startswith("mtp.")
    return name.startswith("model.visual.") or name.startswith("mtp.")


def rename(name: str) -> str:
    """Checkpoint name -> the MLX spelling the qwen36 family is repacked from."""
    if DRAFT_HEAD:
        # The qwen36 MTP sidecar is its own one-layer model: `mtp.` stripped,
        # nothing else, matching prepare_ornith_mtp.py's output.
        if name.startswith("mtp."):
            return name[len("mtp."):]
        raise ValueError(f"unexpected tensor outside the draft head: {name}")
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    prefix = "model.language_model."
    if name.startswith(prefix):
        return "language_model.model." + name[len(prefix):]
    raise ValueError(f"unexpected tensor outside the language model: {name}")


# Zero-centred RMSNorm: transformers' Qwen3_5MoeRMSNorm stores gamma - 1 and
# applies (1 + weight); the runtime, like the MLX checkpoints it was built
# against, applies the stored weight as is. Folded here. The gated
# linear-attention norm (Qwen3_5MoeRMSNormGated) is initialised at one and
# applied plainly, so it is not in this list -- its stored values centre on
# 0.9, where these centre on 0.
UNIT_OFFSET_NORM_SUFFIXES = (
    ".input_layernorm",
    ".post_attention_layernorm",
    ".self_attn.q_norm",
    ".self_attn.k_norm",
    "language_model.model.norm",
)


def fold_unit_offset(out_name: str, value: np.ndarray) -> np.ndarray:
    stem = out_name[: -len(".weight")] if out_name.endswith(".weight") else out_name
    # The draft head has no gated norm; every norm in it is zero-centred
    # (pre_fc_norm_embedding, pre_fc_norm_hidden, the layer norms, norm).
    if DRAFT_HEAD and is_draft_norm(out_name):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    if stem.endswith(UNIT_OFFSET_NORM_SUFFIXES):
        return (value.astype(np.float32) + 1.0).astype(value.dtype)
    return value


# Kept at bf16 in both widths. Suffixes of the renamed stem (no `.weight`).
KEEP_BF16 = (
    ".mlp.gate",                 # router
    ".mlp.shared_expert_gate",   # 1 row of D
    ".linear_attn.in_proj_a",
    ".linear_attn.in_proj_b",
)


def kept_bf16(name: str) -> bool:
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return stem.endswith(KEEP_BF16)


def quant_bits(name: str, width: int) -> int | None:
    """Bits for a renamed tensor, or None to copy it through at bf16.

    Mirrors the runtime's slot model for this family: embedding 8 (embed
    and head), router 8, attention / shared expert / routed expert at the
    build width. The bf16 keeps above override their slot; the runtime sizes
    those tensors from their own dtype.
    """
    if not name.endswith(".weight"):
        return None                                   # A_log, dt_bias
    if name.endswith("conv1d.weight") or name.endswith("norm.weight"):
        return None
    if DRAFT_HEAD and is_draft_norm(name):
        return None
    if DRAFT_HEAD:
        # Same policy as the Ornith draft: every weight at the build width,
        # norms bf16. The MTP loader sizes every slot from the base width.
        return width
    if kept_bf16(name):
        return None
    if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
        return HEAD_BITS
    return width


def outputs_for(name: str, shape: list[int]) -> list[tuple[str, list[int]]]:
    new = rename(name)
    if new.endswith(".mlp.experts.gate_up_proj"):
        # [experts, 2F, hidden]: gate rows first, then up.
        stem = new[: -len("experts.gate_up_proj")] + "switch_mlp."
        experts, fused, hidden = shape
        return [(stem + "gate_proj.weight", [experts, fused // 2, hidden]),
                (stem + "up_proj.weight", [experts, fused // 2, hidden])]
    if new.endswith(".mlp.experts.down_proj"):
        stem = new[: -len("experts.down_proj")] + "switch_mlp."
        return [(stem + "down_proj.weight", list(shape))]
    return [(new, list(shape))]


def write_config(config: dict, out: Path, tensor_names, width: int) -> dict:
    """config.json with the `quantization` block NVMAIRepack reads: a base
    width plus every tensor whose width differs, keyed by stem."""
    overrides = {}
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        bits = quant_bits(name, width)
        if bits is None or bits == width:
            continue
        overrides[name[: -len(".weight")]] = {"bits": bits, "group_size": GROUP_SIZE}
    config = dict(config)
    if DRAFT_HEAD:
        config["model_type"] = "qwen3_5_mtp"
        config["architectures"] = ["Qwen3_5MoeMTP"]
    config["quantization"] = {
        "bits": width, "group_size": GROUP_SIZE, "mode": "affine", **overrides,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))
    return config


# --- transport ---------------------------------------------------------------


# Small fetches retry like the shard download does; a single TLS hiccup on
# the index fetch ended one 70 GB build before it started.
RETRY = ["--retry", "5", "--retry-delay", "5", "--retry-all-errors"]


def fetch_json(remote: str) -> dict:
    raw = subprocess.run(["curl", "-sfL", "--max-time", "120", *RETRY, f"{BASE}/{remote}"],
                         capture_output=True, check=True).stdout
    return json.loads(raw)


def fetch_header(shard: str) -> dict:
    url = f"{BASE}/{shard}"
    raw = subprocess.run(["curl", "-sfL", "--max-time", "60", *RETRY, "-r", "0-7", url],
                         capture_output=True, check=True).stdout
    size = struct.unpack("<Q", raw[:8])[0]
    body = subprocess.run(["curl", "-sfL", "--max-time", "180", *RETRY, "-r", f"8-{8 + size - 1}", url],
                          capture_output=True, check=True).stdout
    return json.loads(body)


_in_flight: subprocess.Popen | None = None


def download(shard: str, work: Path) -> Path:
    """Fetch one shard, resuming a partial file rather than restarting it.

    The curl child is tracked so a failure elsewhere can stop it: a download
    that outlives the converter keeps appending to a file the next run
    resumes, and the shard then fails to deserialize.
    """
    global _in_flight
    dest = work / shard
    dest.parent.mkdir(parents=True, exist_ok=True)
    _in_flight = subprocess.Popen(
        ["curl", "-fL", "--retry", "5", "--retry-delay", "5",
         "--retry-all-errors", "-C", "-", "--silent", "--show-error",
         "-o", str(dest), f"{BASE}/{shard}"])
    code = _in_flight.wait()
    _in_flight = None
    if code != 0:
        raise subprocess.CalledProcessError(code, "curl")
    return dest


def stop_download() -> None:
    proc = _in_flight
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def fetch_tokenizer(out: Path) -> None:
    for name, required in TOKENIZER_FILES:
        result = subprocess.run(["curl", "-sfL", "--max-time", "300", *RETRY, f"{BASE}/{name}"],
                                capture_output=True)
        if result.returncode != 0 or not result.stdout:
            if required:
                raise SystemExit(f"cannot fetch {name} from {REPO}; the "
                                 "snapshot would be rejected at repack")
            continue
        (out / name).write_bytes(result.stdout)
        print(f"  {name} ({len(result.stdout) / 1e6:.2f} MB)")


# --- conversion --------------------------------------------------------------


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
                if out_name.endswith("switch_mlp.gate_proj.weight"):
                    piece = value[:, : value.shape[1] // 2, :]
                elif out_name.endswith("switch_mlp.up_proj.weight"):
                    piece = value[:, value.shape[1] // 2:, :]
                else:
                    piece = value
                piece = np.ascontiguousarray(piece)
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
    """Every tensor's fate, from the index alone. Checks group alignment."""
    wm = index["weight_map"]
    shards = sorted({s for n, s in wm.items() if not skipped(n)})
    headers = {s: fetch_header(s) for s in shards}
    counts: dict[str, int] = {}
    bf16_bytes = 0
    total_out = 0
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__" or skipped(name):
                continue
            for out_name, out_shape in outputs_for(name, meta["shape"]):
                bits = quant_bits(out_name, width)
                n = int(np.prod(out_shape))
                if bits is None:
                    kind = "bf16"
                    bf16_bytes += n * 2
                    total_out += n * 2
                else:
                    if out_shape[-1] % GROUP_SIZE:
                        raise SystemExit(f"{out_name}: last dim {out_shape[-1]} "
                                         f"not a multiple of {GROUP_SIZE}")
                    kind = f"{bits}-bit"
                    total_out += n * bits // 8 + (n // GROUP_SIZE) * 4
                counts[kind] = counts.get(kind, 0) + 1
    print(f"{len(shards)} shards, {sum(len(h) - 1 for h in headers.values())} tensors")
    for kind, n in sorted(counts.items()):
        print(f"  {kind:>6}: {n} tensors")
    print(f"  bf16 kept: {bf16_bytes / 1e6:.0f} MB")
    print(f"  output: ~{total_out / 1e9:.1f} GB at {width}-bit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="agentworld",
                    help="which release to convert (pinned repo and commit)")
    ap.add_argument("--head-bits", type=int, choices=(4, 8), default=8,
                    help="embedding and lm_head width (default 8; the head is ~0.5 GB "
                         "at 8-bit and ~3.5 ms of every 35B token)")
    ap.add_argument("--draft-head", action="store_true",
                    help="convert the mtp.* draft head only, as a qwen3_5_mtp sidecar snapshot")
    ap.add_argument("--plan", action="store_true", help="classify from the index, download nothing")
    ap.add_argument("--bits", type=int, choices=(4, 8), nargs="+", default=[4],
                    help="one width, or both to write two snapshots from one download")
    ap.add_argument("--output", type=Path,
                    help="snapshot directory; with two widths, a prefix that gets -4bit/-8bit")
    ap.add_argument("--work", type=Path, help="scratch for in-flight shards")
    args = ap.parse_args()
    select_model(args.model)
    global DRAFT_HEAD, HEAD_BITS
    DRAFT_HEAD = args.draft_head
    HEAD_BITS = args.head_bits

    config = fetch_json("config.json")
    if config.get("model_type") != "qwen3_5_moe":
        raise SystemExit(f"unexpected model_type {config.get('model_type')!r}")
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
    # Shards that carry only skipped tensors (vision, MTP) are not fetched.
    shards = sorted({s for n, s in index["weight_map"].items() if not skipped(n)})
    if len(widths) == 1:
        outputs = {widths[0]: args.output}
    else:
        outputs = {w: Path(f"{args.output}-{w}bit") for w in widths}
    writers = {w: OutputWriter(out) for w, out in outputs.items()}

    # Fetch shard N+1 while shard N converts; each shard is deleted once
    # converted, so at most two are on disk.
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
    try:
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            done += 1
            print(f"[{done}/{len(shards)}] {item.name}", flush=True)
            convert_shard(item, writers)
            item.unlink()
    except BaseException:
        stop_download()
        raise
    for width, writer in writers.items():
        out = outputs[width]
        writer.finish()
        write_config(config, out, writer.index.keys(), width)
        if not DRAFT_HEAD:            # a draft is prompted through its target's tokenizer
            print(f"tokenizer ({width}-bit):")
            fetch_tokenizer(out)
        print(f"\naffine snapshot written to {out}")
        print(f"  {writer.shard_no} shards, {writer.total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
