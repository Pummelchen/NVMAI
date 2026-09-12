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
import re
import signal
import struct
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

# Every Qwen3.5-MoE 35B-A3B release this converter builds. Pinned commits:
# the install receipt records the source, and a moved `main` must not
# silently change what "AgentWorld 4-bit" means.
MODELS = {
    "agentworld": ("Qwen/Qwen-AgentWorld-35B-A3B", "60d2b0434a53d2e62a7c00a489586815d94ebffb"),
    "qwen36":     ("Qwen/Qwen3.6-35B-A3B",         "995ad96eacd98c81ed38be0c5b274b04031597b0"),
    "ornith15":   ("ornith-ai/Ornith-1.5-35B-A3B",  "10fbf86fed7ecee4a061f8b499a618f46001cac1"),
    # KAT-Coder-V2.5-Dev is a Qwen3.6-35B-A3B fine-tune, so it has the same
    # geometry and the same tensor names (`model.language_model.*`, `lm_head`)
    # as the three above -- verified against its own index rather than assumed:
    # 31,333 tensors, none outside the namespaces this converter knows, no
    # `model.visual.*` and no `mtp.*`, so `skipped()` removes nothing here. Its
    # config declares a `vision_config` and the multimodal wrapper, which is why
    # the checkpoint was checked for vision tensors rather than trusted to lack
    # them.
    "katcoder":   ("Kwaipilot/KAT-Coder-V2.5-Dev",  "7be56fe773e72b6f5ca93c1ae45d828ddb893922"),
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


# Checkpoints ship the routed experts in one of two shapes, and only the first
# was handled here:
#
#   fused     `...mlp.experts.gate_up_proj`  [experts, 2F, hidden]
#             `...mlp.experts.down_proj`     [experts, hidden, F]
#   per-expert `...mlp.experts.<E>.gate_proj`  [F, hidden]
#              `...mlp.experts.<E>.up_proj`
#              `...mlp.experts.<E>.down_proj`  [hidden, F]
#
# The repacker and the runtime only understand the fused spelling, so a
# per-expert checkpoint must be stacked into it. KAT-Coder-V2.5-Dev is the
# per-expert shape; Qwen 3.6 / AgentWorld / Ornith are the fused one.
PER_EXPERT_RE = re.compile(
    r"^(?P<stem>.*\.mlp\.experts)\.(?P<expert>\d+)\.(?P<role>gate_proj|up_proj|down_proj)\.weight$")


def per_expert_routed(out_name: str) -> tuple[str, int, str] | None:
    """(fused target name, expert index, role) for a per-expert routed weight."""
    m = PER_EXPERT_RE.match(out_name)
    if not m:
        return None
    return (f"{m.group('stem').replace('.mlp.experts', '.mlp.switch_mlp')}"
            f".{m.group('role')}.weight", int(m.group("expert")), m.group("role"))


def routed_slices(name: str, value: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The routed-expert outputs this source tensor supplies.

    A fused source supplies all experts at once; a per-expert source supplies
    one expert, which the caller files into its slot. Both end up under the
    fused `switch_mlp` spelling, so the repacker sees exactly what it saw
    before.
    """
    routed = per_expert_routed(rename(name))
    if routed is not None:
        return [(routed[0], value)]
    return [(out_name, value) for out_name, _ in outputs_for(name, list(value.shape))]


class FusedExperts:
    """Stacks per-expert routed weights into the fused `switch_mlp` spelling.

    The repacker packs routed experts from *one tensor per layer per role*
    (`switch_mlp.{gate,up,down}_proj.weight`), which is how the fused
    checkpoints ship them. A per-expert checkpoint (KAT-Coder-V2.5-Dev) ships
    256 separate tensors per layer per role instead, so each is filed into its
    slot and the layer emitted once every expert has arrived.

    It has to happen here rather than in the repacker because the fused form is
    what the runtime's expert streaming reads. Without it the experts are
    treated as resident weights, the manifest declares `expertsPerLayer = 0`,
    and the model stops streaming from SSD -- the one thing this runtime is for.

    Only one layer's tensors are held: `release` drops an entry as soon as its
    last expert lands, so the peak is one layer's three projections (about
    1.5 GiB at 4-bit) plus the writer's open block.
    """

    def __init__(self) -> None:
        self._layers: dict[tuple[int, str], dict] = {}
        self._seen: set[str] = set()

    def add(self, name: str, value: np.ndarray, width: int,
            writer: "OutputWriter") -> None:
        if name in self._seen:
            raise ValueError(f"duplicate source tensor {name}")
        self._seen.add(name)
        # The expert index is read from the *source* name: `routed_slices`
        # retargets a per-expert tensor to its fused name, which no longer
        # carries one. Counting experts is what decides when a layer is done.
        routed = per_expert_routed(rename(name))
        for out_name, piece in routed_slices(name, value):
            target = self._layers.get((width, out_name))
            if target is None:
                target = {"stack": [], "experts": set()}
                self._layers[(width, out_name)] = target
            target["stack"].append(np.ascontiguousarray(piece))
            if routed is not None:
                target["experts"].add(routed[1])

    def release(self, experts_per_layer: int,
                writers: dict[int, "OutputWriter"]) -> None:
        """Emit every layer whose experts have all arrived, and forget it."""
        done = [key for key, target in self._layers.items()
                if len(target["experts"]) >= experts_per_layer]
        for key in done:
            width, out_name = key
            emit_fused(out_name, self._layers.pop(key)["stack"], width, writers)
        if done:
            for writer in writers.values():
                writer.flush()

    def pending(self) -> list[str]:
        """Fused tensors still waiting for experts, as `width:name (n/m)`.

        Non-empty at the end means the checkpoint disagrees with its own
        `num_experts`, or a shard was missed. Either way the install would be
        wrong, so `main` refuses to write the index.
        """
        return [f"{width}-bit {out_name} ({len(target['experts'])} experts)"
                for (width, out_name), target in sorted(self._layers.items())]


def emit_fused(out_name: str, stack: list[np.ndarray], width: int,
               writers: dict[int, "OutputWriter"]) -> None:
    """Quantize one fused tensor exactly as a fused checkpoint's would be."""
    if len(stack) == 1 and stack[0].ndim == 3:
        # The fused source supplies the whole expert axis in one tensor.
        fused = stack[0]
    else:
        # Stacking supplies the leading expert axis the per-expert form lacks.
        # A short stack means an expert went missing, so the shape check below
        # is what catches an incomplete layer rather than shipping one silently.
        fused = np.stack(stack)
    writer = writers[width]
    bits = quant_bits(out_name, width)
    stem = out_name[: -len(".weight")]
    if bits is None:
        writer.add(out_name, fold_unit_offset(out_name, fused))
        return
    packed, scales, biases = quantize_affine(fused, bits)
    writer.add(stem + ".weight", packed)
    writer.add(stem + ".scales", scales)
    writer.add(stem + ".biases", biases)



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
# the index fetch ended one 70 GB build before it started. `--http1.1` for the
# same reason the shard download forces it (see `download`).
RETRY = ["--retry", "5", "--retry-delay", "5", "--retry-all-errors", "--http1.1"]


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


_in_flight: "set[subprocess.Popen]" = set()
_in_flight_lock = threading.Lock()


def download(shard: str, work: Path) -> Path:
    """Fetch one shard whole, retrying from empty if an attempt fails.

    The curl child is tracked so a failure elsewhere can stop it: a download
    that outlives the converter keeps writing a shard for a run that has gone
    away. Several downloads run at once (see `prefetch_shards`), so every live
    child is tracked, not just the newest.

    **HTTP/1.1 is forced deliberately.** Hugging Face resets HTTP/2 streams on
    these 5.3 GB shards every few seconds (`curl: (92) HTTP/2 stream N reset by
    server (error 0x8 CANCEL)`), and a reset ends the transfer, so the run
    churns through retries instead of downloading. Measured on this link with
    the same 25 MB range: HTTP/2 managed 214 KB/s and stopped early, HTTP/1.1
    did 908 KB/s over the whole range.

    **Concurrency is what beats the throttle, not a bigger pipe.** This host
    limits each connection, and the link is not saturated by one: while a
    running job was pulling 205 KB/s, a second connection pulled 488 KB/s
    beside it. Four connections measured ~900 KB/s combined when each was
    capped lower. So the converter runs a small pool of these, and the pool
    size is the lever that matters.
    """
    dest = work / shard
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Every attempt starts from empty, and that is a deliberate trade of bytes
    # for correctness: `--remove-on-error` and `-C -` are mutually exclusive in
    # curl, and resume is the unsafe half here. A host that truncates a response
    # (`curl: (18) end of response with N bytes missing`) can also ignore the
    # Range a later `-C -` sends, so curl appends a fresh copy to the truncated
    # prefix and exits 0 on a file of the wrong length -- a shard that then
    # decodes into silently wrong weights. Re-fetching a dropped partial is
    # cheap next to that, and the pool keeps several shards in flight.
    #
    # curl's own retries cover a short hiccup; this loop covers an outage that
    # outlasts them (a DNS failure once ended a build at shard 6 of 16). The
    # wait backs off because a reset storm is not fixed by retrying faster.
    for attempt in range(1, 11):
        proc = subprocess.Popen(
            ["curl", "-fL", "--http1.1", "--retry", "20", "--retry-delay", "15",
             "--retry-all-errors", "--remove-on-error",
             "--silent", "--show-error", "-o", str(dest), f"{BASE}/{shard}"])
        with _in_flight_lock:
            _in_flight.add(proc)
        try:
            code = proc.wait()
        finally:
            with _in_flight_lock:
                _in_flight.discard(proc)
        if code == 0:
            return dest
        wait = min(30 * attempt, 300)
        print(f"    {shard}: curl exit {code} (attempt {attempt}/10), waiting {wait} s",
              flush=True)
        time.sleep(wait)
    raise subprocess.CalledProcessError(code, "curl")


def stop_download() -> None:
    """Terminate every live download. Several run at once in the worker pool."""
    with _in_flight_lock:
        procs = list(_in_flight)
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# How far ahead to download, and with how many connections. The host throttles
# per connection, so this is the lever on how long a conversion takes: one
# connection measured ~205 KB/s while a second beside it added ~490 KB/s, and
# four reached ~900 KB/s combined. Depth 3 holds about four 5.3 GB shards on
# disk (~22 GB) and leaves the main thread free to convert one while the others
# arrive. Raising it spends disk on bandwidth the link may not have.
PREFETCH_DEPTH = 3
FETCHERS = 3


def prefetch_shards(shards: list[str], fetch, fetchers: int = FETCHERS,
                    depth: int = PREFETCH_DEPTH):
    """Yield downloaded shards as they arrive, fetching the next ones meanwhile.

    A generator so the caller converts on its own thread while the pool keeps
    the link busy. Each shard is claimed from a shared queue, so N fetchers
    split the list instead of each downloading all of it; an exception from any
    fetcher is re-raised in the caller, where `stop_download` can clean up.

    Shards are yielded in *completion* order, not list order. That is safe here
    because a source shard is independent of the others: the converter writes
    each output tensor once, and the snapshot's index is built from what was
    actually written. Order only changes which layer completes first.
    """
    ready: Queue = Queue(maxsize=depth)
    pending: Queue = Queue()
    for shard in shards:
        pending.put(shard)

    def fetcher() -> None:
        try:
            while True:
                try:
                    shard = pending.get_nowait()
                except Exception:                        # noqa: BLE001
                    break
                ready.put(fetch(shard))
        except Exception as exc:                         # noqa: BLE001
            ready.put(exc)
        finally:
            ready.put(None)

    for _ in range(fetchers):
        threading.Thread(target=fetcher, daemon=True).start()
    sentinels = 0
    while True:
        item = ready.get()
        if isinstance(item, Exception):
            # The generator closes here, so the remaining fetchers unwind on
            # their next queue operation.
            raise item
        if item is None:
            # Count the sentinels that actually arrived; do not read a shared
            # counter. A fetcher posts its sentinel from a `finally`, so a
            # sentinel can reach this loop before another fetcher's shards are
            # queued, and ending the run on the first one would drop them.
            sentinels += 1
            if sentinels == fetchers:
                return
            continue
        yield item


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


def convert_shard(path: Path, writers: dict[int, OutputWriter],
                  fused: FusedExperts, experts_per_layer: int) -> None:
    """One source shard into every requested width; the tensor is read once."""
    with safe_open(path, framework="np") as src:
        for name in src.keys():
            if skipped(name):
                continue
            value = src.get_tensor(name)
            if per_expert_routed(rename(name)) is not None:
                # Emitted per completed layer rather than here: a per-expert
                # checkpoint supplies one expert per tensor, and the repacker
                # needs the whole layer's stack in one tensor.
                for width, writer in writers.items():
                    fused.add(name, value, width, writer)
                continue
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
    fused.release(experts_per_layer, writers)


def plan(index: dict, width: int) -> None:
    """Every tensor's fate, from the index alone. Checks group alignment."""
    wm = index["weight_map"]
    shards = sorted({s for n, s in wm.items() if not skipped(n)})
    headers = {s: fetch_header(s) for s in shards}
    counts: dict[str, int] = {}
    bf16_bytes = 0
    total_out = 0
    fused_sources = 0
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__" or skipped(name):
                continue
            if per_expert_routed(rename(name)) is not None:
                # Counted per source tensor here, but the snapshot carries one
                # fused tensor per layer per role; say so rather than let the
                # plan imply the repacker will see 30,720 separate experts.
                fused_sources += 1
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
    if fused_sources:
        print(f"  routed experts: {fused_sources} per-expert sources are fused "
              f"into one tensor per layer and role by the converter")
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
    # Routed experts that ship per-expert rather than fused are stacked back
    # into the one-tensor-per-layer shape the repacker plans from. The expert
    # count is the checkpoint's own `num_experts`, and it is only used here to
    # decide when a layer is complete.
    fused = FusedExperts()
    experts_per_layer = int(config["text_config"]["num_experts"])
    if experts_per_layer <= 0:
        raise SystemExit(f"unusable num_experts {experts_per_layer} in the checkpoint")

    # Fetch ahead with a small pool; see `prefetch_shards` for why concurrency
    # is the lever and how deep it goes.
    #
    # SIGTERM (pkill, a parent script dying) is not an exception in Python:
    # without this the curl children outlive the converter and keep writing
    # shards the next run resumes. Turn it into one so the handler below stops
    # them.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    done = 0
    try:
        for path in prefetch_shards(shards, lambda s: download(s, work)):
            done += 1
            print(f"[{done}/{len(shards)}] {path.name}", flush=True)
            convert_shard(path, writers, fused, experts_per_layer)
            path.unlink()
    except BaseException:
        stop_download()
        raise
    leftovers = fused.pending()
    if leftovers:
        # A layer whose experts did not all arrive would otherwise be dropped,
        # and the repacker would plan an install with an incomplete expert set
        # -- or silently treat the layer as resident. Fail before the index.
        raise SystemExit("incomplete routed-expert layers: "
                         + "; ".join(leftovers))
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
