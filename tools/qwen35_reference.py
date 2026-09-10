#!/usr/bin/env python3.13
"""A stateful numpy reference for Qwen3.5-2B, one token at a time.

Two jobs, and the second is the reason it exists.

**It checks the converter.** `prepare_qwen35.py` writes a snapshot from
the bf16 original with a great deal of quiet folding -- +1 into 148 norms, a
fused output gate, a tied embedding, K/V promoted to 8 bits inside the
4-bit build -- and until something runs a forward pass, none of that has
been tested by anything but inspection. This project has already shipped one
converter whose zero-centred norms were unfolded, and found it only when a
model answered nonsense.

**It is the oracle for the CPU engine.** The engine is debugged against this
file, layer by layer, at position 0 first and then across a sequence, which
is how every other family in this repo was brought up.

It is a sibling of `qwen38_reference.py` and deliberately shaped like it.
Qwen3.5-2B is the simpler cousin: the same Gated DeltaNet and the same full
attention every fourth layer, without the hyper-connections, the PLE, the
sparse-attention indexer or the mixture of experts.

    python3.13 tools/qwen35_reference.py                 # logits at position 0
    python3.13 tools/qwen35_reference.py --check         # known continuations
    python3.13 tools/qwen35_reference.py --dump acts.npz --tokens 5
    python3.13 tools/qwen35_reference.py --snapshot .build/qwen35-2b-affine-4bit
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / ".build/qwen35-2b-affine-8bit"
P = "language_model.model."


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class Shard:
    """A safetensors file, read directly.

    The library's numpy path cannot decode BF16, and every scale and bias in
    this snapshot is BF16 -- so the header is parsed and the payload
    memory-mapped here instead. That is not a workaround so much as the right
    shape for this job: a row of the tied embedding can be sliced out of the
    map without materializing 508 M parameters to read one vector.
    """

    def __init__(self, path: Path):
        self.file = open(path, "rb")
        size = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(size))
        self.base = 8 + size
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def raw(self, name: str, rows: slice | None = None) -> np.ndarray:
        meta = self.header[name]
        start, end = meta["data_offsets"]
        shape = meta["shape"]
        dtype = {"BF16": np.uint16, "F32": np.float32, "U32": np.uint32,
                 "F16": np.float16, "I32": np.int32}[meta["dtype"]]
        block = np.frombuffer(self.map, dtype=dtype,
                              count=(end - start) // np.dtype(dtype).itemsize,
                              offset=self.base + start).reshape(shape)
        if rows is not None:
            block = block[rows]
        return block

    def get(self, name: str, rows: slice | None = None) -> np.ndarray:
        """Values as float32, BF16 widened by bit pattern rather than by a
        library that may not know the type."""
        meta = self.header[name]
        block = self.raw(name, rows)
        if meta["dtype"] == "BF16":
            return (block.astype(np.uint32) << 16).view(np.float32)
        return block


class Weights:
    """The affine snapshot, dequantized on demand and kept.

    Reading it here rather than re-deriving from the bf16 original is the
    point: the engine will consume exactly these numbers, so a disagreement
    between this reference and the engine is a forward-pass bug and never a
    quantization difference.
    """

    def __init__(self, snapshot: Path):
        self.dir = Path(snapshot)
        self.config = json.loads((self.dir / "config.json").read_text())
        self.quantization = self.config["quantization"]
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.map = index["weight_map"]
        self.shards = {file: Shard(self.dir / file) for file in set(self.map.values())}
        self.cache: dict[str, np.ndarray] = {}

    def shard(self, name: str) -> Shard:
        return self.shards[self.map[name]]

    def _bits(self, stem: str) -> int:
        entry = self.quantization.get(stem)
        if isinstance(entry, dict):
            return entry["bits"]
        return self.quantization["bits"]

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        stem = name.removesuffix(".weight")
        shard = self.shard(name)
        if stem + ".scales" in self.map:
            value = dequantize(shard.raw(name),
                               self.shard(stem + ".scales").get(stem + ".scales"),
                               self.shard(stem + ".biases").get(stem + ".biases"),
                               self._bits(stem), self.quantization["group_size"])
        else:
            value = shard.get(name).astype(np.float32)
        self.cache[name] = value
        return value

    def rows(self, name: str, start: int, stop: int) -> np.ndarray:
        """A row range of a quantized matrix, without materializing the rest."""
        stem = name.removesuffix(".weight")
        window = slice(start, stop)
        return dequantize(
            self.shard(name).raw(name, window),
            self.shard(stem + ".scales").get(stem + ".scales", window),
            self.shard(stem + ".biases").get(stem + ".biases", window),
            self._bits(stem), self.quantization["group_size"])

    def row(self, name: str, index: int) -> np.ndarray:
        return self.rows(name, index, index + 1)[0]


def dequantize(packed, scales, biases, bits: int, group: int) -> np.ndarray:
    """The converter's layout: `bits`-wide unsigned lanes packed low-first
    inside each uint32, one BF16 scale and bias per group."""
    lanes = 32 // bits
    mask = (1 << bits) - 1
    rows = packed.shape[0]
    out = np.zeros((rows, packed.shape[1] * lanes), dtype=np.float32)
    for lane in range(lanes):
        out[:, lane::lanes] = ((packed >> (bits * lane)) & mask).astype(np.float32)
    grouped = out.reshape(rows, -1, group)
    return (grouped * scales.astype(np.float32)[..., None]
            + biases.astype(np.float32)[..., None]).reshape(rows, -1)


class Reference:
    def __init__(self, snapshot: Path = DEFAULT):
        self.w = Weights(snapshot)
        c = self.w.config
        self.layers = c["num_hidden_layers"]
        self.hidden = c["hidden_size"]
        self.eps = c["rms_norm_eps"]
        self.interval = c["full_attention_interval"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.rope_theta = float(c.get("rope_theta", 10_000_000.0))
        self.rotary = int(c.get("partial_rotary_factor", 0.25) * self.head_dim)
        self.hk = c["linear_num_key_heads"]
        self.hv = c["linear_num_value_heads"]
        self.dk = c["linear_key_head_dim"]
        self.dv = c["linear_value_head_dim"]
        self.conv_k = c["linear_conv_kernel_dim"]
        self.reset()

    def is_attention(self, layer: int) -> bool:
        return (layer + 1) % self.interval == 0

    def reset(self):
        self.position = 0
        self.gdn_conv: dict[int, np.ndarray] = {}
        self.gdn_state: dict[int, np.ndarray] = {}
        self.kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def rms_norm(self, x, gamma):
        return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps) * gamma

    def rope(self, vec, position):
        """NeoX rotation over the first `rotary` dimensions only.

        Partial rotary: 64 of 256 here. Rotating the whole head is the
        single most likely way to get a plausible-looking wrong model.
        """
        out = vec.copy()
        half = self.rotary // 2
        inv = self.rope_theta ** (-np.arange(half, dtype=np.float64) * 2.0 / self.rotary)
        angle = position * inv
        cos, sin = np.cos(angle), np.sin(angle)
        a = vec[..., :half]
        b = vec[..., half:self.rotary]
        out[..., :half] = a * cos - b * sin
        out[..., half:self.rotary] = a * sin + b * cos
        return out

    # ---------------------------------------------------------------- blocks

    def gdn(self, layer: int, x):
        prefix = f"{P}layers.{layer}.linear_attn."
        g = self.w.get
        qkv = g(prefix + "in_proj_qkv.weight") @ x
        z = g(prefix + "in_proj_z.weight") @ x
        a = g(prefix + "in_proj_a.weight") @ x
        b = g(prefix + "in_proj_b.weight") @ x

        conv_dim = qkv.shape[0]
        tail = self.gdn_conv.get(layer)
        if tail is None:
            tail = np.zeros((self.conv_k - 1, conv_dim), np.float32)
        window = np.concatenate([tail, qkv[None, :]], axis=0)
        conv_w = g(prefix + "conv1d.weight").reshape(-1, self.conv_k)
        conv_out = silu((conv_w * window.T).sum(axis=1))
        self.gdn_conv[layer] = window[1:]

        key_dim = self.hk * self.dk
        q = conv_out[:key_dim].reshape(self.hk, self.dk)
        k = conv_out[key_dim:2 * key_dim].reshape(self.hk, self.dk)
        v = conv_out[2 * key_dim:].reshape(self.hv, self.dv)

        def l2(t):
            return t / np.sqrt((t ** 2).sum(axis=-1, keepdims=True) + self.eps)

        q = np.repeat(l2(q), self.hv // self.hk, axis=0)
        k = np.repeat(l2(k), self.hv // self.hk, axis=0)

        beta = sigmoid(b)
        decay = np.exp(-np.exp(g(prefix + "A_log").astype(np.float64))
                       * softplus(a + g(prefix + "dt_bias")))
        state = self.gdn_state.get(layer)
        if state is None:
            state = np.zeros((self.hv, self.dv, self.dk), np.float32)
        state = state * decay[:, None, None]
        kv_mem = (state * k[:, None, :]).sum(axis=-1)
        delta = (v - kv_mem) * beta[:, None]
        state = state + k[:, None, :] * delta[:, :, None]
        y = (state * q[:, None, :]).sum(axis=-1)
        self.gdn_state[layer] = state

        y = y / np.sqrt(self.dv)
        # The gated linear-attention norm is NOT zero-centred, unlike every
        # other norm in this checkpoint, so the converter must not fold +1
        # into it and this must not add one either.
        yn = self.rms_norm(y, g(prefix + "norm.weight"))
        # SiLU, not sigmoid. This one line was the whole difference between a
        # model that answers and one that does not, and it is the family
        # hazard this project already knows about: the gate is `silu` in the
        # Qwen3-Next/3.6 lineage and `sigmoid` in Qwen3.8-Flash-Next, whose
        # reference this file was ported from. NVMAI's own kernel carries the
        # same choice as a function constant (`FC_GDN_SIGMOID_GATE`), so the
        # engine has to select it per family too.
        #
        # With sigmoid, "Once upon a" predicted a bare space and "The capital
        # of France is" predicted a colon; with silu, " time" and " Paris",
        # both top-1. Nothing else about the forward pass changed.
        out = yn * silu(z.reshape(self.hv, self.dv))
        return g(prefix + "out_proj.weight") @ out.reshape(-1)

    def attention(self, layer: int, x):
        prefix = f"{P}layers.{layer}.self_attn."
        g = self.w.get
        # q_proj is twice as wide: the output gate is fused into it, and
        # reading it as a plain q projection produces a model that runs and
        # is wrong.
        qg = (g(prefix + "q_proj.weight") @ x).reshape(self.heads, 2, self.head_dim)
        q, gate = qg[:, 0, :], qg[:, 1, :]
        k = (g(prefix + "k_proj.weight") @ x).reshape(self.kv_heads, self.head_dim)
        v = (g(prefix + "v_proj.weight") @ x).reshape(self.kv_heads, self.head_dim)

        q = self.rms_norm(q, g(prefix + "q_norm.weight"))
        k = self.rms_norm(k, g(prefix + "k_norm.weight"))
        q = self.rope(q, self.position)
        k = self.rope(k, self.position)

        keys, values = self.kv.get(layer, (None, None))
        keys = k[None] if keys is None else np.concatenate([keys, k[None]], 0)
        values = v[None] if values is None else np.concatenate([values, v[None]], 0)
        self.kv[layer] = (keys, values)

        group = self.heads // self.kv_heads
        out = np.empty((self.heads, self.head_dim), np.float32)
        scale = 1.0 / np.sqrt(self.head_dim)
        for h in range(self.heads):
            scores = (keys[:, h // group, :] @ q[h]) * scale
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[h] = weights @ values[:, h // group, :]
        out = out * sigmoid(gate)
        return g(prefix + "o_proj.weight") @ out.reshape(-1)

    def mlp(self, layer: int, x):
        prefix = f"{P}layers.{layer}.mlp."
        g = self.w.get
        return g(prefix + "down_proj.weight") @ (
            silu(g(prefix + "gate_proj.weight") @ x) * (g(prefix + "up_proj.weight") @ x))

    # ---------------------------------------------------------------- forward

    def step(self, token: int, dump: dict | None = None):
        g = self.w.get
        h = self.w.row(f"{P}embed_tokens.weight", token).astype(np.float32)
        if dump is not None:
            dump["embed"] = h.copy()
        for layer in range(self.layers):
            prefix = f"{P}layers.{layer}."
            xn = self.rms_norm(h, g(prefix + "input_layernorm.weight"))
            mixed = (self.attention(layer, xn) if self.is_attention(layer)
                     else self.gdn(layer, xn))
            h = h + mixed
            xn = self.rms_norm(h, g(prefix + "post_attention_layernorm.weight"))
            h = h + self.mlp(layer, xn)
            if dump is not None:
                dump[f"layer{layer}"] = h.copy()
        h = self.rms_norm(h, g(f"{P}norm.weight"))
        if dump is not None:
            dump["final"] = h.copy()
        # Tied: the embedding is the output projection. Dequantizing all
        # 508 M parameters would cost two gigabytes, so the logits are taken
        # in row blocks.
        logits = self.logits(h)
        self.position += 1
        return logits

    def logits(self, h, block: int = 8192):
        """The tied head, in row blocks.

        The embedding *is* the output projection here, and dequantizing all
        508 M parameters at once would cost two gigabytes for one vector.
        """
        name = f"{P}embed_tokens.weight"
        rows = self.w.shard(name).header[name]["shape"][0]
        out = np.empty(rows, np.float32)
        for start in range(0, rows, block):
            stop = min(start + block, rows)
            out[start:stop] = self.w.rows(name, start, stop) @ h
        return out


# Continuations a 2B model has no excuse for getting wrong, with the token
# ids taken straight from the snapshot's own vocabulary so no tokenizer
# library is needed. This is the check that says the *converter* worked: it
# exercises every fold, the fused gate, the tie and the quantization at once,
# and it is how the SiLU-versus-sigmoid gate bug was found.
CHECKS = [
    (["Once", "\u0120upon", "\u0120a"], "\u0120time"),
    (["The", "\u0120capital", "\u0120of", "\u0120France", "\u0120is"], "\u0120Paris"),
    (["The", "\u0120quick", "\u0120brown", "\u0120fox", "\u0120jumps",
      "\u0120over", "\u0120the", "\u0120lazy"], "\u0120dog"),
]


def check(snapshot: Path) -> int:
    vocab = json.loads((snapshot / "vocab.json").read_text())
    inverse = {i: t for t, i in vocab.items()}
    failures = 0
    for words, expected in CHECKS:
        reference = Reference(snapshot)
        logits = None
        for word in words:
            logits = reference.step(vocab[word])
        top = int(np.argmax(logits))
        prompt = "".join(w.replace("\u0120", " ") for w in words)
        mark = "ok " if top == vocab[expected] else "FAIL"
        failures += top != vocab[expected]
        print(f"  {mark} {prompt:44s} -> {inverse[top]!r} ({logits[top]:.2f}), "
              f"wanted {expected!r}")
    print("all continuations correct" if not failures
          else f"{failures} of {len(CHECKS)} wrong")
    return 0 if not failures else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(DEFAULT))
    ap.add_argument("--token", type=int, default=100, help="first token id")
    ap.add_argument("--tokens", type=int, default=1, help="steps to run")
    ap.add_argument("--dump", default=None, help="write per-layer activations here")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--check", action="store_true",
                    help="run known continuations as a self-test")
    args = ap.parse_args()

    if args.check:
        return check(Path(args.snapshot))

    reference = Reference(Path(args.snapshot))
    print(f"{args.snapshot}: {reference.layers} layers, hidden {reference.hidden}, "
          f"attention at "
          f"{[l for l in range(reference.layers) if reference.is_attention(l)]}")
    token = args.token
    dump: dict | None = {} if args.dump else None
    for step in range(args.tokens):
        logits = reference.step(token, dump=dump if step == 0 else None)
        order = np.argsort(-logits)[: args.top]
        print(f"  position {step}: token {token} -> "
              + ", ".join(f"{int(i)}:{logits[i]:.3f}" for i in order))
        token = int(order[0])
    if args.dump:
        np.savez(args.dump, **dump)
        print(f"wrote {args.dump} ({len(dump)} arrays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
