#!/usr/bin/env python3.13
"""A stateful numpy reference for Qwen3.5-2B, one token at a time.

Two jobs, and the second is the reason it exists.

**It checks the converter.** `prepare_qwen35_2b.py` writes a snapshot from
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
    python3.13 tools/qwen35_reference.py --dump acts.npz --tokens 5
    python3.13 tools/qwen35_reference.py --snapshot .build/qwen35-2b-affine-4bit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / ".build/qwen35-2b-affine-8bit"
P = "language_model.model."


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


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
        self.files = {name: safe_open(str(self.dir / file), framework="np")
                      for name, file in
                      {f: f for f in set(self.map.values())}.items()}
        self.cache: dict[str, np.ndarray] = {}

    def _bits(self, stem: str) -> int:
        entry = self.quantization.get(stem)
        if isinstance(entry, dict):
            return entry["bits"]
        return self.quantization["bits"]

    def get(self, name: str) -> np.ndarray:
        if name in self.cache:
            return self.cache[name]
        handle = self.files[self.map[name]]
        raw = handle.get_tensor(name)
        stem = name.removesuffix(".weight")
        if stem + ".scales" in self.map:
            scales = self.files[self.map[stem + ".scales"]].get_tensor(stem + ".scales")
            biases = self.files[self.map[stem + ".biases"]].get_tensor(stem + ".biases")
            value = dequantize(raw, scales, biases, self._bits(stem),
                               self.quantization["group_size"])
        else:
            value = raw.astype(np.float32)
        self.cache[name] = value
        return value

    def row(self, name: str, index: int) -> np.ndarray:
        """One row of a quantized matrix, without materializing the rest.

        The tied embedding is 508 M parameters; dequantizing all of it to
        read one token would cost two gigabytes for one vector.
        """
        stem = name.removesuffix(".weight")
        handle = self.files[self.map[name]]
        raw = handle.get_tensor(name)[index : index + 1]
        scales = self.files[self.map[stem + ".scales"]].get_tensor(
            stem + ".scales")[index : index + 1]
        biases = self.files[self.map[stem + ".biases"]].get_tensor(
            stem + ".biases")[index : index + 1]
        return dequantize(raw, scales, biases, self._bits(stem),
                          self.quantization["group_size"])[0]


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
        out = yn * sigmoid(z.reshape(self.hv, self.dv))
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
        name = f"{P}embed_tokens.weight"
        stem = name.removesuffix(".weight")
        handle = self.w.files[self.w.map[name]]
        packed = handle.get_tensor(name)
        scales = self.w.files[self.w.map[stem + ".scales"]].get_tensor(stem + ".scales")
        biases = self.w.files[self.w.map[stem + ".biases"]].get_tensor(stem + ".biases")
        bits = self.w._bits(stem)
        group = self.w.quantization["group_size"]
        rows = packed.shape[0]
        out = np.empty(rows, np.float32)
        for start in range(0, rows, block):
            stop = min(start + block, rows)
            weights = dequantize(packed[start:stop], scales[start:stop],
                                 biases[start:stop], bits, group)
            out[start:stop] = weights @ h
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=str(DEFAULT))
    ap.add_argument("--token", type=int, default=100, help="first token id")
    ap.add_argument("--tokens", type=int, default=1, help="steps to run")
    ap.add_argument("--dump", default=None, help="write per-layer activations here")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

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
