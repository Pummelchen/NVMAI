"""A stateful numpy reference for Qwen3.8-Flash-Next, one token at a time.

The stateless checks in `qwen38_full_forward.py` only exercise position 0,
where every carried state is empty. This one threads all four of them -- the
KV cache, the delta-rule recurrent state, the Gated DeltaNet convolution tail
and the PLE convolution history -- so it can tell a wrong layer from a wrong
hand-off between tokens, which is the failure the first harness is blind to.

Weights come from the install, so quantization is common to both sides and any
disagreement is in the forward pass.
"""
import json
from pathlib import Path

import numpy as np

from gturbo_reader import GTurboWeights, PackedExperts

HC, D, EPS = 4, 2560, 1e-6
HC_DIM = HC * D
NUM_LAYERS = 48
FULL_ATTENTION_EVERY = 4
PLE_LAYERS = {1}
HK, HV, DK, DV, GDN_K = 16, 48, 128, 128, 4
N_HEADS, N_KV_HEADS, HEAD_DIM, N_ROT = 24, 2, 256, 64
ROPE_THETA = 10_000_000.0
TOP_K, PLE_K, PLE_DILATION = 10, 4, 3
INDEXER_HEADS, INDEXER_DIM = 4, 128
INDEXER_RATIO, INDEXER_BUDGET = 4, 2048
P = "model.language_model."


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def grouped_rms_norm(x_flat, gamma):
    x = x_flat.reshape(HC, D)
    x = x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS)
    return x.reshape(-1) * gamma


def rms_norm(x, gamma):
    return x / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + EPS) * gamma


def rope_full(vec, position):
    """NeoX rotation over the whole vector (the indexer ropes all 128 dims)."""
    half = vec.shape[-1] // 2
    inv = ROPE_THETA ** (-np.arange(half, dtype=np.float64) / half)
    angle = position * inv
    cos, sin = np.cos(angle), np.sin(angle)
    out = vec.copy()
    a, b = vec[:half], vec[half:]
    out[:half] = a * cos - b * sin
    out[half:] = a * sin + b * cos
    return out


def ngram_rows(context, multipliers, offsets, vocab_sizes,
               ngram_size, heads_per_ngram, eos_id):
    """Row ids for one token. `context[s]` is the token s positions back."""
    ctx = np.empty(ngram_size, dtype=np.int64)
    ctx[0] = context[0]
    cut = False
    for s in range(1, ngram_size):
        raw = context[s] if s < len(context) else -1
        t = -1 if cut else raw
        cut = cut or t < 0 or t == eos_id
        ctx[s] = eos_id if cut else t
    with np.errstate(over="ignore", invalid="ignore"):
        terms = ctx.astype(np.uint64) * np.asarray(
            multipliers[:ngram_size], dtype=np.uint64)
        rows = np.empty(heads_per_ngram * (ngram_size - 1), dtype=np.uint64)
        for n in range(2, ngram_size + 1):
            mixed = terms[0]
            for j in range(1, n):
                mixed = mixed ^ terms[j]
            base = (n - 2) * heads_per_ngram
            v = np.asarray(vocab_sizes[base:base + heads_per_ngram], dtype=np.uint64)
            o = np.asarray(offsets[base:base + heads_per_ngram], dtype=np.uint64)
            rows[base:base + heads_per_ngram] = (mixed % v) + o
    return rows.astype(np.uint32)


class Reference:
    def __init__(self, model_dir, budget=INDEXER_BUDGET):
        self.budget = budget
        self.dir = Path(model_dir)
        self.w = GTurboWeights(model_dir)
        self.experts = PackedExperts(model_dir)
        self.ple = json.loads((self.dir / "ple_constants.json").read_text())
        self.table = np.memmap(self.dir / "ngram_table.bin", dtype=np.float16,
                               mode="r").reshape(-1, self.ple["ple_head_dim"])
        self.reset()

    def reset(self):
        self.position = 0
        self.tokens = []
        self.gdn_conv = {}     # layer -> [K-1, conv_dim]
        self.gdn_state = {}    # layer -> [HV, DV, DK]
        self.kv = {}           # layer -> (keys [n,KV,HD], values)
        self.ple_conv = np.zeros(((PLE_K - 1) * PLE_DILATION, HC_DIM), np.float32)
        self.indexer_raw = {}

    # ---------------------------------------------------------------- blocks
    def hc_read(self, prefix, wide):
        xn = grouped_rms_norm(wide, self.w.get(prefix + "hc_norm"))
        lo = silu(self.w.get(prefix + "input_mix_weight_down.weight") @ xn / HC)
        gate = sigmoid(self.w.get(prefix + "input_mix_weight_up.weight") @ lo)
        return (xn * gate).reshape(HC, D).mean(axis=0)

    def hc_write(self, prefix, wide, block_out):
        xn = grouped_rms_norm(wide, self.w.get(prefix + "hc_norm"))
        inject = 2.0 * sigmoid(
            (self.w.get(prefix + "block_inject_weight.weight") @ xn) / HC)
        return (wide.reshape(HC, D) + block_out[None, :] * inject[:, None]).reshape(-1)

    def gdn(self, layer, x):
        prefix = f"{P}layers.{layer}.linear_attn."
        g = self.w.get
        qkv = g(prefix + "in_proj_qkv.weight") @ x
        z = g(prefix + "in_proj_z.weight") @ x
        a = g(prefix + "in_proj_a.weight") @ x
        b = g(prefix + "in_proj_b.weight") @ x

        conv_dim = qkv.shape[0]
        tail = self.gdn_conv.get(layer)
        if tail is None:
            tail = np.zeros((GDN_K - 1, conv_dim), np.float32)
        window = np.concatenate([tail, qkv[None, :]], axis=0)   # [K, conv_dim]
        conv_w = g(prefix + "conv1d.weight").reshape(-1, GDN_K)
        conv_out = silu((conv_w * window.T).sum(axis=1))
        self.gdn_conv[layer] = window[1:]

        key_dim = HK * DK
        q = conv_out[:key_dim].reshape(HK, DK)
        k = conv_out[key_dim:2 * key_dim].reshape(HK, DK)
        v = conv_out[2 * key_dim:].reshape(HV, DV)

        def l2(t):
            return t / np.sqrt((t ** 2).sum(axis=-1, keepdims=True) + EPS)

        q = np.repeat(l2(q), HV // HK, axis=0)
        k = np.repeat(l2(k), HV // HK, axis=0)

        beta = sigmoid(b)
        decay = np.exp(-np.exp(g(prefix + "A_log").astype(np.float64))
                       * softplus(a + g(prefix + "dt_bias")))
        state = self.gdn_state.get(layer)
        if state is None:
            state = np.zeros((HV, DV, DK), np.float32)
        state = state * decay[:, None, None]
        kv_mem = (state * k[:, None, :]).sum(axis=-1)           # [HV, DV]
        delta = (v - kv_mem) * beta[:, None]
        state = state + k[:, None, :] * delta[:, :, None]
        y = (state * q[:, None, :]).sum(axis=-1)                # [HV, DV]
        self.gdn_state[layer] = state

        y = y / np.sqrt(DV)
        yn = rms_norm(y, g(prefix + "norm.weight"))
        out = yn * sigmoid(z.reshape(HV, DV))
        return g(prefix + "out_proj.weight") @ out.reshape(-1)

    def rope(self, vec, position):
        """NeoX-style rotation over the first N_ROT dimensions."""
        out = vec.copy()
        half = N_ROT // 2
        inv = ROPE_THETA ** (-np.arange(half, dtype=np.float64) * 2.0 / N_ROT)
        angle = position * inv
        cos, sin = np.cos(angle), np.sin(angle)
        a = vec[..., :half]
        b = vec[..., half:N_ROT]
        out[..., :half] = a * cos - b * sin
        out[..., half:N_ROT] = a * sin + b * cos
        return out

    def indexer_keys(self, layer, x):
        """Raw indexer key for this token, cached before norm and rope."""
        prefix = f"{P}layers.{layer}.self_attn.indexer."
        return self.w.get(prefix + "index_k_proj.weight") @ x

    def qsa_keep(self, layer, x, n_kv, budget):
        """The cells this query keeps, as a boolean [n_kv].

        Mirrors the reference: pool raw keys into blocks of `r`, norm, rope at
        the block's own position, score with the indexer heads (relu per head,
        then sum), force the query's ragged tail in, and keep the top
        `budget + r - 1` cells by (score descending, index ascending).
        """
        prefix = f"{P}layers.{layer}.self_attn.indexer."
        g = self.w.get
        r = INDEXER_RATIO
        width = budget + r - 1
        if n_kv <= width:
            return np.ones(n_kv, dtype=bool)

        raw = np.stack(self.indexer_raw[layer][:n_kv])          # [n_kv, D]
        n_blocks = (n_kv + r - 1) // r
        pooled = np.empty((n_blocks, INDEXER_DIM), np.float32)
        for b in range(n_blocks):
            first = b * r
            members = raw[first:min(first + r, n_kv)]
            pooled[b] = members.mean(axis=0)
        pooled = rms_norm(pooled, g(prefix + "k_layernorm"))
        for b in range(n_blocks):
            pooled[b] = rope_full(pooled[b], b * r)

        q = (g(prefix + "index_q_proj.weight") @ x).reshape(
            INDEXER_HEADS, INDEXER_DIM)
        q = rms_norm(q, g(prefix + "q_layernorm"))
        q = np.stack([rope_full(q[h], self.position) for h in range(INDEXER_HEADS)])
        scores = np.maximum(q @ pooled.T, 0.0).sum(axis=0)      # [n_blocks]

        keep = np.zeros(n_kv, dtype=bool)
        complete = (n_kv // r) * r
        keep[complete:] = True                                  # the tail
        remaining = width - (n_kv - complete)
        order = sorted(range(complete // r), key=lambda b: (-scores[b], b))
        for b in order:
            if remaining <= 0:
                break
            take = min(r, remaining)
            keep[b * r: b * r + take] = True
            remaining -= take
        return keep

    def qsa(self, layer, x):
        prefix = f"{P}layers.{layer}.self_attn."
        g = self.w.get
        qg = (g(prefix + "q_proj.weight") @ x).reshape(N_HEADS, 2, HEAD_DIM)
        q, gate = qg[:, 0, :], qg[:, 1, :]
        k = (g(prefix + "k_proj.weight") @ x).reshape(N_KV_HEADS, HEAD_DIM)
        v = (g(prefix + "v_proj.weight") @ x).reshape(N_KV_HEADS, HEAD_DIM)

        q = rms_norm(q, g(prefix + "q_norm"))
        k = rms_norm(k, g(prefix + "k_norm"))
        q = self.rope(q, self.position)
        k = self.rope(k, self.position)

        self.indexer_raw.setdefault(layer, []).append(self.indexer_keys(layer, x))
        keys, values = self.kv.get(layer, (None, None))
        keys = k[None] if keys is None else np.concatenate([keys, k[None]], 0)
        values = v[None] if values is None else np.concatenate([values, v[None]], 0)
        self.kv[layer] = (keys, values)

        n_kv = keys.shape[0]
        keep = self.qsa_keep(layer, x, n_kv, self.budget)
        group = N_HEADS // N_KV_HEADS
        out = np.empty((N_HEADS, HEAD_DIM), np.float32)
        scale = 1.0 / np.sqrt(HEAD_DIM)
        for h in range(N_HEADS):
            kk = keys[keep, h // group, :]
            scores = (kk @ q[h]) * scale
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()
            out[h] = weights @ values[keep, h // group, :]
        out = out * sigmoid(gate)
        return g(prefix + "o_proj.weight") @ out.reshape(-1)

    def moe(self, layer, x):
        prefix = f"{P}layers.{layer}."
        g = self.w.get
        logits = g(prefix + "mlp.gate.weight") @ x
        gates = np.exp(logits - logits.max())
        gates /= gates.sum()
        chosen = np.argsort(-gates)[:TOP_K]
        scores = gates[chosen]
        scores = scores / max(scores.sum(), 6.103515625e-5)
        routed = np.zeros_like(x)
        for expert, score in zip(chosen, scores):
            e = int(expert)
            gate_p = self.experts.tensor(layer, e, "gate") @ x
            up = self.experts.tensor(layer, e, "up") @ x
            routed += score * (self.experts.tensor(layer, e, "down")
                               @ (silu(gate_p) * up))
        shared = g(prefix + "mlp.shared_expert.down_proj.weight") @ (
            silu(g(prefix + "mlp.shared_expert.gate_proj.weight") @ x)
            * (g(prefix + "mlp.shared_expert.up_proj.weight") @ x))
        return routed + sigmoid(
            g(prefix + "mlp.shared_expert_gate.weight") @ x) * shared

    def ple_block(self, layer, wide):
        prefix = f"{P}layers.{layer}.ple."
        g = self.w.get
        context = self.tokens[::-1][:self.ple["ngram_size"]]
        rows = ngram_rows(context, self.ple["layer_multipliers"],
                          self.ple["ngram_heads_offsets"],
                          self.ple["ngram_heads_vocab_sizes"],
                          self.ple["ngram_size"], self.ple["heads_per_ngram"],
                          self.ple["eos_token_id"])
        emb = self.table[rows].astype(np.float32).reshape(-1)

        key = g(prefix + "key_proj.weight") @ emb
        value = g(prefix + "value_proj.weight") @ emb
        key_w = grouped_rms_norm(key, g(prefix + "norm_key"))
        query = grouped_rms_norm(wide, g(prefix + "norm_query"))
        s = (key_w.reshape(HC, D) * query.reshape(HC, D)).sum(-1) / np.sqrt(D)
        gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
        gated = (value[None, :] * gate[:, None]).reshape(-1)
        normalized = grouped_rms_norm(gated, g(prefix + "norm_conv"))

        window = np.concatenate([self.ple_conv, normalized[None, :]], axis=0)
        # Reshaped like the GDN conv above. A depthwise weight is [C, 1, K] in
        # the checkpoint and [C, K] once a repack squeezes it; both have the
        # same channel-major byte layout, which is why the runtime reads either
        # (its kernel takes the tap count explicitly) and only numpy indexing
        # by shape can tell them apart.
        conv_w = g(prefix + "conv1d").reshape(-1, PLE_K)
        # Tap k reads (K-1-k)*dilation positions back, i.e. row k*dilation of
        # the padded window.
        conv = sum(conv_w[:, k] * window[k * PLE_DILATION]
                   for k in range(PLE_K))
        self.ple_conv = window[1:]
        return wide + gated + silu(conv)

    # ----------------------------------------------------------------- stack
    def step(self, token):
        """One token through the stack; returns (per-layer entries, logits)."""
        self.tokens.append(int(token))
        wide = np.tile(
            self.w.get(P + "embed_tokens.weight")[int(token)].astype(np.float32), HC)
        entries = []
        for layer in range(NUM_LAYERS):
            entries.append(wide.copy())
            if layer in PLE_LAYERS:
                wide = self.ple_block(layer, wide)
            prefix = f"{P}layers.{layer}."
            attn_in = self.hc_read(prefix + "attn_hyper_connection.", wide)
            block = (self.qsa(layer, attn_in)
                     if (layer + 1) % FULL_ATTENTION_EVERY == 0
                     else self.gdn(layer, attn_in))
            wide = self.hc_write(prefix + "attn_hyper_connection.", wide, block)
            mlp_in = self.hc_read(prefix + "mlp_hyper_connection.", wide)
            wide = self.hc_write(prefix + "mlp_hyper_connection.", wide,
                                 self.moe(layer, mlp_in))
        self.position += 1
        mixed = self.hc_read(P + "hyper_connection_mixer.", wide)
        return entries, wide, self.w.get("lm_head.weight") @ mixed
