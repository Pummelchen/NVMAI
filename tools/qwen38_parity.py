"""Stage-by-stage parity for Qwen3.8-Flash-Next.

Reads the activations a run dumped with `NVMAI_ACT_DUMP` and recomputes the
same stages in numpy from the installed weights, following the reference
implementation (`Rocktalk-Holdings/mlx-qwen4exp`). Both sides use the same
quantized weights, so any disagreement is in the forward pass.

Usage:  python3 tools/qwen38_parity.py <model-dir> <dump-dir> <token-id>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gturbo_reader import GTurboWeights, PackedExperts  # noqa: E402

HC = 4
D = 2560
HC_DIM = HC * D
EPS = 1e-6


def load(dump: Path, name: str) -> np.ndarray:
    return np.fromfile(dump / f"{name}.f16", dtype=np.float16).astype(np.float32)


def grouped_rms_norm(x_flat: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Reduce over D per stream, then scale the flat [hc*D] view."""
    x = x_flat.reshape(HC, D)
    x = x / np.sqrt((x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True) + EPS)
    return x.reshape(-1) * gamma


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def report(label: str, mine: np.ndarray, reference: np.ndarray) -> bool:
    mine = mine.astype(np.float32)
    reference = reference.astype(np.float32)
    if mine.shape != reference.shape:
        print(f"  {label:24s} SHAPE {mine.shape} vs {reference.shape}")
        return False
    scale = max(np.abs(reference).max(), 1e-9)
    rel = np.abs(mine - reference).max() / scale
    cos = float(mine @ reference / (np.linalg.norm(mine) * np.linalg.norm(reference) + 1e-30))
    # fp16 activations through a quantized GEMV carry a few 1e-3 of relative
    # error; a wired-up-wrong stage misses by whole orders of magnitude, so the
    # gap between pass and fail is not a close call.
    ok = rel < 0.05 and cos > 0.99
    print(f"  {label:24s} {'ok  ' if ok else 'FAIL'} "
          f"max_rel={rel:.4f} cos={cos:.5f}")
    return ok


def main():
    model_dir, dump_dir, token = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    w = GTurboWeights(model_dir)
    P = "model.language_model."

    print(f"token {token}")
    # --- embedding: hc identical copies of the row -----------------------
    embed = load(dump_dir, "embed")
    table = w.get(P + "embed_tokens.weight")
    row = table[token].astype(np.float32)
    report("embed (wide)", embed, np.tile(row, HC))

    # --- layer 0 attention read gate -------------------------------------
    pre = embed  # what the runner had when it encoded the entry
    hc_norm = w.get(P + "layers.0.attn_hyper_connection.hc_norm")
    down = w.get(P + "layers.0.attn_hyper_connection.input_mix_weight_down.weight")
    up = w.get(P + "layers.0.attn_hyper_connection.input_mix_weight_up.weight")
    xn = grouped_rms_norm(pre, hc_norm)
    lo = silu(down @ xn / HC)
    gate = sigmoid(up @ lo)
    mixed = (xn * gate).reshape(HC, D).mean(axis=0)
    report("L0 attn read gate", load(dump_dir, "L0_attn_in"), mixed)

    # --- layer 0 attention write gate ------------------------------------
    inj_w = w.get(P + "layers.0.attn_hyper_connection.block_inject_weight.weight")
    inject = 2.0 * sigmoid((inj_w @ xn) / HC)
    block_out = load(dump_dir, "L0_attn_out")
    combined = pre.reshape(HC, D) + block_out[None, :] * inject[:, None]
    report("L0 attn write gate", load(dump_dir, "L0_hidden_post_attn"),
           combined.reshape(-1))

    # --- layer 0 mlp read gate -------------------------------------------
    post = load(dump_dir, "L0_hidden_post_attn")
    hc_norm_m = w.get(P + "layers.0.mlp_hyper_connection.hc_norm")
    down_m = w.get(P + "layers.0.mlp_hyper_connection.input_mix_weight_down.weight")
    up_m = w.get(P + "layers.0.mlp_hyper_connection.input_mix_weight_up.weight")
    xn_m = grouped_rms_norm(post, hc_norm_m)
    lo_m = silu(down_m @ xn_m / HC)
    gate_m = sigmoid(up_m @ lo_m)
    mixed_m = (xn_m * gate_m).reshape(HC, D).mean(axis=0)
    report("L0 mlp read gate", load(dump_dir, "L0_mlp_in"), mixed_m)

    # --- layer 0 gated-delta block, at position 0 -------------------------
    # Position 0 is the whole point of checking here: the conv history and the
    # recurrent state are both zero, so the block reduces to closed-form
    # algebra and a mismatch cannot be blamed on carried state.
    report("L0 GDN block", load(dump_dir, "L0_attn_out"),
           gdn_first_token(w, P + "layers.0.linear_attn.",
                           load(dump_dir, "L0_attn_in")))

    # --- layer 0 MoE block ------------------------------------------------
    experts = PackedExperts(model_dir)
    report("L0 MoE block", load(dump_dir, "L0_mlp_out"),
           moe_block(w, experts, 0, load(dump_dir, "L0_mlp_in")))

    # --- layer 0 mlp write gate, which is what layer 1 starts from --------
    inj_m = w.get(P + "layers.0.mlp_hyper_connection.block_inject_weight.weight")
    inject_m = 2.0 * sigmoid((inj_m @ xn_m) / HC)
    mlp_out = load(dump_dir, "L0_mlp_out")
    wide = (post.reshape(HC, D) + mlp_out[None, :] * inject_m[:, None]).reshape(-1)
    report("L0 mlp write gate", load(dump_dir, "L1_entry"), wide)

    # --- layer 1 PLE block, read through layer 1's gate -------------------
    # The block's own output is never resident on its own -- the next thing
    # that touches it is the attention read gate -- so the check runs through
    # that gate, using the residual the runner actually had.
    after_ple = ple_block(w, 1, load(dump_dir, "L1_entry"),
                          load(dump_dir, "ple_embedding"))
    report("L1 PLE + read gate", load(dump_dir, "L1_attn_in"),
           hc_read(w, P + "layers.1.attn_hyper_connection.", after_ple))

    # --- layer 3 QSA block, at position 0 ---------------------------------
    # One key: the softmax is a no-op, so this isolates the projections, the
    # q||gate interleave and the per-head sigmoid gate from the mask and the
    # indexer, which only matter once the context is longer.
    report("L3 QSA block", load(dump_dir, "L3_attn_out"),
           qsa_first_token(w, P + "layers.3.self_attn.",
                           load(dump_dir, "L3_attn_in")))


HK, HV, DK, DV, CONV_K = 16, 48, 128, 128, 4


def gdn_first_token(w, prefix: str, x: np.ndarray) -> np.ndarray:
    """The Gated DeltaNet block for the first token of a sequence."""
    qkv = w.get(prefix + "in_proj_qkv.weight") @ x
    z = w.get(prefix + "in_proj_z.weight") @ x
    a = w.get(prefix + "in_proj_a.weight") @ x
    b = w.get(prefix + "in_proj_b.weight") @ x

    # Causal depthwise conv with an all-zero history: only the last tap,
    # which is the one that reads the current position, contributes.
    conv_w = w.get(prefix + "conv1d.weight").reshape(-1, CONV_K)
    conv_out = silu(conv_w[:, CONV_K - 1] * qkv)

    key_dim = HK * DK
    q = conv_out[:key_dim].reshape(HK, DK)
    k = conv_out[key_dim:2 * key_dim].reshape(HK, DK)
    v = conv_out[2 * key_dim:].reshape(HV, DV)

    def l2(t):
        return t / np.sqrt((t ** 2).sum(axis=-1, keepdims=True) + EPS)

    q, k = l2(q), l2(k)
    # 16 key heads serve 48 value heads, three value heads to each.
    q = np.repeat(q, HV // HK, axis=0)
    k = np.repeat(k, HV // HK, axis=0)

    beta = sigmoid(b)
    g = np.exp(-np.exp(w.get(prefix + "A_log").astype(np.float64))
               * softplus(a + w.get(prefix + "dt_bias")))
    # With a zero state the recurrence collapses: delta = beta * v, the state
    # becomes k (x) delta, and the readout is delta scaled by <q, k>.
    _ = g  # the decay multiplies a zero state at t = 0
    delta = v * beta[:, None]
    y = delta * (q * k).sum(axis=-1)[:, None]

    y = y / np.sqrt(DV)
    gamma = w.get(prefix + "norm.weight")
    yn = y / np.sqrt((y ** 2).mean(axis=-1, keepdims=True) + EPS) * gamma
    out = yn * sigmoid(z.reshape(HV, DV))
    return w.get(prefix + "out_proj.weight") @ out.reshape(-1)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)







TOP_K = 10
NUM_EXPERTS = 512


def moe_block(w, experts, layer: int, x: np.ndarray) -> np.ndarray:
    """Softmax router over all experts, top-k renormalized, plus the shared
    expert gated by a sigmoid scalar."""
    P = f"model.language_model.layers.{layer}."
    logits = w.get(P + "mlp.gate.weight") @ x
    gates = np.exp(logits - logits.max())
    gates /= gates.sum()
    chosen = np.argsort(-gates)[:TOP_K]
    scores = gates[chosen]
    scores = scores / max(scores.sum(), 6.103515625e-5)

    routed = np.zeros_like(x)
    for expert, score in zip(chosen, scores):
        gate = experts.tensor(layer, int(expert), "gate") @ x
        up = experts.tensor(layer, int(expert), "up") @ x
        routed += score * (experts.tensor(layer, int(expert), "down")
                           @ (silu(gate) * up))

    shared = w.get(P + "mlp.shared_expert.down_proj.weight") @ (
        silu(w.get(P + "mlp.shared_expert.gate_proj.weight") @ x)
        * (w.get(P + "mlp.shared_expert.up_proj.weight") @ x))
    shared_gate = sigmoid(w.get(P + "mlp.shared_expert_gate.weight") @ x)
    return routed + shared_gate * shared


PLE_K, PLE_DILATION = 4, 3


def hc_read(w, prefix: str, wide: np.ndarray) -> np.ndarray:
    """The gated read that collapses the wide residual to one block input."""
    xn = grouped_rms_norm(wide, w.get(prefix + "hc_norm"))
    lo = silu(w.get(prefix + "input_mix_weight_down.weight") @ xn / HC)
    gate = sigmoid(w.get(prefix + "input_mix_weight_up.weight") @ lo)
    return (xn * gate).reshape(HC, D).mean(axis=0)


def ple_block(w, layer: int, wide: np.ndarray, embedding) -> np.ndarray:
    """The n-gram block, for the first token (zero convolution history)."""
    P = f"model.language_model.layers.{layer}.ple."
    key = w.get(P + "key_proj.weight") @ embedding
    value = w.get(P + "value_proj.weight") @ embedding
    key_w = grouped_rms_norm(key, w.get(P + "norm_key"))
    query = grouped_rms_norm(wide, w.get(P + "norm_query"))
    s = (key_w.reshape(HC, D) * query.reshape(HC, D)).sum(axis=-1) / np.sqrt(D)
    gate = sigmoid(np.sign(s) * np.sqrt(np.maximum(np.abs(s), 1e-6)))
    gated = value[None, :] * gate[:, None]
    normalized = grouped_rms_norm(gated.reshape(-1), w.get(P + "norm_conv"))
    # With no history only the tap that reads the current position survives.
    conv = w.get(P + "conv1d")[:, PLE_K - 1] * normalized
    return wide + gated.reshape(-1) + silu(conv)


N_HEADS, N_KV_HEADS, HEAD_DIM = 24, 2, 256


def qsa_first_token(w, prefix: str, x: np.ndarray) -> np.ndarray:
    """Quantized sparse attention for the first token of a sequence.

    With a single key the softmax is 1 and RoPE is the identity at position 0,
    so the result is the value vector for each head's KV group, scaled by that
    head's sigmoid gate.
    """
    # q_proj emits query and gate interleaved per head, not as two halves.
    qg = (w.get(prefix + "q_proj.weight") @ x).reshape(N_HEADS, 2, HEAD_DIM)
    gate = qg[:, 1, :]
    v = (w.get(prefix + "v_proj.weight") @ x).reshape(N_KV_HEADS, HEAD_DIM)

    out = v[np.arange(N_HEADS) // (N_HEADS // N_KV_HEADS)]
    out = out * sigmoid(gate)
    return w.get(prefix + "o_proj.weight") @ out.reshape(-1)


if __name__ == "__main__":
    main()
