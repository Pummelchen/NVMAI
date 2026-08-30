"""Full-stack parity for Qwen3.8-Flash-Next at position 0.

Runs all 48 layers in numpy from the installed weights and compares the wide
residual at every layer entry against what the runtime dumped. Position 0 is
chosen because every carried state -- the KV cache, the delta-rule state, the
two convolution histories -- is empty there, so the whole stack reduces to
closed-form algebra and a divergence is unambiguously a bug in one layer's
math rather than in state threading.

Usage:  python3 tools/qwen38_full_forward.py <model-dir> <dump-dir>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gturbo_reader import GTurboWeights, PackedExperts  # noqa: E402
from qwen38_parity import (  # noqa: E402
    HC, D, EPS, grouped_rms_norm, silu, sigmoid, hc_read, load,
    gdn_first_token, moe_block, ple_block, qsa_first_token)

P = "model.language_model."
NUM_LAYERS = 48
FULL_ATTENTION_EVERY = 4
PLE_LAYERS = {1}


def hc_write(w, prefix: str, wide: np.ndarray, block_out: np.ndarray) -> np.ndarray:
    xn = grouped_rms_norm(wide, w.get(prefix + "hc_norm"))
    inject = 2.0 * sigmoid((w.get(prefix + "block_inject_weight.weight") @ xn) / HC)
    return (wide.reshape(HC, D) + block_out[None, :] * inject[:, None]).reshape(-1)


def main():
    model_dir, dump_dir = sys.argv[1], Path(sys.argv[2])
    w = GTurboWeights(model_dir)
    experts = PackedExperts(model_dir)
    token = int((dump_dir / "token.txt").read_text().strip())

    wide = np.tile(w.get(P + "embed_tokens.weight")[token].astype(np.float32), HC)
    first_bad = None
    for layer in range(NUM_LAYERS):
        dumped = load(dump_dir, f"L{layer}_entry")
        cos = float(wide @ dumped
                    / (np.linalg.norm(wide) * np.linalg.norm(dumped) + 1e-30))
        flag = "ok  " if cos > 0.999 else "FAIL"
        if cos <= 0.999 and first_bad is None:
            first_bad = layer
        print(f"  L{layer:02d} entry  {flag} cos={cos:.5f}")
        if first_bad is not None:
            break

        if layer in PLE_LAYERS:
            wide = ple_block(w, layer, wide, load(dump_dir, "ple_embedding"))

        prefix = f"{P}layers.{layer}."
        attn_in = hc_read(w, prefix + "attn_hyper_connection.", wide)
        if (layer + 1) % FULL_ATTENTION_EVERY == 0:
            block = qsa_first_token(w, prefix + "self_attn.", attn_in)
        else:
            block = gdn_first_token(w, prefix + "linear_attn.", attn_in)
        wide = hc_write(w, prefix + "attn_hyper_connection.", wide, block)

        mlp_in = hc_read(w, prefix + "mlp_hyper_connection.", wide)
        wide = hc_write(w, prefix + "mlp_hyper_connection.", wide,
                        moe_block(w, experts, layer, mlp_in))

    if first_bad is not None:
        print(f"\nfirst divergence at layer {first_bad}")
        return

    dumped = load(dump_dir, "stack_out")
    cos = float(wide @ dumped / (np.linalg.norm(wide) * np.linalg.norm(dumped) + 1e-30))
    print(f"  stack out   {'ok  ' if cos > 0.999 else 'FAIL'} cos={cos:.5f}")

    mixed = hc_read(w, P + "hyper_connection_mixer.", wide)
    logits = w.get("lm_head.weight") @ mixed
    top = np.argsort(-logits)[:5]
    print(f"\n  reference top-5 tokens: {top.tolist()}")
    print(f"  logits: {logits[top].round(3).tolist()}")


if __name__ == "__main__":
    main()
