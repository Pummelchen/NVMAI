> **Category:** Guides
> **Status:** draft v1 — to be reviewed before posting
> **Wiki source:** System-Design (installation format), Getting-Started (receipts), FAQ (MLX question)

# Installs and verified receipts

Every NVMAI model is a `.gturbo` directory, and every `.gturbo` directory carries a `verified-install.json` receipt. This article is what the format is, why the receipt exists, and the one rule that trips people up (moving the directory).

## What a `.gturbo` install contains

`NVMAIRepack` converts a pinned Hugging Face checkpoint into a directory with:

- **Aligned tensor data** — the weights in NVMAI's own format, laid out for SSD streaming.
- **A manifest** — what's inside, in what layout, what precision.
- **Tokenizer assets** — the model's own tokenizer and chat template, installed with it.
- **`verified-install.json`** — the receipt, bound to the absolute path.

The model *is* that directory. There is no separate "model file" and "metadata"; the install and its attestation travel together.

## The format: no MLX, no GGUF

NVMAI does not use MLX and does not depend on an MLX release existing. Inference is native Swift over hand-written Metal kernels; the only package dependencies are a tokenizer and an HTTP server.

What it actually requires of a checkpoint is a **quantization layout**: affine, group size 64, BF16 scale and bias, packed into `u32` words. MLX happens to write exactly that layout, which is why an `mlx-community` release can be repacked directly and why tensor names here mention MLX. That is a *convenience*, not a precondition. An ordinary bf16/fp16 safetensors checkpoint is a perfectly good source — and usually a **better** one, because quantizing once from the original weights beats inheriting somebody else's quantization error. The quantizer lives in this repo.

GGUF is the weakest source: its k-quants use a different super-block structure, so converting means dequantizing and re-quantizing — lossy twice.

> **The real cost of adding a model is the architecture, not the format.** A model that fits a family NVMAI already implements is largely a conversion job. A new architecture needs its config, its Metal kernels, and a numerical parity pass against a reference. That's a different order of work.

## The verified receipt, and why it's bound to a path

The receipt is a **trusted attestation**: it records a hash of the payload against the manifest and binds it to the **absolute path** the install lives at. This is not a checksum for corruption detection. It's an integrity guarantee that the model you load is the model you installed, *and that it's still at the place it was verified in*.

That path binding is what catches a moved or swapped directory:

- You move the directory → the load fails with `trusted receipt invalid: model directory mismatch`.
- That is **not corruption**. The weights are fine. The receipt just can't vouch for a path it didn't verify.

So you **re-issue** the receipt in place (it re-hashes the payload against the manifest and rebinds it to the new path):

```bash
swift run -c release NVMAIRepack \
  --verify-install \
  --input-gturbo /new/path/to/model
```

**Never hand-edit `verified-install.json` to match a new path.** Editing it *forges* the attestation instead of re-establishing it — the path binding is precisely the thing that detects a moved or swapped directory, so removing it defeats the whole mechanism. Re-issue, don't edit.

## The install lifecycle

| Command | What it does |
| --- | --- |
| `NVMAIRepack --output <dir>` | Stream a verified install from a pinned checkpoint (no second full checkpoint staged). |
| `... --resume` | Continue an interrupted download; completed verified ranges are preserved. |
| `... --discard-partial --output <dir>` | Throw out an incomplete install. |
| `... --verify-install --input-gturbo <dir>` | Re-hash the payload, check the receipt (rebinds after a move). |

Installs stream **verified ranges** directly into the final directory — the installer does not download a second full copy and then copy it over. An interrupted install can resume exactly where it left off, and the ranges it already wrote are the ones that were verified.

## Where to go next

- First install, end to end: [Getting Started](#02)
- Why the expert weights are the part on the SSD: [SSD expert streaming](#03)
- How big the budget the install feeds: [The RAM budget](#04)

*Version at time of writing: NVMAI 5.1.*
