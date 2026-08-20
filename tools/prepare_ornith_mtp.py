#!/usr/bin/env python3
"""Prepare Ornith 1.5's official native MTP layer for NVMAI import.

The official MLX releases omit MTP. This tool consumes only shard 16 from the
pinned original checkpoint, verifies its index/config/tensor contract, applies
Qwen3.5's zero-centered RMSNorm transform, and emits an MLX-affine snapshot.
NVMAIRepack then imports that snapshot into its bounded SSD expert layout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

try:
    import ml_dtypes
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError as error:
    raise SystemExit(
        "missing dependency; install numpy, safetensors, and ml_dtypes in a "
        f"project-local virtual environment ({error})"
    ) from error


SOURCE_REPO = "ornith-ai/Ornith-1.5-35B-A3B"
SOURCE_REVISION = "e4dfb35a93d4b6822a811a7676f3488514abe7e2"
SOURCE_INDEX_SHA256 = "0ad9fbfbb1a7514773ed881d3c9403019bd7ae9ff4c6af3cfc65ca7da784d78f"
SOURCE_SHARD = "model-00016-of-00016.safetensors"
GROUP_SIZE = 64
EXPERTS = 256
NORM_NAMES = {
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.norm.weight",
}
BASE_SHAPES = {
    "mtp.fc.weight": (2048, 4096),
    "mtp.layers.0.input_layernorm.weight": (2048,),
    "mtp.layers.0.mlp.gate.weight": (256, 2048),
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2048, 512),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (512, 2048),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (512, 2048),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2048),
    "mtp.layers.0.post_attention_layernorm.weight": (2048,),
    "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2048),
    "mtp.layers.0.self_attn.o_proj.weight": (2048, 4096),
    "mtp.layers.0.self_attn.q_norm.weight": (256,),
    "mtp.layers.0.self_attn.q_proj.weight": (8192, 2048),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2048),
    "mtp.norm.weight": (2048,),
    "mtp.pre_fc_norm_embedding.weight": (2048,),
    "mtp.pre_fc_norm_hidden.weight": (2048,),
}


def source_shapes() -> dict[str, tuple[int, ...]]:
    expected = dict(BASE_SHAPES)
    for expert in range(EXPERTS):
        prefix = f"mtp.layers.0.mlp.experts.{expert}"
        expected[f"{prefix}.gate_proj.weight"] = (512, 2048)
        expected[f"{prefix}.up_proj.weight"] = (512, 2048)
        expected[f"{prefix}.down_proj.weight"] = (2048, 512)
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    text = config.get("text_config", config)
    expected = {
        "hidden_size": 2048, "num_hidden_layers": 40, "num_experts": 256,
        "num_experts_per_tok": 8, "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256,
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False,
        "vocab_size": 248320,
    }
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError("source config is not qwen3_5_moe")
    mismatches = [f"{key}={text.get(key)!r}, expected {value!r}"
                  for key, value in expected.items() if text.get(key) != value]
    if mismatches:
        raise ValueError("unsupported Ornith MTP config: " + "; ".join(mismatches))
    return config


def validate_index(path: Path, shard_name: str) -> set[str]:
    if shard_name != SOURCE_SHARD:
        raise ValueError(f"source shard must be named {SOURCE_SHARD}")
    actual_digest = sha256(path)
    if actual_digest != SOURCE_INDEX_SHA256:
        raise ValueError(
            f"source index SHA-256 {actual_digest} does not match pinned "
            f"revision {SOURCE_REVISION} ({SOURCE_INDEX_SHA256})")
    index = json.loads(path.read_text())
    mapped = {name: shard for name, shard in index["weight_map"].items()
              if name.startswith("mtp.")}
    expected = set(source_shapes())
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))[:3]
        extra = sorted(set(mapped) - expected)[:3]
        raise ValueError(f"MTP tensor set mismatch: missing={missing}, extra={extra}")
    if set(mapped.values()) != {shard_name}:
        raise ValueError("the pinned MTP tensors are not all in the selected shard")
    return expected


def validate_shard(path: Path, expected: set[str]) -> None:
    with safe_open(path, framework="np") as source:
        keys = {name for name in source.keys() if name.startswith("mtp.")}
        if keys != expected:
            raise ValueError("source shard MTP keys do not match the pinned index")
        shapes = source_shapes()
        for name in sorted(expected):
            tensor = source.get_slice(name)
            if tuple(tensor.get_shape()) != shapes[name] or tensor.get_dtype() != "BF16":
                raise ValueError(
                    f"unexpected {name}: shape={tensor.get_shape()} "
                    f"dtype={tensor.get_dtype()}")


def quantize_affine(value: np.ndarray, bits: int) -> tuple[np.ndarray, ...]:
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


def add_quantized(output: dict[str, np.ndarray], name: str,
                  value: np.ndarray, bits: int) -> None:
    weight, scales, biases = quantize_affine(value, bits)
    output[name + ".weight"] = weight
    output[name + ".scales"] = scales
    output[name + ".biases"] = biases


def convert_tensors(shard: Path, bits: int) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    with safe_open(shard, framework="np") as source:
        for source_name in sorted(BASE_SHAPES):
            destination = source_name.removeprefix("mtp.")
            value = source.get_tensor(source_name)
            if source_name in NORM_NAMES:
                output[destination] = (
                    value.astype(np.float32) + np.float32(1)
                ).astype(ml_dtypes.bfloat16)
            else:
                add_quantized(output, destination.removesuffix(".weight"), value, bits)
        for role in ("gate_proj", "up_proj", "down_proj"):
            values = [source.get_tensor(
                f"mtp.layers.0.mlp.experts.{expert}.{role}.weight"
            ).astype(np.float32) for expert in range(EXPERTS)]
            add_quantized(output, f"layers.0.mlp.switch_mlp.{role}",
                          np.stack(values, axis=0), bits)
    return output


def write_snapshot(stage: Path, tensors: dict[str, np.ndarray],
                   config: dict, bits: int) -> None:
    stage.mkdir(parents=False)
    model_path = stage / "model.safetensors"
    save_file(tensors, model_path)
    total_size = sum(value.nbytes for value in tensors.values())
    index = {"metadata": {"total_size": total_size},
             "weight_map": {name: model_path.name for name in sorted(tensors)}}
    (stage / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    config["model_type"] = "qwen3_5_mtp"
    config["architectures"] = ["Qwen3_5MoeMTP"]
    config["quantization"] = {
        "group_size": GROUP_SIZE, "bits": bits, "mode": "affine"}
    (stage / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=f"Pinned source: {SOURCE_REPO}@{SOURCE_REVISION}")
    parser.add_argument("--source-shard", required=True, type=Path)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.source_shard, args.source_config, args.source_index):
        if not path.is_file():
            raise SystemExit(f"source file does not exist: {path}")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if stage.exists():
        raise SystemExit(f"staging path already exists: {stage}")
    try:
        config = validate_config(args.source_config)
        expected = validate_index(args.source_index, args.source_shard.name)
        validate_shard(args.source_shard, expected)
        tensors = convert_tensors(args.source_shard, args.bits)
        write_snapshot(stage, tensors, config, args.bits)
        os.replace(stage, output)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    print(f"wrote {len(tensors)} tensors to {output}")
    print(f"source: {SOURCE_REPO}@{SOURCE_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
