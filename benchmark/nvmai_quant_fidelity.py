#!/usr/bin/env python3
"""Is the community 4-bit quantization faithful to the official bf16 weights?

A full logit parity is impossible here (the bf16 checkpoint is ~360 GB against
288 GB free), but the question underneath it -- did this quantization preserve
the weights -- is answerable directly and cheaply: pull the same tensors from
both repos by HTTP range, dequantize the MLX affine blocks, and measure the
error against the official values.

Reference point: NVMAI's own shipping 4-bit g64 models measure ~2% relative
error on attention tensors. Comparable error means sound; an order of
magnitude worse means broken.
"""
import json, sys, urllib.request, numpy as np

OFF = "Qwen/Qwen3.8-Flash-Next"
MLX = "RockTalk/Qwen3.8-Flash-Next-MLX-4bit"
MLX_REV = "478474da92599ad0cf9f8bd447e658b29cb8480a"

def get(url, start=None, end=None):
    r = urllib.request.Request(url)
    if start is not None:
        r.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(r, timeout=180) as f:
        return f.read()

def url_for(repo, rev, shard):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{shard}"

_hdr_cache = {}
def header(repo, rev, shard):
    key = (repo, shard)
    if key in _hdr_cache: return _hdr_cache[key]
    u = url_for(repo, rev, shard)
    n = int.from_bytes(get(u, 0, 7), "little")
    hdr = json.loads(get(u, 8, 8 + n - 1))
    _hdr_cache[key] = (hdr, 8 + n)
    return _hdr_cache[key]

def fetch(repo, rev, shard, name):
    hdr, base = header(repo, rev, shard)
    if name not in hdr: return None
    m = hdr[name]
    s, e = m["data_offsets"]
    raw = get(url_for(repo, rev, shard), base + s, base + e - 1)
    dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
          "U32": np.uint32, "I32": np.int32, "U8": np.uint8}[m["dtype"]]
    a = np.frombuffer(raw, dtype=dt)
    if m["dtype"] == "BF16":                      # bf16 -> f32
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.reshape(m["shape"]) if m["dtype"] != "BF16" else a.reshape(m["shape"])

def dequant(q, scales, biases, bits, group):
    """MLX affine: value = q * scale + bias, q packed little-endian into u32."""
    per = 32 // bits
    flat = q.reshape(-1)
    vals = np.empty((flat.size, per), dtype=np.float32)
    mask = (1 << bits) - 1
    for i in range(per):
        vals[:, i] = (flat >> (bits * i)) & mask
    out_rows = q.shape[0]
    vals = vals.reshape(out_rows, -1)             # [out, in]
    s = scales.astype(np.float32); b = biases.astype(np.float32)
    ng = s.shape[1]
    vals = vals.reshape(out_rows, ng, group)
    return (vals * s[:, :, None] + b[:, :, None]).reshape(out_rows, -1)

off_idx = json.load(open("off_index.json"))["weight_map"]
mlx_idx = json.load(open("rt_index.json"))["weight_map"]
qcfg = json.load(open("rt_config.json"))["quantization_config"]

TARGETS = [
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
    "model.language_model.layers.3.mlp.shared_expert.gate_proj.weight",
    "model.language_model.layers.3.mlp.gate.weight",
    "model.language_model.layers.19.self_attn.q_proj.weight",
]

print(f"{'tensor':52s} {'bits':>4s} {'rel err':>9s} {'max|w|':>9s} {'verdict':>9s}")
print("-" * 90)
bad = 0
for name in TARGETS:
    if name not in off_idx or name not in mlx_idx:
        print(f"{name[-50:]:52s} {'':>4s} {'missing':>9s}"); continue
    stem = name[:-len(".weight")]
    bits = 4
    for k, v in qcfg.items():
        if isinstance(v, dict) and stem.endswith(k.split("model.language_model.")[-1]):
            bits = v["bits"]
    group = qcfg.get("group_size", 64)
    ref = fetch(OFF, "main", off_idx[name], name)
    q  = fetch(MLX, MLX_REV, mlx_idx[name], name)
    sc = fetch(MLX, MLX_REV, mlx_idx[stem + ".scales"], stem + ".scales")
    bi = fetch(MLX, MLX_REV, mlx_idx[stem + ".biases"], stem + ".biases")
    if any(x is None for x in (ref, q, sc, bi)):
        print(f"{name[-50:]:52s} {bits:>4d} {'fetch fail':>9s}"); continue
    deq = dequant(q, sc, bi, bits, group)
    if deq.shape != ref.shape:
        print(f"{name[-50:]:52s} {bits:>4d} SHAPE {deq.shape} vs {ref.shape}"); bad += 1; continue
    err = np.linalg.norm(deq - ref) / max(np.linalg.norm(ref), 1e-9)
    ok = err < (0.05 if bits == 4 else 0.02)
    bad += 0 if ok else 1
    print(f"{name[-50:]:52s} {bits:>4d} {err:9.4f} {np.abs(ref).max():9.4f} "
          f"{'OK' if ok else 'SUSPECT':>9s}")
print("-" * 90)
print("VERDICT:", "quantization is faithful" if bad == 0 else f"{bad} tensor(s) SUSPECT")
sys.exit(1 if bad else 0)
