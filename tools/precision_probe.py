"""Does 4-bit quantization change the discrete decisions these tensors make?

The router and the QSA indexer do not produce values that get averaged; they
produce *rankings*, and a ranking either matches or it does not. This fetches
the bf16 originals by HTTP range request -- a few MB, not whole shards, so it
does not compete with the conversion for bandwidth -- quantizes each to 4 and
8 bits with the shipped quantizer, and measures how often the selection moves.
"""
import json, struct, subprocess, sys
import numpy as np, ml_dtypes
sys.path.insert(0, "tools")
import importlib.util
spec = importlib.util.spec_from_file_location("pq", "tools/prepare_qwen38.py")
pq = importlib.util.module_from_spec(spec); spec.loader.exec_module(pq)

SP = "/private/tmp/claude-501/-Users-andreborchert-Downloads-NVMAI/d4ac9bc1-da72-4471-8662-cfbcc02dd766/scratchpad"
BASE = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main"
wm = json.load(open(f"{SP}/q38_index.json"))["weight_map"]
_headers = {}

def header(shard):
    if shard not in _headers:
        url = f"{BASE}/{shard}"
        raw = subprocess.run(["curl","-sfL","--max-time","60","-r","0-7",url],
                             capture_output=True, check=True).stdout
        n = struct.unpack("<Q", raw[:8])[0]
        body = subprocess.run(["curl","-sfL","--max-time","180","-r",f"8-{8+n-1}",url],
                              capture_output=True, check=True).stdout
        _headers[shard] = (json.loads(body), 8 + n)
    return _headers[shard]

def fetch(name):
    shard = wm[name]
    head, data_start = header(shard)
    meta = head[name]
    lo, hi = meta["data_offsets"]
    url = f"{BASE}/{shard}"
    raw = subprocess.run(["curl","-sfL","--max-time","600","-r",
                          f"{data_start+lo}-{data_start+hi-1}", url],
                         capture_output=True, check=True).stdout
    assert len(raw) == hi-lo, f"{name}: got {len(raw)} want {hi-lo}"
    arr = np.frombuffer(raw, dtype=ml_dtypes.bfloat16).astype(np.float32)
    return arr.reshape(meta["shape"])

def roundtrip(w, bits):
    """Quantize with the shipped quantizer, then dequantize, as the GPU does."""
    packed, scale, bias = pq.quantize_affine(w, bits)
    G = pq.GROUP_SIZE
    lanes = 32 // bits
    shape = (*w.shape[:-1], w.shape[-1] // lanes, lanes)
    out = np.zeros(w.shape, dtype=np.float32).reshape(*w.shape[:-1], -1)
    q = np.zeros((*w.shape[:-1], w.shape[-1]), dtype=np.uint32)
    for lane in range(lanes):
        q[..., lane::lanes] = (packed >> np.uint32(bits*lane)) & np.uint32((1<<bits)-1)
    q = q.reshape(*w.shape[:-1], w.shape[-1]//G, G).astype(np.float32)
    deq = q * scale.astype(np.float32)[..., None] + bias.astype(np.float32)[..., None]
    return deq.reshape(w.shape)

rng = np.random.default_rng(0x38C0)

def topk_agreement(w_ref, w_q, k, trials=400, hidden=None):
    """Fraction of trials whose top-k selection is identical, and mean overlap."""
    n = w_ref.shape[-1]
    x = rng.standard_normal((trials, n), dtype=np.float32)      # post-RMSNorm proxy
    a = x @ w_ref.T
    b = x @ w_q.T
    ta = np.argpartition(-a, k, axis=-1)[:, :k]
    tb = np.argpartition(-b, k, axis=-1)[:, :k]
    exact, overlap = 0, 0.0
    for i in range(trials):
        sa, sb = set(ta[i].tolist()), set(tb[i].tolist())
        overlap += len(sa & sb)/k
        exact += (sa == sb)
    rel = np.abs(b-a).mean()/ (np.abs(a).mean()+1e-12)
    return exact/trials, overlap/trials, rel

print(f"{'tensor':38} {'bits':>4} {'exact top-k':>12} {'overlap':>9} {'rel err':>9}")
tests = [
  ("layers.3.mlp.gate.weight", 10, "router: picks 10 of 512 experts"),
  ("layers.3.self_attn.indexer.index_qk_proj.weight", None, "QSA indexer projection"),
  ("layers.3.attn_hyper_connection.block_inject_weight.weight", None, "HC write gate (4 values)"),
]
for suffix, k, label in tests:
    name = "model.language_model." + suffix
    w = fetch(name)
    print(f"\n{label}  shape={list(w.shape)}")
    kk = k if k else max(1, w.shape[0]//4)
    for bits in (4, 8):
        wq = roundtrip(w, bits)
        ex, ov, rel = topk_agreement(w, wq, min(kk, w.shape[0]-1))
        print(f"  {'':36} {bits:>4} {100*ex:11.1f}% {100*ov:8.1f}% {100*rel:8.2f}%")
