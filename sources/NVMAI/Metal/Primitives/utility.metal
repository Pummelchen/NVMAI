#include <metal_stdlib>
using namespace metal;

// Local copy of the tanh-GELU used by the shared-expert kernels. Kept local
// (rather than a cross-module static inline from moe.metal) so reordering the
// concatenated shader library can never break this file's resolution.
static inline float utility_gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same gelu_pytorch_tanh activation without compiling a private shader
// module.
[[kernel, max_total_threads_per_threadgroup(256)]]
void gelu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half(utility_gelu_pytorch_tanh(g) * u);
}

// SwiGLU counterpart for architectures with silu hidden activation (Qwen 3.6).
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_mul_fp16(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     count [[buffer(3)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    const float u = float(up[tid]);
    out[tid] = half((g / (1.0f + exp(-g))) * u);
}

// out[i] *= sigmoid(gate[i]) — Qwen 3.6 full-attention output gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_gate_mul_fp16(
    device half*       out  [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[tid]);
    out[tid] = half(float(out[tid]) / (1.0f + exp(-g)));
}

// y[i] *= sigmoid(gate[0]) — Qwen 3.6 shared-expert scalar gate.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_scalar_mul_fp16(
    device half*       y    [[buffer(0)]],
    device const half* gate [[buffer(1)]],
    constant uint&     count [[buffer(2)]],
    uint               tid  [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float g = float(gate[0]);
    y[tid] = half(float(y[tid]) / (1.0f + exp(-g)));
}

// Qwen 3.6 q_proj emits per-head [query(D) ; gate(D)] pairs. Split them into
// contiguous q [H, D] and gate [H, D] so the per-head norm, RoPE, and
// attention kernels see their usual layout.
[[kernel, max_total_threads_per_threadgroup(256)]]
void split_q_gate_fp16(
    device const half* packed [[buffer(0)]],   // [H, 2*D]
    device half*       q      [[buffer(1)]],   // [H, D]
    device half*       gate   [[buffer(2)]],   // [H, D]
    constant uint&     heads  [[buffer(3)]],
    constant uint&     dim    [[buffer(4)]],
    uint               tid   [[thread_position_in_grid]]
) {
    const uint total = heads * dim;
    if (tid >= total) return;
    const uint h = tid / dim;
    const uint d = tid % dim;
    q[tid] = packed[h * 2u * dim + d];
    gate[tid] = packed[h * 2u * dim + dim + d];
}

// hidden[i] += delta[i] — plain pre-norm residual add for architectures
// without a fused sandwich tail.
[[kernel, max_total_threads_per_threadgroup(256)]]
void residual_add_fp16(
    device half*       hidden [[buffer(0)]],
    device const half* delta  [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    hidden[tid] = half(float(hidden[tid]) + float(delta[tid]));
}

// Row-wise concatenation used by Qwen3.5-MoE MTP's 4096 -> 2048 adapter.
[[kernel, max_total_threads_per_threadgroup(256)]]
void concat_rows_fp16(
    device const half* lhs  [[buffer(0)]],
    device const half* rhs  [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     rows [[buffer(3)]],
    constant uint&     dim  [[buffer(4)]],
    uint               tid  [[thread_position_in_grid]]
) {
    const uint total = rows * dim;
    if (tid >= total) return;
    const uint row = tid / dim;
    const uint col = tid % dim;
    out[row * 2u * dim + col] = lhs[tid];
    out[row * 2u * dim + dim + col] = rhs[tid];
}

// ============================================================================
// Hyper-connection (Gated Residual) stream plumbing — Qwen3.8-Flash-Next.
//
// The residual is S parallel streams of D (4 x 2560). Everything else in the
// block is per-D; only these two steps see the stream axis, so they are the
// only new kernels the residual needs — the projections and gating reuse the
// existing INT4 GEMV and elementwise primitives.
// ============================================================================

// out[d] = (1/S) * sum_s mix[s*D + d] * normed[s*D + d]
//
// The gated read: each stream's normalized value is scaled by its own mix
// weight, then the streams are averaged into the single D-wide vector the
// attention or MLP block consumes.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_mix_reduce_fp16(
    device const half* mix     [[buffer(0)]],   // [S * D]
    device const half* normed  [[buffer(1)]],   // [S * D]
    device       half* out     [[buffer(2)]],   // [D]
    constant     uint& D       [[buffer(3)]],
    constant     uint& S       [[buffer(4)]],
    uint               tid     [[thread_position_in_grid]]
) {
    if (tid >= D) return;
    float acc = 0.0f;
    for (uint s = 0; s < S; ++s) {
        const uint i = s * D + tid;
        acc += float(mix[i]) * float(normed[i]);
    }
    out[tid] = half(acc / float(S));
}

// streams[s*D + d] += block_out[d] * inject[s]
//
// The gated write: one block output is injected back into every stream, each
// with its own learned weight. Accumulates in place, so the residual carries
// forward exactly as a plain transformer's would -- the streams differ only in
// how much of each block they take.
[[kernel, max_total_threads_per_threadgroup(256)]]
void hc_stream_inject_fp16(
    device       half* streams   [[buffer(0)]],  // [S * D], updated in place
    device const half* block_out [[buffer(1)]],  // [D]
    device const half* inject    [[buffer(2)]],  // [S]
    constant     uint& D         [[buffer(3)]],
    constant     uint& S         [[buffer(4)]],
    uint               tid       [[thread_position_in_grid]]
) {
    if (tid >= D * S) return;
    const uint s = tid / D;
    const uint d = tid - s * D;
    streams[tid] = half(float(streams[tid])
                        + float(block_out[d]) * float(inject[s]));
}

// out[i] = sigmoid(x[i]) — standalone gate, distinct from the fused
// sigmoid_gate_mul which multiplies into an existing value.
[[kernel, max_total_threads_per_threadgroup(256)]]
void sigmoid_fp16(
    device const half* x     [[buffer(0)]],
    device       half* out   [[buffer(1)]],
    constant     uint& count [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    out[tid] = half(1.0f / (1.0f + exp(-float(x[tid]))));
}

// out[i] = silu(x[i]) = x * sigmoid(x) — standalone, where silu_mul_fp16
// fuses a second operand this path does not have.
[[kernel, max_total_threads_per_threadgroup(256)]]
void silu_fp16(
    device const half* x     [[buffer(0)]],
    device       half* out   [[buffer(1)]],
    constant     uint& count [[buffer(2)]],
    uint               tid   [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const float v = float(x[tid]);
    out[tid] = half(v / (1.0f + exp(-v)));
}
