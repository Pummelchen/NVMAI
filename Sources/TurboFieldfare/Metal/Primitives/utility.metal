#include <metal_stdlib>
using namespace metal;

// Kept in the shared library so both INT4 and INT8 shared-expert paths use
// the same Gemma activation without compiling a private shader module.
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
    out[tid] = half(gelu_pytorch_tanh(g) * u);
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

// hidden[i] += delta[i] — plain pre-norm residual add for architectures
// without Gemma's fused sandwich tail.
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
