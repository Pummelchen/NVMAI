#include <metal_stdlib>
using namespace metal;

// ============================================================================
// fused.metal — current decode fusions:
//   fused_qkv_epilogue:     per-head Q/K/V RMSNorm + NeoX RoPE after projection.
// ============================================================================

constant constexpr uint kFusedThreads        = 256;
constant constexpr uint kFusedMaxSimdGroups  = kFusedThreads / 32;  // 8
constant constexpr uint kFusedMaxHeadDim     = 512;
constant uint FC_FUSED_D [[function_constant(80)]];
constant uint FC_FUSED_N [[function_constant(81)]];
constant uint FC_FUSED_HEAD_DIM [[function_constant(82)]];
constant uint FC_FUSED_NUM_Q_HEADS [[function_constant(83)]];
constant uint FC_FUSED_NUM_KV_HEADS [[function_constant(84)]];
constant uint FC_FUSED_ROTARY [[function_constant(85)]];
constant bool FC_FUSED_USE_FC [[function_constant(86)]];

static inline bool fused_use_fc() {
    return is_function_constant_defined(FC_FUSED_USE_FC) && FC_FUSED_USE_FC;
}

static inline uint fused_fc_head_dim(constant uint& head_dim) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_HEAD_DIM)) ? FC_FUSED_HEAD_DIM : head_dim;
}

static inline uint fused_fc_num_q_heads(constant uint& num_q_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_Q_HEADS)) ? FC_FUSED_NUM_Q_HEADS : num_q_heads;
}

static inline uint fused_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_NUM_KV_HEADS)) ? FC_FUSED_NUM_KV_HEADS : num_kv_heads;
}

static inline uint fused_fc_rotary(constant uint& rotary) {
    return (fused_use_fc() && is_function_constant_defined(FC_FUSED_ROTARY)) ? FC_FUSED_ROTARY : rotary;
}

inline void fused_rope_neox_pair(thread float& x0,
                                 thread float& x1,
                                 uint pair_index,
                                 uint head_dim,
                                 float position,
                                 float theta_base)
{
    const float exponent = -float(2u * pair_index) / float(head_dim);
    const float freq     = pow(theta_base, exponent);
    const float angle    = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// ============================================================================
// fused_qkv_epilogue — per-head Q/K/V post-processing.
//
// Replaces the cb1 chain after Q/K/V projection:
//     rmsnorm_bf16w_perhead(Q, q_norm) -> rmsnorm_bf16w_perhead(K, k_norm)
//     -> rmsnorm_no_scale_perhead(V) -> rope_neox(Q) -> rope_neox(K)
//
// One threadgroup owns one logical head. Q and K use BF16 weights shared across
// heads; V uses no-scale RMSNorm. For Q/K, the normalized per-head values are
// rounded to half in threadgroup memory before RoPE, preserving the standalone
// kernel boundary where RMSNorm writes FP16 and RoPE reads FP16.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kFusedThreads)]]
void fused_qkv_epilogue(
    device       half*   q              [[buffer(0)]],  // [num_q_heads, head_dim]
    device       half*   k              [[buffer(1)]],  // [num_kv_heads, head_dim]
    device       half*   v              [[buffer(2)]],  // [num_kv_heads, head_dim]
    device const bfloat* q_weight       [[buffer(3)]],  // [head_dim]
    device const bfloat* k_weight       [[buffer(4)]],  // [head_dim]
    constant     uint&   head_dim       [[buffer(5)]],
    constant     uint&   num_q_heads    [[buffer(6)]],
    constant     uint&   num_kv_heads   [[buffer(7)]],
    constant     uint&   position       [[buffer(8)]],
    constant     float&  theta_base     [[buffer(9)]],
    constant     uint&   rotated_pairs  [[buffer(10)]],
    constant     float&  rms_eps        [[buffer(11)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]],
    uint  head_group       [[threadgroup_position_in_grid]]
) {
    // K23: the half4 reinterpretation below requires 16-byte alignment of
    // these threadgroup arrays; declare it explicitly so the alignment holds
    // by construction instead of depending on the compiler's placement.
    alignas(16) threadgroup half  head_tg[kFusedMaxHeadDim];
    threadgroup float partial[kFusedMaxSimdGroups];

    const uint HD = fused_fc_head_dim(head_dim);
    const uint NQ = fused_fc_num_q_heads(num_q_heads);
    const uint NKV = fused_fc_num_kv_heads(num_kv_heads);
    const uint RP = fused_fc_rotary(rotated_pairs);

    const bool is_q = head_group < NQ;
    const bool is_k = !is_q && head_group < (NQ + NKV);
    const bool is_v = !is_q && !is_k && head_group < (NQ + 2u * NKV);
    if (!is_q && !is_k && !is_v) return;

    const uint local_head = is_q ? head_group : (head_group - NQ) % NKV;
    device half* dst = is_q ? (q + local_head * HD)
                    : (is_k ? (k + local_head * HD)
                            : (v + local_head * HD));
    device const half* src = dst;
    device const bfloat* w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]);
        acc = fma(xv, xv, acc);
    }
    acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float sum = (simd_lane_id < simdgroups) ? partial[simd_lane_id] : 0.0f;
        sum = simd_sum(sum);
        if (simd_lane_id == 0) {
            partial[0] = rsqrt(sum / float(HD) + rms_eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv = partial[0];
    for (uint i = lid; i < HD; i += lsize) {
        float xv = float(src[i]) * inv;
        if (!is_v) {
            xv *= float(w[i]);
        }
        head_tg[i] = half(xv);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_v) {
        for (uint i = lid; i < HD; i += lsize) {
            dst[i] = head_tg[i];
        }
        return;
    }

    const uint half_dim = HD / 2u;
    for (uint pair = lid; pair < half_dim; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[half_dim + pair]);
        if (pair < RP) {
            fused_rope_neox_pair(x0, x1, pair, HD, float(position), theta_base);
        }
        dst[pair] = half(x0);
        dst[half_dim + pair] = half(x1);
    }
}
