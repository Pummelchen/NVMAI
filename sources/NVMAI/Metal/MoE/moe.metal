#include <metal_stdlib>
using namespace metal;

constant constexpr uint kMoEGroupSize = 64;
constant constexpr uint kMaxStreamedExperts = 8;
constant constexpr float kGeluSqrt2OverPi = 0.7978845608028654f;
constant constexpr float kGeluCubicCoeff = 0.044715f;
// Max hidden size for the phase-1 threadgroup-staged activation (covers
// qwen36_35B_A3B's 2048; the cooperative load is guarded for smaller D).
constant constexpr uint kMoEXMaxD = 2816;

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];
// Hidden activation for expert FFNs: unset/false = gelu_pytorch_tanh,
// true = silu (Qwen 3.6 SwiGLU). Applies to routed decode, routed prefill,
// and the fused INT8 shared-expert kernel.
constant bool FC_MOE_ACT_SILU [[function_constant(4)]];
constant uint FC_MOE_WEIGHT_BITS [[function_constant(5)]];
constant bool FC_MOE_EVENT_GATED [[function_constant(6)]];

static inline bool moe_io_ready(device const uint* io_status) {
    return !(is_function_constant_defined(FC_MOE_EVENT_GATED) && FC_MOE_EVENT_GATED)
        || io_status[0] == 1u || io_status[0] == 3u;
}

static inline uint moe_weight_bits() {
    return is_function_constant_defined(FC_MOE_WEIGHT_BITS)
        ? FC_MOE_WEIGHT_BITS : 4u;
}

static inline uint moe_affine_value(device const uint8_t* packed,
                                    uint element, uint bits) {
    const uint bit = element * bits;
    const uint byte = bit >> 3;
    const uint shift = bit & 7u;
    uint word = uint(packed[byte]);
    if (shift + bits > 8u) word |= uint(packed[byte + 1u]) << 8;
    return (word >> shift) & ((1u << bits) - 1u);
}

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

static inline float gelu_pytorch_tanh(float x) {
    const float x3 = x * x * x;
    float inner = kGeluSqrt2OverPi * (x + kGeluCubicCoeff * x3);
    // Clamping avoids Metal tanh producing NaN at large magnitudes while being
    // equivalent to the saturated result at FP32 precision.
    inner = clamp(inner, -20.0f, 20.0f);
    return 0.5f * x * (1.0f + tanh(inner));
}

struct ExpertResidencyGPU {
    uint slot;
    uint state;
    ulong generation;
};

/// Classifies the router's exact top-k result against the CPU-published cache
/// map. Only eight positions are touched; one lane avoids atomics and preserves
/// router order in both compact lists.
kernel void moe_classify_expert_residency(
    device const uint* topk_indices [[buffer(0)]],
    device const ExpertResidencyGPU* residency [[buffer(1)]],
    device uint* hit_count [[buffer(2)]],
    device uint* hit_positions [[buffer(3)]],
    device uint* miss_count [[buffer(4)]],
    device uint* miss_positions [[buffer(5)]],
    device uint* miss_experts [[buffer(6)]],
    device uint* resolved_slots [[buffer(7)]],
    device ulong* resolved_generations [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    constant uint& num_experts [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (lane != 0) return;
    uint hits = 0, misses = 0;
    for (uint position = 0; position < top_k; ++position) {
        const uint expert = min(topk_indices[position], num_experts - 1u);
        const ExpertResidencyGPU entry = residency[expert];
        if (entry.state == 2u && entry.slot != 0xffffffffu) {
            hit_positions[hits++] = position;
            resolved_slots[position] = entry.slot;
            resolved_generations[position] = entry.generation;
        } else {
            miss_positions[misses] = position;
            miss_experts[misses] = expert;
            ++misses;
            resolved_slots[position] = 0xffffffffu;
            resolved_generations[position] = entry.generation;
        }
    }
    hit_count[0] = hits;
    miss_count[0] = misses;
}

static inline float moe_hidden_activation(float x) {
    if (is_function_constant_defined(FC_MOE_ACT_SILU) && FC_MOE_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return gelu_pytorch_tanh(x);
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

constant uint FC_ROUTER_BITS [[function_constant(44)]];

static inline uint router_affine_bits() {
    return is_function_constant_defined(FC_ROUTER_BITS) ? FC_ROUTER_BITS : 8u;
}

static inline uint router_affine_value(device const uint8_t* packed,
                                       uint element,
                                       uint bits) {
    const uint bit_offset = element * bits;
    const uint byte_offset = bit_offset >> 3;
    const uint shift = bit_offset & 7u;
    uint word = uint(packed[byte_offset]);
    if (shift + bits > 8u) word |= uint(packed[byte_offset + 1u]) << 8u;
    return (word >> shift) & ((1u << bits) - 1u);
}

static inline void router_gemv_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint bits = router_affine_bits();
    const uint n_groups = DD / kMoEGroupSize;
    const uint row_bytes = DD * bits / 8u;
    device const uint8_t* W_row = W + uint(e) * row_bytes;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint idx = g * kMoEGroupSize + lane * 2u;
        const float q0 = float(router_affine_value(W_row, idx, bits));
        const float q1 = float(router_affine_value(W_row, idx + 1u, bits));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        // Equal scores at the boundary are still considered so the inner
        // loop's lower-index tie-break (matching the reference) applies.
        if (s < top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[8];
    for (uint i = 0; i < 8; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < 8; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// two 16-bit weight halves (sub-tensor offsets are only guaranteed 2-byte
// aligned, so a 4-byte `uint*` load would be misaligned — undefined); the
// remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        // Read the 4-byte weight chunk as two ushorts: row stride N/2,
        // sub-tensor weightsOffset, and byte_base are all even, so a
        // `ushort*` load is always aligned even when the row is only
        // 2-byte aligned overall.
        device const ushort* wp = (device const ushort*)(W_row + byte_base);
        const uint w4 = uint(wp[0]) | (uint(wp[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint8_t byte = W_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float moe_affine_gemv_row_simd(
    device const uint8_t* W, device const bfloat* S,
    device const bfloat* B, device const half* x,
    uint row, uint N, uint lane
) {
    const uint bits = moe_weight_bits();
    const uint groups = N / kMoEGroupSize;
    const uint row_bytes = N * bits / 8u;
    device const uint8_t* wr = W + row * row_bytes;
    device const bfloat* sr = S + row * groups;
    device const bfloat* br = B + row * groups;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const uint e = g * kMoEGroupSize + lane * 2u;
        const float x0 = float(x[e]);
        const float x1 = float(x[e + 1u]);
        const float q0 = float(moe_affine_value(wr, e, bits));
        const float q1 = float(moe_affine_value(wr, e + 1u, bits));
        acc = fma(float(sr[g]), q0 * x0 + q1 * x1, acc);
        acc = fma(float(br[g]), x0 + x1, acc);
    }
    return simd_sum(acc);
}

static inline float2 moe_affine_gate_up_rows_simd(
    device const uint8_t* gateW, device const bfloat* gateS,
    device const bfloat* gateB, device const uint8_t* upW,
    device const bfloat* upS, device const bfloat* upB,
    device const half* x, uint row, uint N, uint lane
) {
    return float2(moe_affine_gemv_row_simd(gateW, gateS, gateB, x, row, N, lane),
                  moe_affine_gemv_row_simd(upW, upS, upB, x, row, N, lane));
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

// Threadgroup-staged activation for phase-1: all rows of a threadgroup share
// the same x vector, but each SIMD row loop re-loads the full x from L2.
// Staging it in threadgroup memory once per threadgroup (cooperatively, then
// a barrier) removes those re-reads — measured 38% -> 56% of peak bandwidth
// on the M3 with 16 rows/threadgroup (NVMAIBench moe_phase1_xsh16).
static inline float2 moe_int4_gate_up_rows_simd_tgmem_u16load(
    threadgroup const half* x,
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    uint row,
    uint N,
    uint lane
) {
    const uint n_groups = N / kMoEGroupSize;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    const uint full_blocks = n_groups / 4;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * 4u + (lane >> 3);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const float e0 = float(x[elem]), e1 = float(x[elem + 1u]);
        const float e2 = float(x[elem + 2u]), e3 = float(x[elem + 3u]);
        const float e4 = float(x[elem + 4u]), e5 = float(x[elem + 5u]);
        const float e6 = float(x[elem + 6u]), e7 = float(x[elem + 7u]);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * 4u; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint8_t gbv = gW_row[g * (kMoEGroupSize / 2) + lane];
        const uint8_t ubv = uW_row[g * (kMoEGroupSize / 2) + lane];
        const float x0 = float(x[g * kMoEGroupSize + lane * 2u]);
        const float x1 = float(x[g * kMoEGroupSize + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device half* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_int4_gate_up_rows_simd_dev_vec_u16load(
        gW, gS, gB, uW, uS, uB, x, f, D, lane);
    if (lane == 0) acts[slot * F + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    // 16 rows per threadgroup with the activation staged in threadgroup
    // memory. Each simdgroup redundantly loads the full x (the shared input),
    // then a barrier makes it visible to the row loops.
    constexpr uint rows_per_tg = 16;
    threadgroup half xt[kMoEXMaxD];
    const uint DD = moe_fc_d(D);
    for (uint i = lane; i < DD; i += 32u) {
        xt[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= moe_fc_top_k(top_k) * moe_fc_f(F)) return;
    const uint slot = rowg / moe_fc_f(F);
    const uint f = rowg % moe_fc_f(F);

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_tgmem_u16load(
        xt, base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        f, DD, lane);
    if (lane == 0) acts[slot * moe_fc_f(F) + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    device const uint* io_status [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    threadgroup half xt[kMoEXMaxD];
    const uint DD = moe_fc_d(D);
    for (uint i = lane; i < DD; i += 32u) {
        xt[i] = x[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * moe_fc_f(F)) return;
    const uint active_idx = rowg / moe_fc_f(F);
    const uint slot = active_slots[active_idx];
    if (slot >= moe_fc_top_k(top_k)) return;
    const uint f = rowg % moe_fc_f(F);

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    const float2 gu = moe_int4_gate_up_rows_simd_tgmem_u16load(
        xt, base + re.gate_W_off,
        (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off),
        base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off),
        f, DD, lane);
    if (lane == 0) acts[slot * moe_fc_f(F) + f] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load_r16(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase1_gate_up_act_u16load_r8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* io_status [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane);
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg_idx == 0 && lane == 0) y[d] = residual[d];
        return;
    }

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = moe_int4_gemv_row_simd_dev_vec(
        dW, dS, dB, act_slot, d, FF, lane);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        float acc = float(residual[d]);
        acc += partial[0]; acc += partial[1]; acc += partial[2]; acc += partial[3];
        acc += partial[4]; acc += partial[5]; acc += partial[6]; acc += partial[7];
        y[d] = half(acc);
    }
}

kernel void moe_affine_phase1_gate_up_act(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], device const uint* io_status [[buffer(7)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F), TK = moe_fc_top_k(top_k);
    const uint rowg = tg * rows_per_tg + sg;
    if (rowg >= TK * FF) return;
    const uint slot = rowg / FF, row = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_affine_gate_up_rows_simd(
        base + re.gate_W_off, (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off), base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off), x, row, DD, lane);
    if (lane == 0) acts[slot * FF + row] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_affine_phase1_gate_up_act_subset(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* x [[buffer(2)]], device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]], device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]], device const uint* io_status [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (!moe_io_ready(io_status)) return;
    constexpr uint rows_per_tg = 16;
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F), TK = moe_fc_top_k(top_k);
    const uint rowg = tg * rows_per_tg + sg;
    if (rowg >= active_count * FF) return;
    const uint slot = active_slots[rowg / FF];
    if (slot >= TK) return;
    const uint row = rowg % FF;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_affine_gate_up_rows_simd(
        base + re.gate_W_off, (device const bfloat*)(base + re.gate_s_off),
        (device const bfloat*)(base + re.gate_b_off), base + re.up_W_off,
        (device const bfloat*)(base + re.up_s_off),
        (device const bfloat*)(base + re.up_b_off), x, row, DD, lane);
    if (lane == 0) acts[slot * FF + row] = half(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_affine_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& re [[buffer(1)]],
    device const half* acts [[buffer(2)]], device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]], device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& F [[buffer(7)]],
    device const uint* io_status [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    if (!moe_io_ready(io_status)) {
        if (sg == 0 && lane == 0) y[d] = residual[d];
        return;
    }
    device const uint8_t* base = routed.blob[sg];
    const float value = moe_affine_gemv_row_simd(
        base + re.down_W_off, (device const bfloat*)(base + re.down_s_off),
        (device const bfloat*)(base + re.down_b_off), acts + sg * FF,
        d, FF, lane);
    if (lane == 0) partial[sg] = float(routing_w[sg]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint i = 0; i < 8u; ++i) acc += partial[i];
        y[d] = half(acc);
    }
}
