#include <metal_stdlib>
using namespace metal;

// ============================================================================
// gdn.metal — gated-DeltaNet linear attention (Qwen 3.6 layers with layer-mask
// value 2). Decode processes one token; prefill processes a bounded chunk of
// rows with the recurrence sequential inside the kernel.
//
// Dataflow per layer (dimensions are runtime parameters; Qwen 3.6 uses
// C = 8192 conv channels, K = 4 taps, Hk = 16 x Dk = 128, Hv = 32 x Dv = 128):
//   mixed_qkv = in_proj_qkv(x)                       (existing INT4 GEMV)
//   conv_out  = silu(causal_depthwise_conv(mixed))    gdn_conv_mix_*
//   q, k      = per-head no-weight RMS norm + scale   gdn_qk_norm_*
//   y, S      = gated delta rule recurrence           gdn_delta_*
//   out       = rmsnorm(y) * silu(z)                  gdn_gated_norm_*
//   out_proj(out)                                     (existing INT4 GEMV)
//
// The recurrence follows the mlx-vlm reference (gated_delta.py): per value
// head h with state S[Dv, Dk] (FP32, persistent across tokens):
//   g    = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
//   beta = sigmoid(b[h])
//   S    = S * g
//   kv   = S @ k
//   S   += outer((v - kv) * beta, k)
//   y    = S @ q
// ============================================================================

constant constexpr float kGdnRmsEps = 1e-6f;

static inline float gdn_silu(float x) {
    return x / (1.0f + exp(-x));
}

static inline float gdn_softplus(float x) {
    // log1p(exp(x)) with the standard large-x shortcut; matches
    // mlx.nn.softplus to FP32 precision.
    return (x > 20.0f) ? x : log(1.0f + exp(x));
}

// ----------------------------------------------------------------------------
// Causal depthwise conv, decode. One thread per channel. The tail buffer holds
// the previous K-1 pre-activation rows and is shifted in place (each thread
// owns its channel column exclusively).
// ----------------------------------------------------------------------------
kernel void gdn_conv_mix_decode(
    device half*        tail     [[buffer(0)]],   // [K-1, C] shifted in place
    device const half*  qkv      [[buffer(1)]],   // [C] current raw row
    device const bfloat* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [C] silu(conv)
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    if (tid >= C) return;

    const uint history = K - 1u;
    float acc = float(qkv[tid]) * float(conv_w[tid * K + (K - 1u)]);
    for (uint j = 0; j < history; ++j) {
        acc = fma(float(tail[j * C + tid]), float(conv_w[tid * K + j]), acc);
    }
    out[tid] = half(gdn_silu(acc));

    for (uint j = 0; j + 1u < history; ++j) {
        tail[j * C + tid] = tail[(j + 1u) * C + tid];
    }
    if (history > 0u) {
        tail[(history - 1u) * C + tid] = qkv[tid];
    }
}

// ----------------------------------------------------------------------------
// Causal depthwise conv, prefill. Threads (channel, row); the incoming tail is
// read-only here — gdn_conv_tail_update refreshes it after the chunk.
// ----------------------------------------------------------------------------
kernel void gdn_conv_mix_prefill(
    device const half*  tail     [[buffer(0)]],   // [K-1, C] state entering chunk
    device const half*  qkv      [[buffer(1)]],   // [T, C] raw rows
    device const bfloat* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [T, C]
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    constant uint&      rows     [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    const uint T = rows;
    const uint ch = gid.x;
    const uint t = gid.y;
    if (ch >= C || t >= T) return;

    const uint history = K - 1u;
    float acc = 0.0f;
    for (uint j = 0; j < K; ++j) {
        // Input row index in the virtual sequence [tail rows | chunk rows].
        const int src = int(t) + int(j) - int(history);
        float value;
        if (src >= 0) {
            value = float(qkv[uint(src) * C + ch]);
        } else {
            value = float(tail[uint(int(history) + src) * C + ch]);
        }
        acc = fma(value, float(conv_w[ch * K + j]), acc);
    }
    out[t * C + ch] = half(gdn_silu(acc));
}

// After a prefill chunk: tail := last K-1 raw rows of [old tail | chunk].
kernel void gdn_conv_tail_update(
    device half*        tail     [[buffer(0)]],   // [K-1, C]
    device const half*  qkv      [[buffer(1)]],   // [T, C] raw rows
    constant uint&      channels [[buffer(2)]],
    constant uint&      taps     [[buffer(3)]],
    constant uint&      rows     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint history = taps - 1u;
    const uint T = rows;
    const uint ch = gid.x;
    const uint j = gid.y;
    if (ch >= C || j >= history) return;

    // New tail row j corresponds to virtual row (T + j) - history.
    const int src = int(T) + int(j) - int(history);
    half value;
    if (src >= 0) {
        value = qkv[uint(src) * C + ch];
    } else {
        value = tail[uint(int(history) + src) * C + ch];
    }
    // Threads for row j read only tail rows < j's source (all sources are
    // < history), and every source row index is strictly less than every
    // destination row index only when T >= history. For T < history the
    // shift moves rows downward (src < j), so read-before-write hazards
    // would occur between threads in different threadgroups. Stage through
    // a register and rely on the separate raw-row case being the common
    // path; for the tiny T < history case the kernel is dispatched with a
    // single threadgroup per channel column (grid.y == history <= 8), and
    // the loop below performs the ordered shift instead.
    if (T >= history) {
        tail[j * C + ch] = value;
        return;
    }
    // T < history: only thread j == 0 performs the whole ordered shift for
    // this channel to avoid cross-thread hazards.
    if (j != 0u) return;
    for (uint row = 0; row < history; ++row) {
        const int rowSrc = int(T) + int(row) - int(history);
        half rowValue;
        if (rowSrc >= 0) {
            rowValue = qkv[uint(rowSrc) * C + ch];
        } else {
            rowValue = tail[uint(int(history) + rowSrc) * C + ch];
        }
        tail[row * C + ch] = rowValue;
    }
}

// ----------------------------------------------------------------------------
// Per-head no-weight RMS norm over q and k slices of conv_out, with the
// delta-rule scales folded in: q *= 1/Dk (inv_scale^2), k *= 1/sqrt(Dk).
// One threadgroup per (head, row); head_dim <= 1024 threads.
// conv_out layout per row: [q: Hk*Dk][k: Hk*Dk][v: Hv*Dv].
// ----------------------------------------------------------------------------
kernel void gdn_qk_norm(
    device half*   conv_out  [[buffer(0)]],
    constant uint& kHeads    [[buffer(1)]],
    constant uint& keyDim    [[buffer(2)]],
    constant uint& rowStride [[buffer(3)]],   // C, elements per row
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_idx  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    const uint tid = tpos.x;
    const uint Hk = kHeads;
    const uint Dk = keyDim;
    const uint headIndex = tg.x;          // 0..<2*Hk: q heads then k heads
    const uint row = tg.y;
    if (headIndex >= 2u * Hk) return;

    const bool isQ = headIndex < Hk;
    const uint head = isQ ? headIndex : headIndex - Hk;
    const uint base = row * rowStride + (isQ ? 0u : Hk * Dk) + head * Dk;

    float sumsq = 0.0f;
    for (uint i = tid; i < Dk; i += 128u) {
        const float x = float(conv_out[base + i]);
        sumsq = fma(x, x, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (simd_lane == 0) partial[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 4u; ++i) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0] / float(Dk);
    const float invRms = rsqrt(mean + kGdnRmsEps);
    const float scale = isQ ? (1.0f / float(Dk)) : rsqrt(float(Dk));

    for (uint i = tid; i < Dk; i += 128u) {
        const float x = float(conv_out[base + i]);
        conv_out[base + i] = half(x * invRms * scale);
    }
}

// ----------------------------------------------------------------------------
// Gated delta rule. Decode: one token. Threadgroups (Hv, Dv/4); threads
// (32, 4): thread (lane, dvSub) owns state[dv][lane*4 .. lane*4+3].
// State is FP32 [Hv, Dv, Dk], persistent. a/b are the projection outputs
// [Hv] half; A_log/dt_bias are BF16 [Hv].
// ----------------------------------------------------------------------------
kernel void gdn_delta_step_decode(
    device const half*   conv_out [[buffer(0)]],   // [C] normed q,k + raw v
    device const half*   a_proj   [[buffer(1)]],   // [Hv]
    device const half*   b_proj   [[buffer(2)]],   // [Hv]
    device const bfloat* A_log    [[buffer(3)]],   // [Hv]
    device const bfloat* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = h / (Hv / Hk);
    device const half* q = conv_out + hk * Dk;
    device const half* k = conv_out + Hk * Dk + hk * Dk;
    device const half* v = conv_out + 2u * Hk * Dk + h * Dv;

    const float g = exp(-exp(float(A_log[h]))
                        * gdn_softplus(float(a_proj[h]) + float(dt_bias[h])));
    const float beta = 1.0f / (1.0f + exp(-float(b_proj[h])));

    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[8];
    float kv = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = srow[idx] * g;
        kv = fma(s[i], float(k[idx]), kv);
    }
    kv = simd_sum(kv);

    const float delta = (float(v[dv]) - kv) * beta;

    float out = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = fma(float(k[idx]), delta, s[i]);
        out = fma(s[i], float(q[idx]), out);
        srow[idx] = s[i];
    }
    out = simd_sum(out);
    if (lane == 0) y[h * Dv + dv] = half(out);
}

// Prefill: identical math with the token loop inside the kernel. q/k/v/a/b
// advance by their row strides each step; state persists in registers across
// the whole chunk and is written back once.
kernel void gdn_delta_step_prefill(
    device const half*   conv_out [[buffer(0)]],   // [T, C]
    device const half*   a_proj   [[buffer(1)]],   // [T, Hv]
    device const half*   b_proj   [[buffer(2)]],   // [T, Hv]
    device const bfloat* A_log    [[buffer(3)]],   // [Hv]
    device const bfloat* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [T, Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    constant uint&       rows     [[buffer(11)]],
    constant uint&       rowStride [[buffer(12)]], // C, conv_out elements/row
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint T = rows;
    const uint C = rowStride;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = h / (Hv / Hk);
    const float expA = exp(float(A_log[h]));
    const float dtb = float(dt_bias[h]);

    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[8];
    for (uint i = 0; i < nPerLane; ++i) {
        s[i] = srow[lane * nPerLane + i];
    }

    for (uint t = 0; t < T; ++t) {
        device const half* q = conv_out + t * C + hk * Dk;
        device const half* k = conv_out + t * C + Hk * Dk + hk * Dk;
        device const half* v = conv_out + t * C + 2u * Hk * Dk + h * Dv;

        const float g = exp(-expA * gdn_softplus(float(a_proj[t * Hv + h]) + dtb));
        const float beta = 1.0f / (1.0f + exp(-float(b_proj[t * Hv + h])));

        float kv = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] *= g;
            kv = fma(s[i], float(k[idx]), kv);
        }
        kv = simd_sum(kv);

        const float delta = (float(v[dv]) - kv) * beta;

        float out = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] = fma(float(k[idx]), delta, s[i]);
            out = fma(s[i], float(q[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) y[t * Hv * Dv + h * Dv + dv] = half(out);
    }

    for (uint i = 0; i < nPerLane; ++i) {
        srow[lane * nPerLane + i] = s[i];
    }
}

// ----------------------------------------------------------------------------
// Gated output norm: out = rmsnorm(y; weight, eps) * silu(z), per value head.
// One threadgroup per (head, row), 128 threads. Norm statistics span one
// head's Dv elements; silu/product in FP32 (matches the reference's
// _precise_swiglu).
// ----------------------------------------------------------------------------
kernel void gdn_gated_norm(
    device const half*   y        [[buffer(0)]],   // [T, Hv * Dv]
    device const half*   z        [[buffer(1)]],   // [T, Hv * Dv]
    device const bfloat* weight   [[buffer(2)]],   // [Dv]
    device half*         out      [[buffer(3)]],   // [T, Hv * Dv]
    constant uint&       vHeads   [[buffer(4)]],
    constant uint&       valueDim [[buffer(5)]],
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_idx  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    const uint tid = tpos.x;
    const uint Hv = vHeads;
    const uint Dv = valueDim;
    const uint head = tg.x;
    const uint row = tg.y;
    if (head >= Hv) return;

    const uint base = row * Hv * Dv + head * Dv;

    float sumsq = 0.0f;
    for (uint i = tid; i < Dv; i += 128u) {
        const float x = float(y[base + i]);
        sumsq = fma(x, x, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (simd_lane == 0) partial[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 4u; ++i) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0] / float(Dv);
    const float invRms = rsqrt(mean + kGdnRmsEps);

    for (uint i = tid; i < Dv; i += 128u) {
        const float normed = float(y[base + i]) * invRms * float(weight[i]);
        const float gate = gdn_silu(float(z[base + i]));
        out[base + i] = half(normed * gate);
    }
}
