#include <metal_stdlib>
using namespace metal;

constant constexpr uint kAffineGroupSize = 64;
constant uint FC_AFFINE_BITS [[function_constant(100)]];

static inline uint affine_quant_bits() {
    return is_function_constant_defined(FC_AFFINE_BITS) ? FC_AFFINE_BITS : 4u;
}

static inline uint affine_quant_value(device const uint* packed,
                                      uint element,
                                      uint bits) {
    const ulong bit_offset = ulong(element) * ulong(bits);
    const uint word = uint(bit_offset >> 5);
    const uint shift = uint(bit_offset & 31u);
    const uint mask = (1u << bits) - 1u;
    uint value = packed[word] >> shift;
    if (shift + bits > 32u) {
        value |= packed[word + 1u] << (32u - shift);
    }
    return value & mask;
}

kernel void affine_quant_gemv_simd(
    device const uint*   weights [[buffer(0)]],
    device const bfloat* scales  [[buffer(1)]],
    device const bfloat* biases  [[buffer(2)]],
    device const half*   x       [[buffer(3)]],
    device half*         y       [[buffer(4)]],
    constant uint&       rows    [[buffer(5)]],
    constant uint&       columns [[buffer(6)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_threadgroup = 8;
    const uint row = tg_idx * rows_per_threadgroup + sg_idx;
    if (row >= rows) return;

    const uint bits = affine_quant_bits();
    const uint groups_per_row = columns / kAffineGroupSize;
    const uint words_per_row = (columns * bits) / 32u;
    device const uint* row_weights = weights + row * words_per_row;
    device const bfloat* row_scales = scales + row * groups_per_row;
    device const bfloat* row_biases = biases + row * groups_per_row;

    float accumulator = 0.0f;
    for (uint group = 0; group < groups_per_row; ++group) {
        const uint first = group * kAffineGroupSize + lane * 2u;
        const float x0 = float(x[first]);
        const float x1 = float(x[first + 1u]);
        const float q0 = float(affine_quant_value(row_weights, first, bits));
        const float q1 = float(affine_quant_value(row_weights, first + 1u, bits));
        accumulator = fma(float(row_scales[group]), q0 * x0 + q1 * x1,
                          accumulator);
        accumulator = fma(float(row_biases[group]), x0 + x1, accumulator);
    }
    accumulator = simd_sum(accumulator);
    if (lane == 0u) y[row] = half(accumulator);
}

kernel void affine_quant_embedding_lookup(
    device const uint*   table     [[buffer(0)]],
    device const bfloat* scales    [[buffer(1)]],
    device const bfloat* biases    [[buffer(2)]],
    device half*         output    [[buffer(3)]],
    constant uint&       token_id  [[buffer(4)]],
    constant uint&       dimension [[buffer(5)]],
    constant float&      out_scale [[buffer(6)]],
    uint element [[thread_position_in_grid]]
) {
    if (element >= dimension) return;
    const uint bits = affine_quant_bits();
    const uint groups_per_row = dimension / kAffineGroupSize;
    const uint words_per_row = (dimension * bits) / 32u;
    device const uint* row_weights = table + token_id * words_per_row;
    device const bfloat* row_scales = scales + token_id * groups_per_row;
    device const bfloat* row_biases = biases + token_id * groups_per_row;
    const uint group = element / kAffineGroupSize;
    const float value = float(affine_quant_value(row_weights, element, bits))
        * float(row_scales[group]) + float(row_biases[group]);
    output[element] = half(value * out_scale);
}
