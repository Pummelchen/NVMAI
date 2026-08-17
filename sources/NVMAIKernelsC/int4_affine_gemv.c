// Time-critical inner loop for CPU-side expert evaluation.
//
// The Swift implementation of this loop ran 2.3x slower than the equivalent C
// (1.19 ms vs 0.49 ms per production expert). The cost was not the algorithm:
// building a SIMD8<Float> element by element does not lower to a vector load,
// so every group paid scalar inserts for both the unpacked nibbles and the
// activations. Here the nibbles come out of one 16-byte load with a mask, a
// shift and a zip, which is what the hardware is actually good at.

#include "include/nvmai_kernels.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define NVMAI_GROUP_SIZE 64

static inline float nvmai_bf16(uint16_t bits) {
    union { uint32_t u; float f; } c;
    c.u = ((uint32_t)bits) << 16;
    return c.f;
}

void nvmai_int4_affine_gemv(const uint8_t *weights,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            const float *x,
                            size_t rows,
                            size_t n,
                            float *out) {
    const size_t groups = n / NVMAI_GROUP_SIZE;
    const size_t row_bytes = n / 2;

#if defined(__ARM_NEON)
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *w_row = weights + r * row_bytes;
        const uint16_t *s_row = scales + r * groups;
        const uint16_t *b_row = biases + r * groups;
        float acc = 0.0f;

        for (size_t g = 0; g < groups; ++g) {
            const uint8_t *wg = w_row + g * (NVMAI_GROUP_SIZE / 2);
            const float *xg = x + g * NVMAI_GROUP_SIZE;
            float32x4_t dot = vdupq_n_f32(0.0f);
            float32x4_t xs = vdupq_n_f32(0.0f);

            // 32 bytes per group; 16 bytes -> 32 nibbles per iteration.
            for (size_t k = 0; k < NVMAI_GROUP_SIZE / 2; k += 16) {
                const uint8x16_t packed = vld1q_u8(wg + k);
                const uint8x16_t lo = vandq_u8(packed, nibble_mask);
                const uint8x16_t hi = vshrq_n_u8(packed, 4);
                // Element 2k is the low nibble of byte k and 2k+1 the high
                // one, so zipping lo with hi restores element order.
                const uint8x16x2_t z = vzipq_u8(lo, hi);

                for (int half = 0; half < 2; ++half) {
                    const uint8x16_t q8 = z.val[half];
                    const uint16x8_t q16_lo = vmovl_u8(vget_low_u8(q8));
                    const uint16x8_t q16_hi = vmovl_u8(vget_high_u8(q8));
                    const float32x4_t q0 =
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_lo)));
                    const float32x4_t q1 =
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16_lo)));
                    const float32x4_t q2 =
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(q16_hi)));
                    const float32x4_t q3 =
                        vcvtq_f32_u32(vmovl_u16(vget_high_u16(q16_hi)));

                    const float *xp = xg + k * 2 + half * 16;
                    const float32x4_t x0 = vld1q_f32(xp);
                    const float32x4_t x1 = vld1q_f32(xp + 4);
                    const float32x4_t x2 = vld1q_f32(xp + 8);
                    const float32x4_t x3 = vld1q_f32(xp + 12);

                    dot = vfmaq_f32(dot, q0, x0);
                    dot = vfmaq_f32(dot, q1, x1);
                    dot = vfmaq_f32(dot, q2, x2);
                    dot = vfmaq_f32(dot, q3, x3);
                    xs = vaddq_f32(xs, x0);
                    xs = vaddq_f32(xs, x1);
                    xs = vaddq_f32(xs, x2);
                    xs = vaddq_f32(xs, x3);
                }
            }
            acc += nvmai_bf16(s_row[g]) * vaddvq_f32(dot)
                 + nvmai_bf16(b_row[g]) * vaddvq_f32(xs);
        }
        out[r] = acc;
    }
#else
    // Portable fallback. NVMAI targets Apple Silicon, so this exists to keep
    // the file compilable rather than as a path anyone is expected to take.
    for (size_t r = 0; r < rows; ++r) {
        const uint8_t *w_row = weights + r * row_bytes;
        float acc = 0.0f;
        for (size_t g = 0; g < groups; ++g) {
            const uint8_t *wg = w_row + g * (NVMAI_GROUP_SIZE / 2);
            const float *xg = x + g * NVMAI_GROUP_SIZE;
            float dot = 0.0f, xsum = 0.0f;
            for (size_t k = 0; k < NVMAI_GROUP_SIZE / 2; ++k) {
                const float x0 = xg[k * 2], x1 = xg[k * 2 + 1];
                dot += (float)(wg[k] & 0x0F) * x0 + (float)(wg[k] >> 4) * x1;
                xsum += x0 + x1;
            }
            acc += nvmai_bf16(scales[r * groups + g]) * dot
                 + nvmai_bf16(biases[r * groups + g]) * xsum;
        }
        out[r] = acc;
    }
#endif
}
