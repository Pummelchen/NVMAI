#ifndef NVMAI_KERNELS_H
#define NVMAI_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/// `out[r] = sum_i (q[r][i] * scale[r][g(i)] + bias[r][g(i)]) * x[i]`
///
/// Affine INT4 GEMV over a `rows`-by-`n` matrix in the packed `.gturbo`
/// layout: nibbles low-first (element 2k in the low half of byte k), one BF16
/// scale and one BF16 bias per group of 64 elements, `scales` and `biases`
/// each `rows * (n / 64)` entries in row-major order.
///
/// `n` must be a multiple of 64. `out` is written, not accumulated.
///
/// Accumulation is factored as `scale * sum(q*x) + bias * sum(x)` per group,
/// matching how moe.metal factors the same product, so the two implementations
/// round alike at group boundaries.
void nvmai_int4_affine_gemv(const uint8_t *weights,
                            const uint16_t *scales,
                            const uint16_t *biases,
                            const float *x,
                            size_t rows,
                            size_t n,
                            float *out);

#endif /* NVMAI_KERNELS_H */
