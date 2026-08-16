import Foundation

/// CPU reference for the `sample` kernel's selection stage.
///
/// The sampler's own suites validate it by properties (argmax lands where
/// expected, a fixed seed replays, ids stay in range) and by agreement between
/// the general and top-64 kernels. Neither anchors the selection *arithmetic* —
/// top-p truncation, top-k capping, temperature reweighting and the inverse-CDF
/// draw — to an independent implementation, so an error shared by both GPU
/// paths would pass. This is that independent implementation.
///
/// It mirrors `logit.metal`'s order exactly, which is mlx-lm's: top-p is taken
/// against the full normalized distribution, top-k caps the survivors, then
/// temperature reweights only the final categorical draw. The RNG is the
/// kernel's xorshift64* — the shader comments call it bit-reproducible across
/// hardware and compilers, which is precisely what makes a CPU reference
/// possible here.
public enum SampleSelectionRef {
    /// xorshift64*, matching `xorshift64` in logit.metal.
    public static func xorshift64(_ state: inout UInt64) -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state &* 2_685_821_657_736_338_717
    }

    /// Top 24 bits as a [0,1) float, matching `uniform01` in logit.metal.
    public static func uniform01(_ state: inout UInt64) -> Float {
        let bits = UInt32(truncatingIfNeeded: xorshift64(&state) >> 40)
        return Float(bits) * (1.0 / 16_777_216.0)
    }

    /// The token id the kernel should choose for `probs`.
    ///
    /// `probs` is the softmaxed distribution the kernel reads — this reference
    /// covers selection only, not the softmax that produces it, which
    /// `LogitSoftcapSoftmaxRef` already covers.
    ///
    /// - Parameters:
    ///   - temperature: `0` selects greedy argmax, matching the kernel.
    ///   - topK: `0` means "no explicit cap"; the kernel then works over the
    ///     whole vocabulary.
    public static func select(probs: [Float],
                              temperature: Float,
                              topK: Int,
                              topP: Float,
                              seed: UInt64) -> UInt32 {
        precondition(!probs.isEmpty, "probs must be non-empty")
        if temperature == 0 {
            // Greedy: argmax, lowest index wins a tie (simd_min on the index).
            var best = 0
            for i in 1..<probs.count where probs[i] > probs[best] { best = i }
            return UInt32(best)
        }

        // Descending selection, ties resolved by lower index — the kernel
        // reduces with simd_max on the value then simd_min on the index.
        let k = topK > 0 ? min(topK, probs.count) : probs.count
        var claimed = [Bool](repeating: false, count: probs.count)
        var idx: [Int] = []
        var val: [Float] = []
        for _ in 0..<k {
            var bestI = -1
            var bestV = -Float.infinity
            for i in 0..<probs.count where !claimed[i] {
                if probs[i] > bestV {
                    bestV = probs[i]
                    bestI = i
                }
            }
            guard bestI >= 0, bestV.isFinite else { break }
            claimed[bestI] = true
            idx.append(bestI)
            val.append(bestV)
        }
        guard !idx.isEmpty else { return 0 }

        // Top-p against the full-vocabulary normalization, before top-k caps.
        var kept = idx.count
        if topP > 0, topP < 1 {
            var cumulative: Float = 0
            var cut = kept
            for i in 0..<kept {
                cumulative += val[i]
                if cumulative >= topP { cut = i + 1; break }
            }
            kept = cut
        }

        // Temperature applies only to the surviving categorical draw.
        let invTemp = 1.0 / temperature
        var weights = Array(val[0..<kept])
        if temperature != 1 {
            for i in 0..<kept { weights[i] = pow(weights[i], invTemp) }
        }
        let surviving = weights.reduce(0, +)

        // One discarded step before drawing, matching the kernel's comment
        // about small seeds being a few bits short of full entropy.
        var rng = seed
        _ = xorshift64(&rng)
        let u = uniform01(&rng) * surviving

        var run: Float = 0
        var picked = idx[0]
        for i in 0..<kept {
            run += weights[i]
            if u <= run { picked = idx[i]; break }
        }
        return UInt32(picked)
    }
}
