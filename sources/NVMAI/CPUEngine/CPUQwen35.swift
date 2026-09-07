import Foundation

/// Qwen3.5-2B on the CPU, one token at a time.
///
/// The side-engine's model: small enough to stay resident beside a 35B, and
/// run on cores the main engine leaves idle. It is a decode-only engine —
/// there is no batched prefill, because everything it is for (distilling a
/// session, checking a claim against the store) is a short prompt and a
/// short answer, and a prompt is just decode with the output thrown away.
///
/// **Width is a scheduling decision.** Measured on this machine, a 35B
/// generation slows by 3% when this reads at one thread and by 31% at four.
/// So `threads` is settable, and the caller — which knows whether someone is
/// waiting on the GPU — chooses: one while a client generation is in flight,
/// four in the gaps.
///
/// Correctness is defined by `tools/qwen35_reference.py`, which is checked
/// in turn by continuations a 2B has no excuse for getting wrong.
public final class CPUQwen35 {

    public let snapshot: AffineSnapshot
    public let configuration: AffineSnapshot.Configuration
    /// Rows of the GEMVs to split across performance cores. See the note
    /// above: this is the knob that decides what the side-engine costs the
    /// model the person is waiting for.
    public var threads: Int

    private let prefix = "language_model.model."
    private var position = 0

    // Small tensors, read once and kept: 24 of each norm, the convolution
    // taps, and the delta rule's two per-head vectors.
    private var inputNorm: [[Float]] = []
    private var postNorm: [[Float]] = []
    private var finalNorm: [Float] = []
    private var gdnNorm: [Int: [Float]] = [:]
    private var convTaps: [Int: [Float]] = [:]
    private var aLog: [Int: [Float]] = [:]
    private var dtBias: [Int: [Float]] = [:]
    private var queryNorm: [Int: [Float]] = [:]
    private var keyNorm: [Int: [Float]] = [:]
    /// The delta rule's two scalar-per-head projections, which the converter
    /// deliberately leaves at BF16: sixteen rows each, so quantizing them
    /// would save nothing and they feed an exponential, where a rounding
    /// error does not stay small.
    private var deltaA: [Int: [Float]] = [:]
    private var deltaB: [Int: [Float]] = [:]

    // Carried state. Every one of these is a place a wrong hand-off between
    // tokens can hide, which is why sequence parity is a separate gate from
    // position 0.
    private var recurrent: [Int: [Float]] = [:]     // [Hv][Dv][Dk]
    private var convolution: [Int: [Float]] = [:]   // [K-1][convDim]
    private var keys: [Int: [Float]] = [:]          // [position][kvHeads * headDim]
    private var values: [Int: [Float]] = [:]

    public init(snapshot: AffineSnapshot, threads: Int? = nil) throws {
        self.snapshot = snapshot
        configuration = snapshot.configuration
        self.threads = threads ?? Int8AffineGEMV.preferredThreads
        for layer in 0..<configuration.layers {
            inputNorm.append(try snapshot.floats("\(prefix)layers.\(layer).input_layernorm.weight"))
            postNorm.append(try snapshot.floats(
                "\(prefix)layers.\(layer).post_attention_layernorm.weight"))
            if configuration.isAttention(layer) {
                queryNorm[layer] = try snapshot.floats(
                    "\(prefix)layers.\(layer).self_attn.q_norm.weight")
                keyNorm[layer] = try snapshot.floats(
                    "\(prefix)layers.\(layer).self_attn.k_norm.weight")
            } else {
                let stem = "\(prefix)layers.\(layer).linear_attn."
                gdnNorm[layer] = try snapshot.floats(stem + "norm.weight")
                convTaps[layer] = try snapshot.floats(stem + "conv1d.weight")
                aLog[layer] = try snapshot.floats(stem + "A_log")
                dtBias[layer] = try snapshot.floats(stem + "dt_bias")
                deltaA[layer] = try snapshot.floats(stem + "in_proj_a.weight")
                deltaB[layer] = try snapshot.floats(stem + "in_proj_b.weight")
            }
        }
        finalNorm = try snapshot.floats("\(prefix)norm.weight")
    }

    /// Forget the conversation. Every carried state goes, and the next token
    /// is position zero again.
    public func reset() {
        position = 0
        recurrent.removeAll(keepingCapacity: true)
        convolution.removeAll(keepingCapacity: true)
        keys.removeAll(keepingCapacity: true)
        values.removeAll(keepingCapacity: true)
    }

    // MARK: - one token

    /// Returns the logits over the whole vocabulary.
    public func step(token: Int) throws -> [Float] {
        var h = try embedding(of: token)
        for layer in 0..<configuration.layers {
            var normalized = h
            CPUOps.rmsNorm(&normalized, gamma: inputNorm[layer],
                           epsilon: configuration.normEpsilon)
            let mixed = configuration.isAttention(layer)
                ? try attention(layer: layer, x: normalized)
                : try gatedDeltaNet(layer: layer, x: normalized)
            for index in h.indices { h[index] += mixed[index] }

            normalized = h
            CPUOps.rmsNorm(&normalized, gamma: postNorm[layer],
                           epsilon: configuration.normEpsilon)
            let feed = try mlp(layer: layer, x: normalized)
            for index in h.indices { h[index] += feed[index] }
        }
        CPUOps.rmsNorm(&h, gamma: finalNorm, epsilon: configuration.normEpsilon)
        position += 1
        return try head(h)
    }

    // MARK: - blocks

    private func matrix(_ name: String) throws -> AffineSnapshot.Matrix {
        try snapshot.matrix(name)
    }

    private func project(_ matrix: AffineSnapshot.Matrix, _ x: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: matrix.rows)
        x.withUnsafeBufferPointer { input in
            out.withUnsafeMutableBufferPointer { output in
                CPUOps.gemv(matrix, x: input.baseAddress!,
                            out: output.baseAddress!, threads: threads)
            }
        }
        return out
    }

    /// `out = W · x` for a small dense row-major matrix. Only the delta
    /// rule's two per-head projections come through here.
    private func dense(_ weights: [Float], rows: Int, x: [Float]) -> [Float] {
        let columns = weights.count / rows
        var out = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var total: Float = 0
            let base = row * columns
            for column in 0..<columns { total += weights[base + column] * x[column] }
            out[row] = total
        }
        return out
    }

    private func embedding(of token: Int) throws -> [Float] {
        let table = try matrix("\(prefix)embed_tokens.weight")
        precondition(token >= 0 && token < table.rows, "token \(token) out of range")
        return dequantize(row: token, of: table)
    }

    /// The tied head: the embedding table is the output projection, so the
    /// logits are one GEMV over 248320 rows — and the largest single read of
    /// every token, which is why it dominates the 1.9 GB per-token figure.
    private func head(_ h: [Float]) throws -> [Float] {
        project(try matrix("\(prefix)embed_tokens.weight"), h)
    }

    private func mlp(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).mlp."
        var gate = project(try matrix(stem + "gate_proj.weight"), x)
        let up = project(try matrix(stem + "up_proj.weight"), x)
        for index in gate.indices { gate[index] = CPUOps.silu(gate[index]) * up[index] }
        return project(try matrix(stem + "down_proj.weight"), gate)
    }

    private func attention(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).self_attn."
        let heads = configuration.heads
        let kvHeads = configuration.keyValueHeads
        let dim = configuration.headDim

        // q_proj is twice as wide: the output gate is fused into it, packed
        // per head as [query ; gate]. Reading it as a plain query projection
        // gives a model that runs and is wrong.
        let packed = project(try matrix(stem + "q_proj.weight"), x)
        var query = [Float](repeating: 0, count: heads * dim)
        var gate = [Float](repeating: 0, count: heads * dim)
        for head in 0..<heads {
            let source = head * 2 * dim
            for index in 0..<dim {
                query[head * dim + index] = packed[source + index]
                gate[head * dim + index] = packed[source + dim + index]
            }
        }
        var key = project(try matrix(stem + "k_proj.weight"), x)
        let value = project(try matrix(stem + "v_proj.weight"), x)

        CPUOps.rmsNormPerSlice(&query, width: dim, gamma: queryNorm[layer]!,
                               epsilon: configuration.normEpsilon)
        CPUOps.rmsNormPerSlice(&key, width: dim, gamma: keyNorm[layer]!,
                               epsilon: configuration.normEpsilon)
        CPUOps.applyRoPE(&query, headDim: dim, rotaryDim: configuration.rotaryDim,
                         position: position, theta: configuration.ropeTheta)
        CPUOps.applyRoPE(&key, headDim: dim, rotaryDim: configuration.rotaryDim,
                         position: position, theta: configuration.ropeTheta)

        keys[layer, default: []].append(contentsOf: key)
        values[layer, default: []].append(contentsOf: value)
        let cached = keys[layer]!.count / (kvHeads * dim)

        let group = heads / kvHeads
        let scale = 1 / Float(dim).squareRoot()
        var out = [Float](repeating: 0, count: heads * dim)
        var scores = [Float](repeating: 0, count: cached)
        for head in 0..<heads {
            let kvHead = head / group
            for step in 0..<cached {
                let base = step * kvHeads * dim + kvHead * dim
                var total: Float = 0
                for index in 0..<dim { total += keys[layer]![base + index] * query[head * dim + index] }
                scores[step] = total * scale
            }
            CPUOps.softmaxInPlace(&scores)
            for step in 0..<cached {
                let weight = scores[step]
                if weight == 0 { continue }
                let base = step * kvHeads * dim + kvHead * dim
                for index in 0..<dim { out[head * dim + index] += weight * values[layer]![base + index] }
            }
        }
        for index in out.indices { out[index] *= CPUOps.sigmoid(gate[index]) }
        return project(try matrix(stem + "o_proj.weight"), out)
    }

    private func gatedDeltaNet(layer: Int, x: [Float]) throws -> [Float] {
        let stem = "\(prefix)layers.\(layer).linear_attn."
        let hk = configuration.linearKeyHeads
        let hv = configuration.linearValueHeads
        let dk = configuration.linearKeyHeadDim
        let dv = configuration.linearValueHeadDim
        let kernel = configuration.convKernel

        let mixed = project(try matrix(stem + "in_proj_qkv.weight"), x)
        let z = project(try matrix(stem + "in_proj_z.weight"), x)
        let a = dense(deltaA[layer]!, rows: hv, x: x)
        let b = dense(deltaB[layer]!, rows: hv, x: x)
        let convDim = mixed.count

        // Causal depthwise convolution over the last `kernel` tokens. Tap
        // `k` reads `kernel - 1 - k` positions back, so the last tap is this
        // token.
        var tail = convolution[layer] ?? [Float](repeating: 0, count: (kernel - 1) * convDim)
        let taps = convTaps[layer]!
        var convolved = [Float](repeating: 0, count: convDim)
        for channel in 0..<convDim {
            var total: Float = 0
            for step in 0..<(kernel - 1) {
                total += taps[channel * kernel + step] * tail[step * convDim + channel]
            }
            total += taps[channel * kernel + kernel - 1] * mixed[channel]
            convolved[channel] = CPUOps.silu(total)
        }
        // Slide the window: drop the oldest row, append this token's input.
        tail.removeFirst(convDim)
        tail.append(contentsOf: mixed)
        convolution[layer] = tail

        let keyWidth = hk * dk
        var query = Array(convolved[0..<keyWidth])
        var key = Array(convolved[keyWidth..<(2 * keyWidth)])
        let value = Array(convolved[(2 * keyWidth)...])
        CPUOps.l2NormalizePerSlice(&query, width: dk, epsilon: configuration.normEpsilon)
        CPUOps.l2NormalizePerSlice(&key, width: dk, epsilon: configuration.normEpsilon)

        var state = recurrent[layer] ?? [Float](repeating: 0, count: hv * dv * dk)
        var readout = [Float](repeating: 0, count: hv * dv)
        let repeats = hv / hk
        for head in 0..<hv {
            let keyHead = head / repeats
            let beta = CPUOps.sigmoid(b[head])
            let decay = expf(-expf(aLog[layer]![head])
                             * CPUOps.softplus(a[head] + dtBias[layer]![head]))
            let stateBase = head * dv * dk
            let keyBase = keyHead * dk
            for row in 0..<dv {
                let rowBase = stateBase + row * dk
                // Decay, read the stored value for this key, correct it
                // towards the new one, and write the correction back as an
                // outer product. This is the delta rule.
                var stored: Float = 0
                for column in 0..<dk {
                    let decayed = state[rowBase + column] * decay
                    state[rowBase + column] = decayed
                    stored += decayed * key[keyBase + column]
                }
                let correction = (value[head * dv + row] - stored) * beta
                var sum: Float = 0
                for column in 0..<dk {
                    let updated = state[rowBase + column] + key[keyBase + column] * correction
                    state[rowBase + column] = updated
                    sum += updated * query[keyBase + column]
                }
                readout[head * dv + row] = sum
            }
        }
        recurrent[layer] = state

        let inverse = 1 / Float(dv).squareRoot()
        for index in readout.indices { readout[index] *= inverse }
        CPUOps.rmsNormPerSlice(&readout, width: dv, gamma: gdnNorm[layer]!,
                               epsilon: configuration.normEpsilon)
        // SiLU, not sigmoid. The gate is `silu` in this lineage and
        // `sigmoid` in Qwen3.8-Flash-Next; getting it wrong produces a model
        // that runs, keeps healthy activations, and predicts a bare space
        // for "Once upon a".
        for index in readout.indices { readout[index] *= CPUOps.silu(z[index]) }
        return project(try matrix(stem + "out_proj.weight"), readout)
    }

    // MARK: - one row of a quantized matrix

    /// Unpacks a single row. Used for the embedding lookup, where reading two
    /// gigabytes to fetch one vector would be absurd.
    private func dequantize(row: Int, of matrix: AffineSnapshot.Matrix) -> [Float] {
        let lanes = 32 / matrix.bits
        let mask = UInt32((1 << matrix.bits) - 1)
        let wordsPerRow = matrix.columns / lanes
        let groupsPerRow = matrix.columns / matrix.groupSize
        var out = [Float](repeating: 0, count: matrix.columns)
        let words = matrix.weights.baseAddress!.assumingMemoryBound(to: UInt32.self)
        for word in 0..<wordsPerRow {
            let packed = words[row * wordsPerRow + word]
            for lane in 0..<lanes {
                let column = word * lanes + lane
                let level = Float((packed >> (matrix.bits * lane)) & mask)
                let group = row * groupsPerRow + column / matrix.groupSize
                out[column] = level * bfloat(matrix.scales[group])
                    + bfloat(matrix.biases[group])
            }
        }
        return out
    }

    @inline(__always)
    private func bfloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
