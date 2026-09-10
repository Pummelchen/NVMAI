import Foundation

/// Choosing the next token, on the CPU.
///
/// Greedy is the default and the only mode the memory work uses: distilling
/// a session and checking a claim both want the model's best answer, and a
/// deterministic side-engine is one whose output can be compared between
/// runs. But a CPU-served model is a served model, and a client that sends a
/// temperature expects it to mean something.
///
/// Deliberately the simple algorithms. Sampling 248,320 logits costs a
/// fraction of the 1.9 GB of weight reads that produced them, so the clever
/// version would save nothing measurable and could get the distribution
/// wrong.
public struct CPUSampler: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    /// Nil is deterministic: the same prompt gives the same answer, which is
    /// what makes a regression visible.
    public var seed: UInt64?

    public init(temperature: Float = 0, topP: Float = 1, topK: Int = 0,
                seed: UInt64? = nil) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.seed = seed
    }

    public var isGreedy: Bool { temperature <= 0 }

    /// unchecked-invariant: `state` is only touched from `next()`, and a
    /// sampler belongs to one generation, which is one task.
    public final class Generator: @unchecked Sendable {
        private var state: UInt64
        init(seed: UInt64) { state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }
        func next() -> Float {
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Float(state >> 40) / Float(1 << 24)
        }
    }

    public func makeGenerator() -> Generator {
        Generator(seed: seed ?? UInt64(Date().timeIntervalSince1970 * 1000))
    }

    public func pick(_ logits: [Float], using generator: Generator) -> Int {
        guard !isGreedy else {
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            return best
        }
        // Top-k first, because it bounds the sort; top-p then trims what is
        // left by mass. Both are no-ops at their defaults.
        let limit = topK > 0 ? min(topK, logits.count) : logits.count
        var order = Array(logits.indices)
        if limit < logits.count {
            order.sort { logits[$0] > logits[$1] }
            order = Array(order.prefix(limit))
        } else {
            order.sort { logits[$0] > logits[$1] }
        }
        let peak = logits[order[0]]
        var weights = [Float]()
        weights.reserveCapacity(order.count)
        var total: Float = 0
        for index in order {
            let value = expf((logits[index] - peak) / max(temperature, 1e-4))
            weights.append(value)
            total += value
        }
        var cutoff = order.count
        if topP < 1 {
            var mass: Float = 0
            for position in weights.indices {
                mass += weights[position] / total
                if mass >= topP { cutoff = position + 1; break }
            }
        }
        var remaining: Float = 0
        for position in 0..<cutoff { remaining += weights[position] }
        let target = generator.next() * remaining
        var running: Float = 0
        for position in 0..<cutoff {
            running += weights[position]
            if running >= target { return order[position] }
        }
        return order[0]
    }
}
