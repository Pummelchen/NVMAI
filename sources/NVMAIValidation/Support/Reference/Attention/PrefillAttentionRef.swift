import Foundation

/// CPU reference for the chunked-prefill attention kernels.
///
/// Lifted out of PrefillAttentionTests so it sits with the other references
/// and is reviewed alongside them, following GDNReference. The input bundle
/// moves with it: it is plain data describing the attention problem, not test
/// scaffolding.
public enum PrefillAttentionRef {
    public struct Inputs {
        public var q: [Float]
        public var k: [Float]
        public var v: [Float]
        public var qStride: Int
        public var kvStride: Int
        public var oStride: Int
        public var headDim: Int
        public var qHeads: Int
        public var kvHeads: Int
        public var start: Int
        public var chunk: Int
        public var kvValid: Int
        public var window: Int
        public var scale: Float

        public init(q: [Float],
                    k: [Float],
                    v: [Float],
                    qStride: Int,
                    kvStride: Int,
                    oStride: Int,
                    headDim: Int,
                    qHeads: Int,
                    kvHeads: Int,
                    start: Int,
                    chunk: Int,
                    kvValid: Int,
                    window: Int,
                    scale: Float) {
            self.q = q
            self.k = k
            self.v = v
            self.qStride = qStride
            self.kvStride = kvStride
            self.oStride = oStride
            self.headDim = headDim
            self.qHeads = qHeads
            self.kvHeads = kvHeads
            self.start = start
            self.chunk = chunk
            self.kvValid = kvValid
            self.window = window
            self.scale = scale
        }
    }

    public static func apply(_ fixture: Inputs) -> [Float] {
        var out = [Float](repeating: 0, count: fixture.chunk * fixture.qHeads * fixture.headDim)
        let qPerKV = fixture.qHeads / fixture.kvHeads
        for t in 0..<fixture.chunk {
            let absQ = fixture.start + t
            let first: Int
            if fixture.window == 0 {
                first = 0
            } else {
                first = max(0, absQ + 1 - fixture.window)
            }
            let last = min(fixture.kvValid, absQ + 1)
            for qh in 0..<fixture.qHeads {
                let kvh = qh / qPerKV
                var scores: [Float] = []
                scores.reserveCapacity(last - first)
                for key in first..<last {
                    var score: Float = 0
                    for d in 0..<fixture.headDim {
                        let qv = fixture.q[t * fixture.qStride + qh * fixture.headDim + d]
                        let kv = fixture.k[key * fixture.kvStride + kvh * fixture.headDim + d]
                        score += qv * kv
                    }
                    scores.append(score * fixture.scale)
                }
                let maxScore = scores.max() ?? -.infinity
                var denom: Float = 0
                for score in scores {
                    denom += Foundation.exp(score - maxScore)
                }
                for d in 0..<fixture.headDim {
                    var acc: Float = 0
                    for (i, key) in (first..<last).enumerated() {
                        let w = Foundation.exp(scores[i] - maxScore)
                        acc += w * fixture.v[key * fixture.kvStride + kvh * fixture.headDim + d]
                    }
                    out[(t * fixture.qHeads + qh) * fixture.headDim + d] = denom > 0 ? acc / denom : 0
                }
            }
        }
        return out
    }
}
