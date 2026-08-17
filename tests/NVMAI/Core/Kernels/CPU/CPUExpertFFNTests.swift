import Testing
import Foundation
@testable import NVMAI
import NVMAIValidationSupport

/// `CPUExpertFFN` reads packed expert bytes and computes the FFN in one pass.
/// These drive it and the independent reference (`MoeRef`/`DequantInt4GemvRef`,
/// which work from decoded `Int4AffineRow`s) from the *same* quantised weights,
/// so agreement means the packed-byte walk and the arithmetic are both right.
@Suite struct CPUExpertFFNTests {
    /// Small but structurally faithful: both dimensions are multiples of the
    /// 64-element group so the offset maths is exercised, not special-cased.
    static let d = 128
    static let f = 64

    private static func rows(count: Int, n: Int, seed: UInt64)
        -> [Quantization.Int4AffineRow] {
        var state = seed
        func next() -> Float {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33)))
                / Float(Int32.max)
        }
        return (0..<count).map { _ in
            Quantization.quantizeInt4Affine((0..<n).map { _ in next() })
        }
    }

    /// Lay rows out exactly as the packer does, so the test also pins the
    /// contiguous ordering `Offsets` assumes.
    private static func pack(gate: [Quantization.Int4AffineRow],
                             up: [Quantization.Int4AffineRow],
                             down: [Quantization.Int4AffineRow],
                             offsets: CPUExpertFFN.Offsets) -> [UInt8] {
        var blob = [UInt8](repeating: 0, count: offsets.strideBytes)
        func write(_ rows: [Quantization.Int4AffineRow],
                   _ wOff: Int, _ sOff: Int, _ bOff: Int) {
            var w = wOff, s = sOff, b = bOff
            for row in rows {
                blob.replaceSubrange(w..<(w + row.packed.count), with: row.packed)
                w += row.packed.count
                for value in row.scales {
                    blob[s] = UInt8(value & 0xFF); blob[s + 1] = UInt8(value >> 8); s += 2
                }
                for value in row.biases {
                    blob[b] = UInt8(value & 0xFF); blob[b + 1] = UInt8(value >> 8); b += 2
                }
            }
        }
        write(gate, offsets.gate, offsets.gateScales, offsets.gateBiases)
        write(up, offsets.up, offsets.upScales, offsets.upBiases)
        write(down, offsets.down, offsets.downScales, offsets.downBiases)
        return blob
    }

    @Test func offsetsCoverTheBlockExactlyWithNoOverlap() {
        let o = CPUExpertFFN.Offsets(d: 2048, f: 512)
        // The production 4-bit expert stride, from packed_experts/layout.json.
        #expect(o.strideBytes == 1_769_472)
        #expect(o.gate == 0)
        #expect(o.gateScales == 524_288)
        #expect(o.up == 589_824)
        #expect(o.down == 1_179_648)
        let bounds = [o.gate, o.gateScales, o.gateBiases, o.up, o.upScales,
                      o.upBiases, o.down, o.downScales, o.downBiases]
        #expect(bounds == bounds.sorted(), "tensors must be laid out in order")
    }

    @Test func matchesTheReferenceFFN() {
        let d = Self.d, f = Self.f
        let offsets = CPUExpertFFN.Offsets(d: d, f: f)
        let gate = Self.rows(count: f, n: d, seed: 1)
        let up = Self.rows(count: f, n: d, seed: 2)
        let down = Self.rows(count: d, n: f, seed: 3)
        let blob = Self.pack(gate: gate, up: up, down: down, offsets: offsets)

        var state: UInt64 = 99
        let x: [Float] = (0..<d).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1
            return Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33)))
                / Float(Int32.max)
        }

        // Reference: decode rows, run the FFN with silu (Qwen 3.6's SwiGLU).
        let gateOut = DequantInt4GemvRef.apply(weightRows: gate, x: x, n: d)
        let upOut = DequantInt4GemvRef.apply(weightRows: up, x: x, n: d)
        let act = (0..<f).map { i -> Float in
            let g = gateOut[i]
            return (g / (1 + Foundation.exp(-g))) * upOut[i]
        }
        let expected = DequantInt4GemvRef.apply(weightRows: down, x: act, n: f)

        var out = [Float](repeating: 0, count: d)
        var scratch = [Float](repeating: 0, count: CPUExpertFFN.scratchFloats(d: d, f: f))
        blob.withUnsafeBytes { raw in
            x.withUnsafeBufferPointer { xp in
                scratch.withUnsafeMutableBufferPointer { sp in
                    out.withUnsafeMutableBufferPointer { op in
                        CPUExpertFFN.accumulate(
                            expert: raw.baseAddress!, offsets: offsets,
                            x: xp.baseAddress!, d: d, f: f, routeWeight: 1.0,
                            scratch: sp.baseAddress!, out: op.baseAddress!)
                    }
                }
            }
        }

        let scale = expected.map { abs($0) }.max() ?? 1
        for i in 0..<d {
            #expect(abs(out[i] - expected[i]) <= 2e-4 * max(scale, 1),
                    "row \(i): \(out[i]) vs \(expected[i])")
        }
    }

    /// The routing weight must scale the contribution and *accumulate*, since
    /// callers sum several experts into one buffer.
    @Test func accumulatesScaledByRouteWeight() {
        let d = Self.d, f = Self.f
        let offsets = CPUExpertFFN.Offsets(d: d, f: f)
        let blob = Self.pack(gate: Self.rows(count: f, n: d, seed: 7),
                             up: Self.rows(count: f, n: d, seed: 8),
                             down: Self.rows(count: d, n: f, seed: 9),
                             offsets: offsets)
        let x = [Float](repeating: 0.05, count: d)

        func run(weight: Float, into out: inout [Float]) {
            var scratch = [Float](repeating: 0, count: CPUExpertFFN.scratchFloats(d: d, f: f))
            blob.withUnsafeBytes { raw in
                x.withUnsafeBufferPointer { xp in
                    scratch.withUnsafeMutableBufferPointer { sp in
                        out.withUnsafeMutableBufferPointer { op in
                            CPUExpertFFN.accumulate(
                                expert: raw.baseAddress!, offsets: offsets,
                                x: xp.baseAddress!, d: d, f: f, routeWeight: weight,
                                scratch: sp.baseAddress!, out: op.baseAddress!)
                        }
                    }
                }
            }
        }

        var once = [Float](repeating: 0, count: d)
        run(weight: 1.0, into: &once)
        var twice = [Float](repeating: 0, count: d)
        run(weight: 0.75, into: &twice)
        run(weight: 0.25, into: &twice)

        let scale = once.map { abs($0) }.max() ?? 1
        for i in 0..<d {
            #expect(abs(once[i] - twice[i]) <= 1e-4 * max(scale, 1))
        }
    }
}
