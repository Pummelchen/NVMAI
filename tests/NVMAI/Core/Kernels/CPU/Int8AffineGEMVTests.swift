import Testing
import Foundation
@testable import NVMAI

/// The decode inner loop of the CPU side-engine.
///
/// The kernel walks packed bytes and group scales itself; the reference
/// dequantises each row through `Quantization.dequantizeInt8Affine` and does
/// a plain dot product. Driving both from the *same* quantised weights means
/// agreement is evidence about the byte walk and the arithmetic, not about
/// the quantiser — which is the property that made the 4-bit kernel's tests
/// worth having.
@Suite struct Int8AffineGEMVTests {

    private static func pseudorandom(_ count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(bitPattern: UInt32(truncatingIfNeeded: state >> 33)))
                / Float(Int32.max)
        }
    }

    /// Rows laid out as the snapshot lays them: weights row-major, then all
    /// scales row-major, then all biases.
    private static func run(rows rowValues: [[Float]], x: [Float]) -> [Float] {
        let n = x.count
        let quantised = rowValues.map { Quantization.quantizeInt8Affine($0) }
        var weights: [UInt8] = []
        var scales: [UInt16] = []
        var biases: [UInt16] = []
        for row in quantised {
            weights.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }
        var out = [Float](repeating: .nan, count: rowValues.count)
        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        out.withUnsafeMutableBufferPointer { o in
                            Int8AffineGEMV.apply(
                                weights: w.baseAddress!, scales: s.baseAddress!,
                                biases: b.baseAddress!, x: xp.baseAddress!,
                                rows: rowValues.count, n: n, out: o.baseAddress!)
                        }
                    }
                }
            }
        }
        return out
    }

    private static func reference(rows rowValues: [[Float]], x: [Float]) -> [Float] {
        rowValues.map { row in
            let dequantised = Quantization.dequantizeInt8Affine(
                Quantization.quantizeInt8Affine(row), n: row.count)
            return zip(dequantised, x).reduce(0) { $0 + $1.0 * $1.1 }
        }
    }

    /// One group: the narrowest shape the format allows, so an off-by-one in
    /// the group stride has nowhere to hide.
    @Test func singleGroupMatchesTheReference() {
        let x = Self.pseudorandom(64, seed: 11)
        let rows = (0..<3).map { Self.pseudorandom(64, seed: UInt64(100 + $0)) }
        let got = Self.run(rows: rows, x: x), want = Self.reference(rows: rows, x: x)
        for (a, b) in zip(got, want) {
            #expect(abs(a - b) <= 1e-3 * max(1, abs(b)))
        }
    }

    /// A production shape: 2048 wide is the attention and gate/up projection,
    /// 6144 the feed-forward's inner dimension.
    @Test func productionWidthsMatchTheReference() {
        for n in [2048, 6144] {
            let x = Self.pseudorandom(n, seed: 7)
            let rows = (0..<5).map { Self.pseudorandom(n, seed: UInt64(n + $0)) }
            let got = Self.run(rows: rows, x: x), want = Self.reference(rows: rows, x: x)
            for (index, (a, b)) in zip(got, want).enumerated() {
                #expect(abs(a - b) <= 1e-3 * max(1, abs(b)),
                        "row \(index) at n=\(n): \(a) vs \(b)")
            }
        }
    }

    /// A constant row quantises to scale 1, bias = value, and must come back
    /// as `value * sum(x)` exactly — the case where the bias term carries the
    /// whole answer and the quantised term contributes nothing.
    @Test func constantRowIsCarriedByTheBiasTerm() {
        let x = Self.pseudorandom(128, seed: 3)
        let rows = [[Float](repeating: 0.375, count: 128)]
        let got = Self.run(rows: rows, x: x)
        let want = 0.375 * x.reduce(0, +)
        #expect(abs(got[0] - want) <= 1e-3 * max(1, abs(want)))
    }

    /// Rows are independent, so a caller may thread over row ranges by
    /// advancing every pointer together. This pins that contract: the same
    /// weights split into two calls give the same answer as one call.
    @Test func rowRangesAreIndependent() {
        let n = 256
        let x = Self.pseudorandom(n, seed: 21)
        let rows = (0..<8).map { Self.pseudorandom(n, seed: UInt64(500 + $0)) }
        let whole = Self.run(rows: rows, x: x)
        let first = Self.run(rows: Array(rows[0..<3]), x: x)
        let rest = Self.run(rows: Array(rows[3...]), x: x)
        for (a, b) in zip(whole, first + rest) {
            #expect(a == b)
        }
    }

    /// Values that survive both quantisers exactly must give the same answer
    /// at both widths. This is what stops the 4-bit and 8-bit builds of one
    /// model from drifting apart, and it is why the two kernels factor the
    /// accumulation identically.
    @Test func agreesWithTheFourBitPathOnRepresentableValues() {
        let n = 64
        let row = (0..<n).map { Float($0 % 16) }   // exact 4-bit levels
        let x = Self.pseudorandom(n, seed: 5)
        let eight = Self.run(rows: [row], x: x)[0]
        let four = Quantization.dequantizeInt4Affine(
            Quantization.quantizeInt4Affine(row), n: n)
        let want = zip(four, x).reduce(0) { $0 + $1.0 * $1.1 }
        #expect(abs(eight - want) <= 1e-3 * max(1, abs(want)),
                "8-bit \(eight) vs 4-bit reference \(want)")
    }

    /// Threading splits rows, so its result must be bit-identical to the
    /// single-threaded one -- there is no cross-worker accumulation whose
    /// order could differ. Asserted rather than assumed.
    @Test func threadedMatchesSingleThreadedExactly() {
        let n = 512
        let x = Self.pseudorandom(n, seed: 31)
        let rowValues = (0..<600).map { Self.pseudorandom(n, seed: UInt64(900 + $0)) }
        let quantised = rowValues.map { Quantization.quantizeInt8Affine($0) }
        var weights: [UInt8] = [], scales: [UInt16] = [], biases: [UInt16] = []
        for row in quantised {
            weights.append(contentsOf: row.packed)
            scales.append(contentsOf: row.scales)
            biases.append(contentsOf: row.biases)
        }
        var one = [Float](repeating: .nan, count: rowValues.count)
        var many = [Float](repeating: .nan, count: rowValues.count)
        weights.withUnsafeBufferPointer { w in
            scales.withUnsafeBufferPointer { s in
                biases.withUnsafeBufferPointer { b in
                    x.withUnsafeBufferPointer { xp in
                        one.withUnsafeMutableBufferPointer { o in
                            Int8AffineGEMV.apply(weights: w.baseAddress!, scales: s.baseAddress!,
                                           biases: b.baseAddress!, x: xp.baseAddress!,
                                           rows: rowValues.count, n: n, out: o.baseAddress!)
                        }
                        many.withUnsafeMutableBufferPointer { o in
                            Int8AffineGEMV.threaded(
                                weights: w.baseAddress!, scales: s.baseAddress!,
                                biases: b.baseAddress!, x: xp.baseAddress!,
                                rows: rowValues.count, n: n, out: o.baseAddress!)
                        }
                    }
                }
            }
        }
        #expect(one == many)
        #expect(Int8AffineGEMV.preferredThreads >= 1)
    }
}
