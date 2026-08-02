import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct AffineQuantTests {
    private static func packed(bits: Int, rows: Int, columns: Int) -> ([UInt8], [[UInt32]]) {
        let rowBytes = columns * bits / 8
        var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
        var values = Array(repeating: [UInt32](repeating: 0, count: columns), count: rows)
        let mask = UInt32((1 << bits) - 1)
        for row in 0..<rows {
            for column in 0..<columns {
                let value = UInt32((row * 11 + column * 3) & Int(mask))
                values[row][column] = value
                let bit = column * bits
                let byte = row * rowBytes + bit / 8
                let shift = bit % 8
                var word = UInt32(value) << shift
                var remaining = bits + shift
                var index = byte
                while remaining > 0 {
                    bytes[index] |= UInt8(truncatingIfNeeded: word)
                    word >>= 8
                    remaining -= 8
                    index += 1
                }
            }
        }
        return (bytes, values)
    }

    @Test(arguments: [4, 6, 8])
    func gemvDecodesPackedAffine(bits: Int) throws {
        let rows = 3, columns = 128
        let (packed, values) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let input = [Float16](repeating: 1, count: columns)
        let ctx = try MetalContext()
        let kernel = try AffineQuantGEMV(context: ctx, weightBits: bits)
        let w = ctx.device.makeBuffer(bytes: packed, length: packed.count)!
        let s = ctx.device.makeBuffer(bytes: scales, length: scales.count * 2)!
        let b = ctx.device.makeBuffer(bytes: biases, length: biases.count * 2)!
        let x = Fp16Buffer.make(ctx.device, halves: input)!
        let y = Fp16Buffer.make(ctx.device, count: rows)!
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb, weights: w, scales: s, biases: b,
                      x: x, y: y, m: UInt32(rows), n: UInt32(columns))
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y, count: rows)
        let expected = values.map { Float($0.reduce(0, +)) }
        #expect(RelError.compute(actual: actual, reference: expected) < 0.002)
    }

    @Test(arguments: [4, 6, 8])
    func embeddingDecodesPackedAffine(bits: Int) throws {
        let rows = 3, columns = 128
        let (packed, values) = Self.packed(bits: bits, rows: rows, columns: columns)
        let one = UInt16(truncatingIfNeeded: Float(1).bitPattern >> 16)
        let scales = [UInt16](repeating: one, count: rows * columns / 64)
        let biases = [UInt16](repeating: 0, count: scales.count)
        let ctx = try MetalContext()
        let kernel = try AffineQuantEmbeddingLookup(context: ctx, weightBits: bits)
        let w = ctx.device.makeBuffer(bytes: packed, length: packed.count)!
        let s = ctx.device.makeBuffer(bytes: scales, length: scales.count * 2)!
        let b = ctx.device.makeBuffer(bytes: biases, length: biases.count * 2)!
        let y = Fp16Buffer.make(ctx.device, count: columns)!
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb, table: w, scales: s, biases: b,
                      out: y, tokenId: 1, d: UInt32(columns), outScale: 1)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        let actual = Fp16Buffer.read(y, count: columns)
        #expect(actual == values[1].map { Float($0) })
    }
}
