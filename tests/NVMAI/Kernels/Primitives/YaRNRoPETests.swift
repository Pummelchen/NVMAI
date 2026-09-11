import Testing
import Darwin
@testable import NVMAI
import NVMAIValidationSupport

@Suite struct YaRNRoPETests {
    @Test func referenceParametersMatchTransformersYaRN() {
        let twoX = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 524_288)
        let fourX = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576)
        #expect(abs(twoX.attentionFactor - 1.0693147) < 1e-6)
        #expect(abs(fourX.attentionFactor - 1.1386294) < 1e-6)
        #expect(abs(twoX.inverseFrequencies[16] - 0.0002766993) < 1e-10)
        #expect(abs(fourX.inverseFrequencies[16] - 0.0002569351) < 1e-10)
        #expect(abs(twoX.inverseFrequencies[31] - 8.2740855e-8) < 1e-12)
        #expect(abs(fourX.inverseFrequencies[31] - 4.1370427e-8) < 1e-12)
    }

    @Test(arguments: [524_287, 1_048_575])
    func scalarKernelMatchesCPUAtExtendedPositions(_ position: Int) throws {
        let target = position < 1_000_000 ? 524_288 : 1_048_576
        let parameters = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: target)
        let context = try MetalContext()
        let rope = try RoPE(context: context, yarn: parameters)
        let input = (0..<256).map { Float16(Float($0 % 19) / 19 - 0.5) }
        guard let buffer = Fp16Buffer.make(context.device, halves: input),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        try rope.encodeNeoxSubdim(commandBuffer: commandBuffer, data: buffer,
                                  position: UInt32(position), headDim: 256,
                                  numHeads: 1, rotaryDim: 64,
                                  theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        var expected = input
        for pair in 0..<32 {
            let angle = Float(position) * parameters.inverseFrequencies[pair]
            let cosine = cos(angle) * parameters.attentionFactor
            let sine = sin(angle) * parameters.attentionFactor
            let x0 = Float(input[pair])
            let x1 = Float(input[32 + pair])
            expected[pair] = Float16(x0 * cosine - x1 * sine)
            expected[32 + pair] = Float16(x0 * sine + x1 * cosine)
        }
        let actual = Fp16Buffer.read(buffer, count: input.count).map(Float16.init)
        #expect(actual == expected)
    }

    @Test func chunkedPrefillMatchesScalarYaRN() throws {
        let parameters = YaRNRoPEParameters(
            headDim: 256, partialRotaryFactor: 0.25, theta: 10_000_000,
            targetContextTokens: 1_048_576)
        let context = try MetalContext()
        let scalar = try RoPE(context: context, yarn: parameters)
        let prefill = try PrefillRoPE(context: context, yarn: parameters)
        let tokens = 3, heads = 2, headDim = 256
        let input = (0..<(tokens * heads * headDim)).map {
            Float16(Float($0 % 23) / 23 - 0.5)
        }
        guard let scalarBuffer = Fp16Buffer.make(context.device, halves: input),
              let prefillBuffer = Fp16Buffer.make(context.device, halves: input),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            return
        }
        try scalar.encodeNeoxSubdim(commandBuffer: commandBuffer, data: scalarBuffer,
                                    position: 524_288, headDim: UInt32(headDim),
                                    numHeads: UInt32(heads), rotaryDim: 64,
                                    numTokens: UInt32(tokens), theta: 10_000_000)
        try prefill.encodeNeoxSubdim(
            commandBuffer: commandBuffer, data: prefillBuffer,
            startPosition: 524_288, queryCount: UInt32(tokens),
            headDim: UInt32(headDim), numHeads: UInt32(heads), rotaryDim: 64,
            tokenStrideElements: UInt32(heads * headDim), theta: 10_000_000)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)
        #expect(Fp16Buffer.read(scalarBuffer, count: input.count)
            == Fp16Buffer.read(prefillBuffer, count: input.count))
    }
}
