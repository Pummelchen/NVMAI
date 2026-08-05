import Testing
@testable import NVMAI

@Suite struct RuntimeConfigurationTests {
    @Test func publicContextChoicesReachQwenMaximum() {
        #expect(RuntimeConfiguration.supportedContextTokens
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
        #expect(RuntimeConfiguration.supportedContextTokens.last
            == RuntimeConfiguration.maximumContextTokens)
    }

    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 128)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
    }

    @Test(arguments: [32, 64, 128, 256, 512, 1_024, 2_048, 4_096])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }
}
