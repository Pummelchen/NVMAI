import Darwin
import Testing
@testable import NVMAI

/// One profile per (model, width), resolved family -> table -> environment.
@Suite struct ModelProfileTests {
    static let shipped: [(String, ModelFamily)] = [
        ("qwen3.6-35b-a3b", .qwen36), ("ornith-1.5-35b-a3b", .qwen36),
        ("qwen-agentworld", .qwen36), ("qwen3.8-flash-next", .qwen38flash),
    ]

    @Test func everyShippedInstallHasItsOwnRow() {
        for (id, family) in Self.shipped {
            for bits in [4, 8] {
                let p = ModelProfile.resolve(modelID: id, family: family, weightBits: bits, environment: [:])
                #expect(p.isTabled, "\(id) \(bits)-bit falls back to its family")
                #expect(p.key == ModelProfile.Key(id, bits))
            }
        }
        #expect(ModelProfile.table.count == 8)
    }

    @Test func modelsSharingAFamilyResolveIndependently() {
        // Same family, different keys: editing one row cannot move the other.
        let a = ModelProfile.resolve(modelID: "qwen-agentworld", family: .qwen36, weightBits: 4, environment: [:])
        let q = ModelProfile.resolve(modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(a.key != q.key)
        #expect(ModelProfile.table[a.key] != nil && ModelProfile.table[q.key] != nil)
    }

    @Test func tabledValuesMatchWhatWasMeasured() {
        let q38 = ModelProfile.resolve(modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4, environment: [:])
        #expect(q38.expertCacheBudgetBytes == 12 << 30)
        #expect(q38.prefetchDepth == 2)
        #expect(q38.prefetchIOTier == IOPOL_UTILITY)
        let q38b = ModelProfile.resolve(modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 8, environment: [:])
        #expect(q38b.expertCacheBudgetBytes == Int(9.5 * Double(1 << 30)) && q38b.prefetchIOTier == 0)
        #expect(q38.sampling.temperature == 1.0 && q38.sampling.topP == 0.95)
        #expect(!q38.hcFused && !q38.qsaGPUSelect)
        let q36 = ModelProfile.resolve(modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 8, environment: [:])
        #expect(q36.expertCacheBudgetBytes == 12 << 30)
        let q36four = ModelProfile.resolve(modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: [:])
        #expect(q36four.expertCacheBudgetBytes == 10 << 30)
        #expect(q36.prefetchDepth == 1)
        #expect(q36.prefillChunkTokens == 4_096)
        #expect(q36.sampling == GenerationDefaults.house)
    }

    @Test func unknownModelFallsBackToItsFamily() {
        let p = ModelProfile.resolve(modelID: "qwen3.6-35b-a3b-mtp-4bit", family: .qwen36MTP, weightBits: 4, environment: [:])
        #expect(!p.isTabled)
        let f = RuntimeConfiguration.decodeTuning(family: .qwen36MTP, weightBits: 4)
        #expect(p.expertCacheBudgetBytes == f.expertCacheBudgetBytes)
        #expect(p.prefetchDepth == f.prefetchDepth)
        #expect(p.prefillChunkTokens == nil)
        #expect(p.sampling == GenerationDefaults.forFamily(.qwen36MTP))
    }

    @Test func environmentOverridesTheTable() {
        let env = ["NVMAI_ROUTER_TOPK_SIMD": "0", "NVMAI_HC_FUSED": "1", "NVMAI_PREFETCH_IO_TIER": "throttle",
                   "NVMAI_PREDICTIVE_PREFETCH": "1", "NVMAI_PREFETCH_TOP_M": "3",
                   "NVMAI_QSA_GPU_SELECT": "verify"]
        let p = ModelProfile.resolve(modelID: "qwen3.6-35b-a3b", family: .qwen36, weightBits: 4, environment: env)
        #expect(!p.routerTopKSimd)
        #expect(p.hcFused)
        #expect(p.qsaGPUSelect)
        #expect(p.prefetchDepth == 3)
        #expect(p.prefetchIOTier == IOPOL_THROTTLE)
        let off = ModelProfile.resolve(modelID: "qwen3.8-flash-next", family: .qwen38flash, weightBits: 4,
                                       environment: ["NVMAI_PREDICTIVE_PREFETCH": "0"])
        #expect(off.prefetchDepth == 0)
    }

    @Test func summaryNamesTheKeyAndEveryKnob() {
        let p = ModelProfile.resolve(modelID: "qwen-agentworld", family: .qwen36, weightBits: 8, environment: [:])
        for needle in ["model=qwen-agentworld", "bits=8", "tabled", "budget=", "prefetch=1",
                       "chunk=4096", "topk_simd=true", "hc_fused=false"] {
            #expect(p.summary.contains(needle), Comment(rawValue: needle))
        }
    }
}
