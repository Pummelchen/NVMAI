import Foundation
import Testing
import NVMAI
@testable import NVMAIAppCore

@Suite struct AppRuntimeOptionsTests {
    @Test func defaultsMatchProduction() throws {
        let options = AppRuntimeOptions()
        #expect(options.expertCacheSlots == AppRuntimeOptions.automaticSlotCount)
        #expect(options.expertCachePolicy == .lfu)
        #expect(options.prefillEnabled)
        #expect(options.prefillChunkTokens == 4096)
        #expect(options.rdadvisePolicy == .default)
        #expect(options.modelVerification == .fullSha256)
        #expect(options.kvCachePrecision == .int8)
        #expect(options.ropeScalingMode == .none)
        #expect(options.thinkingMode == .off)

        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime.expertCacheSlots == RuntimeConfiguration.production.expertCacheSlots)
        #expect(runtime.expertCachePolicy == RuntimeConfiguration.production.expertCachePolicy)
        #expect(runtime.prefillConfig.chunkTokens == 4096)
        #expect(runtime.rdadvisePolicy == RuntimeConfiguration.production.rdadvisePolicy)
        #expect(runtime.headPath == RuntimeConfiguration.production.headPath)
        #expect(options.resultSummary ==
            "Cache auto LFU, prefill 4096, 8-bit KV, native RoPE, thinking off, RDADVISE default, full SHA-256")
    }

    @Test func automaticSlotsResolveFromTheModelDirectory() throws {
        // No readable manifest: the production default, never a crash.
        let missing = URL(fileURLWithPath: "/nonexistent/model.gturbo")
        #expect(AppRuntimeOptions.recommendedSlots(forModelAt: missing)
            == RuntimeConfiguration.production.expertCacheSlots)
        let auto = AppRuntimeOptions()
        #expect(auto.effectiveSlots(forModelAt: missing)
            == RuntimeConfiguration.production.expertCacheSlots)
        #expect(auto.effectiveSlots(forModelAt: nil)
            == RuntimeConfiguration.production.expertCacheSlots)
        // An explicit count is never second-guessed.
        #expect(AppRuntimeOptions(expertCacheSlots: 96).effectiveSlots(forModelAt: missing) == 96)
        #expect(AppRuntimeOptions.slotsLabel(for: AppRuntimeOptions.automaticSlotCount)
            .hasPrefix("Auto"))
        try AppRuntimeOptions().validate()
    }

    @Test func everyPublicChoiceMapsToRuntime() throws {
        for slots in AppRuntimeOptions.allowedSlotCounts {
            let runtime = try AppRuntimeOptions(expertCacheSlots: slots)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.expertCacheSlots == slots)
        }
        for chunk in AppRuntimeOptions.allowedPrefillChunkTokens {
            let runtime = try AppRuntimeOptions(prefillChunkTokens: chunk)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.prefillConfig.chunkTokens == chunk)
        }
        for policy in AppRDAdvicePolicy.allCases {
            let runtime = try AppRuntimeOptions(rdadvisePolicy: policy)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.rdadvisePolicy == policy.runtimeValue)
        }
        for precision in KVCachePrecision.allCases {
            let runtime = try AppRuntimeOptions(kvCachePrecision: precision)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.kvCachePrecision == precision)
        }
    }

    @Test func runtimeAndTrustChoicesAreExplicit() throws {
        let options = AppRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: .adaptive,
            modelVerification: .trustedInstall)
        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.rdadvisePolicy == .adaptive)
        #expect(runtime.headPath == .logits)
        #expect(options.modelVerification.runtimeValue == .sizeCheckTrustedReceipt)
    }

    @Test func validationRejectsValuesOutsideClosedSets() {
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(expertCacheSlots: 12).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(prefillChunkTokens: 96).validate()
        }
    }

    @Test func loadedRuntimeKeyTracksOnlyLoadTimeChoices() {
        let directory = URL(fileURLWithPath: "/tmp/model.gturbo")
        let base = AppRuntimeOptions()
        let baseline = AppLoadedRuntimeKey(
            modelDirectory: directory, maxContextTokens: 4096, options: base)

        var variants: [AppRuntimeOptions] = []
        var value = base
        value.expertCacheSlots = 24; variants.append(value)
        value = base; value.expertCachePolicy = .lru; variants.append(value)
        // The prefill pair is load-time: `prefillChunkTokens` becomes the KV
        // cache's `maxPrefillChunkTokens` and sizes the prefill scratch, and
        // `prefillEnabled` picks the prefill policy. Treating them as
        // request-time is what let the Prefill controls change a setting the
        // loaded session could not honour -- no Reload appeared, and the
        // generation failed with "runtime options do not match the loaded
        // session".
        value = base; value.prefillEnabled = false; variants.append(value)
        value = base; value.prefillChunkTokens = 64; variants.append(value)
        value = base; value.rdadvisePolicy = .bounded; variants.append(value)
        value = base; value.modelVerification = .trustedInstall; variants.append(value)
        value = base; value.kvCachePrecision = .int4; variants.append(value)
        value = base; value.ropeScalingMode = .yarn; variants.append(value)
        value = base; value.thinkingMode = .on; variants.append(value)

        for variant in variants {
            #expect(AppLoadedRuntimeKey(
                modelDirectory: directory,
                maxContextTokens: 4096,
                options: variant) != baseline)
        }
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: base,
            forceLogitsHead: true) != baseline)

        // Request-time: applied while each request's prompt is rendered, so it
        // changes what the model reads without touching what the load
        // allocated. A Reload here would throw away a loaded model for a
        // prompt-formatting flag.
        value = base; value.conciseMode = true
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: value) == baseline)
    }

    /// `SessionLoadKey` is what the client refuses a request against, and it has
    /// to draw the same line as the UI key: a concise-only change may run, a
    /// prefill change may not.
    @Test func sessionLoadKeyComparesOnlyTheLoadIdentity() {
        let directory = URL(fileURLWithPath: "/tmp/model.gturbo")
        let base = AppRuntimeOptions()
        let baseline = SessionLoadKey(directory: directory, maxContext: 4096, options: base)

        var concise = base; concise.conciseMode = true
        #expect(SessionLoadKey(directory: directory, maxContext: 4096,
                               options: concise) == baseline)

        var prefill = base; prefill.prefillChunkTokens = 64
        #expect(SessionLoadKey(directory: directory, maxContext: 4096,
                               options: prefill) != baseline)

        #expect(SessionLoadKey(directory: directory, maxContext: 8192,
                               options: base) != baseline)
        #expect(SessionLoadKey(directory: URL(fileURLWithPath: "/tmp/other.gturbo"),
                               maxContext: 4096, options: base) != baseline)
    }
}
