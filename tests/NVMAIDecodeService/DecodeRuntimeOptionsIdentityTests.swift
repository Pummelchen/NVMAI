import Testing
import Foundation
@testable import NVMAIDecodeProtocol

/// `DecodeRuntimeOptions.loadIdentity` is what the decode service compares a
/// generation request against the loaded session with. The line it draws is the
/// whole point: too wide and a request-time control (Concise) is refused with
/// "runtime options do not match the loaded session" while the app offers no
/// Reload; too narrow and a change that *does* size load-time state runs against
/// buffers built for the old value.
@Suite struct DecodeRuntimeOptionsIdentityTests {
    @Test func conciseModeIsNotPartOfTheLoadIdentity() {
        let base = DecodeRuntimeOptions()
        #expect(base.loadIdentity == DecodeRuntimeOptions(conciseMode: true).loadIdentity,
                "concise mode is applied per request and must not require a reload")
    }

    /// Every field that the load sizes or selects state from has to be in the
    /// identity. Each of these is one the app's `AppLoadedRuntimeKey` also
    /// tracks, which is what keeps the app's Reload affordance and the helper's
    /// refusal in step.
    @Test func everyLoadTimeChoiceChangesTheIdentity() {
        let base = DecodeRuntimeOptions()
        let variants: [DecodeRuntimeOptions] = [
            DecodeRuntimeOptions(expertCacheSlots: 24),
            DecodeRuntimeOptions(expertCachePolicy: "lru"),
            DecodeRuntimeOptions(prefillEnabled: false),
            DecodeRuntimeOptions(prefillChunkTokens: 64),
            DecodeRuntimeOptions(rdadvisePolicy: "bounded"),
            DecodeRuntimeOptions(modelVerification: "trusted-install"),
            DecodeRuntimeOptions(thinkingMode: "on"),
            DecodeRuntimeOptions(kvCacheBits: 4),
            DecodeRuntimeOptions(ropeScalingMode: "yarn"),
        ]
        for variant in variants {
            #expect(base.loadIdentity != variant.loadIdentity,
                    "\(variant) should need a reload")
        }
    }
}
