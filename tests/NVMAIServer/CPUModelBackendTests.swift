import Foundation
import Testing
@testable import NVMAIServerCore

/// The CPU serving path's ceilings, which it enforces rather than promises.
@Suite struct CPUModelBackendTests {

    /// Qwen3.5 claims 262,144 positions and the GPU engine honours it. On
    /// the CPU attention is a loop over the cache and the cache is held per
    /// token: at that length the keys and values alone are over three
    /// gigabytes and every token would walk all of them.
    ///
    /// A ceiling that is quietly enforced beats a promise that is quietly
    /// broken, so the backend clamps rather than repeating the checkpoint.
    @Test func theContextCeilingIsEnforcedNotPromised() {
        #expect(CPUModelBackend.contextCeiling < 262_144)
        #expect(CPUModelBackend.contextCeiling >= 8_192,
                "and still enough for the work this engine is for")
    }
}
