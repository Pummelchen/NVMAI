import Testing

@testable import NVMAI
@testable import NVMAIServerCore

@Suite struct ModelIdentityTests {
    @Test func quantizedManifestIDsBecomeStableAPIIDs() {
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "qwen3.6-35b-a3b-4bit",
            family: .qwen36) == "qwen3.6-35b-a3b")
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "ornith-1.5-35b-a3b-8bit",
            family: .qwen36) == "ornith-1.5-35b-a3b")
    }

    @Test func unknownSnapshotFallsBackToRuntimeFamily() {
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "unknown/snapshot",
            family: .qwen36) == "qwen3.6-35b-a3b")
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "unknown/snapshot",
            family: .qwen36MTP) == "qwen3.6-35b-a3b-mtp")
    }
}
