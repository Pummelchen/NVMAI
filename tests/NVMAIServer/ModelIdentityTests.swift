import Testing

@testable import NVMAI
@testable import NVMAIServerCore

@Suite struct ModelIdentityTests {
    @Test func apiIDsAlwaysEndInTheQuantization() {
        // Two installs of the same weights at different widths are different
        // models to anyone choosing between them, so the id says which.
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "qwen3.6-35b-a3b-4bit",
            family: .qwen36,
            weightBits: 4) == "qwen3.6-35b-a3b_4-Bit")
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "ornith-1.5-35b-a3b-8bit",
            family: .qwen36,
            weightBits: 8) == "ornith-1.5-35b-a3b_8-Bit")
    }

    @Test func theManifestSuffixIsNotDuplicated() {
        // The id the manifest ships already spells the width one way; the
        // advertised id spells it another. Appending without stripping would
        // produce `...-4bit_4-Bit`.
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "qwen3.8-flash-next-4bit",
            family: .qwen38flash,
            weightBits: 4) == "qwen3.8-flash-next_4-Bit")
    }

    @Test func widthComesFromTheManifestNotTheName() {
        // A manifest whose id says nothing about quantization still gets a
        // correct suffix, because the width is read from the routed-expert
        // slot rather than parsed out of the string.
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "some-private-checkpoint",
            family: .qwen36,
            weightBits: 8) == "some-private-checkpoint_8-Bit")
    }

    @Test func unknownSnapshotFallsBackToRuntimeFamily() {
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "unknown/snapshot",
            family: .qwen36,
            weightBits: 4) == "qwen3.6-35b-a3b_4-Bit")
        #expect(ServerModelIdentity.apiModelID(
            manifestModelID: "unknown/snapshot",
            family: .qwen36MTP,
            weightBits: 4) == "qwen3.6-35b-a3b-mtp_4-Bit")
    }
}
