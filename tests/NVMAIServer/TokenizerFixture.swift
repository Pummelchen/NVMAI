import Foundation
import Testing

/// Shared ChatML tokenizer fixture for the NVMAIServer test target. The same
/// synthetic byte-level BPE fixture the NVMAI/Core tokenizer tests use,
/// bundled here so server tests exercise the real ChatML pipeline offline.
enum TokenizerFixture {
    static func folder() throws -> URL {
        try #require(Bundle.module.url(
            forResource: "ChatMLTokenizer",
            withExtension: nil,
            subdirectory: "Fixtures"))
    }
}
