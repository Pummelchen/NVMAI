import Foundation
import Testing
@testable import NVMAI

/// ChatML (Qwen) tokenizer coverage against the synthetic fixture: the same
/// byte-level BPE vocab and Qwen3.6 special-token IDs the model ships, loaded
/// from the local `ChatMLTokenizer` fixture (offline, no downloads).
@Suite("Tokenizer")
struct TokenizerTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    // MARK: - Special tokens

    @Test("Special token IDs are within vocab and follow ChatML semantics")
    func specialTokensWithinVocab() {
        let ids: [Int32] = [tok.bosID, tok.eosID, tok.padID, tok.endOfTurnID]
        for id in ids {
            #expect(id >= 0)
            #expect(id < Int32(tok.vocabSize))
        }
        // ChatML: <|endoftext|> doubles as BOS/EOS/PAD; <|im_end|> is the
        // turn marker.
        #expect(tok.bosID == tok.eosID)
        #expect(tok.padID == tok.eosID)
        #expect(tok.endOfTurnID != tok.eosID)
    }

    @Test("Stop-token set covers im_end and endoftext")
    func stopTokens() {
        #expect(tok.stopTokenIDs.contains(tok.eosID))
        #expect(tok.stopTokenIDs.contains(tok.endOfTurnID))
        #expect(tok.stopTokenIDs.count == 2)
    }

    // MARK: - Encode / decode

    @Test("Round-trip ASCII", arguments: [
        "Hello, world.",
        "The quick brown fox jumps over the lazy dog.",
        "code:  let x = 42;  // comment",
        "numbers 0 1 2 3 4 5 6 7 8 9",
    ])
    func roundTripASCII(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("Round-trip multi-byte UTF-8", arguments: [
        "你好，世界。",
        "漢字",
        "🦝 raccoon emoji",
        "mixed 漢 and 🦝 and a",
        "Здравствуй",
    ])
    func roundTripMultibyte(_ text: String) {
        let ids = tok.encode(text, addBOS: false)
        #expect(tok.decode(ids) == text)
    }

    @Test("ChatML never prepends a BOS")
    func bosNeverPrepended() {
        let withBOS = tok.encode("x", addBOS: true)
        let without = tok.encode("x", addBOS: false)
        #expect(withBOS == without)
    }

    @Test("Empty string encodes to empty with or without BOS")
    func encodeEmpty() {
        let none = tok.encode("", addBOS: false)
        #expect(none.isEmpty)
        let withBOS = tok.encode("", addBOS: true)
        #expect(withBOS.isEmpty)
    }

    @Test("Decoding strips special tokens when requested")
    func decodeStripsSpecial() {
        let ids = tok.encode("hi", addBOS: true)
        let withoutSpecial = tok.decode(ids, skipSpecialTokens: true)
        #expect(withoutSpecial == "hi")
    }

    // MARK: - Streaming detokenizer

    @Test("Streaming detokenizer reassembles ASCII")
    func streamingASCII() throws {
        let target = "Hello, world."
        try assertStreams(target)
    }

    @Test("Streaming detokenizer reassembles multi-byte UTF-8", arguments: [
        "漢字",
        "你好",
        "🦝🦝🦝",
        "mixed 漢 and 🦝",
        "Здравствуй мир",
        "ends with emoji 🦝",
        "🦝 starts with emoji",
        "🦝 middle 漢 end",
    ])
    func streamingMultibyte(_ target: String) throws {
        try assertStreams(target)
    }

    @Test("Streaming detokenizer never emits replacement chars")
    func streamingNoMojibake() throws {
        let ids = tok.encode("mixed 漢 and 🦝 text", addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        for id in ids {
            let delta = try detok.push(id)
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"),
                    "delta contained replacement char: '\(delta)'")
        }
        let tail = detok.flush()
        #expect(!tail.unicodeScalars.contains("\u{FFFD}"))
    }

    @Test("Streaming detokenizer preserves long mixed output", arguments: [
        "The quick brown fox jumps over the lazy dog. 0123456789\n",
        "\u{E000}\u{E001}\u{E002}\u{E003}\u{E004}\u{E005}\u{E006}\u{E007}",
        "NVMAI 漢字 Здравствуй 🦝 café Ελληνικά \u{E000}\n",
    ])
    func streamingLongOutput(_ seed: String) throws {
        var text = seed
        var ids = tok.encode(text, addBOS: false)
        while ids.count < 256 {
            text += seed
            ids = tok.encode(text, addBOS: false)
        }

        let prefix = Array(ids.prefix(256))
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in prefix {
            assembled += try detok.push(id)
        }
        assembled += detok.flush()
        // The prefix may truncate a multi-byte codepoint, in which case the
        // reference decoder produces U+FFFD too; matching it is the contract.
        #expect(assembled == tok.decode(prefix))
    }

    @Test("Flush with no tokens yields empty string")
    func streamingEmpty() {
        var detok = GFDetokenizer(tokenizer: tok)
        #expect(detok.flush() == "")
    }

    @Test("Multi-byte characters stream reassemble through the detokenizer")
    func streamingMultibyteCharacters() throws {
        var detok = GFDetokenizer(tokenizer: tok)
        let ids = tok.encode("漢字", addBOS: false)
        var assembled = ""
        for id in ids {
            assembled += try detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == "漢字")
    }

    @Test("Streaming decoder matches batch decode for arbitrary valid token IDs")
    func streamingMatchesBatchForArbitraryIDs() throws {
        var state: UInt64 = 0x4E_56_4D_41_49
        func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }

        // The compact fixture keeps its base ByteLevel vocabulary dense and
        // the production added-token IDs sparse. Exercise both sets without ever
        // sampling one of the intentionally absent padded rows.
        let denseCount = (0..<tok.vocabSize).first {
            tok.tokenizer.convertIdToToken($0) == nil
        }!
        let validIDs = Array(0..<denseCount).map(Int32.init)
            + tok.byteLevelDecoderConfiguration.addedTokens.keys
        for _ in 0..<64 {
            let ids = (0..<128).map { _ in
                validIDs[Int(next() % UInt64(validIDs.count))]
            }
            var detok = GFDetokenizer(tokenizer: tok)
            var streamed = ""
            for id in ids {
                streamed += try detok.push(id)
                #expect(detok.pendingBytes.count <= 3)
            }
            streamed += detok.flush()
            #expect(streamed == tok.decode(ids))
        }
    }

    @Test("Skipped special token does not split a UTF-8 byte sequence")
    func specialTokenInsideUTF8Sequence() throws {
        let ids = tok.encode("🦝", addBOS: false)
        var probe = GFDetokenizer(tokenizer: tok)
        var split: Int?
        for (index, id) in ids.enumerated() where index + 1 < ids.count {
            _ = try probe.push(id)
            if !probe.pendingBytes.isEmpty {
                split = index + 1
                break
            }
        }
        guard let split else {
            Issue.record("fixture did not split the emoji across ByteLevel tokens")
            return
        }

        var sequence = Array(ids[..<split])
        sequence.append(tok.bosID)
        sequence.append(contentsOf: ids[split...])
        var detok = GFDetokenizer(tokenizer: tok)
        var streamed = ""
        for id in sequence { streamed += try detok.push(id) }
        streamed += detok.flush()
        #expect(streamed == "🦝")
        #expect(streamed == tok.decode(sequence))
    }

    @Test("Added token is a byte barrier and matches batch replacement behavior")
    func addedTokenByteBarrier() throws {
        let ids = tok.encode("🦝", addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        var sequence: [Int32] = []
        for id in ids {
            sequence.append(id)
            _ = try detok.push(id)
            if !detok.pendingBytes.isEmpty { break }
        }
        #expect(!detok.pendingBytes.isEmpty)
        sequence.append(tok.toolCallStartID)

        detok = GFDetokenizer(tokenizer: tok)
        var streamed = ""
        for id in sequence { streamed += try detok.push(id) }
        streamed += detok.flush()
        #expect(streamed == tok.decode(sequence))
        #expect(streamed.hasSuffix("<tool_call>"))
    }

    @Test("Padded and negative token IDs fail instead of truncating output")
    func invalidTokenIDsFailClosed() {
        var detok = GFDetokenizer(tokenizer: tok)
        #expect(throws: GFDetokenizerError.invalidTokenID(248_077)) {
            _ = try detok.push(248_077)
        }
        #expect(throws: GFDetokenizerError.invalidTokenID(-1)) {
            _ = try detok.push(-1)
        }
    }

    @Test("Structured markers must remain literal ByteLevel barriers")
    func structuredMarkerContract() {
        var added = tok.byteLevelDecoderConfiguration.addedTokens
        added[tok.toolCallStartID] = .init(content: "<tool_call>", special: true)
        let incompatible = GFByteLevelDecoderConfiguration(addedTokens: added)
        #expect(throws: GFTokenizerError.self) {
            _ = try GFTokenizer(
                tokenizer: tok.tokenizer,
                byteLevelDecoderConfiguration: incompatible)
        }
    }

    // MARK: - Helpers

    private func assertStreams(_ target: String) throws {
        let ids = tok.encode(target, addBOS: false)
        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            assembled += try detok.push(id)
        }
        assembled += detok.flush()
        #expect(assembled == target, "stream reassembly mismatch: got '\(assembled)' want '\(target)'")
    }
}
