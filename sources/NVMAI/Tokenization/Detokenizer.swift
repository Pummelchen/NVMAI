import Foundation
import Tokenizers

/// Streaming detokenizer for generation loops.
///
/// Two challenges drive the design:
///
/// 1. BPE byte-fallback splits multi-byte codepoints (e.g. emoji) across several
///    tokens. Naively decoding each token in isolation yields broken UTF-8.
/// 2. swift-transformers' decoder silently drops byte-fallback tokens that sit
///    at the **end** of the decoded sequence (the bytes are committed only once
///    a non-byte-fallback token follows). For us this matters at `flush()`.
///
/// Strategy:
///   - During `push(_:)` we decode the newly appended stable IDs (a rolling
///     prefix — never the whole history, R7) and emit the delta vs. the
///     previously emitted text. Any trailing byte-fallback IDs are held back.
///   - During `flush()` we append the stable prefix text AND manually assemble
///     the trailing byte-fallback bytes into a UTF-8 string, replacing invalid
///     sequences losslessly. This recovers text the library would otherwise
///     drop on a sequence-ending codepoint.
///
/// The decoder is prefix-stable for the Qwen ChatML fixture: byte-level BPE
/// maps every token id to a fixed string and the configured decoder
/// (`clean_up_tokenization_spaces = false`) concatenates without rewriting, so
/// `decode(prefix + new) == decode(prefix) + decode(new)`. The rolling
/// accumulation therefore reproduces the full-history decode exactly.
struct GFDetokenizer {
    @usableFromInline let tokenizer: any Tokenizer
    @usableFromInline var stableIDs: [Int] = []
    @usableFromInline var trailingByteIDs: [Int] = []
    @usableFromInline var emitted: String = ""

    init(tokenizer: GFTokenizer) {
        self.tokenizer = tokenizer.tokenizer
    }

    mutating func push(_ id: Int32) -> String {
        let tokenID = Int(id)
        let token = tokenizer.convertIdToToken(tokenID) ?? ""
        if Self.isByteFallback(token) {
            trailingByteIDs.append(tokenID)
            return ""
        }

        if !trailingByteIDs.isEmpty {
            stableIDs.append(contentsOf: trailingByteIDs)
            trailingByteIDs.removeAll(keepingCapacity: true)
        }
        stableIDs.append(tokenID)

        // Decode the full stable prefix (not just the newly appended ids) so
        // multi-byte codepoints split across tokens reassemble with their full
        // context. A byte-level tokenizer can still leave a trailing
        // incomplete UTF-8 sequence; hold those bytes back so they don't
        // surface as a replacement character — the next push completes them.
        let current = tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        return commitDelta(Self.completeUTF8Prefix(current))
    }

    /// Longest prefix of `s` that ends on a complete UTF-8 codepoint. A
    /// trailing split codepoint (fewer continuation bytes than its leading
    /// byte expects) is held back rather than emitted as U+FFFD.
    static func completeUTF8Prefix(_ s: String) -> String {
        let bytes = Array(s.utf8)
        guard let last = bytes.last, last & 0x80 != 0 else { return s }
        var lead = bytes.count - 1
        while lead > 0, bytes[lead] & 0xC0 == 0x80 { lead -= 1 }
        let b = bytes[lead]
        let expected: Int
        if b & 0xE0 == 0xC0 { expected = 2 }
        else if b & 0xF0 == 0xE0 { expected = 3 }
        else if b & 0xF8 == 0xF0 { expected = 4 }
        else { return s }
        guard bytes.count - lead < expected else { return s }
        return String(decoding: bytes[..<lead], as: UTF8.self)
    }

    mutating func flush() -> String {
        let stableText = stableIDs.isEmpty
            ? ""
            : tokenizer.decode(tokens: stableIDs, skipSpecialTokens: true)
        let trailingText = assembleByteFallback(trailingByteIDs)
        let fullText = stableText + trailingText
        return commitDelta(fullText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        // A combining mark can extend the last emitted grapheme, so compare
        // the append-only prefix byte-for-byte rather than by Character.
        let currentUTF8 = current.utf8
        var boundary = currentUTF8.startIndex
        for byte in emitted.utf8 {
            guard boundary != currentUTF8.endIndex,
                  currentUTF8[boundary] == byte else {
                // Decoder altered the prefix — resync rather than emit garbage.
                emitted = current
                return ""
            }
            currentUTF8.formIndex(after: &boundary)
        }
        let delta = String(current[boundary...])
        emitted = current
        return delta
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id),
                  let byte = Self.byteFallbackByte(tok) else {
                // A nil token or non-byte token here means a token ID has no
                // corresponding byte — output may be silently truncated.
                continue
            }
            bytes.append(byte)
        }
        // Invalid/split trailing byte sequences are replaced with U+FFFD
        // instead of silently dropped; this only affects a genuinely
        // truncated final codepoint.
        return String(decoding: bytes, as: UTF8.self)
    }

    /// The byte represented by a byte-fallback token, in either of the two
    /// forms this project uses: the `<0xXX>` spelling and the ByteLevel
    /// byte-to-unicode single-character spelling. Returns nil for tokens that
    /// are not single bytes, and for ASCII (which never splits a codepoint).
    @usableFromInline
    static func byteFallbackByte(_ token: String) -> UInt8? {
        if token.count == 6,
           token.hasPrefix("<0x"),
           token.hasSuffix(">"),
           token.dropFirst(3).dropLast().allSatisfy({ $0.isHexDigit }),
           let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16) {
            return byte
        }
        guard token.unicodeScalars.count == 1,
              let scalar = token.unicodeScalars.first,
              let byte = Self.byteLevelCharToByte[scalar.value],
              byte >= 0x80 else {
            return nil
        }
        return byte
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        byteFallbackByte(token) != nil
    }

    /// Inverse GPT-2 / HuggingFace ByteLevel byte-to-unicode mapping, used to
    /// recover the raw byte from a single-character ByteLevel token.
    @usableFromInline
    static let byteLevelCharToByte: [UInt32: UInt8] = {
        var bytes: [Int] = Array(0x21...0x7E) + Array(0xA1...0xAC) + Array(0xAE...0xFF)
        var chars: [Int] = bytes
        var n = 0
        for b in 0...255 where !bytes.contains(b) {
            bytes.append(b)
            chars.append(0x100 + n)
            n += 1
        }
        var table: [UInt32: UInt8] = [:]
        for (b, c) in zip(bytes, chars) {
            table[UInt32(c)] = UInt8(b)
        }
        return table
    }()
}
