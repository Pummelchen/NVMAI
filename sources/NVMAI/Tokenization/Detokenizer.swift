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
    /// Number of `stableIDs` already folded into `decodedText`.
    @usableFromInline var decodedPrefixCount = 0
    /// Rolling decode of `stableIDs[..<decodedPrefixCount]` (plus committed
    /// byte-fallback ids). Built incrementally so `push` never re-decodes the
    /// full history (R7).
    @usableFromInline var decodedText: String = ""

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

        // Decode only the newly appended ids (which always end with a
        // non-byte-fallback token, so the decoder commits every byte) and
        // append to the rolling text.
        let segment = Array(stableIDs[decodedPrefixCount...])
        decodedText += tokenizer.decode(tokens: segment, skipSpecialTokens: true)
        decodedPrefixCount = stableIDs.count
        return commitDelta(decodedText)
    }

    mutating func flush() -> String {
        let stableText = decodedText
        let trailingText = assembleByteFallback(trailingByteIDs)
        let fullText = stableText + trailingText
        return commitDelta(fullText)
    }

    @usableFromInline
    mutating func commitDelta(_ current: String) -> String {
        // `emitted` is the leading `emittedUTF8Count` bytes of `current`
        // whenever the decoder is prefix-stable (it is for the Qwen ChatML
        // fixture; see the type doc). Track the boundary instead of rescanning
        // the whole emitted string each token (R7).
        let currentUTF8 = current.utf8
        let emittedCount = emitted.utf8.count
        guard currentUTF8.count >= emittedCount else {
            // Decoder rewrote the prefix (or produced a shorter string) —
            // resync rather than emit garbage.
            emitted = current
            return ""
        }
        let boundary = currentUTF8.index(currentUTF8.startIndex,
                                         offsetBy: emittedCount)
        let delta = String(current[boundary...])
        emitted = current
        return delta
    }

    @usableFromInline
    func assembleByteFallback(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count)
        for id in ids {
            guard let tok = tokenizer.convertIdToToken(id) else {
                // NOTE: nil token in byte-fallback path means a token ID has no
                // corresponding vocabulary entry — output may be silently truncated.
                continue
            }
            guard Self.isByteFallback(tok),
                  let byte = UInt8(tok.dropFirst(3).dropLast(), radix: 16)
            else { continue }
            bytes.append(byte)
        }
        // Never fails: invalid/split trailing byte sequences are replaced with
        // U+FFFD instead of silently dropped (R13). The stable prefix already
        // committed everything decodable, so this only affects a genuinely
        // truncated final codepoint.
        return String(decoding: bytes, as: UTF8.self)
    }

    @usableFromInline
    static func isByteFallback(_ token: String) -> Bool {
        token.count == 6
            && token.hasPrefix("<0x")
            && token.hasSuffix(">")
            && token.dropFirst(3).dropLast().allSatisfy { $0.isHexDigit }
    }
}
