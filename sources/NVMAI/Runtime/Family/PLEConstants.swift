import Foundation

/// The hash constants that accompany Qwen3.8-Flash-Next's n-gram table.
///
/// These ship as a sidecar JSON rather than tensors because they are integer
/// parameters of the row addressing, not weights: `PLEHash` needs them before
/// any GPU work, and re-deriving them from a seed would be a second
/// implementation of something the checkpoint already states.
public struct PLEConstants: Decodable, Sendable {
    public let layerMultipliers: [Int64]
    public let ngramHeadsOffsets: [Int64]
    public let ngramHeadsVocabSizes: [Int64]
    public let eosTokenID: Int32
    public let ngramSize: Int
    public let headsPerNgram: Int
    public let pleNumHeads: Int
    public let pleHeadDim: Int

    enum CodingKeys: String, CodingKey {
        case layerMultipliers = "layer_multipliers"
        case ngramHeadsOffsets = "ngram_heads_offsets"
        case ngramHeadsVocabSizes = "ngram_heads_vocab_sizes"
        case eosTokenID = "eos_token_id"
        case ngramSize = "ngram_size"
        case headsPerNgram = "heads_per_ngram"
        case pleNumHeads = "ple_n_heads"
        case pleHeadDim = "ple_head_dim"
    }

    public static func load(directoryURL: URL) throws -> PLEConstants {
        let url = directoryURL.appendingPathComponent(
            Qwen38FlashTensors.pleConstantsFile)
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PLEConstants.self, from: data)
    }

    /// Row count of the table these constants address: the last head's offset
    /// plus its own vocabulary.
    public var tableRowCount: UInt64 {
        guard let offset = ngramHeadsOffsets.last,
              let vocab = ngramHeadsVocabSizes.last else { return 0 }
        return UInt64(offset) + UInt64(vocab)
    }

    /// Check the sidecar's geometry against the architecture that will use it.
    ///
    /// `headCount * pleHeadDim` is the number of fp16 values one token gathers
    /// (`PLEHash.headCount` rows of the table's row width), and the PLE block's
    /// embedding buffer is sized from `cfg.ple.embedDim`. Nothing else compares
    /// the two: a sidecar from another model would have the gather write past
    /// that buffer -- host heap corruption, not a GPU fault -- or feed the block
    /// rows of the wrong width, silently. `PLEHash`'s own consistency checks are
    /// preconditions, so this must run *before* `makeHash()` to turn a corrupt
    /// sidecar into a report rather than a trap.
    public func validate(embedDim: Int,
                         ngramSize: Int,
                         headsPerNgram: Int) throws {
        let headCount = headsPerNgram * (self.ngramSize - 1)
        guard self.ngramSize == ngramSize else {
            throw ModelError.archMismatch(field: "ple.ngramSize",
                                          expected: "\(ngramSize)",
                                          actual: "\(self.ngramSize)")
        }
        guard self.headsPerNgram == headsPerNgram else {
            throw ModelError.archMismatch(field: "ple.headsPerNgram",
                                          expected: "\(headsPerNgram)",
                                          actual: "\(self.headsPerNgram)")
        }
        guard ngramHeadsOffsets.count == headCount,
              ngramHeadsVocabSizes.count == headCount else {
            throw ModelError.archMismatch(
                field: "ple_constants.json head tables",
                expected: "\(headCount) entries each",
                actual: "\(ngramHeadsOffsets.count) offsets, "
                    + "\(ngramHeadsVocabSizes.count) vocab sizes")
        }
        guard headCount * pleHeadDim == embedDim else {
            throw ModelError.archMismatch(
                field: "ple geometry (headCount * pleHeadDim)",
                expected: "\(embedDim) values per token",
                actual: "\(headCount) * \(pleHeadDim) = \(headCount * pleHeadDim)")
        }
    }

    public func makeHash() -> PLEHash {
        PLEHash(multipliers: layerMultipliers.map { UInt64(bitPattern: $0) },
                offsets: ngramHeadsOffsets.map { UInt64(bitPattern: $0) },
                vocabSizes: ngramHeadsVocabSizes.map { UInt64(bitPattern: $0) },
                ngramSize: ngramSize,
                headsPerNgram: headsPerNgram,
                eosTokenID: eosTokenID)
    }
}
