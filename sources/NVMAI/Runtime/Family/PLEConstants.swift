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

    public func makeHash() -> PLEHash {
        PLEHash(multipliers: layerMultipliers.map { UInt64(bitPattern: $0) },
                offsets: ngramHeadsOffsets.map { UInt64(bitPattern: $0) },
                vocabSizes: ngramHeadsVocabSizes.map { UInt64(bitPattern: $0) },
                ngramSize: ngramSize,
                headsPerNgram: headsPerNgram,
                eosTokenID: eosTokenID)
    }
}
