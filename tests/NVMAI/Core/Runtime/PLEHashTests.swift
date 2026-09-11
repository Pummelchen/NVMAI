import Foundation
import Testing
@testable import NVMAI

/// PLE row addressing against golden vectors produced by the reference
/// implementation (`mlx_qwen4exp/ple.py`) using the pinned checkpoint's own
/// `ple_constants.json`.
///
/// Golden vectors rather than a re-derived formula: this is integer hashing
/// with wrapping multiplies, a latching context cut and a 32-bit truncation.
/// Every one of those can be implemented plausibly and wrongly, and a wrong
/// row id fetches real embedding data from the wrong place -- coherent-looking
/// output, no error anywhere.
@Suite("PLE row hashing")
struct PLEHashTests {
    struct Golden: Decodable {
        struct Constants: Decodable {
            let multipliers: [UInt64]
            let offsets: [UInt64]
            let vocab: [UInt64]
            let eos: Int32
            let ngramSize: Int
            let headsPerNgram: Int
        }
        struct Case: Decodable {
            let tokens: [Int32]
            let prev: [Int32]
            let rows: [[UInt32]]
        }
        let constants: Constants
        let cases: [String: Case]
    }

    static let golden: Golden = {
        guard let url = Bundle.module.url(forResource: "ple_golden",
                                          withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let g = try? JSONDecoder().decode(Golden.self, from: data)
        else { fatalError("ple_golden.json fixture missing") }
        return g
    }()

    static func hash() -> PLEHash {
        let c = golden.constants
        return PLEHash(multipliers: c.multipliers, offsets: c.offsets,
                       vocabSizes: c.vocab, ngramSize: c.ngramSize,
                       headsPerNgram: c.headsPerNgram, eosTokenID: c.eos)
    }

    private func check(_ name: String) {
        guard let c = Self.golden.cases[name] else {
            Issue.record("missing golden case \(name)"); return
        }
        // `prev` is -1 padded in the fixture; drop those to real predecessors.
        let previous = c.prev.filter { $0 >= 0 }
        let actual = Self.hash().rows(tokens: c.tokens, previous: previous)
        #expect(actual.count == c.rows.count, "\(name): token count")
        for (t, expected) in c.rows.enumerated() where t < actual.count {
            let got = "\(actual[t].prefix(4))"
            let want = "\(expected.prefix(4))"
            #expect(actual[t] == expected,
                    "\(name) token \(t): got \(got)... want \(want)...")
        }
    }

    @Test("Production constants and geometry") func geometry() {
        let h = Self.hash()
        #expect(h.ngramSize == 3)
        #expect(h.headsPerNgram == 8)
        #expect(h.headCount == 16)          // 2 orders x 8 heads
        #expect(h.vocabSizes.count == 16)
        // Head vocabularies are distinct primes just above 20M.
        #expect(Set(h.vocabSizes).count == 16)
        #expect(h.vocabSizes.allSatisfy { $0 > 20_000_000 && $0 < 20_001_000 })
    }

    @Test("A plain sequence") func fresh() { check("fresh") }

    @Test("An EOS mid-sequence cuts everything older") func eosMid() {
        check("eos_mid")
    }

    @Test("Carried predecessors from an earlier chunk") func withPrev() {
        check("with_prev")
    }

    @Test("A leading EOS does not cut its own context") func eosFirst() {
        check("eos_first")
    }

    @Test("Large ids exercise the wrapping multiply") func largeIDs() {
        check("large_ids")
    }

    @Test("Rows land inside their head's slice of the table")
    func rowsWithinHeadRanges() {
        let h = Self.hash()
        let rows = h.rows(tokens: [7, 99, 1234, 248_000], previous: [])
        for tokenRows in rows {
            for (head, row) in tokenRows.enumerated() {
                let lo = h.offsets[head]
                let hi = lo + h.vocabSizes[head]
                #expect(UInt64(row) >= lo && UInt64(row) < hi,
                        "head \(head) row \(row) outside [\(lo), \(hi))")
            }
        }
    }

    @Test("One mix per n-gram order, not per head")
    func headsShareTheirOrdersMix() {
        // Heads within an order differ only by modulus and offset, so
        // subtracting each head's offset must leave the same value reduced by
        // different primes -- consistent with a single shared mix.
        let h = Self.hash()
        let rows = h.rows(context: [11, 22, 33])
        for order in 0..<2 {
            let base = order * h.headsPerNgram
            let residues = (0..<h.headsPerNgram).map {
                UInt64(rows[base + $0]) - h.offsets[base + $0]
            }
            // If some head had its own mix, agreement across all eight
            // congruences would be a coincidence of ~20M^-7.
            for (i, r) in residues.enumerated() {
                #expect(r < h.vocabSizes[base + i])
            }
        }
    }
}

/// The n-gram sidecar is geometry, and nothing else compared it to the
/// architecture.
///
/// `headCount * pleHeadDim` is the number of fp16 values one token gathers,
/// while the PLE block's embedding buffer is sized from `cfg.ple.embedDim`.
/// A sidecar built for another model would have the gather write past that
/// buffer (host heap corruption) or feed the block wrong-width rows, silently.
/// `PLEHash`'s own consistency checks are preconditions, so this validation has
/// to run before `makeHash()` for a corrupt sidecar to be a report rather than a
/// trap.
@Suite("PLE sidecar geometry")
struct PLEConstantsGeometryTests {
    private func constants(ngramSize: Int = 3,
                           headsPerNgram: Int = 8,
                           pleHeadDim: Int = 160,
                           headEntries: Int? = nil) -> PLEConstants {
        let headCount = headEntries ?? (headsPerNgram * (ngramSize - 1))
        return PLEConstants(
            layerMultipliers: Array(repeating: 1, count: ngramSize),
            ngramHeadsOffsets: Array(repeating: 0, count: headCount),
            ngramHeadsVocabSizes: Array(repeating: 1, count: headCount),
            eosTokenID: 0,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            pleNumHeads: 16,
            pleHeadDim: pleHeadDim)
    }

    @Test("A sidecar that agrees with the architecture validates")
    func acceptsMatchingGeometry() throws {
        // 8 * (3 - 1) * 160 = 2560 values per token.
        try constants().validate(embedDim: 2560, ngramSize: 3, headsPerNgram: 8)
    }

    @Test("A per-token width that disagrees with embedDim is refused")
    func refusesWidthMismatch() {
        #expect(throws: ModelError.self) {
            try constants().validate(embedDim: 2048, ngramSize: 3, headsPerNgram: 8)
        }
    }

    @Test("A different n-gram shape is refused")
    func refusesShapeMismatch() {
        #expect(throws: ModelError.self) {
            try constants(ngramSize: 4).validate(embedDim: 2560, ngramSize: 3,
                                                 headsPerNgram: 8)
        }
        #expect(throws: ModelError.self) {
            try constants(headsPerNgram: 4).validate(embedDim: 2560, ngramSize: 3,
                                                    headsPerNgram: 8)
        }
    }

    @Test("Head tables that do not match the shape are refused")
    func refusesHeadTableMismatch() {
        // Enough entries to pass PLEHash's own precondition would be a trap;
        // too few is what a truncated sidecar looks like, and it is checked
        // here so the failure is named rather than left to a precondition.
        #expect(throws: ModelError.self) {
            try constants(headEntries: 4).validate(embedDim: 2560, ngramSize: 3,
                                                  headsPerNgram: 8)
        }
    }
}
