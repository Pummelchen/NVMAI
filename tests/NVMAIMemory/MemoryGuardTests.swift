import Foundation
import Testing
@testable import ContinuityCore
@testable import NVMAIMemory

/// The precedence rule: a fact the model derived never silently supersedes
/// one the person asserted.
///
/// Every case here mirrors one the offline replay measures
/// (`benchmark/memory_sim.py`, policy `guard`), so the Swift rule and the
/// Python oracle cannot drift apart without a test failing. The replay put
/// this at 31 repairs against 9 breaks on twelve recorded runs; these are
/// the shapes those numbers are made of.
@Suite struct MemoryGuardTests {

    private func store() async throws -> (ContinuityStore, MemoryScope) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("guard-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let engine = ContinuityEngine(
            journal: try FileJournal(url: directory.appendingPathComponent("m.ndjson")))
        try await engine.start()
        let store = ContinuityStore(engine: engine, limits: MemoryLimits())
        let scope = try MemoryScope(namespace: "t", user: "u", workspace: "w")
        return (store, scope)
    }

    private func record(_ key: String, _ value: String, user: Bool) throws -> MemoryRecord {
        var record = MemoryRecord(key: try MemoryKey(validating: key), value: value)
        record.isUserAsserted = user
        return record
    }

    /// The measured failure: an extraction invents a value and overwrites
    /// what the person established. With the guard on, the person's value
    /// stays and the disagreement is recorded.
    @Test func modelDoesNotOverwriteTheUser() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("characters/marcus/eyes", "grey", user: true),
                                in: scope, guarding: true, flaggingReversions: true)
        let outcome = try await store.set(
            try record("characters/marcus/eyes", "hazel", user: false),
            in: scope, guarding: true, flaggingReversions: true)

        #expect(outcome == .heldByGuard(existing: "grey"))
        let held = try await store.get(try MemoryKey(validating: "characters/marcus/eyes"),
                                       in: scope)
        #expect(held?.value == "grey")
        #expect(held?.isDisputed == true, "the conflict must be visible, not silent")
    }

    /// A state the person changes must still reach the store: this is how
    /// "the inn burned in chapter 34" supersedes "the inn is standing". The
    /// guard constrains the model, never the person.
    @Test func userSupersedesTheirOwnEarlierFact() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("state/inn", "standing", user: true),
                                in: scope, guarding: true, flaggingReversions: true)
        let outcome = try await store.set(try record("state/inn", "burned", user: true),
                                          in: scope, guarding: true, flaggingReversions: true)

        #expect(outcome == .stored)
        let held = try await store.get(try MemoryKey(validating: "state/inn"), in: scope)
        #expect(held?.value == "burned")
    }

    /// Model over model is untouched: the guard adds a rule about one
    /// direction and changes nothing else.
    @Test func modelStillSupersedesModel() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("decisions/storage", "sqlite", user: false),
                                in: scope, guarding: true, flaggingReversions: true)
        let outcome = try await store.set(
            try record("decisions/storage", "a journal file", user: false),
            in: scope, guarding: true, flaggingReversions: true)

        #expect(outcome == .stored)
        let held = try await store.get(try MemoryKey(validating: "decisions/storage"), in: scope)
        #expect(held?.value == "a journal file")
    }

    /// Agreeing is not conflicting. A model write that restates the person's
    /// value is stored, not held: holding it would mark an address disputed
    /// over nothing and put a false conflict in front of the next session.
    @Test func agreementIsNotAConflict() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("setting/town", "Ashgrove", user: true),
                                in: scope, guarding: true, flaggingReversions: true)
        let outcome = try await store.set(try record("setting/town", "ashgrove.", user: false),
                                          in: scope, guarding: true, flaggingReversions: true)

        #expect(outcome == .stored, "fold-equal values are the same fact")
        let held = try await store.get(try MemoryKey(validating: "setting/town"), in: scope)
        #expect(held?.isDisputed != true)
    }

    /// With the guard off every case behaves as it did before it existed,
    /// which is what makes shipping it off by default meaningful.
    @Test func guardOffIsTodaysBehaviour() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("characters/rosa/eyes", "hazel", user: true),
                                in: scope, guarding: false, flaggingReversions: true)
        let outcome = try await store.set(
            try record("characters/rosa/eyes", "green", user: false),
            in: scope, guarding: false, flaggingReversions: true)

        #expect(outcome == .stored)
        let held = try await store.get(try MemoryKey(validating: "characters/rosa/eyes"),
                                       in: scope)
        #expect(held?.value == "green", "without the guard the last write wins")
    }

    /// A first write has nothing to protect, whoever makes it.
    @Test func firstWriteIsAlwaysStored() async throws {
        let (store, scope) = try await store()
        let outcome = try await store.set(try record("rules/weather", "never rains", user: false),
                                          in: scope, guarding: true, flaggingReversions: true)
        #expect(outcome == .stored)
    }

    /// Authority has to survive the write, because the rule reads it back on
    /// the *next* write rather than at the time. A user fact written, then
    /// challenged much later, must still be protected.
    @Test func authoritySurvivesForLaterWrites() async throws {
        let (store, scope) = try await store()
        _ = try await store.set(try record("rules/ferry", "Sundays only", user: true),
                                in: scope, guarding: true, flaggingReversions: true)
        for filler in 0..<5 {
            _ = try await store.set(
                try record("state/chapter\(filler)", "written", user: false),
                in: scope, guarding: true, flaggingReversions: true)
        }
        let outcome = try await store.set(
            try record("rules/ferry", "runs daily", user: false),
            in: scope, guarding: true, flaggingReversions: true)
        #expect(outcome == .heldByGuard(existing: "Sundays only"))
    }
}
