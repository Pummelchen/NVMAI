import Foundation
import Testing
@testable import NVMAIMemory

/// The guard's whole promise is that a disagreement is *shown* rather than
/// silently resolved. The store recording a dispute is only half of that;
/// the other half is the next session's prompt, and that half was broken.
///
/// Disputing an address updates it, and the "changed in the most recent
/// session" list is sorted by exactly that -- so a freshly disputed key is
/// always on that list, and the marker was rendered only on the *other*
/// list. The result was a guard that held a write, recorded the conflict,
/// and then handed the next session the surviving value with nothing to say
/// anything had been disputed at all.
@Suite struct MemoryPromptDisputeTests {

    private func record(_ key: String, _ value: String, disputed: Bool) throws -> MemoryRecord {
        var record = MemoryRecord(key: try MemoryKey(validating: key), value: value)
        record.isDisputed = disputed
        return record
    }

    private func prompt(_ bootstrap: MemoryBootstrap) throws -> String {
        MemoryPrompt.instructions(
            scope: try MemoryScope(namespace: "t", user: "u", workspace: "w"),
            session: MemorySession(id: "s-1"),
            bootstrap: bootstrap)
    }

    /// The case the guard actually produces.
    @Test func aDisputedRecentRecordIsMarked() throws {
        let held = try record("characters/marcus/eyes", "grey", disputed: true)
        let text = try prompt(MemoryBootstrap(records: [held], omittedCount: 0,
                                              totalBytes: 40, recent: [held]))
        #expect(text.contains("Changed in the most recent session:"))
        #expect(text.contains("[disputed"),
                "a held write must reach the next session as a disagreement")
    }

    /// And the established-facts list still marks its own.
    @Test func aDisputedEstablishedRecordIsMarked() throws {
        let old = try record("rules/ferry", "Sundays only", disputed: true)
        let text = try prompt(MemoryBootstrap(records: [old], omittedCount: 0,
                                              totalBytes: 40))
        #expect(text.contains("[disputed"))
    }

    /// Nothing else grows a marker.
    @Test func anUndisputedRecordIsNotMarked() throws {
        let plain = try record("setting/town", "Ashgrove", disputed: false)
        let text = try prompt(MemoryBootstrap(records: [plain], omittedCount: 0,
                                              totalBytes: 20, recent: [plain]))
        #expect(!text.contains("[disputed"))
    }

    /// Measured on the correction scenario: two sessions of eight produced
    /// no answer at all because the model, told it "has memory" and given a
    /// workspace name but no tool to reach it with, went looking for a
    /// filesystem — one announced it would retrieve what it knew and emitted
    /// a shell command, the other tried to write its decision to a file.
    ///
    /// Saying what is absent is the fix, so the fragment has to say it.
    @Test func withoutToolsTheFragmentSaysThereIsNothingToCall() throws {
        let text = try prompt(MemoryBootstrap(
            records: [try record("decisions/storage", "postgres", disputed: false)],
            omittedCount: 0, totalBytes: 20))
        #expect(text.contains("no memory tools in this request"))
        #expect(text.contains("nothing to fetch"))
        #expect(text.contains("do not read or write files"))
    }

    /// And it must not say that when there *are* tools, which would tell the
    /// model to ignore the ones it was given.
    @Test func withToolsTheFragmentDoesNotDenyThem() throws {
        let text = MemoryPrompt.instructions(
            scope: try MemoryScope(namespace: "t", user: "u", workspace: "w"),
            session: MemorySession(id: "s-1"),
            bootstrap: MemoryBootstrap(records: [], omittedCount: 0, totalBytes: 0),
            tools: ["memory_get", "memory_set"])
        #expect(!text.contains("no memory tools in this request"))
    }
}
