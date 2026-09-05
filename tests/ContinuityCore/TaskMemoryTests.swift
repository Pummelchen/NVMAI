import Foundation
import Testing
@testable import ContinuityCore

@Suite struct TaskMemoryTests {
    @Test func writingTheSameAddressVersionsRatherThanOverwrites() async throws {
        let memory = TaskMemory()
        let task = UUID()
        let first = try await memory.write(taskID: task, namespace: "plot",
                                           key: "brother", value: "missing")
        #expect(first.isNew)
        #expect(first.item.version == 1)

        let second = try await memory.write(taskID: task, namespace: "plot",
                                            key: "brother", value: "found in act three")
        #expect(second.isNew == false)
        #expect(second.previousVersion == 1)
        #expect(second.item.version == 2)

        let history = await memory.history(taskID: task, namespace: "plot", key: "brother")
        #expect(history.map(\.version) == [1, 2])
        #expect(history.first?.value == "missing")
        #expect(history.first?.status == .superseded)
        #expect(history.last?.value == "found in act three")
        #expect(history.last?.status == .active)
    }

    @Test func provenanceSurvivesOnEveryVersion() async throws {
        let memory = TaskMemory()
        let task = UUID()
        let sessionOne = UUID()
        let sessionTwo = UUID()
        try await memory.write(taskID: task, namespace: "decision", key: "storage",
                               value: "valkey",
                               provenance: Provenance(sessionID: sessionOne, author: .model))
        try await memory.write(taskID: task, namespace: "decision", key: "storage",
                               value: "native swift",
                               provenance: Provenance(sessionID: sessionTwo, author: .user))

        let history = await memory.history(taskID: task, namespace: "decision", key: "storage")
        #expect(history[0].provenance?.sessionID == sessionOne)
        #expect(history[0].provenance?.author == .model)
        #expect(history[1].provenance?.sessionID == sessionTwo)
        #expect(history[1].provenance?.author == .user)
    }

    @Test func tasksCannotSeeEachOther() async throws {
        let memory = TaskMemory()
        let novel = UUID()
        let compiler = UUID()
        try await memory.write(taskID: novel, namespace: "plot", key: "twist", value: "a")
        try await memory.write(taskID: compiler, namespace: "plot", key: "twist", value: "b")

        #expect(await memory.value(taskID: novel, namespace: "plot", key: "twist") == "a")
        #expect(await memory.value(taskID: compiler, namespace: "plot", key: "twist") == "b")
        #expect(await memory.query(taskID: novel).count == 1)
        await memory.forget(taskID: novel)
        #expect(await memory.count(taskID: novel) == 0)
        #expect(await memory.count(taskID: compiler) == 1)
    }

    @Test func archivingKeepsTheValueAndRemovesItFromContext() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "plot", key: "subplot",
                               value: "the cousin")
        try await memory.archive(taskID: task, namespace: "plot", key: "subplot")

        let active = await memory.query(taskID: task)
        #expect(active.isEmpty)
        let item = await memory.item(taskID: task, namespace: "plot", key: "subplot")
        #expect(item?.status == .archived)
        #expect(item?.value == "the cousin")
        let archived = await memory.query(taskID: task, MemoryQuery(statuses: [.archived]))
        #expect(archived.count == 1)
    }

    @Test func disputedItemsStayVisible() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "fact", key: "age", value: "41")
        try await memory.setStatus(taskID: task, namespace: "fact", key: "age",
                                   status: .disputed)
        let visible = await memory.query(taskID: task)
        #expect(visible.count == 1)
        #expect(visible.first?.status == .disputed)
    }

    /// The version chain is the history of what an address said. A status
    /// change says nothing new, so it must not appear as a second version
    /// with the same number.
    @Test func aStatusChangeDoesNotAddToTheVersionChain() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "fact", key: "photo",
                               value: "he has not seen it")
        try await memory.write(taskID: task, namespace: "fact", key: "photo",
                               value: "he recognises it")
        try await memory.setStatus(taskID: task, namespace: "fact", key: "photo",
                                   status: .disputed)

        let history = await memory.history(taskID: task, namespace: "fact", key: "photo")
        #expect(history.map(\.version) == [1, 2])
        #expect(history.last?.status == .disputed)
        #expect(history.first?.value == "he has not seen it")

        // A later value change still versions normally.
        try await memory.write(taskID: task, namespace: "fact", key: "photo",
                               value: "chapter 30 was the error")
        let extended = await memory.history(taskID: task, namespace: "fact", key: "photo")
        #expect(extended.map(\.version) == [1, 2, 3])
    }

    @Test func optimisticWritesDetectAConcurrentChange() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "one")
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "two")

        await #expect(throws: ContinuityError.versionConflict(namespace: "n", key: "k",
                                                              expected: 1, actual: 2)) {
            try await memory.write(taskID: task, namespace: "n", key: "k",
                                   value: "three", expectedVersion: 1)
        }
        // The correct version goes through, and the value is unchanged by the
        // rejected attempt.
        #expect(await memory.value(taskID: task, namespace: "n", key: "k") == "two")
        try await memory.write(taskID: task, namespace: "n", key: "k",
                               value: "three", expectedVersion: 2)
        #expect(await memory.value(taskID: task, namespace: "n", key: "k") == "three")
    }

    @Test func expectingVersionZeroAssertsTheAddressIsNew() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k",
                               value: "first", expectedVersion: 0)
        await #expect(throws: ContinuityError.self) {
            try await memory.write(taskID: task, namespace: "n", key: "k",
                                   value: "again", expectedVersion: 0)
        }
    }

    @Test func queriesFilterAndOrder() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "plot.act1", key: "open",
                               value: "a storm", importance: 0.2, tags: ["scene"])
        try await memory.write(taskID: task, namespace: "plot.act2", key: "turn",
                               value: "the brother returns", importance: 0.9,
                               tags: ["scene", "pivot"])
        try await memory.write(taskID: task, namespace: "style", key: "voice",
                               value: "close third person", importance: 0.5)

        let byPrefix = await memory.query(taskID: task,
                                          MemoryQuery(namespacePrefix: "plot"))
        #expect(byPrefix.count == 2)
        // Prefixes respect segment boundaries, so "plot" never drags in a
        // namespace that merely starts with the same letters.
        let plotting = await memory.query(taskID: task,
                                          MemoryQuery(namespacePrefix: "plo"))
        #expect(plotting.isEmpty)

        let byTag = await memory.query(taskID: task, MemoryQuery(tags: ["pivot"]))
        #expect(byTag.map(\.key) == ["turn"])

        let byText = await memory.query(taskID: task, .search("BROTHER"))
        #expect(byText.map(\.key) == ["turn"])

        let ranked = await memory.query(taskID: task, MemoryQuery(order: .relevance))
        #expect(ranked.map(\.key) == ["turn", "voice", "open"])

        let byAddress = await memory.query(taskID: task, MemoryQuery(order: .address))
        #expect(byAddress.map(\.address) == ["plot.act1.open", "plot.act2.turn", "style.voice"])

        let limited = await memory.query(taskID: task, MemoryQuery(limit: 1))
        #expect(limited.count == 1)
    }

    @Test func dependenciesResolveTransitively() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "constraint", key: "no_network",
                               value: "the core opens no sockets")
        try await memory.write(taskID: task, namespace: "constraint", key: "in_process",
                               value: "same binary", dependencies: ["constraint.no_network"])
        try await memory.write(taskID: task, namespace: "decision", key: "storage",
                               value: "native swift actors",
                               dependencies: ["constraint.in_process"])

        let seed = await memory.item(taskID: task, namespace: "decision", key: "storage")!
        let resolved = await memory.dependencies(taskID: task, of: [seed])
        #expect(Set(resolved.map(\.address)) == ["constraint.in_process",
                                                 "constraint.no_network"])
    }

    @Test func addressesAreValidated() async throws {
        let memory = TaskMemory()
        let task = UUID()
        for bad in ["", "Plot", "plot..act", "plot act", "plot/act"] {
            await #expect(throws: ContinuityError.self,
                          "namespace '\(bad)' should be rejected") {
                try await memory.write(taskID: task, namespace: bad, key: "k", value: "v")
            }
        }
        for bad in ["", "a.b", "Key", "with space"] {
            await #expect(throws: ContinuityError.self, "key '\(bad)' should be rejected") {
                try await memory.write(taskID: task, namespace: "n", key: bad, value: "v")
            }
        }
        try await memory.write(taskID: task, namespace: "plot.act-1", key: "a_key_2",
                               value: "fine")
    }

    @Test func oversizedValuesAreRefused() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxValueBytes: 32))
        await #expect(throws: ContinuityError.self) {
            try await memory.write(taskID: UUID(), namespace: "n", key: "k",
                                   value: String(repeating: "x", count: 33))
        }
    }

    @Test func historyIsBoundedOldestFirst() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxVersionsPerAddress: 3))
        let task = UUID()
        for index in 1...6 {
            try await memory.write(taskID: task, namespace: "n", key: "k", value: "v\(index)")
        }
        let history = await memory.history(taskID: task, namespace: "n", key: "k")
        // Three archived versions plus the live one, and the oldest are gone.
        #expect(history.map(\.version) == [3, 4, 5, 6])
        #expect(history.first?.value == "v3")
    }

    @Test func snapshotRoundTripsIncludingHistory() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "one")
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "two")
        let snapshot = await memory.snapshot()

        let restored = TaskMemory()
        await restored.restore(snapshot)
        #expect(await restored.value(taskID: task, namespace: "n", key: "k") == "two")
        #expect(await restored.history(taskID: task, namespace: "n", key: "k")
                    .map(\.version) == [1, 2])
    }

    @Test func observerReportsSupersessionBeforeTheWrite() async throws {
        let collector = MutationCollector()
        let memory = TaskMemory()
        await memory.setObserver { await collector.add($0) }
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "one")
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "two")

        let labels = await collector.labels
        #expect(labels == ["written", "versioned", "written"])
    }
}

actor MutationCollector {
    private(set) var mutations: [MemoryMutation] = []
    var labels: [String] {
        mutations.map {
            switch $0 {
            case .versioned: return "versioned"
            case .written: return "written"
            case .statusChanged: return "statusChanged"
            }
        }
    }
    func add(_ mutation: MemoryMutation) { mutations.append(mutation) }
}

/// The store's byte budget.
///
/// This replaced a bound derived from the largest permitted value, which at a
/// 64 KiB cap and 200-byte facts refused writes at about a third of a percent
/// of the memory it claimed to allow.
@Suite struct MemoryBudgetTests {
    @Test func bytesAreCountedNotEstimated() async throws {
        let memory = TaskMemory()
        let task = UUID()
        #expect(await memory.byteCount(taskID: task) == 0)

        let value = String(repeating: "x", count: 1000)
        try await memory.write(taskID: task, namespace: "n", key: "k", value: value)
        let held = await memory.byteCount(taskID: task)
        // The value plus a bounded, honest overhead, not a multiple of the
        // value cap.
        #expect(held >= 1000)
        #expect(held < 1000 + 512)
    }

    @Test func aRewriteChargesForTheRetainedVersion() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k",
                               value: String(repeating: "a", count: 500))
        let afterFirst = await memory.byteCount(taskID: task)
        try await memory.write(taskID: task, namespace: "n", key: "k",
                               value: String(repeating: "b", count: 500))
        let afterSecond = await memory.byteCount(taskID: task)
        // History is not free, and a budget that pretended it was would be
        // overrun by exactly the amount of history kept.
        #expect(afterSecond > afterFirst)
        #expect(afterSecond >= 1000)
    }

    @Test func theBudgetRefusesRatherThanEvicts() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxValueBytes: 4096,
                                                     maxBytesPerTask: 8192))
        let task = UUID()
        let chunk = String(repeating: "x", count: 1000)
        var written = 0
        var refused = false
        for index in 0..<20 {
            do {
                try await memory.write(taskID: task, namespace: "n", key: "k\(index)",
                                       value: chunk)
                written += 1
            } catch let error as ContinuityError {
                guard case .storeFull = error else {
                    Issue.record("expected storeFull, got \(error)")
                    return
                }
                refused = true
                break
            }
        }
        #expect(refused)
        #expect(written > 0)
        #expect(await memory.byteCount(taskID: task) <= 8192)
        // Nothing already written was dropped to make room. Silently losing a
        // fact the model deliberately wrote is worse than refusing a new one.
        #expect(await memory.count(taskID: task) == written)
        #expect(await memory.value(taskID: task, namespace: "n", key: "k0") == chunk)
    }

    @Test func archivingAndForgettingReleaseTheirBytes() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k",
                               value: String(repeating: "x", count: 2000))
        #expect(await memory.byteCount(taskID: task) > 2000)
        // Archiving keeps the value on purpose, so it keeps costing.
        try await memory.archive(taskID: task, namespace: "n", key: "k")
        #expect(await memory.byteCount(taskID: task) > 2000)
        // Forgetting the task is the delete, and it returns the bytes.
        await memory.forget(taskID: task)
        #expect(await memory.byteCount(taskID: task) == 0)
    }

    @Test func aTrimmedVersionChainReturnsItsBytes() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxVersionsPerAddress: 2))
        let task = UUID()
        let chunk = String(repeating: "x", count: 1000)
        for _ in 0..<10 {
            try await memory.write(taskID: task, namespace: "n", key: "k", value: chunk)
        }
        // Two retained versions plus the live item, not ten.
        let held = await memory.byteCount(taskID: task)
        #expect(held < 4000)
        #expect(await memory.history(taskID: task, namespace: "n", key: "k").count == 3)
    }

    @Test func restoringRecomputesTheCounters() async throws {
        let memory = TaskMemory()
        let task = UUID()
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "one")
        try await memory.write(taskID: task, namespace: "n", key: "k", value: "two")
        let snapshot = await memory.snapshot()
        let expected = await memory.byteCount(taskID: task)

        let restored = TaskMemory()
        await restored.restore(snapshot)
        #expect(await restored.byteCount(taskID: task) == expected)
        // And the budget keeps working after a restore rather than starting
        // from zero and allowing double the memory.
        #expect(await restored.utilization(taskID: task) > 0)
    }

    @Test func theItemCountIsStillABackstop() async throws {
        let memory = TaskMemory(limits: MemoryLimits(maxBytesPerTask: 1 << 30,
                                                     maxItemsPerTask: 3))
        let task = UUID()
        for index in 0..<3 {
            try await memory.write(taskID: task, namespace: "n", key: "k\(index)", value: "v")
        }
        await #expect(throws: ContinuityError.self) {
            try await memory.write(taskID: task, namespace: "n", key: "k4", value: "v")
        }
        // An existing address can still be corrected at the ceiling; only new
        // ones are refused.
        try await memory.write(taskID: task, namespace: "n", key: "k0", value: "corrected")
    }
}
