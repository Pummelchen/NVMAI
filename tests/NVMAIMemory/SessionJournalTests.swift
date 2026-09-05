import Foundation
import Testing
@testable import NVMAIMemory

/// The journal's contract, and the content filter that makes it affordable.
@Suite struct SessionJournalTests {
    private func scope(_ workspace: String = "repo-a") throws -> MemoryScope {
        try MemoryScope(namespace: "nvmai", user: "local", workspace: workspace)
    }

    private func turn(_ session: String,
                      _ index: Int,
                      prompt: String = "why is FooManager here?",
                      reply: String = "It prevents a race in background sync.",
                      at seconds: TimeInterval = 0) -> JournalTurn {
        JournalTurn(session: session,
                    workspace: "repo-a",
                    index: index,
                    timestamp: Date(timeIntervalSince1970: 1_000 + seconds),
                    prompt: prompt,
                    reply: reply,
                    model: "qwen3.6",
                    promptTokens: 120,
                    completionTokens: 30,
                    latencyMilliseconds: 4_200,
                    stopReason: "stop")
    }

    // MARK: - Filter

    @Test func filterKeepsOrdinaryProseWhole() {
        let filter = JournalFilter()
        let text = "We're keeping FooManager because it prevents a race in background sync."
        let (kept, dropped) = filter.filter(text)
        #expect(kept == text)
        #expect(dropped == 0)
    }

    @Test func filterKeepsShortCodeBlocks() {
        // A three-line snippet is usually the point of the message.
        let filter = JournalFilter()
        let text = "Use this:\n```swift\nlet a = 1\nlet b = 2\n```\nThat is all."
        let (kept, dropped) = filter.filter(text)
        #expect(kept.contains("let a = 1"))
        #expect(dropped == 0)
    }

    @Test func filterCollapsesFileDumpsAndCommandOutput() {
        // A pasted file or a test log is context for that turn, not a durable
        // record, and it is what would otherwise make a turn megabytes.
        let filter = JournalFilter(maximumMessageBytes: 4_096, maximumBlockLines: 5)
        let dump = (0..<400).map { "line \($0) of build output" }.joined(separator: "\n")
        let text = "Tests failed:\n```\n\(dump)\n```\nAny idea why?"
        let (kept, dropped) = filter.filter(text)

        #expect(kept.contains("Tests failed:"))
        #expect(kept.contains("Any idea why?"))
        #expect(!kept.contains("line 200 of build output"))
        #expect(kept.contains("lines"))
        #expect(dropped > 5_000)
        #expect(kept.utf8.count < 500)
    }

    @Test func filterTruncatesAnOversizedMessageFromBothEnds() {
        let filter = JournalFilter(maximumMessageBytes: 400)
        let text = String(repeating: "prose ", count: 400)
        let (kept, dropped) = filter.filter(text)
        #expect(kept.utf8.count <= 500)
        #expect(kept.contains("bytes omitted"))
        #expect(dropped > 0)
    }

    @Test func filterHandlesAnUnclosedFence() {
        // Streamed replies get cut off mid-block; that must not lose the prose
        // before it or crash the filter.
        let filter = JournalFilter(maximumBlockLines: 2)
        let text = "Here it is:\n```swift\nlet a = 1\nlet b = 2\nlet c = 3\nlet d = 4"
        let (kept, _) = filter.filter(text)
        #expect(kept.contains("Here it is:"))
        #expect(kept.contains("lines omitted"))
    }

    @Test func aFilteredTurnStaysWithinAFewKilobytes() {
        // The sizing claim the storage decision rests on.
        let filter = JournalFilter()
        let prompt = "Please refactor the sync layer.\n```swift\n"
            + (0..<500).map { "    let x\($0) = compute(\($0))" }.joined(separator: "\n")
            + "\n```"
        let reply = "I refactored it; the race is gone and 983 tests pass."
        let (keptPrompt, _) = filter.filter(prompt)
        let (keptReply, _) = filter.filter(reply)
        let turn = JournalTurn(session: "s", workspace: "w", index: 0,
                               prompt: keptPrompt, reply: keptReply)
        #expect(turn.byteCount < 5_120)
    }

    // MARK: - Store behaviour

    @Test func recordsAndReadsBackNewestFirst() async throws {
        let journal = InMemoryJournal()
        let scope = try scope()
        for index in 0..<3 {
            await journal.record(turn("s1", index, prompt: "q\(index)", at: TimeInterval(index)),
                                 in: scope)
        }

        let turns = await journal.turns(session: "s1", limit: 10, in: scope)
        #expect(turns.map(\.prompt) == ["q2", "q1", "q0"])
        #expect(turns.first?.promptTokens == 120)
        #expect(turns.first?.stopReason == "stop")
    }

    @Test func workspacesAreIsolated() async throws {
        let journal = InMemoryJournal()
        await journal.record(turn("s1", 0), in: try scope("repo-a"))
        #expect(await journal.turns(session: "s1", limit: 10, in: try scope("repo-b")).isEmpty)
        #expect(await journal.sessions(limit: 10, in: try scope("repo-b")).isEmpty)
    }

    @Test func trimsTurnsPerSession() async throws {
        let journal = InMemoryJournal(limits: JournalLimits(turnsPerSession: 5))
        let scope = try scope()
        for index in 0..<20 {
            await journal.record(turn("s1", index, prompt: "q\(index)", at: TimeInterval(index)),
                                 in: scope)
        }

        let turns = await journal.turns(session: "s1", limit: 50, in: scope)
        #expect(turns.count == 5)
        // The newest five survive; an old turn of a long session is the least
        // useful thing the journal holds.
        #expect(turns.first?.prompt == "q19")
        #expect(turns.last?.prompt == "q15")
    }

    @Test func trimsSessionsPerWorkspace() async throws {
        let journal = InMemoryJournal(limits: JournalLimits(sessionsPerWorkspace: 3))
        let scope = try scope()
        for index in 0..<6 {
            await journal.record(turn("s\(index)", 0, at: TimeInterval(index)), in: scope)
        }

        let sessions = await journal.sessions(limit: 50, in: scope)
        #expect(sessions.count == 3)
        #expect(sessions.map(\.session) == ["s5", "s4", "s3"])
        // The dropped sessions take their turns with them, so the footprint
        // really does have a ceiling.
        #expect(await journal.turns(session: "s0", limit: 10, in: scope).isEmpty)
        #expect(await journal.allTurns(in: scope).count == 3)
    }

    @Test func sessionSummariesSpanTheirTurns() async throws {
        let journal = InMemoryJournal()
        let scope = try scope()
        await journal.record(turn("s1", 0, at: 0), in: scope)
        await journal.record(turn("s1", 1, at: 60), in: scope)

        let summary = try #require(await journal.sessions(limit: 5, in: scope).first)
        #expect(summary.turnCount == 2)
        #expect(summary.firstSeen == Date(timeIntervalSince1970: 1_000))
        #expect(summary.lastSeen == Date(timeIntervalSince1970: 1_060))
        #expect(summary.model == "qwen3.6")
    }

    @Test func searchFindsTurnsAcrossSessions() async throws {
        let journal = InMemoryJournal()
        let scope = try scope()
        await journal.record(turn("s1", 0, prompt: "why FooManager?", at: 0), in: scope)
        await journal.record(turn("s2", 0, prompt: "unrelated question",
                                  reply: "unrelated answer", at: 60), in: scope)

        let hits = await journal.search("foomanager", limit: 10, in: scope)
        #expect(hits.count == 1)
        #expect(hits.first?.session == "s1")
        // Matching the reply counts too: the answer is often where the useful
        // sentence lives.
        #expect(await journal.search("background sync", limit: 10, in: scope).count == 1)
        #expect(await journal.search("", limit: 10, in: scope).isEmpty)
    }

    @Test func journalLimitsDoNotShareCuratedMemoryBudget() {
        // The two stores are budgeted apart on purpose: a busy week of
        // sessions must not evict the facts the model wrote deliberately.
        let limits = JournalLimits(turnsPerSession: 200, sessionsPerWorkspace: 100)
        #expect(limits.approximateMaximumBytes > 0)
        #expect(MemoryLimits().maximumValueBytes != limits.approximateMaximumBytes)
    }
}
