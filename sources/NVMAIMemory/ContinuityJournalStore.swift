import Foundation
import ContinuityCore

/// The engine-authored journal, backed by the same in-process engine as the
/// curated store.
///
/// The two stores stay distinct where it matters. Curated memory is the
/// model's, written deliberately through tools, and it is what gets injected
/// into a prompt. The journal is the engine's, written for every turn at no
/// token cost, and it is never injected. Sharing one engine means they share
/// one file and one restart path, not one key space: a turn lands in the
/// session log, a fact lands in task memory, and neither can evict the other.
public actor ContinuityJournalStore: SessionJournal {
    private let engine: ContinuityEngine
    private let store: ContinuityStore
    private let limits: JournalLimits
    /// Continuity sessions opened for journal turns, by the caller's own id.
    private var sessions: [String: UUID] = [:]
    private var turnCounts: [UUID: Int] = [:]

    public init(engine: ContinuityEngine, store: ContinuityStore, limits: JournalLimits = .init()) {
        self.engine = engine
        self.store = store
        self.limits = limits
    }

    public func record(_ turn: JournalTurn, in scope: MemoryScope) async {
        guard let taskID = try? await store.taskID(for: scope),
              let sessionID = await session(for: turn.session, taskID: taskID,
                                            model: turn.model) else { return }
        try? await engine.recordUserPrompt(sessionID: sessionID, text: turn.prompt)
        try? await engine.recordAssistantResponse(sessionID: sessionID,
                                                  text: turn.reply,
                                                  model: turn.model,
                                                  inputTokens: turn.promptTokens,
                                                  outputTokens: turn.completionTokens,
                                                  latencyMilliseconds: turn.latencyMilliseconds,
                                                  finishReason: turn.stopReason)
        let count = (turnCounts[sessionID] ?? 0) + 1
        turnCounts[sessionID] = count
        if count > limits.turnsPerSession {
            await engine.pruneTurns(sessionID: sessionID, keeping: limits.turnsPerSession)
            turnCounts[sessionID] = limits.turnsPerSession
        }
        await engine.pruneSessions(taskID: taskID, keeping: limits.sessionsPerWorkspace)
    }

    public func turns(session: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        guard let taskID = try? await store.taskID(for: scope),
              let resolved = await engine.session(externalID: session, taskID: taskID)
        else { return [] }
        let all = await engine.turns(taskID: taskID)
        return all
            .filter { $0.sessionID == resolved.id }
            .enumerated()
            .map { Self.journalTurn(from: $0.element, index: $0.offset, session: session,
                                    scope: scope) }
            .reversed()
            .prefix(limit)
            .map { $0 }
    }

    public func sessions(limit: Int, in scope: MemoryScope) async -> [JournalSessionSummary] {
        guard let taskID = try? await store.taskID(for: scope) else { return [] }
        let all = await engine.turns(taskID: taskID)
        var bySession: [UUID: [SessionTurn]] = [:]
        for turn in all { bySession[turn.sessionID, default: []].append(turn) }
        var summaries: [JournalSessionSummary] = []
        for session in await engine.sessions(taskID: taskID) {
            let turns = bySession[session.id] ?? []
            summaries.append(JournalSessionSummary(
                session: session.externalID ?? session.id.uuidString,
                workspace: scope.workspace,
                firstSeen: turns.first?.timestamp ?? session.startedAt,
                lastSeen: turns.last?.completedAt ?? turns.last?.timestamp
                    ?? session.endedAt ?? session.startedAt,
                turnCount: turns.count,
                model: session.model))
        }
        return summaries.sorted { $0.lastSeen > $1.lastSeen }.prefix(limit).map { $0 }
    }

    public func search(_ text: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        guard !text.isEmpty, let taskID = try? await store.taskID(for: scope) else { return [] }
        let labels = await sessionLabels(taskID: taskID)
        let all = await engine.turns(taskID: taskID)
        var counters: [UUID: Int] = [:]
        var matches: [JournalTurn] = []
        for turn in all {
            let index = counters[turn.sessionID] ?? 0
            counters[turn.sessionID] = index + 1
            let haystack = turn.prompt + " " + (turn.response ?? "")
            guard haystack.range(of: text, options: .caseInsensitive) != nil else { continue }
            matches.append(Self.journalTurn(from: turn, index: index,
                                            session: labels[turn.sessionID]
                                                ?? turn.sessionID.uuidString,
                                            scope: scope))
        }
        return matches.reversed().prefix(limit).map { $0 }
    }

    // MARK: - Internals

    private func sessionLabels(taskID: UUID) async -> [UUID: String] {
        var labels: [UUID: String] = [:]
        for session in await engine.sessions(taskID: taskID) {
            labels[session.id] = session.externalID ?? session.id.uuidString
        }
        return labels
    }

    /// Reuses the session the memory store already opened for this id, so a
    /// turn and the facts written during it belong to the same session rather
    /// than to two that merely share a name.
    private func session(for id: String, taskID: UUID, model: String?) async -> UUID? {
        if let known = sessions[id] { return known }
        if let shared = await store.continuitySession(for: id) {
            sessions[id] = shared
            return shared
        }
        if let existing = await engine.session(externalID: id, taskID: taskID) {
            sessions[id] = existing.id
            return existing.id
        }
        guard let opened = try? await engine.beginSession(taskID: taskID, model: model,
                                                          externalID: id) else { return nil }
        sessions[id] = opened.id
        return opened.id
    }

    private static func journalTurn(from turn: SessionTurn,
                                    index: Int,
                                    session: String,
                                    scope: MemoryScope) -> JournalTurn {
        JournalTurn(session: session,
                    workspace: scope.workspace,
                    index: index,
                    timestamp: turn.timestamp,
                    prompt: turn.prompt,
                    reply: turn.response ?? "",
                    model: turn.responseRecord?.model,
                    promptTokens: turn.responseRecord?.inputTokens ?? 0,
                    completionTokens: turn.responseRecord?.outputTokens ?? 0,
                    latencyMilliseconds: turn.responseRecord?.latencyMilliseconds ?? 0,
                    stopReason: turn.responseRecord?.finishReason)
    }
}
