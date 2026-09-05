import Foundation

/// The journal held in process: reference implementation and test double,
/// with the same trim rules the Valkey one applies.
public actor InMemoryJournal: SessionJournal {
    private struct Key: Hashable {
        let scope: MemoryScope
        let session: String
    }

    private var turns: [Key: [JournalTurn]] = [:]
    private var order: [MemoryScope: [String]] = [:]
    private let limits: JournalLimits

    public init(limits: JournalLimits = .init()) {
        self.limits = limits
    }

    public func record(_ turn: JournalTurn, in scope: MemoryScope) async {
        let key = Key(scope: scope, session: turn.session)
        var existing = turns[key] ?? []
        existing.append(turn)
        // Newest wins when the session overflows: an old turn of a long
        // session is the least useful thing the journal holds.
        if existing.count > limits.turnsPerSession {
            existing.removeFirst(existing.count - limits.turnsPerSession)
        }
        turns[key] = existing

        var sessions = order[scope] ?? []
        sessions.removeAll { $0 == turn.session }
        sessions.insert(turn.session, at: 0)
        if sessions.count > limits.sessionsPerWorkspace {
            for stale in sessions.suffix(sessions.count - limits.sessionsPerWorkspace) {
                turns[Key(scope: scope, session: stale)] = nil
            }
            sessions.removeLast(sessions.count - limits.sessionsPerWorkspace)
        }
        order[scope] = sessions
    }

    public func turns(session: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        let stored = turns[Key(scope: scope, session: session)] ?? []
        return Array(stored.reversed().prefix(max(0, limit)))
    }

    public func sessions(limit: Int, in scope: MemoryScope) async -> [JournalSessionSummary] {
        let sessions = order[scope] ?? []
        return sessions.prefix(max(0, limit)).compactMap { session in
            let stored = turns[Key(scope: scope, session: session)] ?? []
            guard let first = stored.first, let last = stored.last else { return nil }
            return JournalSessionSummary(session: session,
                                         workspace: scope.workspace,
                                         firstSeen: first.timestamp,
                                         lastSeen: last.timestamp,
                                         turnCount: stored.count,
                                         model: last.model)
        }
    }

    public func search(_ text: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        let needle = text.lowercased()
        guard !needle.isEmpty else { return [] }
        let sessions = order[scope] ?? []
        var matches: [JournalTurn] = []
        for session in sessions {
            for turn in (turns[Key(scope: scope, session: session)] ?? []).reversed()
            where turn.prompt.lowercased().contains(needle)
                || turn.reply.lowercased().contains(needle) {
                matches.append(turn)
                if matches.count >= max(0, limit) { return matches }
            }
        }
        return matches
    }

    /// Every turn in a scope, for tests.
    public func allTurns(in scope: MemoryScope) -> [JournalTurn] {
        (order[scope] ?? []).flatMap { turns[Key(scope: scope, session: $0)] ?? [] }
    }
}
