import Foundation

/// The session journal on Valkey.
///
/// Its own key space, separate from curated memory, with its own trim policy,
/// so a long week of sessions can never crowd out the facts the model wrote
/// deliberately. Those are the store with the higher value per byte, and a
/// shared budget would let the cheaper one evict the dearer.
///
///     <prefix>:{<scope>}:j:<session>      list of turns, newest first
///     <prefix>:{<scope>}:j:index          sorted set: session -> last seen
///
/// A turn is a few kilobytes with the filter applied, which is what makes a
/// list in Valkey the right home: no file format, no separate index, and the
/// same durability as the rest of the store.
public actor ValkeyJournal: SessionJournal {
    private let connection: ValkeyConnection
    private let prefix: String
    private let limits: JournalLimits
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(connection: ValkeyConnection,
                prefix: String = "nvmai:mem",
                limits: JournalLimits = .init()) {
        self.connection = connection
        self.prefix = prefix
        self.limits = limits
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    private func scopeTag(_ scope: MemoryScope) -> String {
        "{\(scope.namespace)/\(scope.user)/\(scope.workspace)}"
    }

    private func turnsKey(_ session: String, _ scope: MemoryScope) -> String {
        "\(prefix):\(scopeTag(scope)):j:\(session)"
    }

    private func indexKey(_ scope: MemoryScope) -> String {
        "\(prefix):\(scopeTag(scope)):j:index"
    }

    /// Records a turn. Swallows failures by design: the journal is a
    /// by-product of serving, and a completion must never fail because it
    /// could not be written down.
    public func record(_ turn: JournalTurn, in scope: MemoryScope) async {
        guard let payload = try? encoder.encode(turn),
              let json = String(data: payload, encoding: .utf8) else { return }
        do {
            try await connection.connect()
            let key = turnsKey(turn.session, scope)
            _ = try await connection.send(["LPUSH", key, json])
            _ = try await connection.send(["LTRIM", key, "0", String(limits.turnsPerSession - 1)])
            _ = try await connection.send(["ZADD", indexKey(scope),
                                           String(turn.timestamp.timeIntervalSince1970),
                                           turn.session])
            try await trimSessions(in: scope)
        } catch {
            // Nothing to do and nowhere useful to throw: the caller is a
            // completion that has already succeeded.
        }
    }

    /// Drops the oldest sessions past the workspace limit, and their turns
    /// with them, so the journal's footprint has a ceiling.
    private func trimSessions(in scope: MemoryScope) async throws {
        let reply = try await connection.send(["ZRANGE", indexKey(scope), "0",
                                               String(-limits.sessionsPerWorkspace - 1)])
        let stale = reply.arrayValue?.compactMap(\.stringValue) ?? []
        guard !stale.isEmpty else { return }
        for session in stale {
            _ = try? await connection.send(["DEL", turnsKey(session, scope)])
            _ = try? await connection.send(["ZREM", indexKey(scope), session])
        }
    }

    public func turns(session: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        do {
            try await connection.connect()
            let reply = try await connection.send(["LRANGE", turnsKey(session, scope), "0",
                                                   String(max(0, limit) - 1)])
            return (reply.arrayValue ?? []).compactMap(decode)
        } catch {
            return []
        }
    }

    public func sessions(limit: Int, in scope: MemoryScope) async -> [JournalSessionSummary] {
        do {
            try await connection.connect()
            let reply = try await connection.send(["ZRANGE", indexKey(scope), "0",
                                                   String(max(0, limit) - 1), "REV"])
            let ids = reply.arrayValue?.compactMap(\.stringValue) ?? []
            var summaries: [JournalSessionSummary] = []
            for id in ids {
                let stored = await turns(session: id, limit: limits.turnsPerSession, in: scope)
                guard let newest = stored.first, let oldest = stored.last else { continue }
                summaries.append(JournalSessionSummary(session: id,
                                                       workspace: scope.workspace,
                                                       firstSeen: oldest.timestamp,
                                                       lastSeen: newest.timestamp,
                                                       turnCount: stored.count,
                                                       model: newest.model))
            }
            return summaries
        } catch {
            return []
        }
    }

    public func search(_ text: String, limit: Int, in scope: MemoryScope) async -> [JournalTurn] {
        let needle = text.lowercased()
        guard !needle.isEmpty else { return [] }
        // Sessions newest first, scanning until enough matches: the journal
        // is for "when did we last touch this", and the recent answer is
        // almost always the wanted one.
        let recent = await sessions(limit: limits.sessionsPerWorkspace, in: scope)
        var matches: [JournalTurn] = []
        for summary in recent {
            let stored = await turns(session: summary.session,
                                     limit: limits.turnsPerSession,
                                     in: scope)
            for turn in stored where turn.prompt.lowercased().contains(needle)
                || turn.reply.lowercased().contains(needle) {
                matches.append(turn)
                if matches.count >= max(0, limit) { return matches }
            }
        }
        return matches
    }

    private func decode(_ value: RESPValue) -> JournalTurn? {
        guard let json = value.stringValue, let data = json.data(using: .utf8) else { return nil }
        return try? decoder.decode(JournalTurn.self, from: data)
    }
}
