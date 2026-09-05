import Foundation

/// `MemoryStore` on Valkey.
///
/// Key layout, one scope shown:
///
///     <prefix>:{<ns>/<user>/<workspace>}:r:<key>   the record, as JSON
///     <prefix>:{<ns>/<user>/<workspace>}:idx       sorted set: key -> updatedAt
///     <prefix>:{<ns>/<user>/<workspace>}:sessions  capped list of session ids
///
/// The index is what keeps this honest. Listing and searching read the index
/// and then fetch a bounded batch of records; nothing issues `KEYS` or scans
/// the database, so one scope's cost never depends on how much other scopes
/// have stored. The `{...}` hash tag keeps a scope in one slot if the store
/// is ever clustered.
public actor ValkeyMemoryStore: MemoryStore {
    private let connection: ValkeyConnection
    private let prefix: String
    private let limits: MemoryLimits
    private let maximumIndexScan: Int
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(connection: ValkeyConnection,
                prefix: String = "nvmai:mem",
                limits: MemoryLimits = .init(),
                maximumIndexScan: Int = 2_000) {
        self.connection = connection
        self.prefix = prefix
        self.limits = limits
        self.maximumIndexScan = maximumIndexScan
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    // MARK: - Key construction

    /// The scope's key space. Every key this store touches starts here, which
    /// is the single place cross-scope access could go wrong.
    private func scopeTag(_ scope: MemoryScope) -> String {
        "{\(scope.namespace)/\(scope.user)/\(scope.workspace)}"
    }

    private func recordKey(_ key: MemoryKey, _ scope: MemoryScope) -> String {
        "\(prefix):\(scopeTag(scope)):r:\(key.rawValue)"
    }

    private func indexKey(_ scope: MemoryScope) -> String {
        "\(prefix):\(scopeTag(scope)):idx"
    }

    private func sessionsKey(_ scope: MemoryScope) -> String {
        "\(prefix):\(scopeTag(scope)):sessions"
    }

    // MARK: - MemoryStore

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        try await ensureConnected()
        let reply = try await send(["GET", recordKey(key, scope)])
        return decode(reply)
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        try await ensureConnected()
        var stored = record
        // Preserve the original creation time on rewrite, matching the
        // in-memory store: the model is correcting a fact, and "known since"
        // is the useful date.
        if let existing = try? await get(record.key, in: scope) {
            stored.createdAt = existing.createdAt
        }
        stored.updatedAt = Date()
        let payload = try encoder.encode(stored)
        guard let json = String(data: payload, encoding: .utf8) else {
            throw MemoryError.backendUnavailable("record is not representable as UTF-8")
        }
        _ = try await send(["SET", recordKey(record.key, scope), json])
        _ = try await send(["ZADD", indexKey(scope),
                            String(stored.updatedAt.timeIntervalSince1970), record.key.rawValue])
    }

    @discardableResult
    public func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        try await ensureConnected()
        let removed = try await send(["DEL", recordKey(key, scope)])
        _ = try await send(["ZREM", indexKey(scope), key.rawValue])
        return (removed.integerValue ?? 0) > 0
    }

    public func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        try await ensureConnected()
        let reply = try await send(["EXISTS", recordKey(key, scope)])
        return (reply.integerValue ?? 0) > 0
    }

    public func list(prefix keyPrefix: String, limit: Int, in scope: MemoryScope) async throws
        -> [MemoryKey] {
        try await ensureConnected()
        let keys = try await indexedKeys(in: scope)
        return keys
            .filter { keyPrefix.isEmpty || $0.hasPrefix(keyPrefix) }
            .prefix(max(0, min(limit, limits.maximumListResults)))
            .compactMap { try? MemoryKey(validating: $0) }
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        try await ensureConnected()
        let keys = try await indexedKeys(in: scope)
        // Narrow by key prefix before fetching: the prefix is the cheapest
        // filter and the one the model uses most.
        let candidates = keys.filter { query.prefix.map($0.hasPrefix) ?? true }
        let records = try await fetch(keys: Array(candidates.prefix(maximumIndexScan)), in: scope)
        let ranked = MemoryRanking.rank(records, for: query)
        return Array(ranked.prefix(max(0, min(query.limit, limits.maximumSearchResults))))
    }

    @discardableResult
    public func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws
        -> MemoryRecord {
        let existing = try await get(key, in: scope)
        let combined = existing.map { $0.value.isEmpty ? text : $0.value + "\n" + text } ?? text
        try limits.validate(value: combined)
        var record = existing ?? MemoryRecord(key: key, value: "")
        record.value = combined
        try await set(record, in: scope)
        return try await get(key, in: scope) ?? record
    }

    public func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws
        -> MemoryBootstrap {
        try await ensureConnected()
        let payload = try encoder.encode(session)
        if let json = String(data: payload, encoding: .utf8) {
            _ = try? await send(["LPUSH", sessionsKey(scope), json])
            // Keep the last 50: enough to see who has been working here,
            // bounded so an old workspace cannot grow without limit.
            _ = try? await send(["LTRIM", sessionsKey(scope), "0", "49"])
        }
        // Bootstrap reads the most recently updated slice of the index, not
        // the scope: on a store with thousands of facts this is the
        // difference between a bounded read and pulling everything to rank it.
        let keys = try await indexedKeys(in: scope)
        let recent = Array(keys.prefix(max(limits.bootstrapRecords * 5, limits.bootstrapRecords)))
        let records = try await fetch(keys: recent, in: scope)
        return MemoryBootstrap.build(from: records, limits: limits)
    }

    // MARK: - Helpers

    private func ensureConnected() async throws {
        do {
            try await connection.connect()
        } catch let error as ValkeyError {
            throw MemoryError.backendUnavailable(error.description)
        }
    }

    private func send(_ arguments: [String]) async throws -> RESPValue {
        do {
            return try await connection.send(arguments)
        } catch let error as ValkeyError {
            switch error {
            case .timedOut(let operation):
                throw MemoryError.timedOut(operation: operation, milliseconds: 0)
            default:
                throw MemoryError.backendUnavailable(error.description)
            }
        }
    }

    /// Keys in the scope, most recently updated first, bounded.
    private func indexedKeys(in scope: MemoryScope) async throws -> [String] {
        let reply = try await send(["ZRANGE", indexKey(scope), "0", String(maximumIndexScan - 1),
                                    "REV"])
        return reply.arrayValue?.compactMap(\.stringValue) ?? []
    }

    /// Fetches records in batches. `MGET` keeps a search to a couple of round
    /// trips instead of one per key.
    private func fetch(keys: [String], in scope: MemoryScope) async throws -> [MemoryRecord] {
        guard !keys.isEmpty else { return [] }
        var records: [MemoryRecord] = []
        records.reserveCapacity(keys.count)
        for batch in stride(from: 0, to: keys.count, by: 128) {
            let slice = keys[batch..<min(batch + 128, keys.count)]
            let arguments = ["MGET"] + slice.compactMap { raw -> String? in
                guard let key = try? MemoryKey(validating: raw) else { return nil }
                return recordKey(key, scope)
            }
            guard arguments.count > 1 else { continue }
            let reply = try await send(arguments)
            for value in reply.arrayValue ?? [] {
                if let record = decode(value) { records.append(record) }
            }
        }
        return records
    }

    private func decode(_ value: RESPValue) -> MemoryRecord? {
        guard let json = value.stringValue, let data = json.data(using: .utf8) else { return nil }
        return try? decoder.decode(MemoryRecord.self, from: data)
    }
}
