import Foundation

/// A `MemoryStore` held in process.
///
/// It is the reference implementation: every rule the Valkey backend has to
/// honour (scope isolation, value limits, bootstrap bounds, ranking) is
/// implemented here once, in plain Swift, and the conformance tests run
/// against both. It is also what the server falls back to when Valkey is
/// configured but unreachable and `degradeToLocal` is set, and what the
/// tests use so the suite needs no server.
public actor InMemoryStore: MemoryStore {
    private var records: [MemoryScope: [MemoryKey: MemoryRecord]] = [:]
    private var sessions: [MemoryScope: [MemorySession]] = [:]
    private let limits: MemoryLimits

    public init(limits: MemoryLimits = .init()) {
        self.limits = limits
    }

    /// Every record in a scope, for tests and for the consolidation hook.
    public func allRecords(in scope: MemoryScope) -> [MemoryRecord] {
        Array(records[scope]?.values ?? [:].values)
    }

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        records[scope]?[key]
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        var scoped = records[scope] ?? [:]
        // A rewrite keeps the original creation time: the model is updating a
        // fact, not making a new one, and "known since" is the useful date.
        var stored = record
        if let existing = scoped[record.key] {
            stored.createdAt = existing.createdAt
        }
        stored.updatedAt = Date()
        scoped[record.key] = stored
        records[scope] = scoped
    }

    @discardableResult
    public func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        guard records[scope]?[key] != nil else { return false }
        records[scope]?[key] = nil
        return true
    }

    public func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        records[scope]?[key] != nil
    }

    public func list(prefix: String, limit: Int, in scope: MemoryScope) async throws -> [MemoryKey] {
        let scoped = records[scope] ?? [:]
        return scoped.values
            .filter { prefix.isEmpty || $0.key.rawValue.hasPrefix(prefix) }
            .sorted { $0.updatedAt > $1.updatedAt }
            .prefix(max(0, min(limit, limits.maximumListResults)))
            .map(\.key)
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        let scoped = records[scope] ?? [:]
        let ranked = MemoryRanking.rank(Array(scoped.values), for: query)
        return Array(ranked.prefix(max(0, min(query.limit, limits.maximumSearchResults))))
    }

    @discardableResult
    public func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws
        -> MemoryRecord {
        let existing = records[scope]?[key]
        let combined = existing.map { $0.value.isEmpty ? text : $0.value + "\n" + text } ?? text
        try limits.validate(value: combined)
        var record = existing ?? MemoryRecord(key: key, value: "")
        record.value = combined
        record.updatedAt = Date()
        var scoped = records[scope] ?? [:]
        scoped[key] = record
        records[scope] = scoped
        return record
    }

    public func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws
        -> MemoryBootstrap {
        sessions[scope, default: []].append(session)
        let scoped = Array((records[scope] ?? [:]).values)
        return MemoryBootstrap.build(from: scoped, limits: limits)
    }

    /// Sessions recorded in a scope, for tests.
    public func recordedSessions(in scope: MemoryScope) -> [MemorySession] {
        sessions[scope] ?? []
    }
}

/// Size and count bounds shared by every backend.
///
/// These exist because the model chooses what to write. Without a ceiling on
/// value size a single tool call can put a source file in the store, and
/// without one on result counts a search can return the store.
public struct MemoryLimits: Sendable, Equatable {
    public var maximumValueBytes: Int
    public var maximumSearchResults: Int
    public var maximumListResults: Int
    public var bootstrapRecords: Int
    public var bootstrapBytes: Int

    public init(maximumValueBytes: Int = 64 * 1024,
                maximumSearchResults: Int = 50,
                maximumListResults: Int = 200,
                bootstrapRecords: Int = 20,
                bootstrapBytes: Int = 8 * 1024) {
        self.maximumValueBytes = maximumValueBytes
        self.maximumSearchResults = maximumSearchResults
        self.maximumListResults = maximumListResults
        self.bootstrapRecords = bootstrapRecords
        self.bootstrapBytes = bootstrapBytes
    }

    public func validate(value: String) throws {
        let bytes = value.utf8.count
        guard bytes <= maximumValueBytes else {
            throw MemoryError.valueTooLarge(bytes: bytes, limit: maximumValueBytes)
        }
    }
}

extension MemoryBootstrap {
    /// The bounded bootstrap set: most important first, then most recent,
    /// cut by whichever limit binds first.
    ///
    /// Both limits are enforced here rather than at the call site so every
    /// backend gets the same ceiling; the count alone is not enough, because
    /// twenty records of 64 KB would still be 1.2 MB of context.
    static func build(from records: [MemoryRecord], limits: MemoryLimits) -> MemoryBootstrap {
        let ordered = records.sorted { left, right in
            let leftImportance = left.importance ?? 0
            let rightImportance = right.importance ?? 0
            if leftImportance != rightImportance { return leftImportance > rightImportance }
            return left.updatedAt > right.updatedAt
        }
        var chosen: [MemoryRecord] = []
        var bytes = 0
        for record in ordered {
            guard chosen.count < limits.bootstrapRecords else { break }
            let size = record.key.rawValue.utf8.count + record.value.utf8.count
            guard bytes + size <= limits.bootstrapBytes else { continue }
            chosen.append(record)
            bytes += size
        }
        return MemoryBootstrap(records: chosen,
                               omittedCount: records.count - chosen.count,
                               totalBytes: bytes)
    }
}

/// Ranking for `search`, kept apart from storage so the retrieval strategy
/// can change without the backends changing.
///
/// Today: filter by prefix, tags and importance, then score by where the
/// query's terms appear. Deliberately not a vector index; the interface is
/// what allows one later, and adding one now would be a dependency and an
/// index to maintain for a store that holds a few hundred short facts.
public enum MemoryRanking {
    public static func rank(_ records: [MemoryRecord], for query: MemoryQuery) -> [MemoryRecord] {
        let terms = (query.text ?? "")
            .lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
            .filter { $0.count > 2 }
        let candidates = records.filter { record in
            if let prefix = query.prefix, !prefix.isEmpty,
               !record.key.rawValue.hasPrefix(prefix) { return false }
            if let minimum = query.minimumImportance, (record.importance ?? 0) < minimum {
                return false
            }
            if !query.tags.isEmpty {
                let lowered = Set(record.tags.map { $0.lowercased() })
                guard query.tags.contains(where: { lowered.contains($0.lowercased()) }) else {
                    return false
                }
            }
            return true
        }
        guard !terms.isEmpty else {
            return candidates.sorted { scoreWithoutText($0) > scoreWithoutText($1) }
        }
        let scored = candidates.compactMap { record -> (MemoryRecord, Double)? in
            let score = textScore(record, terms: terms)
            return score > 0 ? (record, score) : nil
        }
        return scored
            .sorted { left, right in
                if left.1 != right.1 { return left.1 > right.1 }
                return left.0.updatedAt > right.0.updatedAt
            }
            .map(\.0)
    }

    private static func scoreWithoutText(_ record: MemoryRecord) -> Double {
        (record.importance ?? 0) * 1000 + record.updatedAt.timeIntervalSince1970 / 1_000_000_000
    }

    /// A term in the key counts for more than one in the body: the model
    /// names a memory for what it is about, so "decisions/sync" matching
    /// "sync" is a stronger signal than the word appearing in a sentence.
    private static func textScore(_ record: MemoryRecord, terms: [String]) -> Double {
        let key = record.key.rawValue.lowercased()
        let value = record.value.lowercased()
        let tags = record.tags.map { $0.lowercased() }
        var score = 0.0
        for term in terms {
            if key.contains(term) { score += 3 }
            if tags.contains(where: { $0.contains(term) }) { score += 2 }
            if value.contains(term) { score += 1 }
        }
        guard score > 0 else { return 0 }
        return score + (record.importance ?? 0)
    }
}
