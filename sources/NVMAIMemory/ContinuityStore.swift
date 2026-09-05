import Foundation
import ContinuityCore

/// The durable memory store, backed by the in-process continuity engine.
///
/// This replaced a Valkey client. Nothing above it changed: the server still
/// talks to `MemoryStore`, and the model still sees the same tools. What went
/// away is a second process, a wire protocol, a connection to lose and a
/// cache to size. The store now lives in the same binary as the model that
/// reads it.
///
/// A scope maps to one continuity task, and a `MemoryKey` maps to one
/// address. Keys are normalized on the way in, and the normalized form is
/// what comes back out, so a key the model reads is always a key it can use.
public actor ContinuityStore: MemoryStore {
    private let engine: ContinuityEngine
    private let limits: MemoryLimits
    private var taskIDs: [MemoryScope: UUID] = [:]
    private var sessionIDs: [String: UUID] = [:]
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(engine: ContinuityEngine, limits: MemoryLimits = .init()) {
        self.engine = engine
        self.limits = limits
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
        self.encoder = encoder
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder
    }

    // MARK: - MemoryStore

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        guard let item = await engine.recall(taskID: taskID,
                                             namespace: address.namespace,
                                             key: address.key),
              item.status.isEligibleForContext else { return nil }
        return try record(from: item)
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        let taskID = try await task(for: scope)
        let normalized = Self.normalize(record)
        let address = Self.address(for: normalized.key)
        let payload = String(decoding: try encoder.encode(normalized), as: UTF8.self)
        do {
            try await engine.remember(taskID: taskID,
                                      namespace: address.namespace,
                                      key: address.key,
                                      value: payload,
                                      author: .model,
                                      importance: normalized.importance,
                                      tags: normalized.tags)
        } catch let error as ContinuityError {
            throw Self.translate(error)
        }
    }

    @discardableResult
    public func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        guard let existing = await engine.recall(taskID: taskID,
                                                 namespace: address.namespace,
                                                 key: address.key),
              existing.status.isEligibleForContext else { return false }
        // Archived, not destroyed. A model that deletes a fact in one session
        // and contradicts itself in the next leaves a chain that explains it.
        _ = try? await engine.archive(taskID: taskID,
                                      namespace: address.namespace,
                                      key: address.key)
        return true
    }

    public func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool {
        try await get(key, in: scope) != nil
    }

    public func list(prefix: String, limit: Int, in scope: MemoryScope) async throws -> [MemoryKey] {
        let records = try await allRecords(in: scope)
        let normalizedPrefix = Self.normalizeKeyText(prefix)
        return records
            .filter { normalizedPrefix.isEmpty || $0.key.rawValue.hasPrefix(normalizedPrefix) }
            .sorted { $0.updatedAt > $1.updatedAt }
            .prefix(min(limit, limits.maximumListResults))
            .map(\.key)
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        // Ranking happens here rather than in the engine because the engine
        // stores the record as JSON, and a raw substring match over that would
        // hit field names as readily as content.
        let records = try await allRecords(in: scope)
        var bounded = query
        bounded.limit = min(query.limit, limits.maximumSearchResults)
        return MemoryRanking.rank(records, for: bounded)
    }

    @discardableResult
    public func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws
        -> MemoryRecord {
        let existing = try await get(key, in: scope)
        let combined = existing.map { $0.value.isEmpty ? text : $0.value + "\n" + text } ?? text
        try limits.validate(value: combined)
        var record = existing ?? MemoryRecord(key: key, value: "")
        record.value = combined
        record.updatedAt = Date()
        try await set(record, in: scope)
        return record
    }

    public func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws
        -> MemoryBootstrap {
        let taskID = try await task(for: scope)
        // Session identity is derived from the conversation, so the same id
        // can arrive again after a restart or when a client replays a
        // conversation. Reuse the session rather than opening a second one
        // that splits the same conversation's journal in half.
        if let existing = await engine.session(externalID: session.id, taskID: taskID) {
            sessionIDs[session.id] = existing.id
        } else {
            let continuity = try await engine.beginSession(taskID: taskID,
                                                           model: session.modelID,
                                                           externalID: session.id)
            sessionIDs[session.id] = continuity.id
        }
        let records = try await allRecords(in: scope)
        return MemoryBootstrap.build(from: records, limits: limits)
    }

    // MARK: - Beyond the protocol

    /// The continuity session for a memory session, once one has begun. The
    /// journal uses it so both stores describe the same session rather than
    /// two parallel ones.
    public func continuitySession(for id: String) -> UUID? { sessionIDs[id] }

    public func taskID(for scope: MemoryScope) async throws -> UUID {
        try await task(for: scope)
    }

    // MARK: - Internals

    private func allRecords(in scope: MemoryScope) async throws -> [MemoryRecord] {
        let taskID = try await task(for: scope)
        let items = await engine.recall(taskID: taskID,
                                        ContinuityCore.MemoryQuery(statuses: [.active, .disputed],
                                                                   order: .recency))
        return items.compactMap { try? record(from: $0) }
    }

    private func record(from item: ContinuityCore.MemoryItem) throws -> MemoryRecord {
        guard let data = item.value.data(using: .utf8) else {
            throw MemoryError.backendUnavailable("stored value is not UTF-8")
        }
        if let decoded = try? decoder.decode(MemoryRecord.self, from: data) { return decoded }
        // A value written by something other than this adapter is still worth
        // returning; losing it because it is not in our envelope would be the
        // worse failure.
        let key = try MemoryKey(validating: Self.keyText(for: item))
        return MemoryRecord(key: key,
                            value: item.value,
                            importance: item.importance,
                            tags: item.tags,
                            createdAt: item.createdAt,
                            updatedAt: item.updatedAt)
    }

    private func task(for scope: MemoryScope) async throws -> UUID {
        if let existing = taskIDs[scope] { return existing }
        let id = Self.taskIdentifier(for: scope)
        if await engine.task(id) == nil {
            _ = try await engine.createTask(title: "\(scope.workspace)",
                                            objective: "Durable memory for "
                                                + "\(scope.namespace)/\(scope.user)/"
                                                + "\(scope.workspace)",
                                            id: id)
        }
        taskIDs[scope] = id
        return id
    }

    // MARK: - Address mapping

    struct Address: Equatable {
        let namespace: String
        let key: String
    }

    /// `decisions/sync` becomes namespace `k.decisions`, key `sync`.
    ///
    /// The leading `k` keeps a one-segment key from colliding with a
    /// two-segment one, and it is added on every address, so no key the model
    /// writes can produce it by accident.
    static func address(for key: MemoryKey) -> Address {
        let segments = normalizeKeyText(key.rawValue).split(separator: "/").map(String.init)
        guard let last = segments.last else { return Address(namespace: "k", key: "empty") }
        let leading = segments.dropLast()
        let namespace = (["k"] + leading).joined(separator: ".")
        return Address(namespace: namespace, key: last)
    }

    static func keyText(for item: ContinuityCore.MemoryItem) -> String {
        var segments = item.namespace.split(separator: ".").map(String.init)
        if segments.first == "k" { segments.removeFirst() }
        segments.append(item.key)
        return segments.joined(separator: "/")
    }

    /// The continuity address alphabet is narrower than a memory key's: no
    /// uppercase, and no dots inside a segment. Folding is deliberate and
    /// idempotent, so a key handed back to the model resolves to the same
    /// address when it comes round again.
    static func normalizeKeyText(_ raw: String) -> String {
        String(raw.lowercased().map { character in
            if character == "." { return "-" }
            return character
        })
    }

    static func normalize(_ record: MemoryRecord) -> MemoryRecord {
        guard let key = try? MemoryKey(validating: normalizeKeyText(record.key.rawValue)) else {
            return record
        }
        var copy = record
        copy.key = key
        return copy
    }

    /// A stable task id for a scope, so a restart finds the same memory.
    /// Swift's own hashing is seeded per process and cannot be used for this.
    static func taskIdentifier(for scope: MemoryScope) -> UUID {
        let text = "\(scope.namespace)/\(scope.user)/\(scope.workspace)"
        var high: UInt64 = 0xcbf2_9ce4_8422_2325
        var low: UInt64 = 0x9e37_79b9_7f4a_7c15
        for byte in text.utf8 {
            high ^= UInt64(byte)
            high = high &* 0x0000_0100_0000_01b3
            low = (low &+ UInt64(byte)) &* 0xff51_afd7_ed55_8ccd
            low ^= low >> 33
        }
        var bytes = [UInt8]()
        for shift in stride(from: 56, through: 0, by: -8) {
            bytes.append(UInt8(truncatingIfNeeded: high >> UInt64(shift)))
        }
        for shift in stride(from: 56, through: 0, by: -8) {
            bytes.append(UInt8(truncatingIfNeeded: low >> UInt64(shift)))
        }
        bytes[6] = (bytes[6] & 0x0F) | 0x40
        bytes[8] = (bytes[8] & 0x3F) | 0x80
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3],
                           bytes[4], bytes[5], bytes[6], bytes[7],
                           bytes[8], bytes[9], bytes[10], bytes[11],
                           bytes[12], bytes[13], bytes[14], bytes[15]))
    }

    static func translate(_ error: ContinuityError) -> MemoryError {
        switch error {
        case .valueTooLarge(let bytes, let limit):
            return .valueTooLarge(bytes: bytes, limit: limit)
        case .invalidKey(let value, let reason), .invalidNamespace(let value, let reason):
            return .invalidKey(value, reason)
        default:
            return .backendUnavailable(String(describing: error))
        }
    }
}
