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
///
/// The record is stored in the engine's own fields, not as a JSON blob in the
/// value. That is what lets a text search be answered by the engine instead of
/// by decoding every record in the workspace, keeps the value readable in the
/// journal file, and stops importance and tags being written twice.
public actor ContinuityStore: MemoryStore {
    private let engine: ContinuityEngine
    private let limits: MemoryLimits
    private var taskIDs: [MemoryScope: UUID] = [:]
    private var sessionIDs: [String: UUID] = [:]
    /// Continuity session to the caller's own session name, so a record can
    /// report who wrote it without a lookup per read.
    private var sessionLabels: [UUID: String] = [:]
    private var labelsLoadedFor: Set<UUID> = []

    public init(engine: ContinuityEngine, limits: MemoryLimits = .init()) {
        self.engine = engine
        self.limits = limits
    }

    // MARK: - MemoryStore

    public func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord? {
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        guard let item = await engine.recall(taskID: taskID,
                                             namespace: address.namespace,
                                             key: address.key),
              item.status.isEligibleForContext else { return nil }
        await loadLabels(taskID: taskID)
        return try record(from: item)
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope) async throws {
        try limits.validate(value: record.value)
        let taskID = try await task(for: scope)
        let normalized = Self.normalize(record)
        let address = Self.address(for: normalized.key)
        // The writing session, when the caller named one and a continuity
        // session has been opened for it. That is what makes a record
        // explainable later, so it is carried as provenance rather than
        // duplicated into the value.
        let sessionID = normalized.sourceSession.flatMap { sessionIDs[$0] }
        do {
            if let sessionID {
                try await engine.remember(sessionID: sessionID,
                                          namespace: address.namespace,
                                          key: address.key,
                                          value: normalized.value,
                                          author: Self.author(of: normalized),
                                          importance: normalized.importance,
                                          confidence: normalized.confidence,
                                          tags: normalized.tags)
            } else {
                try await engine.remember(taskID: taskID,
                                          namespace: address.namespace,
                                          key: address.key,
                                          value: normalized.value,
                                          author: Self.author(of: normalized),
                                          importance: normalized.importance,
                                          confidence: normalized.confidence,
                                          tags: normalized.tags)
            }
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
        let taskID = try await task(for: scope)
        let bound = max(0, min(limit, limits.maximumListResults))
        // The prefix is pushed into the engine as a namespace filter, so a
        // list of one category does not walk the whole workspace. Only the
        // last segment cannot be pushed down, because it is a key prefix
        // rather than a namespace, and that residue is filtered here.
        let plan = Self.pushDown(prefix: prefix)
        var query = ContinuityCore.MemoryQuery(namespacePrefix: plan.namespace,
                                               order: .recency)
        if plan.isExact { query.limit = bound }
        let items = await engine.recall(taskID: taskID, query)
        return items
            .filter { plan.matches(Self.keyText(for: $0)) }
            .prefix(bound)
            .compactMap { try? MemoryKey(validating: Self.keyText(for: $0)) }
    }

    public func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord] {
        let taskID = try await task(for: scope)
        // Filtering is pushed into the engine; ranking stays here because it
        // is the memory layer's own policy, shared with the reference store so
        // the two cannot drift.
        let plan = Self.pushDown(prefix: query.prefix ?? "")
        let engineQuery = ContinuityCore.MemoryQuery(namespacePrefix: plan.namespace,
                                                     text: query.text,
                                                     tags: query.tags,
                                                     minimumImportance: query.minimumImportance,
                                                     limit: maximumScan,
                                                     order: .recency)
        let items = await engine.recall(taskID: taskID, engineQuery)
            .filter { plan.matches(Self.keyText(for: $0)) }
        await loadLabels(taskID: taskID)
        let records = items.compactMap { try? record(from: $0) }
        var bounded = query
        bounded.limit = min(query.limit, limits.maximumSearchResults)
        return MemoryRanking.rank(records, for: bounded)
    }

    /// Ceiling on how many records one query may consider.
    ///
    /// The Valkey backend read a bounded slice of a sorted-set index for the
    /// same reason: one scope's cost must not grow with how much it has
    /// stored. Ranking a few thousand short facts is cheap; ranking every
    /// fact of a two-year project on every search is not.
    private let maximumScan = 2_000

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
                                                           externalID: session.id,
                                                           tag: session.tag)
            sessionIDs[session.id] = continuity.id
        }
        // Ranked by what is being asked, not by a static importance. A flat
        // list ranked by importance dropped a character's eye colour out of
        // the window by the fourth session of a novel because running state
        // had been rated higher; the fact was in the store every time. The
        // engine's assembler ranks by priority namespace, then relevance to
        // the request, then importance, and drags dependencies in with
        // what it picks.
        let candidates = max(limits.bootstrapRecords * 8, 200)
        let items = await engine.recall(
            taskID: taskID,
            ContinuityCore.MemoryQuery(statuses: [.active, .disputed],
                                       limit: candidates,
                                       order: .relevance))
        await loadLabels(taskID: taskID)
        guard let task = await engine.task(taskID) else { return .empty }
        var index: [String: ContinuityCore.MemoryItem] = [:]
        for item in items { index[item.address] = item }
        let request = ContextRequest(
            task: task,
            items: items,
            index: index,
            budget: ContextBudget(maxTokens: max(64, limits.bootstrapBytes / 4),
                                  priorityNamespaces: Self.bootstrapPriority,
                                  recentTurnCount: 0),
            focus: session.focus)
        let ordered: [ContinuityCore.MemoryItem]
        if let snapshot = try? DefaultContextAssembler().assemble(request) {
            let byID = Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0) })
            ordered = snapshot.memoryItemIDs.compactMap { byID[$0] }
        } else {
            ordered = items
        }
        let records = ordered.compactMap { try? record(from: $0) }
        return MemoryBootstrap.build(ordered: records, limits: limits,
                                     recent: recentlyChanged(among: items,
                                                             excluding: sessionIDs[session.id]))
    }

    /// The namespaces a session cannot do without, first. Rules and
    /// constraints bound everything; decisions explain the code; a novel's
    /// characters and setting are its rules. Running state comes last: it is
    /// what most recently changed, and it is exactly what a session is most
    /// likely to be about to change again.
    static let bootstrapPriority: [String] = [
        "k.rules", "k.rule", "k.constraints", "k.constraint",
        "k.decisions", "k.decision", "k.gotchas", "k.gotcha",
        "k.project", "k.setting", "k.characters", "k.character",
        "k.architecture", "k.conventions",
    ]

    /// What the most recent other session wrote, newest first, bounded.
    private func recentlyChanged(among items: [ContinuityCore.MemoryItem],
                                 excluding current: UUID?) -> [MemoryRecord] {
        let newest = items
            .filter { $0.provenance?.sessionID != nil && $0.provenance?.sessionID != current }
            .sorted { $0.updatedAt > $1.updatedAt }
        guard let last = newest.first?.provenance?.sessionID else { return [] }
        return newest.filter { $0.provenance?.sessionID == last }
            .prefix(12)
            .compactMap { try? record(from: $0) }
    }

    /// A write that reverts a key to a value it already had before is almost
    /// never a real event and almost always a re-derivation -- an inn that
    /// burned coming back as "standing" because a later session's chapters
    /// were written against a stale bootstrap. The value is written, because
    /// it may be right, and the key is marked disputed so the model sees the
    /// conflict instead of inheriting whichever side it read last.
    ///
    /// Returns true when a reversion was flagged.
    @discardableResult
    /// Who a record is attributed to. The person's own statements are the
    /// only thing in a transcript that is not the model's own output, and
    /// the guard is built entirely on being able to tell them apart.
    static func author(of record: MemoryRecord) -> ProvenanceAuthor {
        record.isUserAsserted ? .user : .model
    }

    /// The outcome of a guarded write, for the caller to log.
    /// `set`, with the precedence rule the guard adds.
    ///
    /// A user-asserted fact may always be replaced by a newer user-asserted
    /// one -- that is how a state the person changes ("the inn burned in
    /// chapter 34") reaches the store. What is refused is the model
    /// overwriting the person, which is the direction that measured as the
    /// single largest source of wrong facts.
    /// The protocol's guarded write. Reversion flagging is off here: that
    /// heuristic was tuned on consolidation, where a value returning to one
    /// it already had is almost always a re-derivation. A deliberate tool
    /// call is not the same act, and flagging it would put a dispute marker
    /// in front of the next session for a write the model meant to make.
    public func set(_ record: MemoryRecord, in scope: MemoryScope,
                    guarding: Bool) async throws -> GuardedWrite {
        try await set(record, in: scope, guarding: guarding, flaggingReversions: false)
    }

    /// The protocol's guarded delete. A model retiring a fact the person
    /// established is the same failure as overwriting it, and the same
    /// answer: the fact stays, the disagreement is recorded, and the next
    /// session sees both.
    public func delete(_ key: MemoryKey, in scope: MemoryScope,
                       guarding: Bool) async throws -> GuardedDelete {
        guard guarding else {
            return try await delete(key, in: scope) ? .deleted : .absent
        }
        let taskID = try await task(for: scope)
        let address = Self.address(for: key)
        let active = await engine.recall(taskID: taskID,
                                         namespace: address.namespace,
                                         key: address.key)
        if let active, active.provenance?.author == .user,
           active.status.isEligibleForContext {
            _ = try? await engine.dispute(taskID: taskID,
                                          namespace: address.namespace,
                                          key: address.key)
            return .heldByGuard
        }
        return try await delete(key, in: scope) ? .deleted : .absent
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope,
                    guarding: Bool,
                    flaggingReversions: Bool) async throws -> GuardedWrite {
        guard guarding, !record.isUserAsserted else {
            let reverted = try await set(record, in: scope,
                                         flaggingReversions: flaggingReversions)
            return reverted ? .reverted : .stored
        }
        let taskID = try await task(for: scope)
        let address = Self.address(for: Self.normalize(record).key)
        let active = await engine.recall(taskID: taskID,
                                         namespace: address.namespace,
                                         key: address.key)
        if let active, active.provenance?.author == .user, active.status != .archived,
           Self.fold(active.value) != Self.fold(record.value) {
            _ = try? await engine.dispute(taskID: taskID,
                                          namespace: address.namespace,
                                          key: address.key)
            return .heldByGuard(existing: active.value)
        }
        let reverted = try await set(record, in: scope,
                                     flaggingReversions: flaggingReversions)
        return reverted ? .reverted : .stored
    }

    public func set(_ record: MemoryRecord, in scope: MemoryScope,
                    flaggingReversions: Bool) async throws -> Bool {
        guard flaggingReversions else { try await set(record, in: scope); return false }
        let taskID = try await task(for: scope)
        let address = Self.address(for: Self.normalize(record).key)
        let history = await engine.history(taskID: taskID, namespace: address.namespace,
                                           key: address.key)
        let incoming = Self.fold(record.value)
        let current = history.last.map { Self.fold($0.value) }
        let reverts = current != nil && current != incoming
            && history.dropLast().contains { Self.fold($0.value) == incoming }
        try await set(record, in: scope)
        if reverts {
            _ = try? await engine.dispute(taskID: taskID, namespace: address.namespace,
                                          key: address.key)
        }
        return reverts
    }

    private static func fold(_ value: String) -> String {
        value.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".\"'"))
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

    private func record(from item: ContinuityCore.MemoryItem) throws -> MemoryRecord {
        let key = try MemoryKey(validating: Self.keyText(for: item))
        var record = MemoryRecord(key: key,
                                  value: item.value,
                                  importance: item.importance,
                                  confidence: item.confidence,
                                  tags: item.tags,
                                  sourceSession: item.provenance?.sessionID
                                      .flatMap { sessionLabels[$0] },
                                  createdAt: item.createdAt,
                                  updatedAt: item.updatedAt)
        record.isDisputed = item.status == .disputed
        // Authority has to survive a read. Without this, every
        // read-modify-write -- `append` is one -- rewrites a fact the person
        // asserted as though the model had derived it, and the guard can
        // never fire on that address again.
        record.isUserAsserted = item.provenance?.author == .user
        return record
    }

    /// Fills the session-name map for a task, once.
    ///
    /// After a restart the map is empty but the sessions themselves carry
    /// their external names, so one pass rebuilds it. Without this a record
    /// written last week would come back saying nobody wrote it.
    private func loadLabels(taskID: UUID) async {
        guard !labelsLoadedFor.contains(taskID) else { return }
        labelsLoadedFor.insert(taskID)
        for session in await engine.sessions(taskID: taskID) {
            if let external = session.externalID { sessionLabels[session.id] = external }
        }
    }

    /// How a memory-key prefix is answered.
    ///
    /// The namespace filter is only ever an optimization: the residual check
    /// runs against the reconstructed key and decides the result on its own.
    /// Keeping it that way means a prefix that lands mid-segment, like `dec`
    /// for `decisions/sync`, cannot silently be matched against the wrong
    /// part of the address.
    struct PrefixPlan: Equatable {
        /// Namespace prefix the engine can filter on, when the prefix names
        /// whole segments. Nil when it does not.
        let namespace: String?
        /// The normalized prefix, matched against the whole key.
        let residual: String
        /// True when the namespace filter alone is exactly the answer, so the
        /// engine may apply the limit itself.
        var isExact: Bool { namespace != nil && residual.hasSuffix("/") }

        func matches(_ keyText: String) -> Bool {
            residual.isEmpty || keyText.hasPrefix(residual)
        }
    }

    static func pushDown(prefix: String) -> PrefixPlan {
        let normalized = normalizeKeyText(prefix)
        guard !normalized.isEmpty else { return PrefixPlan(namespace: nil, residual: "") }
        let segments = normalized.split(separator: "/").map(String.init)
        guard !segments.isEmpty else { return PrefixPlan(namespace: nil, residual: "") }
        if normalized.hasSuffix("/") {
            // Whole segments: "decisions/" is exactly the namespace k.decisions.
            return PrefixPlan(namespace: (["k"] + segments).joined(separator: "."),
                              residual: normalized)
        }
        guard segments.count >= 2 else {
            // One partial segment. It could be a namespace or a bare key, so
            // nothing can be pushed down without risking a wrong answer.
            return PrefixPlan(namespace: nil, residual: normalized)
        }
        let leading = segments.dropLast()
        return PrefixPlan(namespace: (["k"] + leading).joined(separator: "."),
                          residual: normalized)
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

    public struct Address: Equatable {
        public let namespace: String
        public let key: String
    }

    /// `decisions/sync` becomes namespace `k.decisions`, key `sync`.
    ///
    /// The leading `k` keeps a one-segment key from colliding with a
    /// two-segment one, and it is added on every address, so no key the model
    /// writes can produce it by accident.
    public static func address(for key: MemoryKey) -> Address {
        let segments = normalizeKeyText(key.rawValue).split(separator: "/").map(String.init)
        guard let last = segments.last else { return Address(namespace: "k", key: "empty") }
        let leading = segments.dropLast()
        let namespace = (["k"] + leading).joined(separator: ".")
        return Address(namespace: namespace, key: last)
    }

    public static func keyText(for item: ContinuityCore.MemoryItem) -> String {
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
