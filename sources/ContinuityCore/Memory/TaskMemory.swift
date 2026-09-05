import Foundation

/// The semantic state of every task, isolated by task identifier.
///
/// This is the store the model reads and writes through. It is deliberately
/// small: an address, a value, a version chain and provenance. It holds no
/// transcript, because transcripts belong in `SessionLog`, and mixing the two
/// is how a memory system turns into a second, worse copy of the chat history.
///
/// Writes never destroy. Replacing a value pushes the old one into the version
/// chain, so "what did it believe in chapter 40" stays answerable in chapter
/// 100.
public actor TaskMemory {
    private var items: [UUID: [String: MemoryItem]] = [:]
    private var versions: [UUID: [String: [MemoryVersion]]] = [:]
    private let limits: MemoryLimits
    /// Notified after every mutation so the engine can journal it without
    /// this actor knowing that persistence exists.
    private var observer: (@Sendable (MemoryMutation) async -> Void)?

    public init(limits: MemoryLimits = .default) {
        self.limits = limits
    }

    public func setObserver(_ observer: (@Sendable (MemoryMutation) async -> Void)?) {
        self.observer = observer
    }

    // MARK: - Writing

    /// Record a belief at an address.
    ///
    /// - Parameter expectedVersion: when given, the write fails unless the
    ///   address is at that version. Two sessions working the same task will
    ///   otherwise overwrite each other with no trace, and the loser never
    ///   learns it lost.
    @discardableResult
    public func write(taskID: UUID,
                      namespace: String,
                      key: String,
                      value: String,
                      provenance: Provenance? = nil,
                      importance: Double? = nil,
                      tags: [String]? = nil,
                      dependencies: [String]? = nil,
                      expectedVersion: Int? = nil,
                      now: Date = Date()) async throws -> MemoryWriteResult {
        try MemoryAddressValidator.validateNamespace(namespace, limits: limits)
        try MemoryAddressValidator.validateKey(key, limits: limits)
        let byteCount = value.utf8.count
        guard byteCount <= limits.maxValueBytes else {
            throw ContinuityError.valueTooLarge(bytes: byteCount,
                                                limit: limits.maxValueBytes)
        }

        let address = "\(namespace).\(key)"
        var taskItems = items[taskID] ?? [:]
        let existing = taskItems[address]

        if let expected = expectedVersion {
            let actual = existing?.version ?? 0
            guard actual == expected else {
                throw ContinuityError.versionConflict(namespace: namespace, key: key,
                                                      expected: expected, actual: actual)
            }
        }

        if existing == nil && taskItems.count >= limits.maxItemsPerTask {
            throw ContinuityError.valueTooLarge(bytes: taskItems.count,
                                                limit: limits.maxItemsPerTask)
        }

        let result: MemoryWriteResult
        var archived: MemoryVersion?
        if var item = existing {
            archived = archiveVersion(of: item, taskID: taskID, status: .superseded)
            item.value = value
            item.version += 1
            item.updatedAt = now
            item.status = .active
            item.provenance = provenance ?? item.provenance
            if let importance { item.importance = min(max(importance, 0), 1) }
            if let tags { item.tags = tags }
            if let dependencies { item.dependencies = dependencies }
            taskItems[address] = item
            result = MemoryWriteResult(item: item, previousVersion: item.version - 1)
        } else {
            let item = MemoryItem(taskID: taskID,
                                  namespace: namespace,
                                  key: key,
                                  value: value,
                                  version: 1,
                                  createdAt: now,
                                  updatedAt: now,
                                  provenance: provenance,
                                  status: .active,
                                  importance: importance,
                                  tags: tags ?? [],
                                  dependencies: dependencies ?? [])
            taskItems[address] = item
            result = MemoryWriteResult(item: item, previousVersion: nil)
        }
        items[taskID] = taskItems
        if let archived { await notify(.versioned(archived)) }
        await notify(.written(result))
        return result
    }

    /// Move an address to a new status without changing its value.
    ///
    /// Used for archiving and for marking a contradiction. This does not add
    /// to the version chain: the chain is the history of what an address
    /// *said*, and a status change says nothing new. Archiving one here would
    /// put two entries with the same version number in the history, which
    /// reads as a second version that never happened. The transition is
    /// visible in `updatedAt` and in the session log instead.
    @discardableResult
    public func setStatus(taskID: UUID,
                          namespace: String,
                          key: String,
                          status: MemoryStatus,
                          provenance: Provenance? = nil,
                          now: Date = Date()) async throws -> MemoryItem {
        let address = "\(namespace).\(key)"
        guard var item = items[taskID]?[address] else {
            throw ContinuityError.unknownMemoryItem(namespace: namespace, key: key)
        }
        guard item.status != status else { return item }
        item.status = status
        item.updatedAt = now
        if let provenance { item.provenance = provenance }
        items[taskID]?[address] = item
        await notify(.statusChanged(item))
        return item
    }

    /// Retire an address. Not a delete: the value and its history stay, and
    /// the assembler stops offering it.
    @discardableResult
    public func archive(taskID: UUID,
                        namespace: String,
                        key: String,
                        provenance: Provenance? = nil,
                        now: Date = Date()) async throws -> MemoryItem {
        try await setStatus(taskID: taskID, namespace: namespace, key: key,
                      status: .archived, provenance: provenance, now: now)
    }

    // MARK: - Reading

    public func item(taskID: UUID, namespace: String, key: String) -> MemoryItem? {
        items[taskID]?["\(namespace).\(key)"]
    }

    public func value(taskID: UUID, namespace: String, key: String) -> String? {
        item(taskID: taskID, namespace: namespace, key: key)?.value
    }

    public func query(taskID: UUID, _ query: MemoryQuery = .active) -> [MemoryItem] {
        let all = items[taskID].map { Array($0.values) } ?? []
        var matched = all.filter { query.matches($0) }
        matched.sort { lhs, rhs in
            switch query.order {
            case .relevance:
                let left = lhs.importance ?? 0.5
                let right = rhs.importance ?? 0.5
                if left != right { return left > right }
                if lhs.updatedAt != rhs.updatedAt { return lhs.updatedAt > rhs.updatedAt }
                return lhs.address < rhs.address
            case .recency:
                if lhs.updatedAt != rhs.updatedAt { return lhs.updatedAt > rhs.updatedAt }
                return lhs.address < rhs.address
            case .address:
                return lhs.address < rhs.address
            }
        }
        if let limit = query.limit, matched.count > limit {
            matched = Array(matched.prefix(limit))
        }
        return matched
    }

    /// Every version of an address, oldest first, with the live item last.
    public func history(taskID: UUID, namespace: String, key: String) -> [MemoryVersion] {
        let address = "\(namespace).\(key)"
        var chain = versions[taskID]?[address] ?? []
        chain.sort { $0.version < $1.version }
        if let live = items[taskID]?[address] {
            chain.append(MemoryVersion(itemID: live.id,
                                       taskID: taskID,
                                       namespace: live.namespace,
                                       key: live.key,
                                       value: live.value,
                                       version: live.version,
                                       recordedAt: live.updatedAt,
                                       provenance: live.provenance,
                                       status: live.status))
        }
        return chain
    }

    /// Items the given items depend on, transitively, excluding the ones
    /// already present. The assembler uses this so a decision never arrives
    /// without the constraint behind it.
    public func dependencies(taskID: UUID,
                             of seeds: [MemoryItem],
                             maximumDepth: Int = 3) -> [MemoryItem] {
        guard let taskItems = items[taskID] else { return [] }
        var seen = Set(seeds.map(\.address))
        var frontier = seeds
        var collected: [MemoryItem] = []
        var depth = 0
        while !frontier.isEmpty && depth < maximumDepth {
            var next: [MemoryItem] = []
            for item in frontier {
                for address in item.dependencies where !seen.contains(address) {
                    seen.insert(address)
                    guard let resolved = taskItems[address],
                          resolved.status.isEligibleForContext else { continue }
                    collected.append(resolved)
                    next.append(resolved)
                }
            }
            frontier = next
            depth += 1
        }
        return collected
    }

    public func taskIdentifiers() -> [UUID] { Array(items.keys) }

    public func count(taskID: UUID) -> Int { items[taskID]?.count ?? 0 }

    // MARK: - Snapshot and restore

    /// The whole store, for persistence. Values are already `Sendable` and
    /// `Codable`, so the journal never needs to know these types' internals.
    public func snapshot() -> MemorySnapshot {
        MemorySnapshot(items: items.values.flatMap { Array($0.values) },
                       versions: versions.values.flatMap { $0.values.flatMap { $0 } })
    }

    /// Replace everything. Used on startup after a journal replay, never
    /// while a task is live.
    public func restore(_ snapshot: MemorySnapshot) {
        items = [:]
        versions = [:]
        for item in snapshot.items {
            items[item.taskID, default: [:]][item.address] = item
        }
        for version in snapshot.versions {
            let address = "\(version.namespace).\(version.key)"
            versions[version.taskID, default: [:]][address, default: []].append(version)
        }
    }

    /// Drop a task entirely. The one true delete in the API, because a user
    /// who abandons a project is entitled to have it gone.
    public func forget(taskID: UUID) {
        items[taskID] = nil
        versions[taskID] = nil
    }

    // MARK: - Internals

    @discardableResult
    private func archiveVersion(of item: MemoryItem, taskID: UUID,
                                status: MemoryStatus) -> MemoryVersion {
        let version = MemoryVersion(itemID: item.id,
                                    taskID: taskID,
                                    namespace: item.namespace,
                                    key: item.key,
                                    value: item.value,
                                    version: item.version,
                                    recordedAt: item.updatedAt,
                                    provenance: item.provenance,
                                    status: status)
        var chain = versions[taskID]?[item.address] ?? []
        chain.append(version)
        if chain.count > limits.maxVersionsPerAddress {
            chain.removeFirst(chain.count - limits.maxVersionsPerAddress)
        }
        versions[taskID, default: [:]][item.address] = chain
        return version
    }

    private func notify(_ mutation: MemoryMutation) async {
        guard let observer else { return }
        await observer(mutation)
    }
}

/// What changed, for observers.
public enum MemoryMutation: Sendable, Equatable {
    /// A value that has just been displaced. Emitted before the write that
    /// displaced it, so a journal replays the chain in the order it happened.
    case versioned(MemoryVersion)
    case written(MemoryWriteResult)
    case statusChanged(MemoryItem)
}

/// The store's full contents.
public struct MemorySnapshot: Codable, Sendable, Equatable {
    public var items: [MemoryItem]
    public var versions: [MemoryVersion]

    public init(items: [MemoryItem] = [], versions: [MemoryVersion] = []) {
        self.items = items
        self.versions = versions
    }
}
