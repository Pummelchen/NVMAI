import Foundation

/// One piece of what the system currently believes about a task.
///
/// Addressed by namespace and key, which the application chooses:
/// `plot.missing_brother` for a novel, `decision.use_actor` for a codebase,
/// `constraint.marcus.photo_knowledge` for either. The engine stays generic
/// by never interpreting the namespace, only isolating and indexing by it.
public struct MemoryItem: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let taskID: UUID
    public var namespace: String
    public var key: String
    public var value: String
    /// Increments on every write to this address. Version 1 is the first.
    public var version: Int
    public let createdAt: Date
    public var updatedAt: Date
    public var provenance: Provenance?
    public var status: MemoryStatus
    /// Assembler ranking hint, 0...1. Absent means ordinary.
    public var importance: Double?
    /// How sure the writer was, 0...1. Reported on retrieval; nothing filters
    /// on it, because a low-confidence memory is still evidence.
    public var confidence: Double?
    public var tags: [String]
    /// Other addresses this one depends on, as "namespace.key". The assembler
    /// pulls these in with the item so a decision never arrives without the
    /// constraint that produced it.
    public var dependencies: [String]

    public init(id: UUID = UUID(),
                taskID: UUID,
                namespace: String,
                key: String,
                value: String,
                version: Int = 1,
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                provenance: Provenance? = nil,
                status: MemoryStatus = .active,
                importance: Double? = nil,
                confidence: Double? = nil,
                tags: [String] = [],
                dependencies: [String] = []) {
        self.id = id
        self.taskID = taskID
        self.namespace = namespace
        self.key = key
        self.value = value
        self.version = version
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.provenance = provenance
        self.status = status
        self.importance = importance.map { min(max($0, 0), 1) }
        self.confidence = confidence.map { min(max($0, 0), 1) }
        self.tags = tags
        self.dependencies = dependencies
    }

    /// The address, as the model and the logs spell it.
    public var address: String { "\(namespace).\(key)" }

    /// Roughly what this costs in a prompt. Four characters per token is the
    /// usual English approximation and is close enough to budget with; the
    /// assembler needs a bound, not a tokenizer.
    public var estimatedTokens: Int {
        max(1, (address.count + value.count) / 4 + 2)
    }

    /// What this occupies, for the store's byte budget.
    ///
    /// Counted rather than estimated from a worst case, because a budget
    /// derived from the largest permitted value is not a budget: with a 64 KiB
    /// value cap and 200-byte facts it would refuse writes at a fraction of a
    /// percent of the memory it claims to allow.
    ///
    /// `MemoryItem.overheadBytes` covers the identifiers, dates, status and
    /// the dictionary entry holding it. It does not have to be exact; it has
    /// to be an honest constant rather than a multiplier hiding a guess.
    public var storageBytes: Int {
        var total = MemoryItem.overheadBytes
        total += namespace.utf8.count + key.utf8.count + value.utf8.count
        for tag in tags { total += tag.utf8.count + 8 }
        for dependency in dependencies { total += dependency.utf8.count + 8 }
        return total
    }

    /// Two UUIDs, three dates, a version, two optional doubles, a status, and
    /// the dictionary slot. Rounded up.
    public static let overheadBytes = 160
}

public extension MemoryVersion {
    /// What a retained version occupies. Versions are charged to the same
    /// budget as live items: history that is never accounted for is history
    /// that grows until something else fails.
    var storageBytes: Int {
        MemoryItem.overheadBytes + namespace.utf8.count + key.utf8.count + value.utf8.count
    }
}

public enum MemoryStatus: String, Codable, Sendable, CaseIterable {
    /// The current belief at this address.
    case active
    /// Replaced by a later version. Kept, because the question "what did the
    /// system believe when it wrote that" has no answer if old versions are
    /// destroyed.
    case superseded
    /// Two sources disagree and nothing has resolved it. Surfaced to the
    /// model rather than silently picking a winner.
    case disputed
    /// Deliberately retired. Not deleted, and not offered as context.
    case archived

    /// Whether the assembler may put this in a prompt.
    public var isEligibleForContext: Bool {
        self == .active || self == .disputed
    }
}

/// One historical version of an address.
///
/// Stored apart from the live item so reading current state never pays for
/// history, which is the common case by a wide margin.
public struct MemoryVersion: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let itemID: UUID
    public let taskID: UUID
    public let namespace: String
    public let key: String
    public let value: String
    public let version: Int
    public let recordedAt: Date
    public let provenance: Provenance?
    /// What this version became when it stopped being current.
    public let status: MemoryStatus

    public init(id: UUID = UUID(),
                itemID: UUID,
                taskID: UUID,
                namespace: String,
                key: String,
                value: String,
                version: Int,
                recordedAt: Date = Date(),
                provenance: Provenance? = nil,
                status: MemoryStatus = .superseded) {
        self.id = id
        self.itemID = itemID
        self.taskID = taskID
        self.namespace = namespace
        self.key = key
        self.value = value
        self.version = version
        self.recordedAt = recordedAt
        self.provenance = provenance
        self.status = status
    }
}
