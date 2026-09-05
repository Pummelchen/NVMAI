import Foundation

/// One period of interaction with a model.
///
/// A session is not the task. The task is the long-running objective, which
/// may span months and a hundred sessions; the session is one sitting. The
/// distinction is the point of the whole engine: continuity is what survives
/// the end of a session, so conflating the two removes the thing being built.
public struct Session: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public let taskID: UUID
    public let startedAt: Date
    public var endedAt: Date?
    /// The model this session ran against, when the caller knows it.
    public let model: String?
    /// The caller's own name for this session, when it has one.
    ///
    /// An integration usually already has a conversation identifier and needs
    /// to find its session again after a restart. Without somewhere to put
    /// it, every such caller has to keep a side table that the journal cannot
    /// rebuild.
    public let externalID: String?
    /// A short human label for what this session was about, when the
    /// caller could tell: a project name, a client's working directory.
    /// It is a label, not the isolation boundary -- that is the task -- but
    /// it is what a person reading the log later uses to tell a book
    /// session from a coding session at a glance.
    public let tag: String?

    public init(id: UUID = UUID(),
                taskID: UUID,
                startedAt: Date = Date(),
                endedAt: Date? = nil,
                model: String? = nil,
                externalID: String? = nil,
                tag: String? = nil) {
        self.id = id
        self.taskID = taskID
        self.startedAt = startedAt
        self.endedAt = endedAt
        self.model = model
        self.externalID = externalID
        self.tag = tag
    }

    public var isOpen: Bool { endedAt == nil }
}

/// The long-running objective a set of sessions serves.
///
/// Deliberately thin: a title and an objective the assembler always puts
/// first. Everything else about a task is semantic state, which belongs in
/// `TaskMemory` where it can be versioned and superseded, not in a struct
/// that would have to grow a field per kind of task.
public struct ContinuityTask: Identifiable, Codable, Sendable, Equatable {
    public let id: UUID
    public var title: String
    /// What the task is currently trying to achieve. The one piece of state
    /// the context assembler will not drop, whatever the budget.
    public var objective: String
    public let createdAt: Date
    public var updatedAt: Date

    public init(id: UUID = UUID(),
                title: String,
                objective: String = "",
                createdAt: Date = Date(),
                updatedAt: Date = Date()) {
        self.id = id
        self.title = title
        self.objective = objective
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public enum ContinuityError: Error, Equatable, CustomStringConvertible {
    case unknownTask(UUID)
    case unknownSession(UUID)
    /// A session was used with a task it does not belong to. Never silently
    /// repaired: attaching events to the wrong task corrupts both.
    case sessionTaskMismatch(session: UUID, expected: UUID, actual: UUID)
    case sessionAlreadyEnded(UUID)
    case invalidNamespace(String, reason: String)
    case invalidKey(String, reason: String)
    case valueTooLarge(bytes: Int, limit: Int)
    /// The task's store is at its budget. Distinct from `valueTooLarge`
    /// because the caller can fix that one by writing less and cannot fix
    /// this one at all: something has to be archived or forgotten first.
    case storeFull(bytes: Int, limit: Int)
    case tooManyItems(count: Int, limit: Int)
    case unknownMemoryItem(namespace: String, key: String)
    case versionConflict(namespace: String, key: String, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case .unknownTask(let id): return "no such task \(id)"
        case .unknownSession(let id): return "no such session \(id)"
        case .sessionTaskMismatch(let session, let expected, let actual):
            return "session \(session) belongs to task \(actual), not \(expected)"
        case .sessionAlreadyEnded(let id): return "session \(id) has already ended"
        case .invalidNamespace(let value, let reason):
            return "invalid namespace '\(value)': \(reason)"
        case .invalidKey(let value, let reason): return "invalid key '\(value)': \(reason)"
        case .valueTooLarge(let bytes, let limit):
            return "value is \(bytes) bytes; the limit is \(limit)"
        case .storeFull(let bytes, let limit):
            return "the store holds \(bytes) bytes and the budget is \(limit); "
                + "archive or forget something first"
        case .tooManyItems(let count, let limit):
            return "the store holds \(count) items and the limit is \(limit)"
        case .unknownMemoryItem(let namespace, let key):
            return "no memory at \(namespace).\(key)"
        case .versionConflict(let namespace, let key, let expected, let actual):
            return "\(namespace).\(key) is at version \(actual), not \(expected)"
        }
    }
}
