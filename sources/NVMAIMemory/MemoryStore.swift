import Foundation

/// Persistent agent memory: durable facts a model writes in one session and
/// reads in another.
///
/// This is deliberately not the KV cache. The KV cache is per-request model
/// state that the runtime owns; this is a small, model-authored store of
/// things worth keeping after the conversation ends, and nothing in the
/// serving path depends on it being present.
///
/// Everything above this protocol works in terms of `MemoryRecord` and
/// `MemoryScope`. No caller outside `NVMAIMemory` issues a database command,
/// so the backend can be replaced without touching the server.
public protocol MemoryStore: Sendable {
    /// One record, or nil when the key is absent from this scope.
    func get(_ key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord?
    /// Writes a record, replacing any existing value for the key.
    func set(_ record: MemoryRecord, in scope: MemoryScope) async throws
    /// Removes a key. Returns whether something was removed.
    @discardableResult
    func delete(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool
    func exists(_ key: MemoryKey, in scope: MemoryScope) async throws -> Bool
    /// Keys under a prefix, newest first, bounded by `limit`.
    func list(prefix: String, limit: Int, in scope: MemoryScope) async throws -> [MemoryKey]
    /// Records matching a query, ranked by the backend's own strategy.
    func search(_ query: MemoryQuery, in scope: MemoryScope) async throws -> [MemoryRecord]
    /// Appends a line to a record, creating it when absent. Returns the record
    /// as stored afterwards.
    @discardableResult
    func append(_ text: String, to key: MemoryKey, in scope: MemoryScope) async throws -> MemoryRecord
    /// Records the session and returns a bounded bootstrap: the few durable
    /// facts worth having before the first user message. Never the store.
    func sessionInit(_ session: MemorySession, in scope: MemoryScope) async throws -> MemoryBootstrap
}

/// A validated memory key: slash-separated segments of a small, safe
/// alphabet.
///
/// Keys come from the model, so they are parsed rather than trusted. The
/// rejected shapes are the ones that would otherwise escape a scope or make
/// a key that cannot be listed: absolute keys, `..`, empty segments, control
/// characters, and anything past `maximumLength`.
public struct MemoryKey: Hashable, Sendable, CustomStringConvertible, Codable {
    public static let maximumLength = 256
    public static let maximumSegments = 12

    public let rawValue: String
    public var description: String { rawValue }
    /// The key's leading segment, which is how memory is organised in
    /// practice ("decisions", "architecture", "tasks").
    public var category: String { rawValue.split(separator: "/").first.map(String.init) ?? "" }

    public init(validating raw: String) throws {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MemoryError.invalidKey(raw, "empty") }
        guard trimmed.count <= Self.maximumLength else {
            throw MemoryError.invalidKey(raw, "longer than \(Self.maximumLength) characters")
        }
        guard !trimmed.hasPrefix("/") else { throw MemoryError.invalidKey(raw, "must be relative") }
        let segments = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard segments.count <= Self.maximumSegments else {
            throw MemoryError.invalidKey(raw, "more than \(Self.maximumSegments) segments")
        }
        for segment in segments {
            guard !segment.isEmpty else { throw MemoryError.invalidKey(raw, "empty segment") }
            guard segment != "." && segment != ".." else {
                throw MemoryError.invalidKey(raw, "relative segment '\(segment)'")
            }
            guard segment.allSatisfy(Self.isAllowed) else {
                throw MemoryError.invalidKey(raw, "segment '\(segment)' has unsupported characters")
            }
        }
        self.rawValue = trimmed
    }

    /// Letters, digits and `. _ -`, which covers how a model actually names
    /// things while excluding the separators the backend's own key grammar
    /// uses (`:`), whitespace, and anything non-printable.
    private static func isAllowed(_ character: Character) -> Bool {
        character.isLetter || character.isNumber || character == "." || character == "_"
            || character == "-"
    }
}

/// Where a record lives. Two scopes never see each other's keys.
///
/// `workspace` is the repository or project. `user` separates people sharing
/// one server. `namespace` separates whole deployments on one Valkey, so a
/// second NVMAI on the same machine cannot read the first one's memory.
public struct MemoryScope: Hashable, Sendable, Codable {
    public static let maximumComponentLength = 96

    public let namespace: String
    public let user: String
    public let workspace: String

    public init(namespace: String, user: String, workspace: String) throws {
        self.namespace = try Self.validate(namespace, "namespace")
        self.user = try Self.validate(user, "user")
        self.workspace = try Self.validate(workspace, "workspace")
    }

    /// Scope components reach us from a launch flag, an environment variable
    /// and an HTTP header, so they are sanitized on the same terms as keys:
    /// no separators, no traversal, bounded length.
    private static func validate(_ value: String, _ field: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MemoryError.invalidScope(field, "empty") }
        guard trimmed.count <= maximumComponentLength else {
            throw MemoryError.invalidScope(field, "longer than \(maximumComponentLength) characters")
        }
        guard trimmed != "." && trimmed != ".." else {
            throw MemoryError.invalidScope(field, "relative")
        }
        let allowed = trimmed.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "." || $0 == "_" || $0 == "-"
        }
        guard allowed else {
            throw MemoryError.invalidScope(field, "'\(trimmed)' has unsupported characters")
        }
        return trimmed
    }
}

/// A stored memory. `value` is arbitrary UTF-8; when the model writes JSON it
/// is preserved verbatim, so structure is the model's choice and not the
/// store's.
public struct MemoryRecord: Sendable, Codable, Equatable {
    public var key: MemoryKey
    public var value: String
    /// How much this matters, 0...1. Ranks the bootstrap set.
    public var importance: Double?
    /// How sure the writer was, 0...1. Retrieval reports it; nothing filters
    /// on it, because a low-confidence memory is still evidence.
    public var confidence: Double?
    public var tags: [String]
    /// The session that wrote the record, so a reader can tell how it got here.
    public var sourceSession: String?
    public var createdAt: Date
    public var updatedAt: Date

    public init(key: MemoryKey,
                value: String,
                importance: Double? = nil,
                confidence: Double? = nil,
                tags: [String] = [],
                sourceSession: String? = nil,
                createdAt: Date = Date(),
                updatedAt: Date = Date()) {
        self.key = key
        self.value = value
        self.importance = importance.map { min(max($0, 0), 1) }
        self.confidence = confidence.map { min(max($0, 0), 1) }
        self.tags = tags
        self.sourceSession = sourceSession
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

/// A retrieval request. Text matching is substring-and-token based today; the
/// shape is chosen so a semantic backend can answer the same query later
/// without the model-facing tools changing.
public struct MemoryQuery: Sendable, Equatable {
    public var text: String?
    public var prefix: String?
    public var tags: [String]
    public var minimumImportance: Double?
    public var limit: Int

    public init(text: String? = nil,
                prefix: String? = nil,
                tags: [String] = [],
                minimumImportance: Double? = nil,
                limit: Int = 10) {
        self.text = text
        self.prefix = prefix
        self.tags = tags
        self.minimumImportance = minimumImportance
        self.limit = limit
    }
}

/// What a session is told about itself.
public struct MemorySession: Sendable, Equatable, Codable {
    public let id: String
    public let startedAt: Date
    public let modelID: String?

    public init(id: String, startedAt: Date = Date(), modelID: String? = nil) {
        self.id = id
        self.startedAt = startedAt
        self.modelID = modelID
    }
}

/// The bounded set handed to a session before its first user message.
///
/// Bounded twice, by record count and by total bytes, because the failure
/// this guards against is a store that has grown for months quietly eating
/// the context window.
public struct MemoryBootstrap: Sendable, Equatable {
    public let records: [MemoryRecord]
    /// Records that matched but were dropped by the limits, so the prompt can
    /// say memory exists beyond what it shows.
    public let omittedCount: Int
    public let totalBytes: Int

    public init(records: [MemoryRecord], omittedCount: Int, totalBytes: Int) {
        self.records = records
        self.omittedCount = omittedCount
        self.totalBytes = totalBytes
    }

    public static let empty = MemoryBootstrap(records: [], omittedCount: 0, totalBytes: 0)
}

public enum MemoryError: Error, Equatable, CustomStringConvertible {
    case invalidKey(String, String)
    case invalidScope(String, String)
    case valueTooLarge(bytes: Int, limit: Int)
    case backendUnavailable(String)
    case timedOut(operation: String, milliseconds: Int)
    case disabled

    public var description: String {
        switch self {
        case .invalidKey(let key, let why): return "invalid memory key '\(key)': \(why)"
        case .invalidScope(let field, let why): return "invalid memory \(field): \(why)"
        case .valueTooLarge(let bytes, let limit):
            return "memory value is \(bytes) bytes; the limit is \(limit)"
        case .backendUnavailable(let detail): return "memory backend unavailable: \(detail)"
        case .timedOut(let operation, let ms): return "memory \(operation) timed out after \(ms) ms"
        case .disabled: return "memory is disabled"
        }
    }

    /// Whether the failure is the backend's rather than the caller's. These
    /// are the ones the server degrades on instead of surfacing as a tool
    /// error, because the model cannot fix them.
    public var isBackendFailure: Bool {
        switch self {
        case .backendUnavailable, .timedOut, .disabled: return true
        case .invalidKey, .invalidScope, .valueTooLarge: return false
        }
    }
}
