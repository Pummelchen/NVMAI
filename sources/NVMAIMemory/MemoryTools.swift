import Foundation

/// The model-facing memory API: six tools, described the way the model will
/// see them, and one executor that runs a call against a `MemoryStore`.
///
/// The model never gets a database command. It gets logical operations on a
/// scope it does not choose: the scope is resolved by the server from the
/// launch configuration and the request, and passed in here. A call cannot
/// name another workspace, and a key that would escape the scope fails to
/// parse before any backend sees it.
public enum MemoryTools {
    public static let names: Set<String> = [
        "memory_get", "memory_set", "memory_delete",
        "memory_search", "memory_list", "memory_append",
    ]

    public static func isMemoryTool(_ name: String) -> Bool { names.contains(name) }

    /// Tool definitions in the JSON-schema shape the chat templates use.
    ///
    /// Descriptions are part of the behaviour: they are where the model reads
    /// what a key should look like and when a write is worth making.
    public static func definitions() -> [MemoryToolDefinition] {
        [
            MemoryToolDefinition(
                name: "memory_search",
                description: "Search durable project memory. Use before changing an unfamiliar "
                    + "area or when the user refers to an earlier decision.",
                parameters: .object([
                    "query": .string("What to look for, in plain words."),
                    "prefix": .string("Optional key prefix, for example 'decisions/'."),
                    "tags": .stringArray("Optional tags to require."),
                    "limit": .integer("Maximum results, default 10."),
                ], required: [])),
            MemoryToolDefinition(
                name: "memory_get",
                description: "Read one memory by exact key.",
                parameters: .object(["key": .string("The key, for example 'decisions/sync'.")],
                                    required: ["key"])),
            MemoryToolDefinition(
                name: "memory_list",
                description: "List memory keys, optionally under a prefix. Cheap way to see what "
                    + "is known before fetching anything.",
                parameters: .object([
                    "prefix": .string("Optional key prefix."),
                    "limit": .integer("Maximum keys, default 50."),
                ], required: [])),
            MemoryToolDefinition(
                name: "memory_set",
                description: "Write or replace a durable memory. Use for decisions, conventions, "
                    + "constraints, failed approaches and project state. Not for conversation, "
                    + "reasoning, secrets or source code. Update an existing key rather than "
                    + "creating a near-duplicate.",
                parameters: .object([
                    "key": .string("Topic-shaped key, for example 'decisions/sync'."),
                    "value": .string("The fact, stated so it is useful months from now."),
                    "importance": .number("0 to 1; how much this should surface first later."),
                    "confidence": .number("0 to 1; how sure you are."),
                    "tags": .stringArray("Optional tags."),
                ], required: ["key", "value"])),
            MemoryToolDefinition(
                name: "memory_append",
                description: "Add a line to an existing memory, creating it if absent.",
                parameters: .object([
                    "key": .string("The key to extend."),
                    "value": .string("The line to add."),
                ], required: ["key", "value"])),
            MemoryToolDefinition(
                name: "memory_delete",
                description: "Delete a memory that is wrong or obsolete. Prefer updating.",
                parameters: .object(["key": .string("The key to delete.")], required: ["key"])),
        ]
    }

    /// Runs one call and returns what the model should see.
    ///
    /// Failures are reported, never hidden: a write that did not reach the
    /// store comes back as an error so the model does not record a fact it
    /// believes is saved. That is the difference between a memory system and
    /// one that lies.
    public static func execute(name: String,
                               arguments: [String: MemoryToolValue],
                               store: any MemoryStore,
                               scope: MemoryScope,
                               session: MemorySession,
                               limits: MemoryLimits) async -> MemoryToolResult {
        do {
            switch name {
            case "memory_get":
                let key = try key(from: arguments)
                guard let record = try await store.get(key, in: scope) else {
                    return .ok(["found": .bool(false), "key": .string(key.rawValue)])
                }
                return .ok(["found": .bool(true), "record": .record(record)])

            case "memory_set":
                let key = try key(from: arguments)
                guard let value = arguments["value"]?.stringValue else {
                    throw MemoryError.invalidKey(key.rawValue, "missing 'value'")
                }
                let record = MemoryRecord(
                    key: key,
                    value: value,
                    importance: arguments["importance"]?.doubleValue,
                    confidence: arguments["confidence"]?.doubleValue,
                    tags: arguments["tags"]?.stringArrayValue ?? [],
                    sourceSession: session.id)
                try await store.set(record, in: scope)
                return .ok(["stored": .bool(true), "key": .string(key.rawValue)])

            case "memory_append":
                let key = try key(from: arguments)
                guard let value = arguments["value"]?.stringValue else {
                    throw MemoryError.invalidKey(key.rawValue, "missing 'value'")
                }
                let record = try await store.append(value, to: key, in: scope)
                return .ok(["stored": .bool(true), "key": .string(key.rawValue),
                            "bytes": .int(record.value.utf8.count)])

            case "memory_delete":
                let key = try key(from: arguments)
                let removed = try await store.delete(key, in: scope)
                return .ok(["deleted": .bool(removed), "key": .string(key.rawValue)])

            case "memory_list":
                let prefix = arguments["prefix"]?.stringValue ?? ""
                let limit = arguments["limit"]?.intValue ?? 50
                let keys = try await store.list(prefix: prefix, limit: limit, in: scope)
                return .ok(["keys": .stringArray(keys.map(\.rawValue))])

            case "memory_search":
                let query = MemoryQuery(
                    text: arguments["query"]?.stringValue,
                    prefix: arguments["prefix"]?.stringValue,
                    tags: arguments["tags"]?.stringArrayValue ?? [],
                    minimumImportance: arguments["min_importance"]?.doubleValue,
                    limit: arguments["limit"]?.intValue ?? 10)
                let records = try await store.search(query, in: scope)
                return .ok(["results": .records(records)])

            default:
                return .failure("unknown memory tool '\(name)'")
            }
        } catch let error as MemoryError {
            return .failure(error.description)
        } catch {
            return .failure("memory operation failed: \(error)")
        }
    }

    private static func key(from arguments: [String: MemoryToolValue]) throws -> MemoryKey {
        guard let raw = arguments["key"]?.stringValue else {
            throw MemoryError.invalidKey("", "missing 'key'")
        }
        return try MemoryKey(validating: raw)
    }
}

/// A tool definition, in the shape the server converts to its own function
/// definition type. Kept free of any server type so this module stays
/// independent of the engine.
public struct MemoryToolDefinition: Sendable, Equatable {
    public let name: String
    public let description: String
    public let parameters: MemoryToolSchema

    public init(name: String, description: String, parameters: MemoryToolSchema) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

public indirect enum MemoryToolSchema: Sendable, Equatable {
    case string(String)
    case number(String)
    case integer(String)
    case stringArray(String)
    case object([String: MemoryToolSchema], required: [String])

    /// The JSON Schema object a chat template expects.
    public var jsonObject: [String: Any] {
        switch self {
        case .string(let description):
            return ["type": "string", "description": description]
        case .number(let description):
            return ["type": "number", "description": description]
        case .integer(let description):
            return ["type": "integer", "description": description]
        case .stringArray(let description):
            return ["type": "array", "items": ["type": "string"], "description": description]
        case .object(let properties, let required):
            var mapped: [String: Any] = [:]
            for (name, schema) in properties { mapped[name] = schema.jsonObject }
            return ["type": "object", "properties": mapped, "required": required]
        }
    }
}

/// A tool argument as parsed from the model's call.
public enum MemoryToolValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case stringArray([String])
    case null

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var doubleValue: Double? {
        switch self {
        case .number(let value): return value
        case .string(let value): return Double(value)
        default: return nil
        }
    }

    public var intValue: Int? { doubleValue.map(Int.init) }

    public var stringArrayValue: [String]? {
        switch self {
        case .stringArray(let values): return values
        case .string(let value): return [value]
        default: return nil
        }
    }
}

/// What a tool call produced, ready to be rendered as the tool message the
/// model sees next.
public enum MemoryToolResult: Sendable, Equatable {
    case ok([String: MemoryToolOutput])
    case failure(String)

    public var isFailure: Bool {
        if case .failure = self { return true }
        return false
    }

    /// Compact JSON. Records are rendered with their metadata so the model
    /// can judge staleness, and values are returned whole: a truncated
    /// memory would be worse than none.
    public func jsonString() -> String {
        switch self {
        case .failure(let message):
            return jsonEncode(["ok": false, "error": message])
        case .ok(let fields):
            var object: [String: Any] = ["ok": true]
            for (key, value) in fields { object[key] = value.jsonValue }
            return jsonEncode(object)
        }
    }

    private func jsonEncode(_ object: [String: Any]) -> String {
        // Slashes are not escaped: every memory key contains them, and
        // `decisions\/sync` in a tool result is what the model reads back.
        guard let data = try? JSONSerialization.data(withJSONObject: object,
                                                     options: [.sortedKeys,
                                                               .withoutEscapingSlashes]),
              let text = String(data: data, encoding: .utf8) else {
            return "{\"ok\":false,\"error\":\"result is not encodable\"}"
        }
        return text
    }
}

public enum MemoryToolOutput: Sendable, Equatable {
    case bool(Bool)
    case int(Int)
    case string(String)
    case stringArray([String])
    case record(MemoryRecord)
    case records([MemoryRecord])

    var jsonValue: Any {
        switch self {
        case .bool(let value): return value
        case .int(let value): return value
        case .string(let value): return value
        case .stringArray(let values): return values
        case .record(let record): return Self.encode(record)
        case .records(let records): return records.map(Self.encode)
        }
    }

    private static func encode(_ record: MemoryRecord) -> [String: Any] {
        var object: [String: Any] = [
            "key": record.key.rawValue,
            "value": record.value,
            "updated_at": ISO8601DateFormatter().string(from: record.updatedAt),
        ]
        if let importance = record.importance { object["importance"] = importance }
        if let confidence = record.confidence { object["confidence"] = confidence }
        if !record.tags.isEmpty { object["tags"] = record.tags }
        if let session = record.sourceSession { object["source_session"] = session }
        return object
    }
}
