import Foundation
import NVMAI
import NVMAIMemory

/// Bridges the memory subsystem to the server's own types.
///
/// `NVMAIMemory` knows nothing about chat messages, tool definitions or
/// requests, and the serving code knows nothing about Valkey. Everything that
/// has to speak both lives here: deriving a session identity from a
/// conversation, turning memory tools into the tokenizer's function
/// definitions, and turning a model's tool call into a memory operation and
/// its result back into a message.
enum ServerMemory {
    /// A stable session id for a conversation.
    ///
    /// The API is stateless and clients send the whole history each turn, so
    /// there is no session id to read. The first user message plus the
    /// workspace identifies a conversation well enough to keep one session's
    /// memory continuous across its turns, and it changes when a new
    /// conversation starts, which is when a new session should begin.
    static func sessionIdentifier(messages: [GFTokenizer.Message],
                                  workspace: String) -> String {
        let seed = messages.first { $0.role == .user }?.content ?? ""
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in (workspace + "\u{0}" + seed).utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(format: "s-%016llx", hash)
    }

    /// True when this request is the opening turn of a conversation, which is
    /// where the bootstrap belongs. Later turns reuse the session.
    static func isFirstTurn(_ messages: [GFTokenizer.Message]) -> Bool {
        !messages.contains { $0.role == .assistant }
    }

    /// Memory tools in the tokenizer's own definition type.
    static func functionDefinitions(_ definitions: [MemoryToolDefinition])
        -> [GFTokenizer.FunctionDefinition] {
        definitions.map { definition in
            GFTokenizer.FunctionDefinition(
                name: definition.name,
                description: definition.description,
                parameters: jsonValue(from: definition.parameters.jsonObject))
        }
    }

    /// Adds memory tools to a request's own, without displacing them.
    ///
    /// A client's tool of the same name wins: the client executes its tools
    /// and we execute ours, and two definitions of one name would make the
    /// model's call ambiguous.
    static func merging(tools: [GFTokenizer.FunctionDefinition],
                        memory: [GFTokenizer.FunctionDefinition])
        -> [GFTokenizer.FunctionDefinition] {
        let existing = Set(tools.map(\.name))
        return tools + memory.filter { !existing.contains($0.name) }
    }

    /// Converts a parsed tool call's arguments into memory tool values.
    static func arguments(from json: JSONValue) -> [String: MemoryToolValue] {
        guard case .object(let fields) = json else { return [:] }
        var result: [String: MemoryToolValue] = [:]
        for (name, value) in fields {
            result[name] = toolValue(from: value)
        }
        return result
    }

    private static func toolValue(from json: JSONValue) -> MemoryToolValue {
        switch json {
        case .string(let text): return .string(text)
        case .number(let number): return .number(number)
        case .integer(let number): return .number(Double(number))
        case .unsignedInteger(let number): return .number(Double(number))
        case .decimal(let number): return .number(NSDecimalNumber(decimal: number).doubleValue)
        case .bool(let flag): return .bool(flag)
        case .array(let items):
            return .stringArray(items.compactMap { item in
                if case .string(let text) = item { return text }
                return nil
            })
        case .null: return .null
        case .object:
            // Objects are not a memory argument type; rendering it back to
            // text keeps a malformed call debuggable instead of silent.
            return .string(String(describing: json))
        }
    }

    /// The assistant turn that made a set of tool calls, as history.
    static func assistantMessage(content: String, calls: [ParsedToolCall])
        -> GFTokenizer.Message {
        GFTokenizer.Message(
            role: .assistant,
            content: content.isEmpty ? nil : content,
            toolCalls: calls.map {
                GFTokenizer.HistoricalToolCall(id: $0.id, name: $0.name, arguments: $0.arguments)
            })
    }

    /// One tool result, as the message the model reads next.
    static func toolResultMessage(call: ParsedToolCall,
                                  result: MemoryToolResult) -> GFTokenizer.Message {
        GFTokenizer.Message(role: .tool,
                            content: result.jsonString(),
                            toolCallID: call.id,
                            name: call.name)
    }

    private static func jsonValue(from object: Any) -> JSONValue {
        switch object {
        case let dictionary as [String: Any]:
            var mapped: [String: JSONValue] = [:]
            for (key, value) in dictionary { mapped[key] = jsonValue(from: value) }
            return .object(mapped)
        case let array as [Any]:
            return .array(array.map(jsonValue(from:)))
        case let text as String:
            return .string(text)
        case let flag as Bool:
            return .bool(flag)
        case let number as Int:
            return .integer(Int64(number))
        case let number as Double:
            return .number(number)
        default:
            return .null
        }
    }
}
