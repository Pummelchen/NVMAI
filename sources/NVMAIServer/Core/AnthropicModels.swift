import Foundation
import NVMAI

// MARK: - Error envelope

/// The Anthropic Messages API error body: `{"type":"error","error":{...}}`.
public struct AnthropicErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let type: String
        public let message: String
    }

    public let type: String
    public let error: Detail
    public let requestID: String?

    enum CodingKeys: String, CodingKey {
        case type, error
        case requestID = "request_id"
    }

    public init(type: String, message: String, requestID: String? = nil) {
        self.type = "error"
        self.error = Detail(type: type, message: message)
        self.requestID = requestID
    }

    /// The Anthropic rendering of a server request error. The validator
    /// speaks OpenAI (message, param, code); the Anthropic API names the
    /// field inside the message, so the two are folded together here.
    public static func from(_ error: ServerRequestError, requestID: String? = nil) -> AnthropicErrorEnvelope {
        switch error {
        case .invalid(let message, let param, _):
            return AnthropicErrorEnvelope(
                type: "invalid_request_error",
                message: param.map { "\($0): \(message)" } ?? message,
                requestID: requestID)
        case .unknownModel:
            return AnthropicErrorEnvelope(
                type: "not_found_error",
                message: "model: requested model is not available on this server",
                requestID: requestID)
        case .queueFull:
            return AnthropicErrorEnvelope(
                type: "overloaded_error", message: "Overloaded", requestID: requestID)
        case .unsupportedOperation(let operation):
            return AnthropicErrorEnvelope(
                type: "api_error",
                message: "\(operation) is not supported by this backend",
                requestID: requestID)
        case .notFound(let message, _):
            return AnthropicErrorEnvelope(
                type: "not_found_error", message: message, requestID: requestID)
        }
    }

    /// HTTP status for an Anthropic error type.
    public var httpStatus: Int {
        switch error.type {
        case "invalid_request_error": 400
        case "authentication_error": 401
        case "permission_error": 403
        case "not_found_error": 404
        case "request_too_large": 413
        case "rate_limit_error": 429
        case "overloaded_error": 529
        default: 500
        }
    }
}

// MARK: - Request decoding

/// `POST /v1/messages`. Content is kept as `JSONValue` because the block
/// grammar is wide and mostly refused; the mapper reads what it honours and
/// names what it cannot.
public struct AnthropicMessagesRequest: Decodable, Sendable {
    public struct Message: Decodable, Sendable {
        public let role: String
        public let content: JSONValue
    }

    public let model: String
    public let messages: [Message]
    public let maxTokens: Int?
    public let system: JSONValue?
    public let metadata: JSONValue?
    public let stopSequences: [String]?
    public let stream: Bool?
    public let temperature: Float?
    public let topK: Int?
    public let topP: Float?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let thinking: JSONValue?
    public let serviceTier: String?
    public let outputConfig: JSONValue?
    public let outputFormat: JSONValue?
    public let container: JSONValue?
    public let mcpServers: JSONValue?
    public let contextManagement: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, metadata, stream, temperature, tools, thinking, container
        case maxTokens = "max_tokens"
        case stopSequences = "stop_sequences"
        case topK = "top_k"
        case topP = "top_p"
        case toolChoice = "tool_choice"
        case serviceTier = "service_tier"
        case outputConfig = "output_config"
        case outputFormat = "output_format"
        case mcpServers = "mcp_servers"
        case contextManagement = "context_management"
    }
}

/// `POST /v1/messages/count_tokens` accepts the same body without
/// `max_tokens`, `stream` and the sampling controls.
public struct AnthropicCountTokensRequest: Decodable, Sendable {
    public let model: String
    public let messages: [AnthropicMessagesRequest.Message]
    public let system: JSONValue?
    public let tools: [JSONValue]?
    public let toolChoice: JSONValue?
    public let thinking: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, messages, system, tools, thinking
        case toolChoice = "tool_choice"
    }
}

// MARK: - Messages -> chat mapping

public enum AnthropicMapper {
    private static func invalid(_ message: String, _ param: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: "invalid_value")
    }

    private static func unsupported(_ message: String, _ param: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: "unsupported_value")
    }

    /// The text of a `system` prompt: a string or text blocks.
    static func systemText(_ system: JSONValue?) throws -> String? {
        guard let system else { return nil }
        switch system {
        case .string(let text):
            return text
        case .array(let blocks):
            var out: [String] = []
            for (index, block) in blocks.enumerated() {
                guard case .object(let dict) = block,
                      case .string("text")? = dict["type"],
                      case .string(let text)? = dict["text"] else {
                    throw invalid("system blocks must be text blocks", "system.\(index)")
                }
                out.append(text)
            }
            return out.joined(separator: "\n\n")
        default:
            throw invalid("system must be a string or an array of text blocks", "system")
        }
    }

    /// A tool_result's content: a string, or text blocks (other block kinds
    /// inside a result are refused, as NVMAI cannot show the model an image).
    static func toolResultText(_ content: JSONValue?, param: String) throws -> String {
        guard let content else { return "" }
        switch content {
        case .string(let text):
            return text
        case .array(let blocks):
            var out: [String] = []
            for (index, block) in blocks.enumerated() {
                guard case .object(let dict) = block, case .string(let type)? = dict["type"] else {
                    throw invalid("content blocks must have a type", "\(param).content.\(index)")
                }
                guard type == "text", case .string(let text)? = dict["text"] else {
                    throw unsupported("\(type) blocks in tool results are not supported; this server is text-only",
                                      "\(param).content.\(index)")
                }
                out.append(text)
            }
            return out.joined(separator: "\n")
        default:
            throw invalid("tool_result content must be a string or an array of blocks", "\(param).content")
        }
    }

    /// One conversation message, in the order the chat template needs it:
    /// a user message with tool results becomes tool messages (then any text
    /// as a user message); an assistant message becomes one assistant message
    /// carrying its text and tool calls.
    static func chatMessages(for message: AnthropicMessagesRequest.Message,
                             index: Int) throws -> [OpenAIChatMessage] {
        let param = "messages.\(index)"
        guard message.role == "user" || message.role == "assistant" else {
            throw invalid("role must be user or assistant", "\(param).role")
        }
        switch message.content {
        case .string(let text):
            return [OpenAIChatMessage(role: message.role, content: .text(text),
                                      toolCalls: nil, toolCallID: nil, name: nil)]
        case .array(let blocks):
            var text: [String] = []
            var toolCalls: [OpenAIToolCall] = []
            var toolResults: [OpenAIChatMessage] = []
            for (blockIndex, block) in blocks.enumerated() {
                let blockParam = "\(param).content.\(blockIndex)"
                guard case .object(let dict) = block, case .string(let type)? = dict["type"] else {
                    throw invalid("content blocks must have a type", blockParam)
                }
                switch type {
                case "text":
                    guard case .string(let value)? = dict["text"] else {
                        throw invalid("text block requires text", blockParam)
                    }
                    text.append(value)
                case "tool_use":
                    guard message.role == "assistant" else {
                        throw invalid("tool_use blocks belong to assistant messages", blockParam)
                    }
                    guard case .string(let id)? = dict["id"], case .string(let name)? = dict["name"] else {
                        throw invalid("tool_use requires id and name", blockParam)
                    }
                    let input = dict["input"] ?? .object([:])
                    let arguments = (try? input.encoded(sortedKeys: true)) ?? "{}"
                    toolCalls.append(OpenAIToolCall(
                        id: id, type: "function",
                        function: OpenAIFunctionCall(name: name, arguments: arguments)))
                case "tool_result":
                    guard message.role == "user" else {
                        throw invalid("tool_result blocks belong to user messages", blockParam)
                    }
                    guard case .string(let useID)? = dict["tool_use_id"] else {
                        throw invalid("tool_result requires tool_use_id", blockParam)
                    }
                    var result = try toolResultText(dict["content"], param: blockParam)
                    if case .bool(true)? = dict["is_error"], !result.hasPrefix("Error") {
                        result = "Error: " + result
                    }
                    toolResults.append(OpenAIChatMessage(
                        role: "tool", content: .text(result),
                        toolCalls: nil, toolCallID: useID, name: nil))
                case "thinking", "redacted_thinking":
                    // Replayed thoughts from an earlier turn. NVMAI never
                    // renders a model's prior thinking into its prompt.
                    continue
                case "image", "document", "search_result", "server_tool_use",
                     "web_search_tool_result", "container_upload":
                    throw unsupported("\(type) blocks are not supported; this server is text-only", blockParam)
                default:
                    throw unsupported("unsupported content block type \(type)", blockParam)
                }
            }
            var out = toolResults
            if message.role == "assistant" {
                let joined = text.joined(separator: "\n")
                out.append(OpenAIChatMessage(
                    role: "assistant",
                    content: joined.isEmpty && !toolCalls.isEmpty ? nil : .text(joined),
                    toolCalls: toolCalls.isEmpty ? nil : toolCalls,
                    toolCallID: nil, name: nil))
            } else if !text.isEmpty || toolResults.isEmpty {
                out.append(OpenAIChatMessage(
                    role: "user", content: .text(text.joined(separator: "\n")),
                    toolCalls: nil, toolCallID: nil, name: nil))
            }
            return out
        default:
            throw invalid("content must be a string or an array of blocks", "\(param).content")
        }
    }

    /// Function definitions from Anthropic tool objects. Built-in tool types
    /// (bash, text editor, web search, computer use, ...) are refused: nothing
    /// on this server executes them.
    static func tools(_ tools: [JSONValue]?) throws -> [OpenAITool]? {
        guard let tools, !tools.isEmpty else { return nil }
        return try tools.enumerated().map { index, tool in
            let param = "tools.\(index)"
            guard case .object(let dict) = tool else {
                throw invalid("tools must be objects", param)
            }
            if case .string(let type)? = dict["type"], type != "custom" {
                throw unsupported("tool type \(type) is not supported; only custom (function) tools are available", "\(param).type")
            }
            guard case .string(let name)? = dict["name"] else {
                throw invalid("tool requires a name", "\(param).name")
            }
            let description: String?
            if case .string(let text)? = dict["description"] { description = text } else { description = nil }
            let schema = dict["input_schema"] ?? .object(["type": .string("object"), "properties": .object([:])])
            return OpenAITool(type: "function",
                              function: OpenAIFunctionDefinition(name: name, description: description,
                                                                 parameters: schema))
        }
    }

    /// The chat-side tool_choice for an Anthropic one. `auto` and `none` are
    /// honoured; `any` and `tool` force a call, which the decoder cannot do.
    static func toolChoice(_ choice: JSONValue?) throws -> JSONValue? {
        guard let choice else { return nil }
        guard case .object(let dict) = choice, case .string(let type)? = dict["type"] else {
            throw invalid("tool_choice must be an object with a type", "tool_choice")
        }
        if case .bool(true)? = dict["disable_parallel_tool_use"] {
            throw unsupported("disable_parallel_tool_use is not supported", "tool_choice.disable_parallel_tool_use")
        }
        switch type {
        case "auto": return .string("auto")
        case "none": return .string("none")
        case "any", "tool":
            throw unsupported("tool_choice \(type) is not supported; the model chooses whether to call a tool",
                              "tool_choice.type")
        default:
            throw invalid("tool_choice type must be auto, any, tool or none", "tool_choice.type")
        }
    }

    /// Extended thinking is a load-time property of the served model, as it
    /// is for reasoning_effort on the OpenAI path: a request may confirm the
    /// active mode, never switch it.
    static func validateThinking(_ thinking: JSONValue?, maxTokens: Int,
                                 profile: ServerReasoningProfile) throws {
        guard let thinking else { return }
        guard case .object(let dict) = thinking, case .string(let type)? = dict["type"] else {
            throw invalid("thinking must be an object with a type", "thinking")
        }
        switch type {
        case "disabled":
            return
        case "enabled", "adaptive":
            guard profile.thinkingMode == .on else {
                throw unsupported("thinking is a load-time control; this server was started with thinking off. Restart with --thinking on",
                                  "thinking.type")
            }
            if type == "enabled" {
                guard case .integer(let budget)? = dict["budget_tokens"] else {
                    throw invalid("thinking.budget_tokens is required when thinking is enabled", "thinking.budget_tokens")
                }
                guard budget >= 1024 else {
                    throw invalid("budget_tokens must be at least 1024", "thinking.budget_tokens")
                }
                guard budget < maxTokens else {
                    throw invalid("budget_tokens must be less than max_tokens", "thinking.budget_tokens")
                }
            }
        default:
            throw invalid("thinking type must be enabled, adaptive or disabled", "thinking.type")
        }
    }

    /// Build the chat-completions request for a Messages request, so the one
    /// validator and the one generation path serve both APIs.
    public static func chatRequest(_ request: AnthropicMessagesRequest,
                                   profile: ServerReasoningProfile) throws -> OpenAIChatRequest {
        guard let maxTokens = request.maxTokens else {
            throw invalid("field required", "max_tokens")
        }
        guard maxTokens > 0 else {
            throw invalid("max_tokens must be greater than 0", "max_tokens")
        }
        if let temperature = request.temperature, !(0...1).contains(temperature) {
            throw invalid("temperature must be between 0 and 1", "temperature")
        }
        if let topP = request.topP, !(0...1).contains(topP) {
            throw invalid("top_p must be between 0 and 1", "top_p")
        }
        if request.outputFormat != nil {
            throw unsupported("structured output formats are not supported", "output_format")
        }
        if case .object(let config)? = request.outputConfig, config["format"] != nil {
            throw unsupported("structured output formats are not supported", "output_config.format")
        }
        if request.container != nil {
            throw unsupported("containers are not supported", "container")
        }
        if request.mcpServers != nil {
            throw unsupported("MCP servers are not supported", "mcp_servers")
        }
        if request.contextManagement != nil {
            throw unsupported("context management is not supported", "context_management")
        }
        try validateThinking(request.thinking, maxTokens: maxTokens, profile: profile)
        guard !request.messages.isEmpty else {
            throw invalid("at least one message is required", "messages")
        }
        if request.messages.last?.role == "assistant" {
            throw unsupported("a trailing assistant message (prefill) is not supported", "messages")
        }

        var messages: [OpenAIChatMessage] = []
        if let system = try systemText(request.system), !system.isEmpty {
            messages.append(OpenAIChatMessage(role: "system", content: .text(system),
                                              toolCalls: nil, toolCallID: nil, name: nil))
        }
        for (index, message) in request.messages.enumerated() {
            for chat in try chatMessages(for: message, index: index) {
                // Consecutive user text turns combine into one, as the API
                // itself documents; the template renders one turn per role.
                if chat.role == "user", let previous = messages.last, previous.role == "user",
                   case .text(let earlier)? = previous.content, case .text(let later)? = chat.content {
                    messages[messages.count - 1] = OpenAIChatMessage(
                        role: "user", content: .text(earlier + "\n\n" + later),
                        toolCalls: nil, toolCallID: nil, name: nil)
                } else {
                    messages.append(chat)
                }
            }
        }
        let stop: OpenAIStop? = (request.stopSequences?.isEmpty ?? true) ? nil : .many(request.stopSequences ?? [])
        return OpenAIChatRequest(
            model: request.model,
            messages: messages,
            stream: request.stream ?? false,
            streamOptions: nil,
            temperature: request.temperature ?? GenerationDefaults.temperature,
            topP: request.topP ?? GenerationDefaults.topP,
            maxTokens: maxTokens,
            maxCompletionTokens: nil,
            stop: stop,
            seed: nil,
            tools: try tools(request.tools),
            toolChoice: try toolChoice(request.toolChoice),
            parallelToolCalls: nil,
            topK: request.topK ?? GenerationDefaults.topK,
            repetitionPenalty: nil,
            n: 1,
            logprobs: nil,
            presencePenalty: GenerationDefaults.presencePenalty,
            frequencyPenalty: nil,
            reasoningEffort: nil)
    }

    /// The count_tokens body, as a Messages request without generation.
    public static func chatRequest(counting request: AnthropicCountTokensRequest,
                                   profile: ServerReasoningProfile) throws -> OpenAIChatRequest {
        let full = AnthropicMessagesRequest(
            model: request.model, messages: request.messages, maxTokens: 1,
            system: request.system, metadata: nil, stopSequences: nil, stream: false,
            temperature: nil, topK: nil, topP: nil, tools: request.tools,
            toolChoice: request.toolChoice, thinking: nil, serviceTier: nil,
            outputConfig: nil, outputFormat: nil, container: nil, mcpServers: nil,
            contextManagement: nil)
        return try chatRequest(full, profile: profile)
    }
}

// MARK: - Response builders

public enum AnthropicBuilder {
    /// The API version this server implements. Sent back on every response.
    public static let version = "2023-06-01"

    public static func messageID() -> String {
        "msg_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    public static func requestID() -> String {
        "req_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    /// Anthropic's stop_reason for a completion: tool calls first (a turn
    /// that called a tool ends with tool_use whatever else it said), then the
    /// output cap, then a matched stop string, then a natural end.
    public static func stopReason(for completion: ServerCompletion) -> (reason: String, sequence: String?) {
        if !completion.toolCalls.isEmpty { return ("tool_use", nil) }
        if completion.finishReason == "length" { return ("max_tokens", nil) }
        if let stop = completion.stopSequence { return ("stop_sequence", stop) }
        return ("end_turn", nil)
    }

    public static func textBlock(_ text: String) -> [String: Any] {
        ["type": "text", "text": text]
    }

    public static func toolUseBlock(_ call: ParsedToolCall) -> [String: Any] {
        ["type": "tool_use", "id": call.id, "name": call.name,
         "input": call.arguments.foundationObject()]
    }

    /// Content blocks of a completion: the text (when there is any) and one
    /// tool_use block per call. An empty completion is one empty text block,
    /// never an empty content array.
    public static func contentBlocks(_ completion: ServerCompletion) -> [[String: Any]] {
        var blocks: [[String: Any]] = []
        if !completion.content.isEmpty || completion.toolCalls.isEmpty {
            blocks.append(textBlock(completion.content))
        }
        blocks += completion.toolCalls.map(toolUseBlock)
        return blocks
    }

    /// Anthropic counts cache reads apart from input_tokens: the two sum to
    /// what OpenAI reports as prompt_tokens.
    public static func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        let cached = usage.promptTokensDetails.cachedTokens
        return [
            "input_tokens": max(usage.promptTokens - cached, 0),
            "output_tokens": usage.completionTokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cached,
            "service_tier": "standard",
        ]
    }

    public static func messageObject(id: String,
                                     model: String,
                                     content: [[String: Any]],
                                     stopReason: String?,
                                     stopSequence: String?,
                                     usage: [String: Any]) -> [String: Any] {
        [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stopReason.map { $0 as Any } ?? NSNull(),
            "stop_sequence": stopSequence.map { $0 as Any } ?? NSNull(),
            "usage": usage,
        ]
    }

    /// `GET /v1/models` in the Anthropic shape.
    public static func modelList(ids: [String]) -> [String: Any] {
        let created = ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: 0))
        return [
            "data": ids.map { ["type": "model", "id": $0, "display_name": $0, "created_at": created] },
            "has_more": false,
            "first_id": ids.first.map { $0 as Any } ?? NSNull(),
            "last_id": ids.last.map { $0 as Any } ?? NSNull(),
        ]
    }

    public static func modelObject(id: String) -> [String: Any] {
        ["type": "model", "id": id, "display_name": id,
         "created_at": ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: 0))]
    }
}
