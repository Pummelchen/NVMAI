import Foundation
import NVMAI

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String, type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAITextPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
}

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAITextPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAITextPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

    func textValue() throws -> String {
        switch self {
        case .text(let text):
            return text
        case .parts(let parts):
            guard parts.allSatisfy({ $0.type == "text" && $0.text != nil }) else {
                throw ServerRequestError.invalid(
                    message: "only text content parts are supported",
                    param: "messages",
                    code: "unsupported_content")
            }
            return parts.compactMap(\.text).joined()
        }
    }
}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        /// Model creation time. Omitted when unknown rather than lying with a
        /// fabricated epoch (S30).
        public let created: Int?
        public let ownedBy: String

        enum CodingKeys: String, CodingKey {
            case id, object, created
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int?, ownedBy: String) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full",
                                type: "rate_limit_error")
        }
    }
}

public struct ValidatedChatRequest: Sendable {
    public let messages: [GFTokenizer.Message]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    /// Set when the request named the "<model>-fast" alias: the CLI-strip
    /// heuristic runs for this request regardless of NVMAI_STRIP_CLI_PROMPT.
    public let stripCLIPrompt: Bool

    public init(messages: [GFTokenizer.Message],
                tools: [GFTokenizer.FunctionDefinition],
                stream: Bool,
                includeUsage: Bool,
                generationConfig: GenerationConfig,
                maximumCompletionTokens: Int,
                stripCLIPrompt: Bool = false) {
        self.messages = messages
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.stripCLIPrompt = stripCLIPrompt
    }

    /// The post-strip view of this request: the same request carrying the
    /// messages and tools that were actually encoded into the prompt.
    ///
    /// The prompt cache must key on this view, not the raw request. Its
    /// entries describe a KV range that was prefilled from the filtered
    /// messages, and its continuation paths re-render the tail with the same
    /// template — so matching on the raw messages would splice an unfiltered
    /// tail onto a filtered prefix (see `ServerPromptCache`).
    public func replacingMessages(
        _ messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition]
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: stream,
            includeUsage: includeUsage,
            generationConfig: generationConfig,
            maximumCompletionTokens: maximumCompletionTokens,
            stripCLIPrompt: stripCLIPrompt)
    }
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                maxContext: Int = RuntimeConfiguration
                                    .supportedContextTokens.max() ?? 262_144) throws -> ValidatedChatRequest {
        // The "<model>-fast" alias selects the same weights as the base model
        // but enables the CLI-strip heuristic per request (chat-only speed),
        // so tool-using clients keep the base model and chat users opt in.
        let fastModelID = modelID + "-fast"
        let stripCLIPrompt = request.model == fastModelID
        guard request.model == modelID || stripCLIPrompt else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }
        // S17: include_usage is a streaming option; silently ignoring it on a
        // non-stream request hides a client bug.
        if request.streamOptions?.includeUsage == true, request.stream != true {
            throw invalid("stream_options.include_usage requires stream=true",
                          "stream_options", "invalid_value")
        }
        // S16: OpenAI forbids setting both bounds in one request.
        guard request.maxCompletionTokens == nil || request.maxTokens == nil else {
            throw invalid("max_tokens and max_completion_tokens cannot both be set",
                          "max_tokens", "invalid_value")
        }

        let temperature = request.temperature ?? 0.2
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        // No artificial output cap: when the client omits max_tokens /
        // max_completion_tokens, generation is bounded only by the session's
        // configured context window (further clamped to the available context
        // at inference time), so the model replies until it is done.
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? maxContext
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }
        // S11: validate against the session's configured context window, not
        // the hard architectural ceiling.
        let cappedMaximum = min(maximum, maxContext)
        guard cappedMaximum == maximum else {
            throw invalid("maximum completion tokens exceeds the configured context window (\(maxContext))",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "value_too_large")
        }

        // S18: stop strings must be non-empty, unique, and bounded.
        let stopValues = request.stop?.values ?? []
        var stopStrings: [String] = []
        if !stopValues.isEmpty {
            guard stopValues.allSatisfy({ !$0.isEmpty }) else {
                throw invalid("stop strings must not be empty", "stop", "invalid_value")
            }
            guard stopValues.count <= 4 else {
                throw invalid("at most 4 stop strings are supported", "stop", "value_too_large")
            }
            let totalLength = stopValues.reduce(0) { $0 + $1.utf8.count }
            guard totalLength <= 256 else {
                throw invalid("stop strings must total at most 256 bytes", "stop", "value_too_large")
            }
            var seen: Set<String> = []
            stopStrings = stopValues.filter { seen.insert($0).inserted }
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        case .some(.bool(true)):
            // Legacy boolean form of "auto" (S31).
            includeTools = true
        case .some(.bool(false)):
            // Legacy boolean form of "none" (S31).
            includeTools = false
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0)
        }
        let messages = try validateMessages(request.messages)
        // A client-supplied seed makes sampling deterministic.
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: stopStrings)
        return ValidatedChatRequest(messages: messages,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    stripCLIPrompt: stripCLIPrompt)
    }

    private static func validateTool(_ tool: OpenAITool) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard name.range(of: #"^[A-Za-z0-9_-]{1,64}$"#, options: .regularExpression) != nil else {
            throw invalid("tool name must match [A-Za-z0-9_-]{1,64}",
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters)
        let parameters = tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (_, definition) in definitions {
                        // ChatML tool-call parameter names are free-form;
                        // only the schema structure itself is validated.
                        try validateSchemaKeys(definition)
                    }
                } else {
                    try validateSchemaKeys(value)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value)
            }
        default:
            break
        }
    }

    private static func validateMessages(_ input: [OpenAIChatMessage]) throws -> [GFTokenizer.Message] {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        guard input.count <= 1000 else {
            throw invalid("message count exceeds maximum of 1000",
                          "messages", "value_too_large")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            let content = try message.content?.textValue()
            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil,
                      call.function.name.range(
                        of: #"^[A-Za-z0-9_-]{1,64}$"#,
                        options: .regularExpression) != nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
        }
        // S19: a conversation that ends with an assistant tool call that is
        // never answered by a tool result would resume from an unanswerable
        // state; reject it instead of generating tool-response markup.
        if knownCalls.contains(where: { !$0.value.resolved }) {
            throw invalid("conversation ends with an unresolved tool call",
                          "messages", "invalid_tool_call")
        }
        return result
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
