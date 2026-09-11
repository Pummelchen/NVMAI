import Foundation
import NVMAI

// MARK: - Responses API request decoding

/// The OpenAI Responses API request body (`POST /v1/responses`). Every
/// documented field is decoded so that a client's request never fails for
/// naming something this server merely ignores; the mapper below decides
/// which of them it can honour, echo, or must refuse. Text-only: image and
/// file inputs are rejected with a spec-shaped error.
public struct ResponsesAPIRequest: Decodable, Sendable {
    public struct Item: Codable, Sendable, Equatable {
        /// Item kind. Optional: some clients (e.g. OpenCode) omit it and rely
        /// on role+content / call_id+output to convey the kind.
        public let type: String?
        public let id: String?
        public let status: String?
        public let role: String?
        /// String content or an array of parts ({type: input_text, text}).
        public let content: JSONValue?
        public let callID: String?
        public let name: String?
        public let arguments: String?
        /// A function_call_output's result: a string or an array of parts.
        public let output: JSONValue?
        /// Reasoning items round-tripped by a client; carried, never read.
        public let summary: JSONValue?
        public let encryptedContent: String?
        /// The namespace a function call belongs to, when its tool was
        /// declared inside a `namespace` tool.
        public let namespace: String?

        enum CodingKeys: String, CodingKey {
            case type, id, status, role, content, name, arguments, output, summary, namespace
            case callID = "call_id"
            case encryptedContent = "encrypted_content"
        }

        public init(type: String?, id: String? = nil, status: String? = nil,
                    role: String? = nil, content: JSONValue? = nil,
                    callID: String? = nil, name: String? = nil,
                    arguments: String? = nil, output: JSONValue? = nil,
                    summary: JSONValue? = nil, encryptedContent: String? = nil,
                    namespace: String? = nil) {
            self.type = type
            self.id = id
            self.status = status
            self.role = role
            self.content = content
            self.callID = callID
            self.name = name
            self.arguments = arguments
            self.output = output
            self.summary = summary
            self.encryptedContent = encryptedContent
            self.namespace = namespace
        }

        /// Resolved item kind: the explicit `type`, or inferred from the
        /// present fields when the client omits it.
        public var resolvedType: String? {
            if let type { return type }
            if role != nil && content != nil { return "message" }
            if callID != nil && output != nil { return "function_call_output" }
            if name != nil && arguments != nil { return "function_call" }
            return nil
        }
    }

    public struct Tool: Codable, Sendable, Equatable {
        public let type: String
        public let name: String?
        public let description: String?
        public let parameters: JSONValue?
        public let strict: Bool?
        /// A `namespace` tool groups function tools under a name; the model
        /// calls the functions, and the call carries the namespace back.
        public let tools: [Tool]?

        public init(type: String, name: String?, description: String?,
                    parameters: JSONValue?, strict: Bool?, tools: [Tool]? = nil) {
            self.type = type
            self.name = name
            self.description = description
            self.parameters = parameters
            self.strict = strict
            self.tools = tools
        }
    }

    /// The Responses API's nested reasoning options; only `effort` is read.
    public struct Reasoning: Codable, Equatable, Sendable {
        public let effort: String?
        public let summary: String?
    }

    public struct TextConfig: Decodable, Sendable {
        public let format: JSONValue?
        public let verbosity: String?
    }

    public struct StreamOptions: Decodable, Sendable {
        public let includeObfuscation: Bool?

        enum CodingKeys: String, CodingKey {
            case includeObfuscation = "include_obfuscation"
        }
    }

    /// `input` is either a plain string (one user message) or a list of items.
    public enum Input: Decodable, Sendable {
        case text(String)
        case items([Item])

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else {
                self = .items(try container.decode([Item].self))
            }
        }

        public var items: [Item] {
            switch self {
            case .text(let text):
                [Item(type: "message", role: "user", content: .string(text))]
            case .items(let items):
                items
            }
        }
    }

    public let model: String
    public let instructions: String?
    public let input: Input?
    public let tools: [Tool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let temperature: Float?
    public let topP: Float?
    public let topK: Int?
    public let topLogprobs: Int?
    public let presencePenalty: Float?
    public let stream: Bool?
    public let streamOptions: StreamOptions?
    public let store: Bool?
    public let background: Bool?
    public let reasoning: Reasoning?
    public let previousResponseID: String?
    public let metadata: JSONValue?
    public let user: String?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let serviceTier: String?
    public let truncation: String?
    public let text: TextConfig?
    public let include: [String]?
    public let prompt: JSONValue?
    public let conversation: JSONValue?

    enum CodingKeys: String, CodingKey {
        case model, instructions, input, tools, stream, store, temperature, reasoning
        case metadata, user, truncation, text, include, prompt, conversation, background
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case maxOutputTokens = "max_output_tokens"
        case maxToolCalls = "max_tool_calls"
        case topP = "top_p"
        case topK = "top_k"
        case topLogprobs = "top_logprobs"
        case presencePenalty = "presence_penalty"
        case streamOptions = "stream_options"
        case previousResponseID = "previous_response_id"
        case safetyIdentifier = "safety_identifier"
        case promptCacheKey = "prompt_cache_key"
        case serviceTier = "service_tier"
    }

    /// The input as items, however the client wrote it.
    public var inputItems: [Item] { input?.items ?? [] }

    /// Whether the finished response should be kept for `previous_response_id`
    /// and `GET /v1/responses/{id}`. The API's default is true.
    public var stores: Bool { store ?? true }
}

// MARK: - Responses -> chat mapping

public enum ResponsesAPIMapper {
    /// Text of a message's content: a plain string or an array of text parts.
    /// `input_text`, `output_text` and `refusal` parts carry text; images and
    /// files are refused because NVMAI is text-only.
    public static func messageText(_ content: JSONValue?, param: String = "input") throws -> String {
        guard let content else { return "" }
        switch content {
        case .string(let text):
            return text
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                      case .string(let type)? = dict["type"] else {
                    throw ServerRequestError.invalid(
                        message: "content parts must be objects with a type",
                        param: param, code: "invalid_value")
                }
                switch type {
                case "input_text", "output_text":
                    if case .string(let text)? = dict["text"] { out += text }
                case "refusal":
                    if case .string(let text)? = dict["refusal"] { out += text }
                case "input_image", "input_file", "input_audio":
                    throw ServerRequestError.invalid(
                        message: "\(type) parts are not supported; this server is text-only",
                        param: param, code: "unsupported_content")
                default:
                    throw ServerRequestError.invalid(
                        message: "unsupported content part type \(type)",
                        param: param, code: "unsupported_content")
                }
            }
            return out
        default:
            throw ServerRequestError.invalid(
                message: "content must be a string or an array of parts",
                param: param, code: "invalid_value")
        }
    }

    /// The text of a function_call_output's `output`: a string, or a list of
    /// `input_text` parts.
    public static func outputText(_ output: JSONValue?) throws -> String {
        guard let output else { return "" }
        if case .string(let text) = output { return text }
        return try messageText(output, param: "input.output")
    }

    /// Refuse the request features this server cannot honour, before any
    /// mapping. Each refusal names the field and says why, in the API's own
    /// error shape, instead of silently generating something else.
    public static func validateFeatures(_ request: ResponsesAPIRequest) throws {
        if request.background == true {
            throw ServerRequestError.invalid(
                message: "background responses are not supported",
                param: "background", code: "unsupported_value")
        }
        if let format = request.text?.format, case .object(let dict) = format,
           case .string(let type)? = dict["type"], type != "text" {
            throw ServerRequestError.invalid(
                message: "text.format \(type) is not supported; only plain text output is available",
                param: "text.format", code: "unsupported_value")
        }
        if let topLogprobs = request.topLogprobs, topLogprobs > 0 {
            throw ServerRequestError.invalid(
                message: "logprobs are not supported", param: "top_logprobs",
                code: "unsupported_value")
        }
        if request.prompt != nil {
            throw ServerRequestError.invalid(
                message: "prompt templates are not supported", param: "prompt",
                code: "unsupported_value")
        }
        if request.conversation != nil {
            throw ServerRequestError.invalid(
                message: "conversations are not supported; use previous_response_id",
                param: "conversation", code: "unsupported_value")
        }
        if let include = request.include {
            let supported: Set<String> = ["message.output_text.logprobs", "reasoning.encrypted_content"]
            if let bad = include.first(where: { !supported.contains($0) }) {
                throw ServerRequestError.invalid(
                    message: "include value \(bad) is not supported",
                    param: "include", code: "unsupported_value")
            }
        }
    }

    /// The function tools of a request, with namespaced functions flattened
    /// into the list and remembered by namespace. Hosted tool types
    /// (web_search, file_search, code_interpreter, mcp, image_generation,
    /// computer use, shell, apply_patch) have nothing on this server to run
    /// them and are left out; the model never sees them and never calls
    /// them. Codex sends web_search on every turn, so refusing would refuse
    /// Codex.
    public static func functionTools(_ tools: [ResponsesAPIRequest.Tool]?)
        -> (tools: [OpenAITool], namespaces: [String: String]) {
        var out: [OpenAITool] = []
        var namespaces: [String: String] = [:]
        func add(_ tool: ResponsesAPIRequest.Tool, namespace: String?) {
            guard tool.type == "function", let name = tool.name, !name.isEmpty else { return }
            let parameters: JSONValue = tool.parameters
                ?? .object(["type": .string("object"), "properties": .object([:])])
            out.append(OpenAITool(
                type: "function",
                function: OpenAIFunctionDefinition(name: name, description: tool.description,
                                                   parameters: parameters)))
            if let namespace { namespaces[name] = namespace }
        }
        for tool in tools ?? [] {
            if tool.type == "namespace" {
                for nested in tool.tools ?? [] { add(nested, namespace: tool.name) }
            } else {
                add(tool, namespace: nil)
            }
        }
        return (out, namespaces)
    }

    /// Build the chat-completions request equivalent to a responses request.
    /// NVMAI's chat template requires exactly one leading system message and
    /// rejects the developer role, so instructions and developer guidance are
    /// merged into a single opening system message. `priorItems` is the
    /// conversation a `previous_response_id` resolved to; it precedes the
    /// request's own input.
    public static func chatRequest(_ request: ResponsesAPIRequest,
                                   priorItems: [ResponsesAPIRequest.Item] = [],
                                   inputItems: [ResponsesAPIRequest.Item]? = nil) throws -> OpenAIChatRequest {
        try validateFeatures(request)
        var systemParts: [String] = []
        if let instructions = request.instructions, !instructions.isEmpty {
            systemParts.append(instructions)
        }
        var messages: [OpenAIChatMessage] = []
        for item in priorItems + (inputItems ?? request.inputItems) {
            guard let kind = item.resolvedType else {
                throw ServerRequestError.invalid(
                    message: "unsupported input item; cannot determine its type",
                    param: "input", code: "unsupported_input")
            }
            switch kind {
            case "message":
                let role = item.role ?? "user"
                let text = try messageText(item.content)
                if role == "system" || role == "developer" {
                    if !text.isEmpty { systemParts.append(text) }
                } else {
                    messages.append(OpenAIChatMessage(
                        role: role, content: .text(text),
                        toolCalls: nil, toolCallID: nil, name: nil))
                }
            case "function_call":
                guard let callID = item.callID, !callID.isEmpty else {
                    throw ServerRequestError.invalid(
                        message: "function_call item requires call_id",
                        param: "input", code: "invalid_value")
                }
                let function = OpenAIFunctionCall(name: item.name ?? "", arguments: item.arguments ?? "{}")
                messages.append(OpenAIChatMessage(
                    role: "assistant", content: nil,
                    toolCalls: [OpenAIToolCall(id: callID, type: "function", function: function)],
                    toolCallID: nil, name: nil))
            case "function_call_output":
                messages.append(OpenAIChatMessage(
                    role: "tool", content: .text(try outputText(item.output)),
                    toolCalls: nil, toolCallID: item.callID, name: nil))
            case "reasoning":
                // A client replaying an earlier turn returns the reasoning
                // item it was given. The model's thoughts are never part of
                // its prompt, so there is nothing to render; accept and skip.
                continue
            case "item_reference":
                throw ServerRequestError.invalid(
                    message: "item_reference \(item.id ?? "") could not be resolved",
                    param: "input", code: "item_not_found")
            default:
                throw ServerRequestError.invalid(
                    message: "unsupported input item type \(kind)",
                    param: "input", code: "unsupported_input")
            }
        }
        var chatMessages = messages
        if !systemParts.isEmpty {
            chatMessages.insert(OpenAIChatMessage(
                role: "system", content: .text(systemParts.joined(separator: "\n\n")),
                toolCalls: nil, toolCallID: nil, name: nil), at: 0)
        }
        let tools = functionTools(request.tools).tools
        return OpenAIChatRequest(
            model: request.model,
            messages: chatMessages,
            stream: request.stream ?? false,
            streamOptions: nil,
            // Sampling the request omitted stays nil on purpose. Filling it here
            // with the generic `GenerationDefaults` makes the served model's own
            // defaults unreachable: the validator resolves each field as
            // `request.value ?? sampling.value`, where `sampling` is the loaded
            // model's profile. Qwen3.8-Flash-Next's card says temperature 1.0, so
            // a fixed 0.6 here sampled it wrong on this surface while
            // /v1/chat/completions honoured the profile.
            temperature: request.temperature,
            topP: request.topP,
            // Codex and OpenCode omit max_output_tokens; forward nil so the
            // chat validator applies its context-bounded default (no
            // artificial output cap) instead of a fixed token budget.
            maxTokens: request.maxOutputTokens,
            maxCompletionTokens: nil,
            stop: nil,
            seed: nil,
            tools: tools.isEmpty ? nil : tools,
            // The chat validator knows every tool_choice form the server
            // honours ("auto", "none") and refuses the rest by name.
            toolChoice: request.toolChoice,
            // Codex sends parallel_tool_calls=false on every turn. The
            // decoder cannot promise a single call per turn, and refusing
            // would refuse Codex; the value is echoed and not enforced.
            parallelToolCalls: nil,
            topK: request.topK,
            repetitionPenalty: nil,
            n: 1,
            logprobs: nil,
            presencePenalty: request.presencePenalty,
            frequencyPenalty: nil,
            reasoningEffort: request.reasoning?.effort)
    }

    /// A finished response's output, in the shape a later request carries it
    /// back as input. This is what `previous_response_id` chains on. The
    /// reasoning item is left out: a replayed one is skipped on the way back
    /// in, because thoughts are never part of a prompt.
    public static func outputAsInput(completion: ServerCompletion,
                                     responseID: String,
                                     namespaces: [String: String] = [:]) -> [ResponsesAPIRequest.Item] {
        var items: [ResponsesAPIRequest.Item] = []
        let ids = ResponsesAPIBuilder.itemIDs(responseID: responseID, completion: completion)
        if !completion.content.isEmpty {
            items.append(ResponsesAPIRequest.Item(
                type: "message", id: ids.message, status: "completed", role: "assistant",
                content: .array([.object(["type": .string("output_text"),
                                          "text": .string(completion.content),
                                          "annotations": .array([])])])))
        }
        for (index, call) in completion.toolCalls.enumerated() {
            items.append(ResponsesAPIRequest.Item(
                type: "function_call", id: ids.calls[index], status: "completed",
                callID: call.id, name: call.name, arguments: call.argumentsJSON,
                namespace: namespaces[call.name]))
        }
        return items
    }
}

// MARK: - Stored responses

/// Finished responses kept for `previous_response_id`, `GET`, `DELETE` and
/// `input_items`. Bounded and in memory: this is the API's storage contract
/// for a single-user local server, not a database. The oldest entry goes
/// when the cap is reached.
/// unchecked-invariant: every field is guarded by `lock`.
public final class ResponseStore: @unchecked Sendable {
    public struct Entry: Sendable {
        /// The response object as returned to the client, JSON-encoded.
        public let responseJSON: Data
        /// The conversation the response was generated from, as input items
        /// (a prior chain already flattened in).
        public let inputItems: [ResponsesAPIRequest.Item]
        /// The response's output, as the input items a follow-up carries.
        public let outputItems: [ResponsesAPIRequest.Item]
        public let created: Date

        /// What a request naming this response as `previous_response_id`
        /// continues from.
        public var conversation: [ResponsesAPIRequest.Item] { inputItems + outputItems }
    }

    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    public let capacity: Int

    public init(capacity: Int = 256) {
        self.capacity = capacity
    }

    public func put(id: String, entry: Entry) {
        lock.withLock {
            if entries[id] == nil { order.append(id) }
            entries[id] = entry
            while order.count > capacity, let oldest = order.first {
                order.removeFirst()
                entries.removeValue(forKey: oldest)
            }
        }
    }

    public func get(_ id: String) -> Entry? {
        lock.withLock { entries[id] }
    }

    @discardableResult
    public func delete(_ id: String) -> Bool {
        lock.withLock {
            guard entries.removeValue(forKey: id) != nil else { return false }
            order.removeAll { $0 == id }
            return true
        }
    }

    public var count: Int { lock.withLock { entries.count } }

    /// An item of any stored conversation by its id, for `item_reference`.
    public func item(withID id: String) -> ResponsesAPIRequest.Item? {
        lock.withLock {
            for entry in entries.values {
                if let item = (entry.outputItems + entry.inputItems).first(where: { $0.id == id }) {
                    return item
                }
            }
            return nil
        }
    }
}

// MARK: - Response object builders

/// Everything a response object echoes back from its request. Built once
/// per request so the in_progress, completed and stored objects agree.
public struct ResponsesAPIEcho: Sendable {
    public let instructions: String?
    public let maxOutputTokens: Int?
    public let maxToolCalls: Int?
    public let previousResponseID: String?
    public let metadata: JSONValue
    public let user: String?
    public let safetyIdentifier: String?
    public let promptCacheKey: String?
    public let serviceTier: String
    public let truncation: String
    public let temperature: Float?
    public let topP: Float?
    public let store: Bool
    public let tools: [ResponsesAPIRequest.Tool]
    public let namespaces: [String: String]
    public let toolChoice: JSONValue
    public let textVerbosity: String
    public let reasoningEffort: String?
    public let reasoningSummary: String?
    public let parallelToolCalls: Bool

    public init(request: ResponsesAPIRequest, effectiveEffort: ModelReasoningEffort?) {
        instructions = request.instructions
        maxOutputTokens = request.maxOutputTokens
        maxToolCalls = request.maxToolCalls
        previousResponseID = request.previousResponseID
        metadata = request.metadata ?? .object([:])
        user = request.user
        safetyIdentifier = request.safetyIdentifier
        promptCacheKey = request.promptCacheKey
        serviceTier = "default"
        truncation = request.truncation ?? "disabled"
        temperature = request.temperature
        topP = request.topP
        store = request.stores
        tools = request.tools ?? []
        namespaces = ResponsesAPIMapper.functionTools(request.tools).namespaces
        toolChoice = request.toolChoice ?? .string("auto")
        textVerbosity = request.text?.verbosity ?? "medium"
        reasoningEffort = request.reasoning?.effort ?? effectiveEffort?.rawValue
        reasoningSummary = request.reasoning?.summary
        parallelToolCalls = request.parallelToolCalls ?? true
    }
}

public enum ResponsesAPIBuilder {
    public static func responseID() -> String {
        "resp_" + hex()
    }

    static func hex() -> String {
        UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    /// The output item ids of a response, derived from the response id so
    /// that the streamed `output_item.added` and the final object agree.
    public static func itemIDs(responseID: String,
                               completion: ServerCompletion) -> (message: String, calls: [String]) {
        let suffix = String(responseID.dropFirst("resp_".count))
        return ("msg_" + suffix,
                completion.toolCalls.indices.map { "fc_" + suffix + String($0) })
    }

    public static func messageItemID(responseID: String) -> String {
        "msg_" + String(responseID.dropFirst("resp_".count))
    }

    public static func functionCallItemID(responseID: String, index: Int) -> String {
        "fc_" + String(responseID.dropFirst("resp_".count)) + String(index)
    }

    /// Numbered like function calls: a model that thinks again after it has
    /// started answering produces a second reasoning item in the stream.
    public static func reasoningItemID(responseID: String, index: Int) -> String {
        "rs_" + String(responseID.dropFirst("resp_".count)) + String(index)
    }

    /// One streaming event. `sequence_number` is the client's ordering key;
    /// the handler owns the counter.
    public static func event(_ type: String, sequence: Int,
                             _ fields: [String: Any]) -> [String: Any] {
        var object = fields
        object["type"] = type
        object["sequence_number"] = sequence
        return object
    }

    public static func responseObject(id: String,
                                      created: Int,
                                      model: String,
                                      status: String,
                                      output: [[String: Any]],
                                      usage: OpenAIUsage?,
                                      echo: ResponsesAPIEcho,
                                      incompleteReason: String? = nil,
                                      error: (code: String, message: String)? = nil,
                                      completedAt: Int? = nil) -> [String: Any] {
        var object: [String: Any] = [
            "id": id,
            "object": "response",
            "created_at": created,
            "completed_at": completedAt.map { $0 as Any } ?? NSNull(),
            "status": status,
            "background": false,
            "error": error.map { ["code": $0.code, "message": $0.message] as Any } ?? NSNull(),
            "incomplete_details": incompleteReason.map { ["reason": $0] as Any } ?? NSNull(),
            "instructions": echo.instructions.map { $0 as Any } ?? NSNull(),
            "max_output_tokens": echo.maxOutputTokens.map { $0 as Any } ?? NSNull(),
            "max_tool_calls": echo.maxToolCalls.map { $0 as Any } ?? NSNull(),
            "model": model,
            "output": output,
            "parallel_tool_calls": echo.parallelToolCalls,
            "previous_response_id": echo.previousResponseID.map { $0 as Any } ?? NSNull(),
            "prompt_cache_key": echo.promptCacheKey.map { $0 as Any } ?? NSNull(),
            "reasoning": ["effort": echo.reasoningEffort.map { $0 as Any } ?? NSNull(),
                          "summary": echo.reasoningSummary.map { $0 as Any } ?? NSNull()],
            "safety_identifier": echo.safetyIdentifier.map { $0 as Any } ?? NSNull(),
            "service_tier": echo.serviceTier,
            "store": echo.store,
            "temperature": echo.temperature.map { $0 as Any } ?? NSNull(),
            "text": ["format": ["type": "text"], "verbosity": echo.textVerbosity],
            "tool_choice": echo.toolChoice.foundationObject(),
            "tools": echo.tools.map(toolObject),
            "top_logprobs": 0,
            "top_p": echo.topP.map { $0 as Any } ?? NSNull(),
            "truncation": echo.truncation,
            "usage": NSNull(),
            "user": echo.user.map { $0 as Any } ?? NSNull(),
            "metadata": echo.metadata.foundationObject(),
        ]
        if let usage {
            object["usage"] = usageObject(usage)
        }
        return object
    }

    public static func usageObject(_ usage: OpenAIUsage) -> [String: Any] {
        [
            "input_tokens": usage.promptTokens,
            "input_tokens_details": ["cached_tokens": usage.promptTokensDetails.cachedTokens,
                                     "cache_write_tokens": 0],
            "output_tokens": usage.completionTokens,
            "output_tokens_details": ["reasoning_tokens": 0],
            "total_tokens": usage.totalTokens,
        ]
    }

    static func toolObject(_ tool: ResponsesAPIRequest.Tool) -> [String: Any] {
        var object: [String: Any] = ["type": tool.type]
        if let name = tool.name { object["name"] = name }
        if tool.type == "function" || tool.type == "custom" {
            object["description"] = tool.description.map { $0 as Any } ?? NSNull()
            object["parameters"] = tool.parameters?.foundationObject() ?? NSNull()
            object["strict"] = tool.strict ?? false
        }
        if let nested = tool.tools {
            object["tools"] = nested.map(toolObject)
        }
        return object
    }

    public static func outputTextPart(_ text: String) -> [String: Any] {
        ["type": "output_text", "text": text, "annotations": [], "logprobs": []]
    }

    public static func messageItem(id: String,
                                   role: String,
                                   text: String,
                                   status: String) -> [String: Any] {
        ["id": id, "type": "message", "role": role, "status": status,
         "content": [outputTextPart(text)]]
    }

    public static func functionCallItem(id: String,
                                        name: String,
                                        arguments: String,
                                        callID: String,
                                        status: String,
                                        namespace: String? = nil) -> [String: Any] {
        var item: [String: Any] = ["id": id, "type": "function_call", "status": status,
                                   "name": name, "arguments": arguments, "call_id": callID]
        if let namespace { item["namespace"] = namespace }
        return item
    }

    public static func summaryTextPart(_ text: String) -> [String: Any] {
        ["type": "summary_text", "text": text]
    }

    /// The model's thoughts as a reasoning item. They are the whole thought,
    /// not a summary of it, but `summary` is the part of a reasoning item
    /// that Responses clients read and show; the item's other fields carry
    /// hosted-model state (encrypted content) this server has none of.
    public static func reasoningItem(id: String, text: String) -> [String: Any] {
        ["id": id, "type": "reasoning", "summary": [summaryTextPart(text)]]
    }

    /// Output items for a completed generation: the reasoning when there is
    /// any, then the message, then calls.
    public static func outputItems(completion: ServerCompletion,
                                   responseID: String,
                                   namespaces: [String: String] = [:]) -> [[String: Any]] {
        let ids = itemIDs(responseID: responseID, completion: completion)
        var output: [[String: Any]] = []
        if !completion.reasoning.isEmpty {
            output.append(reasoningItem(id: reasoningItemID(responseID: responseID, index: 0),
                                        text: completion.reasoning))
        }
        if !completion.content.isEmpty || completion.toolCalls.isEmpty {
            output.append(messageItem(id: ids.message, role: "assistant",
                                      text: completion.content, status: "completed"))
        }
        for (index, call) in completion.toolCalls.enumerated() {
            output.append(functionCallItem(
                id: ids.calls[index], name: call.name,
                arguments: call.argumentsJSON, callID: call.id, status: "completed",
                namespace: namespaces[call.name]))
        }
        return output
    }

    /// The terminal status of a generation: "incomplete" when the output
    /// cap ended it, which the API reports with `incomplete_details`.
    public static func terminalStatus(for completion: ServerCompletion) -> (status: String, reason: String?) {
        completion.finishReason == "length"
            ? ("incomplete", "max_output_tokens")
            : ("completed", nil)
    }

    /// `GET /v1/responses/{id}/input_items`.
    public static func inputItemsList(_ items: [ResponsesAPIRequest.Item]) throws -> Data {
        let encoder = JSONEncoder()
        let data = try items.map { try JSONSerialization.jsonObject(with: encoder.encode($0)) }
        let ids = items.compactMap(\.id)
        let object: [String: Any] = [
            "object": "list",
            "data": data,
            "first_id": ids.first.map { $0 as Any } ?? NSNull(),
            "last_id": ids.last.map { $0 as Any } ?? NSNull(),
            "has_more": false,
        ]
        return try JSONSerialization.data(withJSONObject: object)
    }
}

// MARK: - JSONValue bridging

extension JSONValue {
    /// The Foundation object `JSONSerialization` writes for this value.
    public func foundationObject() -> Any {
        switch self {
        case .object(let value): value.mapValues { $0.foundationObject() }
        case .array(let value): value.map { $0.foundationObject() }
        case .string(let value): value
        case .integer(let value): value
        case .unsignedInteger(let value): value
        case .decimal(let value): value
        case .number(let value): value
        case .bool(let value): value
        case .null: NSNull()
        }
    }
}
