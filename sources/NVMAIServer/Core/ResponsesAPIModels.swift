import Foundation
import NVMAI

// MARK: - Responses API request decoding

/// Minimal decode of the OpenAI Responses API request body (`POST /v1/responses`).
/// Unknown fields are ignored; only what NVMAI maps onto chat completions is
/// modeled. Text-only: image/video input items are rejected.
public struct ResponsesAPIRequest: Decodable, Sendable {
    public struct Item: Decodable, Sendable {
        /// Item kind. Optional: some clients (e.g. OpenCode) omit it and rely
        /// on role+content / call_id+output to convey the kind.
        public let type: String?
        public let role: String?
        /// String content or an array of parts ({type: input_text, text}).
        public let content: JSONValue?
        public let callID: String?
        public let name: String?
        public let arguments: String?
        public let output: String?

        enum CodingKeys: String, CodingKey {
            case type, role, content, name, arguments, output
            case callID = "call_id"
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

    public struct Tool: Decodable, Sendable {
        public let type: String
        public let name: String?
        public let description: String?
        public let parameters: JSONValue?
    }

    public let model: String
    public let instructions: String?
    public let input: [Item]?
    public let tools: [Tool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let maxOutputTokens: Int?
    public let temperature: Float?
    public let topP: Float?
    public let stream: Bool?
    public let store: Bool?

    enum CodingKeys: String, CodingKey {
        case model, instructions, input, tools, stream, store, temperature
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case maxOutputTokens = "max_output_tokens"
        case topP = "top_p"
    }
}

// MARK: - Responses -> chat mapping

public enum ResponsesAPIMapper {
    /// Text of a message item's content: either a plain string or an array of
    /// input_text parts. Anything else is unsupported (NVMAI is text-only).
    public static func messageText(_ content: JSONValue?) throws -> String {
        guard let content else { return "" }
        switch content {
        case .string(let text):
            return text
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                      case .string(let type)? = dict["type"],
                      type == "input_text" else {
                    throw ServerRequestError.invalid(
                        message: "only text input parts are supported",
                        param: "input", code: "unsupported_content")
                }
                if case .string(let text)? = dict["text"] {
                    out += text
                }
            }
            return out
        default:
            throw ServerRequestError.invalid(
                message: "only text input parts are supported",
                param: "input", code: "unsupported_content")
        }
    }

    /// Build the chat-completions request equivalent to a responses request.
    /// NVMAI's chat template requires exactly one leading system message and
    /// rejects the developer role, so instructions and developer guidance are
    /// merged into a single opening system message.
    public static func chatRequest(_ request: ResponsesAPIRequest) throws -> OpenAIChatRequest {
        var systemParts: [String] = []
        if let instructions = request.instructions, !instructions.isEmpty {
            systemParts.append(instructions)
        }
        var messages: [OpenAIChatMessage] = []
        for item in request.input ?? [] {
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
                let callID = item.callID ?? "call_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "").prefix(24)
                let function = OpenAIFunctionCall(name: item.name ?? "", arguments: item.arguments ?? "{}")
                messages.append(OpenAIChatMessage(
                    role: "assistant", content: nil,
                    toolCalls: [OpenAIToolCall(id: String(callID), type: "function", function: function)],
                    toolCallID: nil, name: nil))
            case "function_call_output":
                messages.append(OpenAIChatMessage(
                    role: "tool", content: .text(item.output ?? ""),
                    toolCalls: nil, toolCallID: item.callID, name: nil))
            default:
                throw ServerRequestError.invalid(
                    message: "unsupported input item type \(item.type)",
                    param: "input", code: "unsupported_input")
            }
        }
        var chatMessages = messages
        if !systemParts.isEmpty {
            chatMessages.insert(OpenAIChatMessage(
                role: "system", content: .text(systemParts.joined(separator: "\n\n")),
                toolCalls: nil, toolCallID: nil, name: nil), at: 0)
        }
        let tools: [OpenAITool]? = (request.tools ?? []).compactMap { tool in
            guard tool.type == "function", let name = tool.name, !name.isEmpty else { return nil }
            let parameters: JSONValue = tool.parameters ?? .object(["type": .string("object"), "properties": .object([:])])
            return OpenAITool(
                type: "function",
                function: OpenAIFunctionDefinition(name: name,
                                                   description: tool.description,
                                                   parameters: parameters))
        }
        return OpenAIChatRequest(
            model: request.model,
            messages: chatMessages,
            stream: request.stream ?? true,
            streamOptions: nil,
            temperature: request.temperature ?? 0.2,
            topP: request.topP,
            // Codex and OpenCode omit max_output_tokens, so the default here
            // is the effective cap for them. Bound it generously (16k) so
            // long reasoning + answer generations can run to completion;
            // maxNewTokens is still clamped to the available context in
            // ServerInference. The previous 2,048 default truncated answers
            // whose thinking pass alone consumed the whole budget.
            maxTokens: request.maxOutputTokens ?? 16_384,
            maxCompletionTokens: nil,
            stop: nil,
            seed: nil,
            tools: (tools?.isEmpty ?? true) ? nil : tools,
            toolChoice: nil,
            // parallel_tool_calls=false is not supported by the chat path;
            // forward nothing so the validator's default applies.
            parallelToolCalls: nil,
            topK: nil,
            repetitionPenalty: nil,
            n: 1,
            logprobs: nil,
            presencePenalty: nil,
            frequencyPenalty: nil)
    }
}

// MARK: - Responses response object builders

public enum ResponsesAPIBuilder {
    public static func responseID() -> String {
        "resp_" + UUID().uuidString.lowercased().replacingOccurrences(of: "-", with: "")
    }

    public static func responseObject(id: String,
                                      created: Int,
                                      model: String,
                                      status: String,
                                      output: [[String: Any]],
                                      usage: OpenAIUsage?,
                                      store: Bool = false,
                                      temperature: Float? = nil) -> [String: Any] {
        var object: [String: Any] = [
            "id": id,
            "object": "response",
            "created_at": created,
            "status": status,
            "error": NSNull(),
            "incomplete_details": NSNull(),
            "instructions": NSNull(),
            "max_output_tokens": NSNull(),
            "model": model,
            "output": output,
            "parallel_tool_calls": true,
            "previous_response_id": NSNull(),
            "reasoning": ["effort": NSNull(), "summary": NSNull()],
            "store": store,
            "temperature": temperature.map { $0 as Any } ?? NSNull(),
            "text": ["format": ["type": "text"]],
            "tool_choice": "auto",
            "tools": [],
            "top_p": NSNull(),
            "truncation": NSNull(),
            "usage": NSNull(),
            "user": NSNull(),
            "metadata": [:],
        ]
        if let usage {
            object["usage"] = [
                "input_tokens": usage.promptTokens,
                "input_tokens_details": ["cached_tokens": usage.promptTokensDetails.cachedTokens],
                "output_tokens": usage.completionTokens,
                "output_tokens_details": ["reasoning_tokens": 0],
                "total_tokens": usage.totalTokens,
            ]
        }
        return object
    }

    public static func messageItem(id: String,
                                   role: String,
                                   text: String,
                                   status: String) -> [String: Any] {
        ["id": id, "type": "message", "role": role, "status": status,
         "content": [["type": "output_text", "text": text, "annotations": []]]]
    }

    public static func functionCallItem(id: String,
                                        name: String,
                                        arguments: String,
                                        callID: String,
                                        outputIndex: Int,
                                        status: String) -> [String: Any] {
        ["id": id, "type": "function_call", "status": status,
         "name": name, "arguments": arguments, "call_id": callID,
         "output_index": outputIndex]
    }

    /// Output items for a completed generation (JSON mode / response.completed).
    public static func outputItems(completion: ServerCompletion,
                                   idPrefix: String) -> [[String: Any]] {
        var output: [[String: Any]] = []
        var index = 0
        if !completion.content.isEmpty {
            output.append(messageItem(id: idPrefix + "_msg0", role: "assistant",
                                      text: completion.content, status: "completed"))
            index += 1
        }
        for call in completion.toolCalls {
            output.append(functionCallItem(
                id: idPrefix + "_fc\(index)", name: call.name,
                arguments: call.argumentsJSON, callID: call.id,
                outputIndex: index, status: "completed"))
            index += 1
        }
        return output
    }
}
