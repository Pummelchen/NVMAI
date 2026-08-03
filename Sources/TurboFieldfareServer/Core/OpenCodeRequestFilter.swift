import Foundation
import TurboFieldfare

public enum OpenCodeRequestProfile: String, Sendable {
    case codingLean = "coding-lean"
    case promptOnly = "prompt-only"
}

public struct OpenCodeFilterAudit: Sendable {
    public let profile: OpenCodeRequestProfile
    public let originalMessages: [GFTokenizer.Message]
    public let originalTools: [GFTokenizer.FunctionDefinition]
    public let originalBodyBytes: Int

    public init(profile: OpenCodeRequestProfile,
                originalMessages: [GFTokenizer.Message],
                originalTools: [GFTokenizer.FunctionDefinition],
                originalBodyBytes: Int) {
        self.profile = profile
        self.originalMessages = originalMessages
        self.originalTools = originalTools
        self.originalBodyBytes = originalBodyBytes
    }
}

public enum OpenCodeRequestFilter {
    public static let clientHeader = "x-nvmai-client"
    public static let profileHeader = "x-nvmai-profile"
    public static let clientName = "opencode"

    public static let codingSystemPrompt = """
    You are a concise coding agent. Solve the user's request using available tools only when needed. Inspect only relevant files, preserve existing conventions, make focused changes, verify results, and report clearly. Never invent tool results.
    """

    private static let codingTools: Set<String> = [
        "apply_patch", "bash", "edit", "glob", "grep", "list", "read", "write",
    ]
    private static let discardedSchemaKeys: Set<String> = [
        "$comment", "$schema", "deprecated", "examples", "readOnly", "title", "writeOnly",
    ]

    public static func resolve(client: String?, profile: String?) throws -> OpenCodeRequestProfile? {
        guard let profile else { return nil }
        guard normalized(client) == clientName else {
            throw ServerRequestError.invalid(
                message: "x-nvmai-profile requires x-nvmai-client: opencode",
                param: profileHeader,
                code: "invalid_lean_profile")
        }
        guard let resolved = OpenCodeRequestProfile(rawValue: normalized(profile) ?? "") else {
            throw ServerRequestError.invalid(
                message: "x-nvmai-profile must be coding-lean or prompt-only",
                param: profileHeader,
                code: "invalid_lean_profile")
        }
        return resolved
    }

    public static func apply(_ request: ValidatedChatRequest,
                             profile: OpenCodeRequestProfile,
                             originalBodyBytes: Int) throws -> ValidatedChatRequest {
        let audit = OpenCodeFilterAudit(
            profile: profile,
            originalMessages: request.messages,
            originalTools: request.tools,
            originalBodyBytes: originalBodyBytes)
        let messages: [GFTokenizer.Message]
        let tools: [GFTokenizer.FunctionDefinition]

        switch profile {
        case .codingLean:
            let conversation = request.messages.filter {
                $0.role != .system && $0.role != .developer
            }
            guard conversation.contains(where: { $0.role == .user }) else {
                throw ServerRequestError.invalid(
                    message: "coding-lean requires a user message",
                    param: "messages",
                    code: "invalid_message")
            }
            messages = [GFTokenizer.Message(role: .system, content: codingSystemPrompt)]
                + conversation
            tools = request.tools.compactMap { tool in
                guard codingTools.contains(tool.name) else { return nil }
                return compact(tool)
            }
        case .promptOnly:
            guard let user = request.messages.last(where: { $0.role == .user }),
                  let content = user.content else {
                throw ServerRequestError.invalid(
                    message: "prompt-only requires a user message with text content",
                    param: "messages",
                    code: "invalid_message")
            }
            messages = [GFTokenizer.Message(role: .user, content: content)]
            tools = []
        }

        return ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: request.stream,
            includeUsage: request.includeUsage,
            generationConfig: request.generationConfig,
            maximumCompletionTokens: request.maximumCompletionTokens,
            filterAudit: audit)
    }

    private static func compact(
        _ tool: GFTokenizer.FunctionDefinition
    ) -> GFTokenizer.FunctionDefinition {
        GFTokenizer.FunctionDefinition(
            name: tool.name,
            description: shortened(tool.description, limit: 240),
            parameters: compactSchema(tool.parameters))
    }

    private static func compactSchema(_ value: JSONValue) -> JSONValue {
        switch value {
        case .object(let object):
            return .object(object.reduce(into: [:]) { result, entry in
                guard !discardedSchemaKeys.contains(entry.key) else { return }
                if entry.key == "description", case .string(let text) = entry.value {
                    result[entry.key] = .string(shortened(text, limit: 160))
                } else {
                    result[entry.key] = compactSchema(entry.value)
                }
            })
        case .array(let values):
            return .array(values.map(compactSchema))
        default:
            return value
        }
    }

    private static func shortened(_ value: String, limit: Int) -> String {
        let collapsed = value.split(whereSeparator: \Character.isWhitespace)
            .joined(separator: " ")
        guard collapsed.count > limit else { return collapsed }
        return String(collapsed.prefix(limit - 1)) + "…"
    }

    private static func normalized(_ value: String?) -> String? {
        value?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}
