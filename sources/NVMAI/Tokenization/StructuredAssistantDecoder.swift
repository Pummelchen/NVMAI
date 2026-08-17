import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    case toolCall(ParsedToolCall)
}

/// unchecked-invariant: one decoder per generation, driven only from that
/// generation's task. Its channel/tool-token state is a running parse of a
/// single token stream and would be meaningless shared, so exclusive
/// ownership -- not locking -- is what makes it safe.
public final class StructuredAssistantDecoder: @unchecked Sendable {
    private enum Channel {
        case thought
        case visible
    }

    private let tokenizer: GFTokenizer
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var channel: Channel = .visible
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        return try consumeChatML(tokenID: tokenID, delta: delta)
    }

    /// ChatML transitions: `<think>`…`</think>` suppress thought text, and
    /// `<tool_call>`…`</tool_call>` buffer tokens for the Qwen parser. Everything
    /// else streams as visible content.
    private func consumeChatML(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = []
            return []
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            toolTokens = nil
            let text = tokenizer.decode(tokens, skipSpecialTokens: false)
            do {
                let call = try QwenToolCallParser().parse(
                    text, allowedTools: allowedTools, id: idGenerator())
                emittedCalls += 1
                return [.toolCall(call)]
            } catch {
                failed = true
                throw error
            }
        }
        if var tokens = toolTokens {
            tokens.append(tokenID)
            guard tokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
                failed = true
                throw ToolCallParserError.oversized
            }
            toolTokens = tokens
            return []
        }
        if tokenID == tokenizer.thinkStartID {
            channel = .thought
            return []
        }
        if tokenID == tokenizer.thinkEndID {
            channel = .visible
            return []
        }
        guard channel != .thought else { return [] }
        return delta.isEmpty ? [] : [.content(delta)]
    }

    public func finish() throws {
        guard !failed, toolTokens == nil else {
            throw ToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
}
