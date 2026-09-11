import Foundation

public enum StructuredAssistantEvent: Equatable, Sendable {
    case content(String)
    /// Text the model wrote inside `<think>`…`</think>`: its reasoning, which
    /// a client shows apart from the answer, or not at all.
    case reasoning(String)
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
    private let parsesToolCalls: Bool
    private let idGenerator: @Sendable () -> String
    private var channel: Channel
    /// Set when a thought closes, until the answer's first character. The
    /// templates strip the newlines between `</think>` and the answer when
    /// they render the turn back (`content.split('</think>')[-1].lstrip('\n')`),
    /// so the answer reported here starts where the template's would.
    private var trimsLeadingNewlines = false
    private var toolTokens: [Int32]?
    private var emittedCalls = 0
    private var failed = false

    /// `startsInThought` is true when the rendered prompt already opened a
    /// `<think>` block: the model's first token is then reasoning, and it
    /// never writes the opening marker itself. `parsesToolCalls` is false for
    /// a prompt rendered without the tool template, where there is no tool to
    /// call; a `<tool_call>` the model writes anyway streams as text, exactly
    /// as it does when no decoder runs at all.
    public init(tokenizer: GFTokenizer,
                allowedTools: Set<String>,
                startsInThought: Bool = false,
                parsesToolCalls: Bool = true,
                idGenerator: @escaping @Sendable () -> String = {
                    "call_" + (0..<24).map { _ in String(format: "%x", UInt8.random(in: 0...15)) }.joined()
                }) {
        self.tokenizer = tokenizer
        self.allowedTools = allowedTools
        self.parsesToolCalls = parsesToolCalls
        self.idGenerator = idGenerator
        self.channel = startsInThought ? .thought : .visible
    }

    /// The decoder a generation's output runs through. Both engines ask this
    /// one question, so the thought split is the same rule wherever a model
    /// runs.
    ///
    /// A tool-templated prompt parses its calls here. Every other prompt gets a
    /// decoder that splits thoughts alone -- `parsesToolCalls: false`, so a
    /// `<tool_call>` a model writes without a tool template still streams as
    /// text.
    ///
    /// Thinking off used to mean *no decoder*, on the reasoning that a prompt
    /// which closes the block cannot produce a thought to split, and the
    /// verbatim path that produced was simpler and byte-stable. Qwen
    /// AgentWorld 35B-A3B 8-bit disproves it: with `--reasoning off` it opens a
    /// `<think>` block of its own, spends the entire token budget inside it,
    /// and that verbatim path streamed the scaffold as the *answer*, so a
    /// client that capped tokens received no answer at all and
    /// `reasoning_content` stayed empty. It is not a decoder: the channel is
    /// decided by the prompt (`startsInThought`), so a generation that opens no
    /// marker still emits exactly the same text, in the same channel, as the
    /// verbatim path did.
    ///
    /// There is no nil case left to return. A loaded tokenizer always resolves
    /// `<think>`/`</think>` -- `resolveChatMLTokens` throws when either is
    /// missing -- so the old `nil` was reachable only through the thinking-off
    /// shortcut this corrects. `allowedTools` is nil when the prompt was
    /// rendered without the tool template.
    public static func forGeneration(tokenizer: GFTokenizer,
                                     promptIDs: [Int32],
                                     allowedTools: Set<String>?) -> StructuredAssistantDecoder {
        let opensThought = promptLeavesThoughtOpen(promptIDs, tokenizer: tokenizer)
        if let allowedTools {
            return StructuredAssistantDecoder(tokenizer: tokenizer,
                                              allowedTools: allowedTools,
                                              startsInThought: opensThought)
        }
        return StructuredAssistantDecoder(tokenizer: tokenizer,
                                          allowedTools: [],
                                          startsInThought: opensThought,
                                          parsesToolCalls: false)
    }

    /// Whether generation begins inside a thought.
    ///
    /// Every supported template's thinking-on generation prompt ends in
    /// `<think>\n`, so the model starts mid-thought and its first `<think>`
    /// is never generated; with thinking off the prompt closes the block
    /// (`<think>\n\n</think>\n\n`). Only the tokens after the last
    /// `<|im_end|>` are read. That is the generation prompt, and an earlier
    /// turn's markers -- a replayed thought, a user quoting the tag -- say
    /// nothing about where this generation starts.
    public static func promptLeavesThoughtOpen(_ promptIDs: [Int32],
                                               tokenizer: GFTokenizer) -> Bool {
        guard let start = tokenizer.thinkStartID, let end = tokenizer.thinkEndID else {
            return false
        }
        let generationPrompt = promptIDs.lastIndex(of: tokenizer.endOfTurnID)
            .map { promptIDs[($0 + 1)...] } ?? promptIDs[...]
        var open = false
        for id in generationPrompt {
            if id == start { open = true } else if id == end { open = false }
        }
        return open
    }

    public func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        return try consumeChatML(tokenID: tokenID, delta: delta)
    }

    /// ChatML transitions: `<think>`…`</think>` carry reasoning, and
    /// `<tool_call>`…`</tool_call>` buffer tokens for the Qwen parser.
    /// Everything else streams in whichever channel is open.
    private func consumeChatML(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        if parsesToolCalls, let events = try consumeToolToken(tokenID: tokenID, delta: delta) {
            return events
        }
        if tokenID == tokenizer.thinkStartID {
            let prefix = try boundaryPrefix(delta, marker: "<think>")
            let events = channelEvents(prefix)
            channel = .thought
            trimsLeadingNewlines = false
            return events
        }
        if tokenID == tokenizer.thinkEndID {
            let prefix = try boundaryPrefix(delta, marker: "</think>")
            let events = channelEvents(prefix)
            trimsLeadingNewlines = channel == .thought
            channel = .visible
            return events
        }
        return channelEvents(delta)
    }

    /// A tool marker, or a token inside an open call; nil for anything else.
    private func consumeToolToken(tokenID: Int32,
                                  delta: String) throws -> [StructuredAssistantEvent]? {
        if tokenID == tokenizer.toolCallStartID {
            guard toolTokens == nil else {
                failed = true
                throw ToolCallParserError.malformed
            }
            let prefix = try boundaryPrefix(delta, marker: "<tool_call>")
            let events = channelEvents(prefix)
            toolTokens = []
            return events
        }
        if tokenID == tokenizer.toolCallEndID {
            guard let tokens = toolTokens else {
                failed = true
                throw ToolCallParserError.malformed
            }
            _ = try boundaryPrefix(delta, marker: "</tool_call>")
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
        guard var tokens = toolTokens else { return nil }
        tokens.append(tokenID)
        guard tokens.count * MemoryLayout<Int32>.size <= QwenToolCallParser.maximumBytes else {
            failed = true
            throw ToolCallParserError.oversized
        }
        toolTokens = tokens
        return []
    }

    /// Routes the detokenizer's final buffered bytes through the current
    /// channel: a thought's tail is reasoning, and an unfinished tool call's
    /// tail never becomes visible.
    public func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        guard toolTokens == nil else { return [] }
        return channelEvents(text)
    }

    private func boundaryPrefix(_ delta: String, marker: String) throws -> String {
        guard delta.hasSuffix(marker) else {
            failed = true
            throw ToolCallParserError.malformed
        }
        return String(delta.dropLast(marker.count))
    }

    private func channelEvents(_ text: String) -> [StructuredAssistantEvent] {
        switch channel {
        case .thought:
            return text.isEmpty ? [] : [.reasoning(text)]
        case .visible:
            var visible = Substring(text)
            if trimsLeadingNewlines {
                visible = visible.drop { $0 == "\n" }
                guard !visible.isEmpty else { return [] }
                trimsLeadingNewlines = false
            }
            return visible.isEmpty ? [] : [.content(String(visible))]
        }
    }

    public func finish() throws {
        guard !failed, toolTokens == nil else {
            throw ToolCallParserError.malformed
        }
    }

    public var hasToolCalls: Bool { emittedCalls > 0 }
}
