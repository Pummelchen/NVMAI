import Foundation
import Testing
@testable import NVMAI

/// StructuredAssistantDecoder in ChatML mode: `<think>` suppression and
/// `<tool_call>` buffering driven by the fixture tokenizer's added-token IDs.
@Suite("ChatML decoder")
struct ChatMLDecoderTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
                                   idGenerator: { "call_fixed" })
    }

    /// Feeds text through the streaming detokenizer so each token carries the
    /// same delta the generation loop would produce.
    private func feed(_ text: String,
                      into decoder: StructuredAssistantDecoder) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var detok = GFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        return events
    }

    private func visibleText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .content(let delta) = event { result += delta }
        }
    }

    @Test("Visible text streams through unchanged")
    func plainText() throws {
        let d = decoder()
        let events = try feed("Hello there!", into: d)
        #expect(visibleText(events) == "Hello there!")
        try d.finish()
        #expect(!d.hasToolCalls)
    }

    private func reasoningText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .reasoning(let delta) = event { result += delta }
        }
    }

    @Test("Think spans become reasoning, text after them is visible")
    func thinkBecomesReasoning() throws {
        let d = decoder()
        let events = try feed("<think>\nhidden reasoning\n</think>\n\nvisible answer", into: d)
        #expect(reasoningText(events) == "\nhidden reasoning\n")
        // The newlines between `</think>` and the answer go, as the template
        // itself strips them when it renders the turn back.
        #expect(visibleText(events) == "visible answer")
        try d.finish()
    }

    @Test("Tool call spans buffer and emit a parsed call")
    func toolCallBuffering() throws {
        let d = decoder()
        let events = try feed(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            into: d)
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Preamble text before the tool call stays visible")
    func preambleThenToolCall() throws {
        let d = decoder()
        let events = try feed(
            "Checking the weather now.\n\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
            into: d)
        #expect(visibleText(events) == "Checking the weather now.\n\n")
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Unknown tool inside a call fails closed")
    func unknownToolFails() {
        let d = decoder(allowedTools: [])
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try feed(
                "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
                into: d)
        }
    }

    @Test("Nested tool-call start is malformed")
    func nestedToolCallStart() throws {
        let d = decoder()
        _ = try d.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        }
    }

    @Test("Tool-call end without a start is malformed")
    func endWithoutStart() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.consume(tokenID: tok.toolCallEndID, delta: "")
        }
    }

    @Test("Finish with an unterminated tool call is malformed")
    func unterminatedToolCall() throws {
        let d = decoder()
        _ = try d.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Byte barrier prefix is emitted before entering thought")
    func byteBarrierBeforeThought() throws {
        let d = decoder()
        let events = try d.consume(
            tokenID: tok.thinkStartID!, delta: "\u{FFFD}<think>")
        #expect(events == [.content("\u{FFFD}")])
        _ = try d.consume(tokenID: tok.thinkEndID!, delta: "</think>")
        try d.finish()
    }

    @Test("Detokenizer tail follows visible and thought channel state")
    func tailRespectsChannel() throws {
        let d = decoder()
        #expect(try d.consumeTail("visible") == [.content("visible")])
        _ = try d.consume(tokenID: tok.thinkStartID!, delta: "<think>")
        #expect(try d.consumeTail("unfinished thought") == [.reasoning("unfinished thought")])
        try d.finish()
    }

    /// Every supported template opens `<think>` in a thinking-on generation
    /// prompt, so the model's first token is already reasoning and it never
    /// writes the opening marker. A decoder that started in the visible
    /// channel would stream the whole thought as the answer.
    @Test("A prompt that left the thought open starts in the thought channel")
    func startsInsideAnOpenThought() throws {
        let d = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [],
                                           startsInThought: true, parsesToolCalls: false)
        let events = try feed("weighing it up\n</think>\n\nthe answer", into: d)
        #expect(reasoningText(events) == "weighing it up\n")
        #expect(visibleText(events) == "the answer")
        // The very first token is already reasoning, never visible text.
        #expect(events.first.map { if case .reasoning = $0 { true } else { false } } == true)
        try d.finish()
    }

    @Test("Tool calls still parse after a thought")
    func toolCallAfterThought() throws {
        let d = StructuredAssistantDecoder(tokenizer: tok, allowedTools: ["get_weather"],
                                           startsInThought: true, idGenerator: { "call_fixed" })
        let events = try feed(
            "the user wants weather\n</think>\n\n<tool_call>\n<function=get_weather>\n"
                + "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            into: d)
        #expect(reasoningText(events) == "the user wants weather\n")
        #expect(visibleText(events).isEmpty)
        #expect(events.last == .toolCall(ParsedToolCall(
            id: "call_fixed", name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#)))
        try d.finish()
    }

    /// Without the tool template there is nothing to call, so a `<tool_call>`
    /// the model writes anyway is text -- as it is when no decoder runs.
    @Test("A thought splitter without tools leaves tool markup as text")
    func thoughtSplitterLeavesToolMarkupAlone() throws {
        let d = StructuredAssistantDecoder(tokenizer: tok, allowedTools: [],
                                           parsesToolCalls: false)
        let events = try feed("<tool_call>\nx\n</tool_call>", into: d)
        #expect(visibleText(events) == "<tool_call>\nx\n</tool_call>")
        try d.finish()
    }

    /// A stray `</think>` with no thought open is not the end of a thought,
    /// so the answer after it keeps its leading newlines.
    @Test("Only a closing thought trims the answer's leading newlines")
    func strayThinkEndDoesNotTrim() throws {
        let d = decoder()
        let events = try feed("</think>\n\nanswer", into: d)
        #expect(visibleText(events) == "\n\nanswer")
        #expect(reasoningText(events).isEmpty)
    }

    /// The fixture's template is the shipped one: thinking on leaves
    /// `<think>\n` open at the end of the generation prompt, thinking off
    /// closes it. Checked on the rendered token ids both engines decide from,
    /// for the plain and the tool-template renders.
    @Test("The rendered prompt leaves a thought open exactly when thinking is on")
    func renderedPromptOpensThoughtOnlyWithThinking() async throws {
        let thinking = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder(),
                                                  thinkingMode: .on)
        let messages = [GFTokenizer.Message(role: .user, content: "Hi <think>")]
        let rendered = try thinking.applyChatTemplate(messages)
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n"))
        for (tokenizer, open) in [(thinking, true), (tok, false)] {
            let plain = tokenizer.encode(try tokenizer.applyChatTemplate(messages), addBOS: false)
            let tooled = try tokenizer.encodeToolChat(messages: messages, tools: [])
            #expect(StructuredAssistantDecoder.promptLeavesThoughtOpen(plain, tokenizer: tokenizer) == open)
            #expect(StructuredAssistantDecoder.promptLeavesThoughtOpen(tooled, tokenizer: tokenizer) == open)
        }
    }

    /// With thinking off and no tool template there is no decoder at all, so
    /// that output keeps the verbatim path it has always had.
    @Test("Thinking off without tools builds no decoder")
    func thinkingOffBuildsNoDecoder() async throws {
        let messages = [GFTokenizer.Message(role: .user, content: "Hi")]
        let offPrompt = tok.encode(try tok.applyChatTemplate(messages), addBOS: false)
        #expect(StructuredAssistantDecoder.forGeneration(
            tokenizer: tok, promptIDs: offPrompt, allowedTools: nil) == nil)
        #expect(StructuredAssistantDecoder.forGeneration(
            tokenizer: tok, promptIDs: offPrompt, allowedTools: ["get_weather"]) != nil)

        let thinking = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder(),
                                                  thinkingMode: .on)
        let onPrompt = thinking.encode(try thinking.applyChatTemplate(messages), addBOS: false)
        let splitter = try #require(StructuredAssistantDecoder.forGeneration(
            tokenizer: thinking, promptIDs: onPrompt, allowedTools: nil))
        let events = try feed("mulling\n</think>\n\nHello", into: splitter)
        #expect(reasoningText(events) == "mulling\n")
        #expect(visibleText(events) == "Hello")
    }

    @Test("Detokenizer tail inside an unfinished tool call is never visible")
    func toolTailIsSuppressed() throws {
        let d = decoder()
        _ = try d.consume(tokenID: tok.toolCallStartID, delta: "<tool_call>")
        #expect(try d.consumeTail("hidden") == [])
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    @Test("Control token without its literal marker fails closed")
    func malformedBoundaryFails() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.consume(tokenID: tok.thinkStartID!, delta: "missing marker")
        }
    }
}
