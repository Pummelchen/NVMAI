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

    @Test("Think spans are suppressed, text after them is visible")
    func thinkSuppression() throws {
        let d = decoder()
        let events = try feed("<think>\nhidden reasoning\n</think>\n\nvisible answer", into: d)
        let text = visibleText(events)
        #expect(!text.contains("hidden reasoning"))
        #expect(text.contains("visible answer"))
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
        #expect(try d.consumeTail("hidden") == [])
        try d.finish()
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
