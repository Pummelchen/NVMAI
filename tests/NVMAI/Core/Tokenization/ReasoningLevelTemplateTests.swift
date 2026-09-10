import Foundation
import Testing
@testable import NVMAI

/// A thinking level earns its place in a family's list only by rendering a
/// prompt no other listed level renders, and a level the family refuses must
/// not be one that would have rendered something new. Both are checked here
/// by rendering, not by reading templates.
///
/// Each fixture pairs a byte copy of an installed `chat_template.jinja`
/// (md5-checked against the install on 2026-09-11; 4- and 8-bit installs
/// ship identical templates) with the synthetic ChatML vocab:
///
///   ChatMLTokenizer            Qwen 3.6 35B-A3B
///   AgentWorldChatMLTokenizer  Qwen-AgentWorld 35B-A3B
///   OrnithChatMLTokenizer      Ornith 1.5 35B-A3B
///   Qwen38ChatMLTokenizer      Qwen3.8-Flash-Next 125B-A6B
///   Qwen35ChatMLTokenizer      Qwen3.5-2B (the CPU engine)
@Suite("Reasoning levels against the installed templates")
struct ReasoningLevelTemplateTests {
    private typealias Message = GFTokenizer.Message

    enum Owner: Sendable {
        case gpu(ModelFamily)
        case cpu(CPUModelFamily)

        var levels: [ReasoningLevel] {
            switch self {
            case .gpu(let family): family.supportedReasoningLevels
            case .cpu(let family): family.supportedReasoningLevels
            }
        }

        func runtime(_ level: ReasoningLevel) throws
            -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
            switch self {
            case .gpu(let family): try family.runtimeReasoning(for: level)
            case .cpu(let family): try family.runtimeReasoning(for: level)
            }
        }
    }

    struct Template: Sendable, CustomTestStringConvertible {
        let fixture: String
        let owner: Owner
        var testDescription: String { fixture }
    }

    static let templates: [Template] = [
        Template(fixture: "ChatMLTokenizer", owner: .gpu(.qwen36)),
        Template(fixture: "AgentWorldChatMLTokenizer", owner: .gpu(.qwen36)),
        Template(fixture: "OrnithChatMLTokenizer", owner: .gpu(.qwen36)),
        Template(fixture: "Qwen38ChatMLTokenizer", owner: .gpu(.qwen38flash)),
        Template(fixture: "Qwen35ChatMLTokenizer", owner: .cpu(.qwen35Dense)),
    ]

    static let binaryFixtures = ["ChatMLTokenizer", "AgentWorldChatMLTokenizer",
                                 "OrnithChatMLTokenizer", "Qwen35ChatMLTokenizer"]

    private static func folder(_ fixture: String) throws -> URL {
        try #require(Bundle.module.url(
            forResource: fixture, withExtension: nil, subdirectory: "Fixtures"))
    }

    /// The Jinja context the tokenizer builds for these settings.
    private static func context(_ thinking: ModelThinkingMode,
                                _ effort: String?) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": thinking.isEnabled]
        if let effort { context["reasoning_effort"] = effort }
        return context
    }

    /// Renders the template itself, bypassing the typed effort enum, so a
    /// value the runtime cannot express (`high`, `max`) can still be put to
    /// the template.
    private static func render(_ tok: GFTokenizer,
                               _ context: [String: any Sendable]) throws -> String {
        let ids = try tok.tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Hi"]],
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: [],
            additionalContext: context)
        return tok.tokenizer.decode(tokens: ids, skipSpecialTokens: false)
    }

    @Test("Every supported level renders its own prompt, on both render paths",
          arguments: templates)
    func supportedLevelsRenderDistinctPrompts(_ template: Template) async throws {
        let levels = template.owner.levels
        var manual: [String] = []
        var jinja: [String] = []
        for level in levels {
            let (thinking, effort) = try template.owner.runtime(level)
            let tok = try await GFTokenizer.load(from: Self.folder(template.fixture),
                                                 thinkingMode: thinking,
                                                 reasoningEffort: effort)
            let messages = [Message(role: .user, content: "Hi")]
            manual.append(try tok.applyChatTemplate(messages))
            let ids = try tok.encodeToolChat(messages: messages, tools: [])
            jinja.append(tok.tokenizer.decode(tokens: ids.map(Int.init),
                                              skipSpecialTokens: false))
        }
        #expect(Set(manual).count == levels.count, "\(zip(levels, manual).map { "\($0): \($1)" })")
        #expect(Set(jinja).count == levels.count)
        // The runtime's ChatML renderer and the template agree on every level.
        #expect(manual == jinja)
    }

    @Test("A refused level would render nothing a supported level does not",
          arguments: templates)
    func refusedLevelsAddNoPrompt(_ template: Template) async throws {
        let tok = try await GFTokenizer.load(from: Self.folder(template.fixture))
        let levels = template.owner.levels
        var supported: Set<String> = []
        for level in levels {
            let (thinking, effort) = try template.owner.runtime(level)
            supported.insert(try Self.render(tok, Self.context(thinking, effort?.rawValue)))
        }
        #expect(supported.count == levels.count)

        for level in ReasoningLevel.allCases where !levels.contains(level) {
            #expect(throws: ReasoningLevelError.self) { try template.owner.runtime(level) }
            // What the level would have to mean to the template: plain `on` is
            // thinking with no effort named; every other level names itself.
            let context = level == .on
                ? Self.context(.on, nil)
                : Self.context(.on, level.rawValue)
            // A template that raises on the value cannot render it at all.
            guard let prompt = try? Self.render(tok, context) else { continue }
            #expect(supported.contains(prompt),
                    "\(template.fixture): \(level.rawValue) renders a prompt no supported level does")
        }
    }

    @Test("Binary templates render the same prompt whatever effort is named",
          arguments: binaryFixtures)
    func binaryTemplatesIgnoreEffort(_ fixture: String) async throws {
        let tok = try await GFTokenizer.load(from: Self.folder(fixture))
        let on = try Self.render(tok, Self.context(.on, nil))
        let off = try Self.render(tok, Self.context(.off, nil))
        #expect(on != off)
        for level in ReasoningLevel.allCases where level != .off && level != .on {
            #expect(try Self.render(tok, Self.context(.on, level.rawValue)) == on)
            #expect(try Self.render(tok, Self.context(.off, level.rawValue)) == off)
        }
    }

    @Test("Qwen3.8 accepts exactly low, medium and xhigh, and on is xhigh")
    func qwen38EffortSetIsClosed() async throws {
        let tok = try await GFTokenizer.load(from: Self.folder("Qwen38ChatMLTokenizer"))
        for name in ["minimal", "high", "max"] {
            #expect(throws: (any Error).self) {
                try Self.render(tok, Self.context(.on, name))
            }
        }
        for effort in ModelReasoningEffort.allCases {
            _ = try Self.render(tok, Self.context(.on, effort.rawValue))
        }
        // Why `.on` is not offered: it selects the default effort and renders
        // byte-identically to naming it.
        #expect(try Self.render(tok, Self.context(.on, nil))
            == Self.render(tok, Self.context(.on, "xhigh")))
    }
}
