import Foundation
import Testing
@testable import NVMAI

@Suite struct ConcisePromptTests {
    @Test func promptSelectionIsPerQuantization() {
        #expect(ConcisePrompt.prompt(forRoutedExpertBits: 4) == ConcisePrompt.standard)
        #expect(ConcisePrompt.prompt(forRoutedExpertBits: 6) == ConcisePrompt.standard)
        #expect(ConcisePrompt.prompt(forRoutedExpertBits: 8) == ConcisePrompt.strengthened)
        #expect(ConcisePrompt.standard != ConcisePrompt.strengthened)
    }

    @Test func appendsAfterExistingSystemGuidance() {
        let messages = [
            GFTokenizer.Message(role: .user, content: "first"),
            GFTokenizer.Message(role: .system, content: "user system"),
            GFTokenizer.Message(role: .user, content: "second"),
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 4)
        #expect(injected[2].role == .system)
        #expect(injected[2].content == "CONCISE")
    }

    @Test func appendsAfterDeveloperGuidanceToo() {
        let messages = [
            GFTokenizer.Message(role: .developer, content: "dev"),
            GFTokenizer.Message(role: .user, content: "question"),
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 3)
        #expect(injected[1].role == .system)
        #expect(injected[1].content == "CONCISE")
    }

    @Test func opensTheMessagesWhenNoSystemGuidanceExists() {
        let messages = [
            GFTokenizer.Message(role: .user, content: "question"),
        ]
        let injected = ConcisePrompt.appendingSystemPrompt("CONCISE", to: messages)
        #expect(injected.count == 2)
        #expect(injected[0].role == .system)
        #expect(injected[0].content == "CONCISE")
        #expect(injected[1].role == .user)
    }

    @Test func standardAndStrengthenedPromptsAreNonEmpty() {
        #expect(!ConcisePrompt.standard.isEmpty)
        #expect(!ConcisePrompt.strengthened.isEmpty)
        #expect(ConcisePrompt.standard.contains("Never:"))
        #expect(ConcisePrompt.strengthened.contains("Never:"))
    }
}
