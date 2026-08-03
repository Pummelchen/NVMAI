import Foundation
import Testing

@testable import TurboFieldfare
@testable import TurboFieldfareServerCore

@Suite("OpenCode request filter")
struct OpenCodeRequestFilterTests {
    @Test func profileSelectionRequiresExplicitOpenCodeClient() throws {
        #expect(try OpenCodeRequestFilter.resolve(client: nil, profile: nil) == nil)
        #expect(try OpenCodeRequestFilter.resolve(
            client: " OpenCode ",
            profile: " CODING-LEAN ") == .codingLean)
        #expect(try OpenCodeRequestFilter.resolve(
            client: "opencode",
            profile: "prompt-only") == .promptOnly)
        #expect(throws: ServerRequestError.self) {
            try OpenCodeRequestFilter.resolve(client: nil, profile: "coding-lean")
        }
        #expect(throws: ServerRequestError.self) {
            try OpenCodeRequestFilter.resolve(client: "opencode", profile: "fast")
        }
    }

    @Test func codingLeanCompactsGuidanceAndToolsWhilePreservingToolHistory() async throws {
        let original = try validatedFixture("opencode-1.15.11-tool-result.json")
        let filtered = try OpenCodeRequestFilter.apply(
            original,
            profile: .codingLean,
            originalBodyBytes: 12_153)

        #expect(filtered.messages.map(\.role) == [.system, .user, .assistant, .tool])
        #expect(filtered.messages[0].content == OpenCodeRequestFilter.codingSystemPrompt)
        #expect(filtered.messages[2].toolCalls == original.messages[2].toolCalls)
        #expect(filtered.messages[3].toolCallID == original.messages[3].toolCallID)
        #expect(filtered.tools.map(\.name) == ["read"])
        #expect(filtered.tools[0].description.count <= 240)
        #expect(filtered.tools[0].parameters.objectValue?["$schema"] == nil)
        #expect(filtered.filterAudit?.profile == .codingLean)
        #expect(filtered.filterAudit?.originalBodyBytes == 12_153)

        let tokenizer = try await GFTokenizer.load()
        let originalIDs = try tokenizer.encodeToolChat(
            messages: original.messages,
            tools: original.tools)
        let filteredIDs = try tokenizer.encodeToolChat(
            messages: filtered.messages,
            tools: filtered.tools)
        #expect(filteredIDs.count * 2 < originalIDs.count)
    }

    @Test func codingLeanWhitelistsOnlyCoreCodingTools() throws {
        let request = try decoded(#"""
        {
          "model":"m",
          "messages":[
            {"role":"system","content":"large client prompt"},
            {"role":"developer","content":"more client guidance"},
            {"role":"user","content":"fix the bug"}
          ],
          "tools":[
            {"type":"function","function":{"name":"read","description":"Read files.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to read."}}}}},
            {"type":"function","function":{"name":"webfetch","description":"Fetch the web.","parameters":{"type":"object","properties":{}}}}
          ]
        }
        """#)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let filtered = try OpenCodeRequestFilter.apply(
            validated,
            profile: .codingLean,
            originalBodyBytes: 512)

        #expect(filtered.messages.map(\.role) == [.system, .user])
        #expect(filtered.messages[1].content == "fix the bug")
        #expect(filtered.tools.map(\.name) == ["read"])
    }

    @Test func promptOnlyKeepsExactlyTheLatestUserPrompt() throws {
        let request = try decoded(#"""
        {
          "model":"m",
          "messages":[
            {"role":"system","content":"client instructions"},
            {"role":"user","content":"old question"},
            {"role":"assistant","content":"old answer"},
            {"role":"user","content":"real prompt"}
          ],
          "tools":[
            {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{}}}}
          ],
          "stream":true,
          "max_completion_tokens":123
        }
        """#)
        let validated = try OpenAIRequestValidator.validate(request, modelID: "m")
        let filtered = try OpenCodeRequestFilter.apply(
            validated,
            profile: .promptOnly,
            originalBodyBytes: 640)

        #expect(filtered.messages == [GFTokenizer.Message(role: .user, content: "real prompt")])
        #expect(filtered.tools.isEmpty)
        #expect(filtered.stream)
        #expect(filtered.maximumCompletionTokens == 123)
        #expect(filtered.filterAudit?.profile == .promptOnly)
    }

    private func validatedFixture(_ name: String) throws -> ValidatedChatRequest {
        let url = try #require(Bundle.module.url(
            forResource: name,
            withExtension: nil,
            subdirectory: "Fixtures"))
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self,
            from: Data(contentsOf: url))
        return try OpenAIRequestValidator.validate(
            request,
            modelID: "gemma-4-26b-a4b-it")
    }

    private func decoded(_ json: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
    }
}
