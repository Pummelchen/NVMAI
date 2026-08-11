import Foundation
import Testing
@testable import NVMAIServerCore

@Suite struct ResponsesAPIMapperTests {
    private func decode(_ json: String) throws -> ResponsesAPIRequest {
        try JSONDecoder().decode(ResponsesAPIRequest.self, from: Data(json.utf8))
    }

    @Test func instructionsAndDeveloperMergeIntoOneLeadingSystemMessage() throws {
        let request = try decode("""
        {
          "model": "qwen3.6-35b-a3b",
          "instructions": "You are a coding agent.",
          "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Be precise."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
          ]
        }
        """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 2)
        #expect(chat.messages[0].role == "system")
        #expect(try chat.messages[0].content?.textValue() == "You are a coding agent.\n\nBe precise.")
        #expect(chat.messages[1].role == "user")
    }

    @Test func functionCallMapsToAssistantToolCalls() throws {
        let request = try decode("""
        {
          "model": "m",
          "input": [
            {"type": "function_call", "call_id": "call_1", "name": "exec_command",
             "arguments": "{\\"command\\": \\"ls\\"}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"}
          ]
        }
        """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 2)
        #expect(chat.messages[0].role == "assistant")
        #expect(chat.messages[0].toolCalls?.first?.function.name == "exec_command")
        #expect(chat.messages[1].role == "tool")
        #expect(chat.messages[1].toolCallID == "call_1")
    }

    @Test func toolsMapToChatFunctionTools() throws {
        let request = try decode("""
        {
          "model": "m",
          "input": [],
          "tools": [
            {"type": "function", "name": "apply_patch", "description": "edit files",
             "parameters": {"type": "object", "properties": {}}}
          ]
        }
        """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.tools?.count == 1)
        #expect(chat.tools?.first?.function.name == "apply_patch")
        #expect(chat.tools?.first?.type == "function")
    }

    @Test func outputTokensAndStreamMapThrough() throws {
        let request = try decode("""
        {"model": "m", "input": [], "max_output_tokens": 512, "stream": true}
        """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.maxTokens == 512)
        #expect(chat.stream == true)
        #expect(chat.messages.isEmpty)
    }

    @Test func nonTextPartsAreRejected() throws {
        let request = try decode("""
        {
          "model": "m",
          "input": [{"type": "message", "role": "user",
                     "content": [{"type": "input_image", "image_url": "x"}]}]
        }
        """)
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatRequest(request)
        }
    }

    @Test func inputItemsWithoutTypeAreInferred() throws {
        // OpenCode omits the item "type" field and relies on role+content.
        let request = try decode("""
        {
          "model": "m",
          "input": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {"call_id": "call_1", "name": "bash", "arguments": "{}"},
            {"call_id": "call_1", "output": "ok"}
          ]
        }
        """)
        let chat = try ResponsesAPIMapper.chatRequest(request)
        #expect(chat.messages.count == 4)
        #expect(chat.messages[0].role == "system")
        #expect(chat.messages[1].role == "user")
        #expect(chat.messages[2].role == "assistant")
        #expect(chat.messages[2].toolCalls?.first?.function.name == "bash")
        #expect(chat.messages[3].role == "tool")
        #expect(chat.messages[3].toolCallID == "call_1")
    }

    @Test func unsupportedInputItemIsRejected() throws {
        let request = try decode("""
        {"model": "m", "input": [{"type": "computer_call", "action": "click"}]}
        """)
        #expect(throws: ServerRequestError.self) {
            _ = try ResponsesAPIMapper.chatRequest(request)
        }
    }
}
