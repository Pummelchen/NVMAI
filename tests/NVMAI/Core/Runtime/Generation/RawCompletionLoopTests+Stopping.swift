import Foundation
import Metal
import Testing

@testable import NVMAI

extension RawCompletionLoopTests {
  @Test func stopsOnEOS() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let idB = tok.encode("b", addBOS: false).first!
    let (collected, result) = try await runLoop(
      seq: [idA, idB], end: tok.eosID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .eos)
    #expect(result.newTokens == 3)
    #expect(collected.tokens.map(\.1) == [idA, idB])
  }

  @Test func stopsOnEndOfTurn() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (_, result) = try await runLoop(
      seq: [idA], end: tok.endOfTurnID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .endOfTurn)
  }

  @Test func stopsOnMaxTokensAndCountsExactly() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (collected, result) = try await runLoop(
      seq: [idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 5, temperature: 0))
    #expect(result.reason == .maxTokens)
    #expect(result.newTokens == 5)
    #expect(collected.tokens.count == 5)
    #expect(collected.tokens.map(\.0) == [0, 1, 2, 3, 4])
  }

  /// `StopReason.toolCalls` used to be decided by comparing the *stop* token to
  /// `toolResponseID`, which is never in `stopTokenIDs`, so this returned
  /// `.endOfTurn` and three callers stayed dead: tool-round prompt-cache reuse,
  /// the `orphanToolResponse` diagnostic and the `rawStop` string. The assertion
  /// fails by name on that code.
  @Test func stopsOnToolCallsAfterACompleteCall() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (_, result) = try await runLoop(
      seq: [idA, tok.toolCallStartID, tok.toolCallEndID], end: tok.endOfTurnID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .toolCalls)
  }

  /// A truncated call is not a tool turn: the model was cut off mid-call, so
  /// nothing parsed and `.endOfTurn` is the honest reason.
  @Test func anUnterminatedToolCallStaysEndOfTurn() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let (_, result) = try await runLoop(
      seq: [idA, tok.toolCallStartID], end: tok.endOfTurnID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .endOfTurn)
  }

  /// A closing marker on its own must not make the turn a tool call, or the
  /// orphan diagnostic would fire on a turn that never opened one.
  @Test func aStrayToolCallCloserStaysEndOfTurn() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let (_, result) = try await runLoop(
      seq: [tok.toolCallEndID], end: tok.endOfTurnID,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0))
    #expect(result.reason == .endOfTurn)
  }

  @Test func stopsOnStopString() async throws {
    let tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    let idA = tok.encode("a", addBOS: false).first!
    let textA = tok.decode([idA], skipSpecialTokens: true)
    let (_, result) = try await runLoop(
      seq: [idA, idA], end: idA,
      config: GenerationConfig(maxNewTokens: 50, temperature: 0, stopStrings: [textA]))
    #expect(result.reason == .stopString)
  }

}
