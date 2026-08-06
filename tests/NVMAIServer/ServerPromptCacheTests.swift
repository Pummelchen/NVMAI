import Foundation
import Testing

@testable import NVMAI
@testable import NVMAIServerCore

@Suite("Server prompt cache")
struct ServerPromptCacheTests {
    private let domain = ServerPromptCacheDomain(
        modelID: "model",
        sourceSnapshotHash: "snapshot",
        runtimeProfileHash: "profile",
        maximumContext: 16_384,
        kvStorage: "fp16",
        fp16RingEnabled: true,
        templateSHA256: "template")

    @Test func textContinuationUsesActualGeneratedHistoryAndOnlyPrefillsSuffix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let initialPrompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        let generated = tokenizer.encode("answer", addBOS: false)
        let kvBacked = initialPrompt + generated
        var cache = ServerPromptCache()
        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: initialPrompt,
                kvBacked: kvBacked,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))

        let continuation = request(messages: initial.messages + [
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(continuation.messages),
            addBOS: false)
        let match = cache.match(
            domain: domain,
            request: continuation,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer)

        guard case .hit(_, let effective, let cached) = match else {
            Issue.record("expected text continuation hit")
            return
        }
        let bridge = tokenizer.encodeTextContinuation(userContent: "second")
        #expect(cached == kvBacked.count)
        #expect(effective == kvBacked + bridge)
        #expect(!rendered.prefix(kvBacked.count).elementsEqual(kvBacked))
        #expect(effective[cached] == tokenizer.endOfTurnID)
    }

    @Test func mismatchedLineageDomainAndUnsafeStopsMiss() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()

        for reason in [StopReason.stopString, .eos] {
            let publication = cache.publish(
                domain: domain,
                request: initial,
                content: "answer",
                calls: [],
                result: rawResult(
                    prompt: prompt,
                    kvBacked: prompt,
                    boundary: tokenizer.eosID,
                    reason: reason))
            // Unsafe stop reasons must not publish a cacheable entry
            // (S17). `publish` returns nil when the entry is rejected.
            #expect(publication == nil)
        }

        cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt + tokenizer.encode("answer", addBOS: false),
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let changed = request(messages: [
            GFTokenizer.Message(role: .user, content: "changed"),
            GFTokenizer.Message(role: .assistant, content: "answer"),
            GFTokenizer.Message(role: .user, content: "second"),
        ])
        let rendered = tokenizer.encode(
            try tokenizer.applyChatTemplate(changed.messages),
            addBOS: false)
        #expect(cache.match(
            domain: domain,
            request: changed,
            renderedPromptIDs: rendered,
            tokenizer: tokenizer) == .miss)
    }

    @Test func tailCompletedStopStringDoesNotPublishPrefix() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var matcher = StreamingStopMatcher(stops: ["🌳stop"])
        #expect(matcher.push("answer 🌳") == "answer ")
        #expect(matcher.push("stop") == "")
        #expect(matcher.isStopped)

        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer ",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn),
            stopStringFiltered: matcher.isStopped)
        // A stop-string-filtered tail must not be published as a cacheable
        // entry: the cached KV would resume from a partial stop string (S17).
        #expect(publication == nil)
    }

    @Test func multiPrefixChoosesLongestExactPrefixAndUsesLRUEviction() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        var cache = ServerPromptCache(maximumEntries: 2)
        let shortPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "short",
            calls: [],
            result: rawResult(
                prompt: [1, 2],
                kvBacked: [1, 2],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let short = try #require(shortPublication)
        let longPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "long",
            calls: [],
            result: rawResult(
                prompt: [1, 2, 3],
                kvBacked: [1, 2, 3],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let long = try #require(longPublication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: [1, 2, 3, 4],
            tokenizer: tokenizer)
        #expect(match == .hit(
            entryID: long.entry.id,
            effectivePromptIDs: [1, 2, 3, 4],
            cachedPromptTokens: 3))

        let newestPublication = cache.publish(
            domain: domain,
            request: initial,
            content: "newest",
            calls: [],
            result: rawResult(
                prompt: [9],
                kvBacked: [9],
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let newest = try #require(newestPublication)
        #expect(newest.evictedEntryIDs == [short.entry.id])
        #expect(cache.entries.map(\.id) == [long.entry.id, newest.entry.id])
    }

    /// An identical-prompt replay of a published entry reports the ENTIRE
    /// prompt as cached (`cachedPromptTokens == rendered.count`): at the
    /// session layer (S12) that leaves nothing to prefill, which is what
    /// "a prompt-cache hit short-circuits prefill" means. The end-to-end
    /// resume-path half of that contract lives inside `ServerModelSession`
    /// (its private init and hardcoded production-arch load make it
    /// untestable with the synthetic toy); this pins the cache layer's half.
    @Test func identicalReplayReportsEntirePromptAsCached() async throws {
        let tokenizer = try await GFTokenizer.load(from: TokenizerFixture.folder())
        let initial = request(messages: [
            GFTokenizer.Message(role: .user, content: "first"),
        ])
        let prompt = tokenizer.encode(
            try tokenizer.applyChatTemplate(initial.messages),
            addBOS: false)
        var cache = ServerPromptCache()
        let publication = cache.publish(
            domain: domain,
            request: initial,
            content: "answer",
            calls: [],
            result: rawResult(
                prompt: prompt,
                kvBacked: prompt,
                boundary: tokenizer.endOfTurnID,
                reason: .endOfTurn))
        let entry = try #require(publication)

        let match = cache.match(
            domain: domain,
            request: initial,
            renderedPromptIDs: prompt,
            tokenizer: tokenizer)
        #expect(match == .hit(
            entryID: entry.entry.id,
            effectivePromptIDs: prompt,
            cachedPromptTokens: prompt.count))
    }

    private func request(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = []
    ) -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: messages,
            tools: tools,
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 16, temperature: 0),
            maximumCompletionTokens: 16)
    }

    private func rawResult(
        prompt: [Int32],
        kvBacked: [Int32],
        boundary: Int32,
        reason: StopReason
    ) -> RawDecodeResult {
        RawDecodeResult(
            prefillTokens: prompt.count,
            cachedPromptTokens: 0,
            computedPrefillTokens: prompt.count,
            prefillSeconds: 0,
            newTokens: 1,
            decodeSeconds: 0,
            reason: reason,
            kvPosition: kvBacked.count,
            kvBackedTokenIDs: kvBacked,
            uncommittedBoundaryTokenIDs: [boundary])
    }
}
