import Foundation
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case unsupportedForDialect(String)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        }
    }
}

/// The binary reasoning switch exposed by compatible Qwen/Ornith chat
/// templates. Ornith 1.5 accepts `enable_thinking=true|false`; it does not
/// define low/medium/high effort levels or a thinking-token budget.
public enum ModelThinkingMode: String, Codable, CaseIterable, Sendable {
    case off
    case on

    public var isEnabled: Bool { self == .on }

    /// Backwards-compatible resolution for processes that still configure the
    /// runtime through `NVMAI_THINKING_MODE`. Unknown values retain the
    /// historical safe default of off.
    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelThinkingMode {
        switch environment["NVMAI_THINKING_MODE"]?.lowercased() {
        case "1", "on", "true", "yes": return .on
        default: return .off
        }
    }
}

/// The reasoning-effort levels defined by chat templates that support them.
/// The Qwen3.8-Flash-Next template accepts `reasoning_effort` while thinking
/// is on and injects an effort-specific instruction into the system block
/// (`xhigh` is its default; `medium` is accepted but injects no text).
/// Ornith 1.5 and Qwen 3.6 templates define no effort levels, so those
/// families reject these values at the surface instead of faking them.
public enum ModelReasoningEffort: String, Codable, CaseIterable, Sendable {
    case low
    case medium
    case xhigh

    /// Environment resolution mirroring `ModelThinkingMode.resolved`:
    /// `NVMAI_REASONING_EFFORT` selects a level, and unset or unknown values
    /// keep the safe default of nil (the template's own default applies).
    public static func resolved(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ModelReasoningEffort? {
        guard let raw = environment["NVMAI_REASONING_EFFORT"]?.lowercased() else {
            return nil
        }
        return ModelReasoningEffort(rawValue: raw)
    }
}

/// Tokenizer wrapper for the compatible Qwen3.5-MoE ChatML model family.
///
/// Loads tokenizer sidecars in a completed `.gturbo/tokenizer/` directory.
/// Exposes typed accessors for the IDs the generator actually needs (BOS / EOS /
/// pad / end-of-turn) and adapts encode/decode to Int32 to match the buffer
/// types kernels consume.
///
/// NVMAI owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
/// unchecked-invariant: immutable after `load`. The stored token ids and the
/// underlying swift-transformers tokenizer are never mutated afterwards, so
/// concurrent encode/decode calls only read.
public struct GFTokenizer: @unchecked Sendable {
    /// Nominal BOS. This is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// Alias of the `<think>` / `</think>` markers.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    public let thinkingMode: ModelThinkingMode
    /// Requested effort override for effort-aware templates; nil means the
    /// template's own default. Cleared when thinking is off because every
    /// supported template ignores effort without thinking.
    public let reasoningEffort: ModelReasoningEffort?
    /// The instruction the bundled template injects at the head of the system
    /// block for the active thinking/effort context (Qwen3.8-style templates
    /// inject an effort sentence; binary templates inject nothing). Derived
    /// from the template so it owns the wording; the manual ChatML renderer
    /// mirrors it.
    private let effortSystemInstruction: String?

    /// Generation-prompt suffix appended after the last message: derived from
    /// the tokenizer's bundled `chat_template.jinja`
    /// (`add_generation_prompt` with thinking disabled) when available,
    /// falling back to the pinned constant otherwise (R6).
    private let generationSuffix: String

    @usableFromInline
    let tokenizer: any Tokenizer
    let byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration

    public static func load(
        from folder: URL,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil
    ) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(
            .local(folder.standardizedFileURL.path, thinkingMode, reasoningEffort))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            thinkingMode: ModelThinkingMode = .off,
                            reasoningEffort: ModelReasoningEffort? = nil,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        guard let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) else {
            throw GFTokenizerError.missingToolTemplate
        }
        return try await load(from: folder,
                              thinkingMode: thinkingMode,
                              reasoningEffort: reasoningEffort)
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(
        from folder: URL,
        thinkingMode: ModelThinkingMode,
        reasoningEffort: ModelReasoningEffort? = nil
    ) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        let decoder = try GFByteLevelDecoderConfiguration.load(
            from: folder.appendingPathComponent("tokenizer.json"),
            tokenizer: underlying)
        return try GFTokenizer(tokenizer: underlying,
                               byteLevelDecoderConfiguration: decoder,
                               thinkingMode: thinkingMode,
                               reasoningEffort: reasoningEffort)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(
        tokenizer: any Tokenizer,
        thinkingMode: ModelThinkingMode = .off,
        reasoningEffort: ModelReasoningEffort? = nil
    ) throws {
        try self.init(
            tokenizer: tokenizer,
            byteLevelDecoderConfiguration: .knownChatMLTokens(tokenizer: tokenizer),
            thinkingMode: thinkingMode,
            reasoningEffort: reasoningEffort)
    }

    init(tokenizer: any Tokenizer,
         byteLevelDecoderConfiguration: GFByteLevelDecoderConfiguration,
         thinkingMode: ModelThinkingMode = .off,
         reasoningEffort: ModelReasoningEffort? = nil) throws {
        self.tokenizer = tokenizer
        self.byteLevelDecoderConfiguration = byteLevelDecoderConfiguration

        let resolved = try Self.resolveChatMLTokens(tokenizer)
        try Self.validateStreamingDecoder(byteLevelDecoderConfiguration,
                                          tokenizer: tokenizer,
                                          resolved: resolved)
        self.bosID = resolved.bosID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
        self.thinkingMode = thinkingMode
        // Every supported template ignores effort while thinking is off, so
        // an off-mode tokenizer stores none rather than an inert value.
        let activeEffort = thinkingMode.isEnabled ? reasoningEffort : nil
        self.reasoningEffort = activeEffort
        let context = Self.templateContext(thinkingEnabled: thinkingMode.isEnabled,
                                           reasoningEffort: activeEffort)
        self.generationSuffix = Self.deriveGenerationSuffix(
            tokenizer, thinkingEnabled: thinkingMode.isEnabled, context: context)
        self.effortSystemInstruction = thinkingMode.isEnabled
            ? Self.deriveEffortSystemInstruction(tokenizer, context: context)
            : nil
    }

    /// The Jinja context shared by every bundled-template render: the binary
    /// switch always, plus the effort override only when one is active (an
    /// absent key selects the template's own default level).
    private static func templateContext(
        thinkingEnabled: Bool,
        reasoningEffort: ModelReasoningEffort?
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = ["enable_thinking": thinkingEnabled]
        if let reasoningEffort {
            context["reasoning_effort"] = reasoningEffort.rawValue
        }
        return context
    }

    private var templateContext: [String: any Sendable] {
        Self.templateContext(thinkingEnabled: thinkingMode.isEnabled,
                             reasoningEffort: reasoningEffort)
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    private static func validateStreamingDecoder(
        _ decoder: GFByteLevelDecoderConfiguration,
        tokenizer: any Tokenizer,
        resolved: ResolvedSpecialTokens
    ) throws {
        let literalMarkers: [(Int32, String)] = [
            (resolved.toolCallStartID, "<tool_call>"),
            (resolved.toolCallEndID, "</tool_call>"),
            (resolved.toolResponseID, "<tool_response>"),
            (resolved.toolResponseEndID, "</tool_response>"),
            (resolved.channelStartID, "<think>"),
            (resolved.channelEndID, "</think>"),
        ]
        for (id, content) in literalMarkers {
            guard let added = decoder.addedTokens[id],
                  added.content == content, !added.special else {
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML control token \(content) must be a literal ByteLevel barrier")
            }
        }

        let filteredMarkers = [resolved.eosID, resolved.endOfTurnID]
        for id in filteredMarkers {
            guard decoder.addedTokens[id]?.special == true else {
                let token = tokenizer.convertIdToToken(Int(id)) ?? "id \(id)"
                throw GFTokenizerError.unsupportedForDialect(
                    "ChatML stop token \(token) must be marked special")
            }
        }
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    /// The model's padded embedding/lm_head row count. The tokenizer's own
    /// vocab (248 077 for Qwen) is smaller; logits buffers and the
    /// embedding/lm_head are sized to the padded rows, and `vocabSize`
    /// reports at least this many.
    private static let paddedLogitsVocabSize = 248_320

    /// Derive the tokenizer's actual vocab by probing `convertIdToToken` for
    /// the first invalid id. Standard vocab files keep ids dense from 0, so
    /// the first nil is the vocab count. Bounded so a pathological tokenizer
    /// cannot make init scan forever; nil means "no reliable derivation".
    private static func derivedVocabSize(_ tokenizer: any Tokenizer) -> Int? {
        // 2,097,152 — far above any shipping vocab.
        let upperBound = 1 << 21
        for id in 0..<upperBound where tokenizer.convertIdToToken(id) == nil {
            return id
        }
        return nil
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            // At least the model's padded embedding/lm_head rows; larger when
            // the tokenizer's own vocab (derived from `convertIdToToken`)
            // exceeds them.
            vocabSize: max(Self.derivedVocabSize(tokenizer) ?? 0,
                           Self.paddedLogitsVocabSize))
    }

    /// Encode UTF-8 text to token IDs.
    ///
    /// ChatML has no BOS, so `addBOS` is a no-op; BOS is never prepended.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Codable, Sendable {
        case system, developer, user, assistant, tool
    }
    public struct HistoricalToolCall: Codable, Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Codable, Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Codable, Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch. Used only
    /// when the tokenizer has no chat template or template rendering fails
    /// (R6); `generationSuffix` carries the template-derived value otherwise.
    private static let fallbackChatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    /// Same as `fallbackChatMLGenerationSuffix`, but for thinking mode ON:
    /// the template's `enable_thinking=true` branch leaves the `<think>`
    /// block open so the model must reason before answering.
    private static let fallbackChatMLGenerationSuffixThinking =
        "<|im_start|>assistant\n<think>\n"

    /// Derive the generation-prompt suffix from the tokenizer's bundled
    /// `chat_template.jinja` (`add_generation_prompt` with thinking per
    /// `thinkingEnabled`), falling back to the pinned constant when no
    /// template is available or rendering fails. The probe renders one empty
    /// user turn both with and without the generation prompt; the generation
    /// prompt is appended after the message loop, so the suffix is the
    /// token-level difference between the two renders.
    private static func deriveGenerationSuffix(_ tokenizer: any Tokenizer,
                                               thinkingEnabled: Bool,
                                               context: [String: any Sendable]) -> String {
        let fallback = thinkingEnabled
            ? Self.fallbackChatMLGenerationSuffixThinking
            : Self.fallbackChatMLGenerationSuffix
        guard tokenizer.hasChatTemplate else {
            return fallback
        }
        let probe: [Tokenizers.Message] = [["role": "user", "content": ""]]
        do {
            let withPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: context)
            let withoutPrompt = try tokenizer.applyChatTemplate(
                messages: probe,
                chatTemplate: nil,
                addGenerationPrompt: false,
                truncation: false,
                maxLength: nil,
                tools: [],
                additionalContext: context)
            guard withPrompt.count > withoutPrompt.count else {
                return fallback
            }
            // The generation prompt is appended after the message loop, so the
            // with-prompt render is the without-prompt render plus the suffix.
            let suffixIDs = Array(withPrompt[withoutPrompt.count...])
            return tokenizer.decode(tokens: suffixIDs, skipSpecialTokens: false)
        } catch {
            return fallback
        }
    }

    /// Derive the instruction the bundled template injects at the head of the
    /// system block for the active context. The probe renders one user turn
    /// with no system message: effort-aware templates open the render with a
    /// synthetic system block holding only the instruction, while binary
    /// templates render no leading system block at all (nil).
    private static func deriveEffortSystemInstruction(
        _ tokenizer: any Tokenizer,
        context: [String: any Sendable]
    ) -> String? {
        guard tokenizer.hasChatTemplate else { return nil }
        let probe: [Tokenizers.Message] = [["role": "user", "content": "x"]]
        guard let ids = try? tokenizer.applyChatTemplate(
            messages: probe,
            chatTemplate: nil,
            addGenerationPrompt: false,
            truncation: false,
            maxLength: nil,
            tools: [],
            additionalContext: context) else { return nil }
        let text = tokenizer.decode(tokens: ids, skipSpecialTokens: false)
        let blockStart = Self.imStartMark + "system\n"
        guard text.hasPrefix(blockStart),
              let blockEnd = text.range(of: Self.imEndMark) else { return nil }
        let instruction = String(
            text[text.index(text.startIndex, offsetBy: blockStart.count)..<blockEnd.lowerBound])
        return instruction.isEmpty ? nil : instruction
    }

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        // Every message is rendered through the ChatML template before
        // encoding; there is no other prompt path.
        try chatMLChatTemplate(messages)
    }

    private func chatMLChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        // Effort-aware templates open the conversation with the derived
        // instruction: inside the leading system block when the chat has one,
        // otherwise as a synthetic system block of its own.
        if let instruction = effortSystemInstruction, messages.first?.role != .system {
            s += Self.imStartMark + "system\n" + instruction + Self.imEndMark + "\n"
        }
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            // The bundled Jinja template trims every message's rendered
            // content (`render_content(...)|trim`); the manual renderer
            // mirrors that exactly so both paths agree byte-for-byte.
            var content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            if index == 0, message.role == .system,
               let instruction = effortSystemInstruction {
                content = content.isEmpty ? instruction : instruction + "\n\n" + content
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        s += generationSuffix
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: templateContext
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        // The template trims user content (`render_content(...)|trim`), so the
        // continuation bridge mirrors it; see `chatMLChatTemplate`.
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return [endOfTurnID] + encode(
            "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                + generationSuffix,
            addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        // The ChatML template's `<think>` stripping depends on each assistant
        // turn's position relative to the last user query, so a re-rendered
        // prefix is not guaranteed to be a token prefix of the full render.
        // Callers (ServerPromptCache) fall back to prefix matching; the
        // tool-result KV continuation is unsupported for ChatML.
        throw GFTokenizerError.unsupportedForDialect("tool-result KV continuation")
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case local(String, ModelThinkingMode, ModelReasoningEffort?)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .local(let path, let thinkingMode, let reasoningEffort):
                return try await GFTokenizer.loadUncached(
                    from: URL(fileURLWithPath: path),
                    thinkingMode: thinkingMode,
                    reasoningEffort: reasoningEffort)
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
