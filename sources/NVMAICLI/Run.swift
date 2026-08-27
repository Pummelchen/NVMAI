import Foundation
import Metal
import NVMAI

private struct MessageJSON: Decodable {
    let role: String
    let content: String?

    enum CodingKeys: String, CodingKey { case role, content }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(String.self, forKey: .role)
        if let value = try container.decodeIfPresent(JSONValue.self, forKey: .content) {
            content = Self.text(value)
        } else {
            content = nil
        }
    }

    private static func text(_ value: JSONValue) -> String? {
        switch value {
        case .string(let s):
            return s
        case .null:
            return nil
        case .array(let parts):
            var out = ""
            for part in parts {
                guard case .object(let dict) = part,
                      case .string(let type)? = dict["type"], type == "input_text",
                      case .string(let text)? = dict["text"] else { continue }
                out += text
            }
            return out
        default:
            return nil
        }
    }
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

/// lint:allow-long the CLI driver: parse messages, load the model, run one
/// completion, print the timing footer. It is the top-level script for a
/// one-shot tool, and its steps have no other caller.
public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        // Reasoning effort is defined per family; check it against the
        // installed manifest before any heavier work. An unreadable manifest
        // is left for the model load below, which reports it better.
        if args.reasoningEffort != nil,
           let family = try? ManifestReader.peekFamily(directoryURL: modelURL) {
            try family.validateReasoning(thinkingMode: args.thinkingMode,
                                         effort: args.reasoningEffort)
        }
        let tokenizer = try await GFTokenizer.load(
            forModelDirectory: modelURL,
            thinkingMode: args.thinkingMode,
            reasoningEffort: args.reasoningEffort)
        // Concise mode injects a per-quantization system prompt. The routed
        // expert bit width comes from the manifest so the right prompt
        // variant is selected before the full model load.
        let concisePrompt: String?
        if args.concise {
            let bits = (try? ManifestReader.load(
                directoryURL: modelURL,
                expecting: .qwen36_35B_A3B).quant?.routedExpert.weightBits) ?? 4
            concisePrompt = ConcisePrompt.prompt(forRoutedExpertBits: bits)
        } else {
            concisePrompt = nil
        }
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            if let concisePrompt {
                let messages = ConcisePrompt.appendingSystemPrompt(
                    concisePrompt,
                    to: [GFTokenizer.Message(role: .user, content: rawPrompt)])
                let rendered = try tokenizer.applyChatTemplate(messages)
                promptIds = tokenizer.encode(rendered, addBOS: false)
            } else {
                promptIds = tokenizer.encode(rawPrompt, addBOS: true)
            }
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            var messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            if let concisePrompt {
                messages = ConcisePrompt.appendingSystemPrompt(concisePrompt, to: messages)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            presencePenalty: GenerationDefaults.presencePenalty,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        let loadRuntime = try RuntimeConfiguration(
            expertCacheSlots: args.expertCacheSlots,
            rdadvisePolicy: RDAdvicePolicyMode.parse(args.rdadvise),
            forceLogitsHead: !config.isPureGreedy,
            decodeExpertExecution: try RuntimeDecodeExpertExecution.environmentValue(),
            expertIOSynchronization: try RuntimeExpertIOSynchronization.environmentValue(),
            expertIOSubmission: try RuntimeExpertIOSubmission.environmentValue())

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            streamingMode: .pread(slotCount: loadRuntime.expertCacheSlots),
            expertCachePolicy: loadRuntime.modelExpertCachePolicy,
            integrityPolicy: .resolved(directoryURL: modelURL))
        let prefillChunkTokens: Int
        switch args.prefillChunk {
        case .fixed(let tokens):
            prefillChunkTokens = tokens
        case .auto:
            prefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens
                .first(where: { $0 >= promptIds.count })
                ?? PrefillRuntimeConfig.maxChunkTokens
        case nil:
            prefillChunkTokens = model.config.family == .qwen36
                ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                : loadRuntime.prefillChunkTokens
        }
        let runtime = try RuntimeConfiguration(
            expertCacheSlots: loadRuntime.expertCacheSlots,
            expertCachePolicy: loadRuntime.expertCachePolicy,
            rdadvisePolicy: loadRuntime.rdadvisePolicy,
            prefillChunkTokens: prefillChunkTokens,
            prefillAttentionPath: loadRuntime.prefillAttentionPath,
            forceLogitsHead: !config.isPureGreedy,
            decodeExpertExecution: loadRuntime.decodeExpertExecution,
            expertIOSynchronization: loadRuntime.expertIOSynchronization,
            expertIOSubmission: loadRuntime.expertIOSubmission,
            kvCachePrecision: args.kvCachePrecision,
            ropeScalingMode: args.ropeScalingMode,
            yarnContextTokens: args.ropeScalingMode == .yarn
                ? args.maxContext : RuntimeConfiguration.defaultYaRNContextTokens)
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if ProcessInfo.processInfo.environment["TURBO_FIELDFARE_PHASES"] == "1" {
            let ms = { (n: UInt64) in String(format: "%.1f", Double(n) / 1e6) }
            let total = stats.decodeSeconds * 1000
            let accounted = Double(runner.totalCb1Nanos + runner.totalIoNanos
                                   + runner.totalCb2Nanos) / 1e6
            var lines = "\n[phases over \(stats.newTokens) tokens, decode "
            lines += String(format: "%.0f", total) + " ms]\n"
            lines += "  cb1 encode+commit: " + ms(runner.totalCb1Nanos) + " ms\n"
            lines += "  expert io await:   " + ms(runner.totalIoNanos) + " ms\n"
            lines += "  cb2 encode+commit: " + ms(runner.totalCb2Nanos) + " ms\n"
            lines += "  unaccounted (GPU waits): "
            lines += String(format: "%.1f", total - accounted) + " ms\n"
            stderr.write(Data(lines.utf8))
        }
        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok/\(String(format: "%.2f", stats.prefillSeconds))s new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}
