import Foundation
import NVMAI

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let mtpModel: String?
    public let mtpMemoryMiB: Int
    public let port: Int
    /// Explicit --model-id value; nil defers to the loaded model's family
    /// default (gemma-4-26b-a4b-it or qwen3.6-35b-a3b).
    public let modelIDOverride: String?
    public var modelID: String { modelIDOverride ?? "gemma-4-26b-a4b-it" }
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryMiB: Int
    public let promptCacheDiskDirectory: String?
    public let promptCacheDiskMiB: Int
    public let prefillChunkTokens: Int?

    public static let usage = """
    usage: NVMAIServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --mtp-model <dir>      Optional native Qwen3.6 MTP sidecar directory.
      --mtp-memory-mib <MiB> Strict incremental MTP budget, 256...512
                             (default 384).
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default derived from the
                             installed model: gemma-4-26b-a4b-it or
                             qwen3.6-35b-a3b).
      --max-context <tokens> 4096, 8192, 16384, 32768, 65536, 131072, or 262144
                             (default 16384).
      --queue-limit <count>  Maximum queued requests (default 4).
      --prompt-cache-mode <off|single-prefix|multi-prefix>
                             Prompt KV reuse mode (default multi-prefix).
      --prompt-cache-entries <count>
                             Maximum retained prefixes, 1...64 (default 4).
      --prompt-cache-memory-mib <MiB>
                             RAM snapshot budget, 0...4096 (default 256).
      --prompt-cache-disk <dir>
                             Optional persistent SSD cache directory.
      --prompt-cache-disk-mib <MiB>
                             SSD snapshot budget, 0...65536 (default 8192).
      --prefill-chunk <tokens>
                             Prefill chunk size: 32, 64, 128, 256, 512,
                             1024, 2048, or 4096 (default 1024 for Qwen).
      --help                 Show this help.
    """

    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var mtpModel: String?
        var mtpMemoryMiB = StreamingMTPMemoryPlan.defaultBudgetMiB
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 16_384
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .multiPrefix
        var promptCacheMaximumEntries = 4
        var promptCacheMemoryMiB = 256
        var promptCacheDiskDirectory: String?
        var promptCacheDiskMiB = 8_192
        var prefillChunkTokens: Int?
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            guard index + 1 < input.count else {
                throw ServerArgumentError.invalid("\(flag) requires a value")
            }
            let value = input[index + 1]
            index += 2
            switch flag {
            case "--model":
                model = value
            case "--mtp-model":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--mtp-model must not be empty")
                }
                mtpModel = value
            case "--mtp-memory-mib":
                guard let parsed = Int(value),
                      StreamingMTPMemoryPlan.allowedBudgetMiB.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--mtp-memory-mib must be between 256 and 512")
                }
                mtpMemoryMiB = parsed
            case "--port":
                guard let parsed = Int(value), (1...65_535).contains(parsed) else {
                    throw ServerArgumentError.invalid("--port must be between 1 and 65535")
                }
                port = parsed
            case "--model-id":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid("--model-id must not be empty")
                }
                modelIDOverride = value
            case "--max-context":
                guard let parsed = Int(value),
                      RuntimeConfiguration.supportedContextTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
            case "--queue-limit":
                guard let parsed = Int(value), parsed > 0 else {
                    throw ServerArgumentError.invalid("--queue-limit must be positive")
                }
                queueLimit = parsed
            case "--prompt-cache-mode":
                guard let parsed = ServerPromptCacheMode(rawValue: value) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-mode must be off, single-prefix, or multi-prefix")
                }
                promptCacheMode = parsed
            case "--prompt-cache-entries":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-entries must be between 1 and 64")
                }
                promptCacheMaximumEntries = parsed
            case "--prompt-cache-memory-mib":
                guard let parsed = Int(value), (0...4_096).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-memory-mib must be between 0 and 4096")
                }
                promptCacheMemoryMiB = parsed
            case "--prompt-cache-disk":
                guard !value.isEmpty else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk must not be empty")
                }
                promptCacheDiskDirectory = value
            case "--prompt-cache-disk-mib":
                guard let parsed = Int(value), (0...65_536).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--prompt-cache-disk-mib must be between 0 and 65536")
                }
                promptCacheDiskMiB = parsed
            case "--prefill-chunk":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedPrefillChunkTokens.contains(parsed) else {
                    throw ServerArgumentError.invalid("--prefill-chunk is not supported")
                }
                prefillChunkTokens = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        return ServerArguments(model: model,
                               mtpModel: mtpModel,
                               mtpMemoryMiB: mtpMemoryMiB,
                               port: port,
                               modelIDOverride: modelIDOverride,
                               maxContext: maxContext,
                               queueLimit: queueLimit,
                               promptCacheMode: promptCacheMode,
                               promptCacheMaximumEntries: promptCacheMaximumEntries,
                               promptCacheMemoryMiB: promptCacheMemoryMiB,
                               promptCacheDiskDirectory: promptCacheDiskDirectory,
                               promptCacheDiskMiB: promptCacheDiskMiB,
                               prefillChunkTokens: prefillChunkTokens)
    }
}

public enum ServerArgumentError: Error, Equatable, CustomStringConvertible {
    case help
    case invalid(String)

    public var description: String {
        switch self {
        case .help: "help"
        case .invalid(let message): message
        }
    }
}
