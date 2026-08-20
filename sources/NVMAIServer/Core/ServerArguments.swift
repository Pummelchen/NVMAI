import Foundation
import NVMAI

public struct ServerArguments: Equatable, Sendable {
    public let model: String
    public let mtpModel: String?
    public let mtpMemoryMiB: Int
    public let port: Int
    /// Explicit --model-id value; nil derives the API ID from the installed
    /// manifest (for example qwen3.6-35b-a3b or ornith-1.5-35b-a3b).
    public let modelIDOverride: String?
    public let maxContext: Int
    public let queueLimit: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryMiB: Int
    public let promptCacheDiskDirectory: String?
    public let promptCacheDiskMiB: Int
    public let prefillChunkTokens: Int?
    public let kvCachePrecision: KVCachePrecision
    public let ropeScalingMode: RuntimeRoPEScalingMode
    public let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use. Slots are derived from it and the
    /// model's own expert stride, so this is the knob and the slot count is the
    /// outcome. `--expert-cache-slots` still wins if both are given.
    public let expertCacheBudgetBytes: Int?
    /// Defer the model load to the first inference request.
    public let lazyLoad: Bool
    /// Release the weights after this many idle seconds; 0 disables unloading.
    public let idleUnloadSeconds: Int

    /// Unloading implies deferring the first load — loading at boot only to
    /// drop it moments later is incoherent. Derived rather than folded into
    /// `lazyLoad` so the struct stays a faithful record of what was typed.
    public var managesResidency: Bool { lazyLoad || idleUnloadSeconds > 0 }

    /// Idle unloading discards the in-memory prefix cache with the session.
    /// With a disk cache configured the entries rehydrate on reload; without
    /// one, every unload costs a full cold prefill on the next request.
    public var unloadDiscardsWarmCache: Bool {
        idleUnloadSeconds > 0
            && promptCacheMode != .off
            && promptCacheDiskDirectory == nil
    }

    public static let usage = """
    usage: NVMAIServer --model <completed .gturbo directory> [options]

      --model <dir>          Required model directory.
      --mtp-model <dir>      Optional native Qwen3.6 MTP sidecar directory.
      --mtp-memory-mib <MiB> Strict incremental MTP budget, 256...512
                             (default 384).
      --port <1...65535>     Loopback port (default 8080).
      --model-id <id>        API model identifier (default derived from the
                             installed model manifest).
      --max-context <tokens> Native: 4096...262144 (default 262144).
                             With YaRN: 524288 or 1048576 (default 1048576).
      --rope-scaling <mode>  Context scaling: none or yarn (default none).
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
                             1024, 2048, or 4096 (default 4096 for supported
                             35B-A3B text models).
      --kv-bits <4|8|16>     KV-cache storage precision (default 8).
      --expert-cache-slots <count>
                             Routed-expert cache slots per layer: 8, 16, 24,
                             32, 64, 96, or 128 (default 64). Environment
                             override: NVMAI_EXPERT_CACHE_SLOTS.
      --ram-budget <size>    Bytes the routed-expert cache may use, e.g. 8G,
                             2G, 512M. Slots are derived from this and the
                             model's expert stride, so this is the knob and
                             the slot count is the result. Default 8G, which
                             holds the measured routing working set; smaller
                             budgets are markedly slower because expert reads
                             bypass the page cache and have no fallback.
                             --expert-cache-slots overrides this.
      --lazy-load            Bind the port immediately and defer the model load
                             to the first inference request (default off).
      --idle-unload-seconds <n>
                             Release the model weights after n seconds with no
                             requests, 0...86400 (default 0, disabled). The
                             next request reloads transparently. Implies
                             --lazy-load. Pair with --prompt-cache-disk, since
                             unloading discards the in-memory prefix cache.
      --help                 Show this help.
    """

    /// lint:allow-long a flag table: one `case` per option plus its
    /// validation. Splitting it into per-group parsers would hide the
    /// exhaustive switch that makes an unhandled flag a compile-visible gap.
    public static func parse(_ input: [String]) throws -> ServerArguments {
        var model: String?
        var mtpModel: String?
        var mtpMemoryMiB = StreamingMTPMemoryPlan.defaultBudgetMiB
        var port = 8080
        var modelIDOverride: String?
        var maxContext = 262_144
        var maxContextWasSet = false
        var queueLimit = 4
        var promptCacheMode: ServerPromptCacheMode = .multiPrefix
        var promptCacheMaximumEntries = 4
        var promptCacheMemoryMiB = 256
        var promptCacheDiskDirectory: String?
        var promptCacheDiskMiB = 8_192
        var prefillChunkTokens: Int?
        var kvCachePrecision: KVCachePrecision = .int8
        var ropeScalingMode: RuntimeRoPEScalingMode = .none
        var expertCacheSlots: Int?
        var expertCacheBudgetBytes: Int?
        var lazyLoad = false
        var idleUnloadSeconds = 0
        var index = 0
        while index < input.count {
            let flag = input[index]
            if flag == "--help" || flag == "-h" { throw ServerArgumentError.help }
            // Valueless flags are consumed before the "requires a value" guard
            // below; otherwise `--lazy-load --port 9999` would swallow --port
            // as this flag's value and then reject it as unknown.
            if flag == "--lazy-load" {
                lazyLoad = true
                index += 1
                continue
            }
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
                      (1...RuntimeConfiguration.maximumContextTokens).contains(parsed) else {
                    throw ServerArgumentError.invalid("--max-context is not supported")
                }
                maxContext = parsed
                maxContextWasSet = true
            case "--rope-scaling":
                guard let parsed = RuntimeRoPEScalingMode(rawValue: value) else {
                    throw ServerArgumentError.invalid("--rope-scaling must be none or yarn")
                }
                ropeScalingMode = parsed
            case "--queue-limit":
                guard let parsed = Int(value), (1...64).contains(parsed) else {
                    throw ServerArgumentError.invalid("--queue-limit must be between 1 and 64")
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
            case "--kv-bits":
                guard let bits = Int(value),
                      let parsed = KVCachePrecision(rawValue: bits) else {
                    throw ServerArgumentError.invalid("--kv-bits must be 4, 8, or 16")
                }
                kvCachePrecision = parsed
            case "--expert-cache-slots":
                guard let parsed = Int(value),
                      RuntimeConfiguration.allowedExpertCacheSlots.contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--expert-cache-slots must be one of \(RuntimeConfiguration.allowedExpertCacheSlots)")
                }
                expertCacheSlots = parsed
            case "--ram-budget":
                guard let parsed = RuntimeConfiguration.parseBudgetBytes(value) else {
                    throw ServerArgumentError.invalid(
                        "--ram-budget must be a positive size such as 2G, 512M or a byte count")
                }
                expertCacheBudgetBytes = parsed
            case "--idle-unload-seconds":
                guard let parsed = Int(value), (0...86_400).contains(parsed) else {
                    throw ServerArgumentError.invalid(
                        "--idle-unload-seconds must be between 0 and 86400")
                }
                idleUnloadSeconds = parsed
            default:
                throw ServerArgumentError.invalid("unknown flag: \(flag)")
            }
        }
        guard let model else { throw ServerArgumentError.invalid("--model is required") }
        if ropeScalingMode == .yarn {
            if !maxContextWasSet {
                maxContext = RuntimeConfiguration.defaultYaRNContextTokens
            }
            guard RuntimeConfiguration.supportedYaRNContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid(
                    "YaRN --max-context must be 524288 or 1048576")
            }
        } else {
            guard RuntimeConfiguration.supportedContextTokens.contains(maxContext) else {
                throw ServerArgumentError.invalid("--max-context is not supported")
            }
        }
        if ropeScalingMode == .yarn, mtpModel != nil {
            throw ServerArgumentError.invalid(
                "--mtp-model cannot be combined with --rope-scaling yarn")
        }
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
                               prefillChunkTokens: prefillChunkTokens,
                               kvCachePrecision: kvCachePrecision,
                               ropeScalingMode: ropeScalingMode,
                               expertCacheSlots: expertCacheSlots,
                               expertCacheBudgetBytes: expertCacheBudgetBytes,
                               lazyLoad: lazyLoad,
                               idleUnloadSeconds: idleUnloadSeconds)
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
