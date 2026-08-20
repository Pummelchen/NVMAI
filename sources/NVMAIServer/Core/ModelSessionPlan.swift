import Foundation
import NVMAI

/// The facts a caller needs before a model is resident.
///
/// All three are derivable from `manifest.json`, which the installer writes
/// next to the weights — so the startup banner reports real values even when
/// the load has been deferred, instead of placeholders that resolve later.
public struct ModelSessionFacts: Sendable, Equatable {
    public let modelID: String
    public let prefillChunkTokens: Int
    public let promptCacheMode: ServerPromptCacheMode
    /// Routed-expert slots per layer in force, so the banner can state the
    /// streaming budget instead of leaving the user to infer it from a flag they
    /// may not have passed.
    public let expertCacheSlots: Int

    public init(modelID: String,
                prefillChunkTokens: Int,
                promptCacheMode: ServerPromptCacheMode,
                expertCacheSlots: Int = 0) {
        self.modelID = modelID
        self.prefillChunkTokens = prefillChunkTokens
        self.promptCacheMode = promptCacheMode
        self.expertCacheSlots = expertCacheSlots
    }
}

enum ServerModelIdentity {
    static func apiModelID(manifestModelID: String,
                           family: ModelFamily) -> String {
        let quantizationSuffixes = ["-4bit", "-8bit", "-6bit"]
        for suffix in quantizationSuffixes where manifestModelID.hasSuffix(suffix) {
            return String(manifestModelID.dropLast(suffix.count))
        }
        if manifestModelID != "unknown/snapshot" {
            return manifestModelID
        }
        switch family {
        case .qwen36: return "qwen3.6-35b-a3b"
        case .qwen36MTP: return "qwen3.6-35b-a3b-mtp"
        }
    }
}

/// Everything needed to build a `ServerModelSession`, in one place.
///
/// Both the eager path and the deferred path construct sessions through
/// `makeSession`, so a parameter added to `ServerModelSession.load` cannot be
/// wired into one path and forgotten in the other.
public struct ModelSessionPlan: Sendable {
    public let modelDirectory: URL
    public let maxContext: Int
    public let promptCacheMode: ServerPromptCacheMode
    public let promptCacheMaximumEntries: Int
    public let promptCacheMemoryLimitBytes: Int
    public let promptCacheDiskDirectory: URL?
    public let promptCacheDiskLimitBytes: Int
    public let prefillChunkTokens: Int?
    public let expertCacheSlots: Int?
    /// Bytes the routed-expert cache may use; slots are derived from it.
    public let expertCacheBudgetBytes: Int?
    public let mtpModelDirectory: URL?
    public let mtpMemoryMiB: Int

    public init(modelDirectory: URL,
                maxContext: Int,
                promptCacheMode: ServerPromptCacheMode,
                promptCacheMaximumEntries: Int,
                promptCacheMemoryLimitBytes: Int,
                promptCacheDiskDirectory: URL?,
                promptCacheDiskLimitBytes: Int,
                prefillChunkTokens: Int?,
                expertCacheSlots: Int?,
                expertCacheBudgetBytes: Int? = nil,
                mtpModelDirectory: URL?,
                mtpMemoryMiB: Int) {
        self.modelDirectory = modelDirectory
        self.maxContext = maxContext
        self.promptCacheMode = promptCacheMode
        self.promptCacheMaximumEntries = promptCacheMaximumEntries
        self.promptCacheMemoryLimitBytes = promptCacheMemoryLimitBytes
        self.promptCacheDiskDirectory = promptCacheDiskDirectory
        self.promptCacheDiskLimitBytes = promptCacheDiskLimitBytes
        self.prefillChunkTokens = prefillChunkTokens
        self.expertCacheSlots = expertCacheSlots
        self.expertCacheBudgetBytes = expertCacheBudgetBytes
        self.mtpModelDirectory = mtpModelDirectory
        self.mtpMemoryMiB = mtpMemoryMiB
    }

    public func makeSession(
        reusingContext: MetalContext? = nil
    ) async throws -> ServerModelSession {
        try await ServerModelSession.load(
            modelDirectory: modelDirectory,
            maxContext: maxContext,
            promptCacheMode: promptCacheMode,
            promptCacheMaximumEntries: promptCacheMaximumEntries,
            promptCacheMemoryLimitBytes: promptCacheMemoryLimitBytes,
            promptCacheDiskDirectory: promptCacheDiskDirectory,
            promptCacheDiskLimitBytes: promptCacheDiskLimitBytes,
            prefillChunkTokens: prefillChunkTokens,
            expertCacheSlots: expertCacheSlots,
            expertCacheBudgetBytes: expertCacheBudgetBytes,
            mtpModelDirectory: mtpModelDirectory,
            mtpMemoryMiB: mtpMemoryMiB,
            reusingContext: reusingContext)
    }

    /// Resolve the banner facts by reading `manifest.json` only — no weights
    /// are mapped and no Metal device is created.
    ///
    /// Throwing here also preserves the eager path's behaviour that a bad
    /// `--model` fails at launch rather than on the first request.
    public func previewFacts(modelIDOverride: String? = nil) throws -> ModelSessionFacts {
        let identity = try ManifestReader.peekIdentity(directoryURL: modelDirectory)
        let family = identity.family
        let defaultModelID = ServerModelIdentity.apiModelID(
            manifestModelID: identity.modelID,
            family: family)
        // Mirrors the precedence in ServerModelSession.load: an explicit
        // --prefill-chunk wins, otherwise qwen36 takes the long-prefill chunk
        // and anything else takes the runtime default. Family is the only
        // input, and family comes from the manifest.
        let resolvedChunk = prefillChunkTokens
            ?? (family == .qwen36
                ? RuntimeConfiguration.qwenLongPrefillChunkTokens
                : RuntimeConfiguration.production.prefillChunkTokens)
        return ModelSessionFacts(
            modelID: modelIDOverride ?? defaultModelID,
            prefillChunkTokens: resolvedChunk,
            promptCacheMode: ServerModelSession.effectivePromptCacheMode(
                requested: promptCacheMode,
                mtpEnabled: mtpModelDirectory != nil))
    }
}
