import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
    /// Aborts an in-flight service-side load. The default implementation is a
    /// no-op: in-process clients stop their load through app-side task
    /// cancellation.
    func cancelLoad()
}

public extension AppModelLifecycleClient {
    func cancelLoad() {}
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}
