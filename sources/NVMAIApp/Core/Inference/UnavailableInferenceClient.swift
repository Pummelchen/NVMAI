import Foundation

/// Placeholder inference client used when the decode-service client cannot be
/// constructed (e.g. the bundled service executable path is unavailable).
/// Every operation fails with the recorded error so the app stays interactive
/// and surfaces the problem in the UI instead of crashing at launch (D4).
public final class UnavailableInferenceClient: AppModelLifecycleClient, @unchecked Sendable {
    private let failure: AppInferenceError

    public init(failure: AppInferenceError) {
        self.failure = failure
    }

    public func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                             options: AppRuntimeOptions, forceLogitsHead: Bool,
                             onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
        onState(.failed(failure))
        throw failure
    }

    public func unload() async {}

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish(throwing: failure)
        }
    }

    public func cancel() {}
}
