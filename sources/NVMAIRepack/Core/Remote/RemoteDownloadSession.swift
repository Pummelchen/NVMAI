import Foundation

public struct RemoteDownloadSessionPolicy: Sendable, Equatable {
    public var requestTimeoutSeconds: TimeInterval
    public var resourceTimeoutSeconds: TimeInterval
    public var waitsForConnectivity: Bool
    public var maximumConnectionsPerHost: Int
    public var maximumRedirects: Int

    public init(requestTimeoutSeconds: TimeInterval = 300,
                resourceTimeoutSeconds: TimeInterval = 7 * 24 * 60 * 60,
                waitsForConnectivity: Bool = true,
                maximumConnectionsPerHost: Int = 1,
                maximumRedirects: Int = 5) {
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.resourceTimeoutSeconds = resourceTimeoutSeconds
        self.waitsForConnectivity = waitsForConnectivity
        self.maximumConnectionsPerHost = maximumConnectionsPerHost
        self.maximumRedirects = maximumRedirects
    }
}

public final class RemoteDownloadSession: @unchecked Sendable {
    public let policy: RemoteDownloadSessionPolicy

    private let configuration: URLSessionConfiguration

    public init(policy: RemoteDownloadSessionPolicy = RemoteDownloadSessionPolicy(),
                protocolClasses: [AnyClass]? = nil) {
        self.policy = policy
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = policy.requestTimeoutSeconds
        configuration.timeoutIntervalForResource = policy.resourceTimeoutSeconds
        configuration.waitsForConnectivity = policy.waitsForConnectivity
        configuration.httpMaximumConnectionsPerHost = policy.maximumConnectionsPerHost
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCache = nil
        if let protocolClasses {
            configuration.protocolClasses = protocolClasses
        }
        self.configuration = configuration
    }

    public func response(for request: URLRequest) async throws -> (Data, URLResponse) {
        guard let configuration = configuration.copy() as? URLSessionConfiguration else {
            throw RepackError.configurationInvalid(detail:
                "could not copy remote metadata session configuration")
        }
        let delegate = MetadataRedirectDelegate(
            originalRequest: request,
            maximumRedirects: policy.maximumRedirects)
        let session = URLSession(configuration: configuration,
                                 delegate: delegate,
                                 delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }
        return try await session.data(for: request)
    }

    public func transfer(request: URLRequest,
                         targetPath: String,
                         expectation: RemoteRangeExpectation,
                         progress: @escaping @Sendable (UInt64) -> Void = { _ in }) async throws
        -> RemoteRangeTransferResult {
        guard let configuration = configuration.copy() as? URLSessionConfiguration else {
            throw RepackError.configurationInvalid(detail:
                "could not copy remote download session configuration")
        }
        return try await RemoteRangeTransfer.run(
            configuration: configuration,
            request: request,
            targetPath: targetPath,
            expectation: expectation,
            maximumRedirects: policy.maximumRedirects,
            progress: progress)
    }

    var configurationSnapshot: RemoteDownloadSessionConfigurationSnapshot {
        RemoteDownloadSessionConfigurationSnapshot(
            requestTimeoutSeconds: configuration.timeoutIntervalForRequest,
            resourceTimeoutSeconds: configuration.timeoutIntervalForResource,
            waitsForConnectivity: configuration.waitsForConnectivity,
            maximumConnectionsPerHost: configuration.httpMaximumConnectionsPerHost,
            requestCachePolicy: configuration.requestCachePolicy,
            hasURLCache: configuration.urlCache != nil)
    }
}

struct RemoteDownloadSessionConfigurationSnapshot: Equatable {
    let requestTimeoutSeconds: TimeInterval
    let resourceTimeoutSeconds: TimeInterval
    let waitsForConnectivity: Bool
    let maximumConnectionsPerHost: Int
    let requestCachePolicy: URLRequest.CachePolicy
    let hasURLCache: Bool
}

private final class MetadataRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private var policy: RemoteMetadataRedirectPolicy

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.policy = RemoteMetadataRedirectPolicy(
            originalRequest: originalRequest,
            maximumRedirects: maximumRedirects)
    }

    func urlSession(_ session: URLSession,
                    task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        lock.lock()
        let redirected = policy.request(proposedRequest: request)
        lock.unlock()
        completionHandler(redirected)
    }
}

struct RemoteMetadataRedirectPolicy {
    private let sourceHost: String?
    private let originalMethod: String?
    private let originalAuthorization: String?
    private let maximumRedirects: Int
    private var redirectCount = 0

    init(originalRequest: URLRequest, maximumRedirects: Int) {
        self.sourceHost = originalRequest.url?.host?.lowercased()
        self.originalMethod = originalRequest.httpMethod
        self.originalAuthorization = originalRequest.value(
            forHTTPHeaderField: "Authorization")
        self.maximumRedirects = maximumRedirects
    }

    mutating func request(proposedRequest: URLRequest) -> URLRequest? {
        redirectCount += 1
        guard redirectCount <= maximumRedirects,
              proposedRequest.url?.scheme?.lowercased() == "https" else {
            return nil
        }
        let host = proposedRequest.url?.host?.lowercased()
        let isSameHost = host == sourceHost
        // Redirect policy (shared with the ranged-GET path in
        // RemoteRangeTransfer.swift): the HF `resolve/...` endpoint answers
        // with a redirect to the object store for LFS-backed files, so both
        // paths must accept the CDN host. Same-host redirects keep the
        // Authorization header; once the host changes (e.g. to the CDN) it is
        // dropped and never re-attached.
        let isHFObjectStore = host.map(RemoteHFObjectStoreHosts.all.contains) ?? false
        guard isSameHost || isHFObjectStore else {
            return nil
        }
        var redirected = proposedRequest
        redirected.httpMethod = originalMethod
        redirected.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        redirected.setValue(
            isSameHost ? originalAuthorization : nil,
            forHTTPHeaderField: "Authorization")
        return redirected
    }
}

/// Hosts that serve Hugging Face object-store content after a `/resolve/...`
/// redirect. Both the metadata HEAD policy (RemoteDownloadSession.swift) and
/// the ranged-GET policy (RemoteRangeTransfer.swift) treat redirects to these
/// hosts as part of the normal resolve flow; the ranged-GET policy additionally
/// follows any HTTPS object-store host, because the CDN fronting a given repo
/// is not guaranteed to be one of these names.
enum RemoteHFObjectStoreHosts {
    static let all: Set<String> = [
        "cdn-lfs.huggingface.co",
        "cas-bridge.hf.co",
    ]
}
