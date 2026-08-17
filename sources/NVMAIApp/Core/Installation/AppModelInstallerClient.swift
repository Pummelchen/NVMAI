import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    /// Re-issues the install receipt for a model whose payload is intact but
    /// whose receipt is bound to a different directory. Re-hashes locally
    /// against the manifest; never touches the network.
    func reattestInstall(outputDirectory: URL) async throws
    func cancel()
}
