import Foundation

public enum RemoteChunkPolicy {
    /// Default ranged-GET chunk size. The default sits inside the valid range
    /// rather than at its ceiling.
    public static let defaultBytes = 64 * 1024 * 1024
    /// Smallest accepted chunk size. Chunks below 4 KiB would fragment the
    /// download into pointless requests.
    public static let minBytes = 4 * 1024
    /// Hard ceiling for a single ranged request (4x the default). Requests
    /// larger than this are split before they are issued.
    public static let maxBytes = 256 * 1024 * 1024
}
