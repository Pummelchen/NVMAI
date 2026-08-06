import Foundation

/// Per-worker scratch budget ceiling. The M1a copy budget caps per-worker
/// staging well under 1 MB; the streaming repacker allocates its tile-bounded
/// scratch directly, so this type only publishes the shared limit.
public enum BoundedScratch {
    public static let defaultLimitBytes: Int = 1_048_576 - 1  // strictly under 1 MB
}
