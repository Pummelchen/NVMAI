import Foundation
import NVMAIKernelsC

/// Reads routed-expert blocks from a packed layer file, in parallel.
///
/// Device ceiling, measured against a 16.88 GiB working set -- deliberately larger
/// than this machine's page cache, so the numbers are the disk's and not RAM's:
///
///     1 thread   2.43 GB/s
///     4 threads  3.92 GB/s
///     8 threads  3.92 GB/s   (saturated)
///
/// Four readers is the knee.
///
/// What this is *not*: a fix for v3.x's fetch rate. That path already parallelises
/// through `DispatchQueue.concurrentPerform` and runs at ~2.6-2.8 GB/s, 66-72% of
/// the ceiling. And at decode's actual batch size this pool is no faster than a
/// plain `pread` -- at 128 slots the hit rate is ~92%, so a layer fetches about one
/// expert, and one read is one read:
///
///     batch 1   pool 3.41 GB/s   serial 3.44
///     batch 4   pool 5.36        serial 3.80
///     batch 8   pool 5.95        serial 3.69
///
/// The pool earns its keep only at 4-8 misses per batch, which is what a *smaller*
/// slot cache produces. It exists to make a low-RAM configuration viable, not to
/// speed up the current one.
///
/// `bypassCache` (the default) keeps expert reads out of the unified buffer cache.
/// That is a footprint decision rather than a speed one: streaming 16.88 GiB with
/// it on left the machine at 78% free memory, so the slot cache stays the only
/// cache and a declared RAM budget means what it says. Allowing the page cache
/// makes repeat reads faster but grows a second, unbounded cache in exactly the
/// memory this project exists to leave alone.
/// unchecked-invariant: the C reader serializes batch publication under its
/// mutex and this wrapper stores no mutable Swift state.
public final class ParallelExpertReader: @unchecked Sendable {
    /// The C reader owns a fixed pool created in `init` and
    /// serialises every batch behind its own mutex, so concurrent `fetch` calls
    /// are safe at the C level. This type adds no Swift mutable state -- the two
    /// stored properties are immutable after init -- so there is nothing here for
    /// a second caller to corrupt.
    private let handle: OpaquePointer

    public let threadCount: Int
    public let expertStride: Int

    public enum Failure: Error, CustomStringConvertible {
        case openFailed(path: String, errno: Int32)
        case readFailed(errno: Int32)

        public var description: String {
            switch self {
            case .openFailed(let path, let code):
                return "parallel expert reader could not open \(path): "
                    + String(cString: strerror(code))
            case .readFailed(let code):
                return "parallel expert read failed: \(String(cString: strerror(code)))"
            }
        }
    }

    /// - Parameters:
    ///   - threads: readers to run concurrently; clamped to 1...16 by the C layer.
    ///     Four saturates the development machine.
    ///   - bypassCache: keep reads out of the page cache. Default on, because a
    ///     bounded footprint is the point of streaming.
    public init(path: String,
                expertStride: Int,
                threads: Int = 4,
                bypassCache: Bool = true) throws {
        precondition(expertStride > 0, "expertStride must be positive")
        var failure: Int32 = 0
        guard let handle = nvmai_expert_reader_create(path,
                                                     expertStride,
                                                     Int32(threads),
                                                     bypassCache ? 1 : 0,
                                                     &failure) else {
            throw Failure.openFailed(path: path, errno: failure)
        }
        self.handle = handle
        self.expertStride = expertStride
        self.threadCount = Int(nvmai_expert_reader_threads(handle))
    }

    deinit {
        nvmai_expert_reader_destroy(handle)
    }

    /// Reads `experts[i]` into `destinations[i]` and returns once all have landed.
    ///
    /// Each destination must be at least `expertStride` bytes. On failure the
    /// destinations hold undefined bytes and must not be used -- a partially
    /// filled expert slot would otherwise be indistinguishable from a valid one.
    public func fetch(experts: [UInt32],
                      into destinations: [UnsafeMutableRawPointer]) throws {
        precondition(experts.count == destinations.count,
                     "experts and destinations must be the same length")
        guard !experts.isEmpty else { return }
        let status = destinations.withUnsafeBufferPointer { dst in
            // `void *const *` imports with an optional element type. A
            // non-optional pointer has identical layout to its optional, so the
            // rebind is a type-level adjustment and moves no bytes.
            dst.withMemoryRebound(to: UnsafeMutableRawPointer?.self) { rebound in
                experts.withUnsafeBufferPointer { ids in
                    nvmai_expert_reader_fetch(handle,
                                              ids.baseAddress,
                                              rebound.baseAddress,
                                              experts.count)
                }
            }
        }
        if status != 0 {
            throw Failure.readFailed(errno: status)
        }
    }

    /// As `fetch(experts:into:)` but with absolute byte offsets.
    ///
    /// The streamer's regions carry a per-layer base and a container offset, so an
    /// expert index alone would address the wrong layer.
    public func fetch(offsets: [UInt64],
                      into destinations: [UnsafeMutableRawPointer]) throws {
        precondition(offsets.count == destinations.count,
                     "offsets and destinations must be the same length")
        guard !offsets.isEmpty else { return }
        let status = destinations.withUnsafeBufferPointer { dst in
            dst.withMemoryRebound(to: UnsafeMutableRawPointer?.self) { rebound in
                offsets.withUnsafeBufferPointer { offs in
                    nvmai_expert_reader_fetch_offsets(handle,
                                                      offs.baseAddress,
                                                      rebound.baseAddress,
                                                      offsets.count)
                }
            }
        }
        if status != 0 {
            throw Failure.readFailed(errno: status)
        }
    }
}
