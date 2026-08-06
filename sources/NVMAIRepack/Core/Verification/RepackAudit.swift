import Foundation

/// Counters tracked during a repack run. Emitted to JSON via the
/// `--copy-audit` flag.
public final class RepackAudit {
    public var sourceBytesRead: UInt64 = 0
    public var outputBytesWritten: UInt64 = 0
    public var intentionalCopyBytes: UInt64 = 0
    public var byteCopyTiles: UInt64 = 0
    public var largestScratchBytes: Int = 0
    public var wholeFileHeapBuffers: Bool = false
    public var bitWidthOverridesHonored: Int = 0
    public var sourceSnapshotSha256: String = ""
    public var tensorsDroppedMultimodal: [String] = []
    public var wallTimeSeconds: Double = 0
    public var outputFiles: [OutputFile] = []
    public var packedExpertLayoutMode: String = "identity"
    public var remoteBytesDownloaded: UInt64 = 0
    public var remoteGapBytesDownloaded: UInt64 = 0
    public var remoteRangeRequests: UInt64 = 0
    public var remoteRangeRetries: UInt64 = 0
    public var remoteRangeStreamingSupported: Bool = false
    public var largestRemoteTransferBytes: Int = 0
    public var remoteRepoID: String?
    public var remoteRequestedRevision: String?
    public var remoteResolvedCommit: String?
    public var remoteRetries: [RemoteRetryRecord] = []

    public init() {}

    public struct OutputFile {
        public let relativePath: String
        public let size: UInt64
        public let sha256: String
    }

    public struct RemoteRetryRecord {
        public let label: String
        public let attempt: Int
        public let detail: String
    }

    public func recordTile(bytes: Int) {
        byteCopyTiles &+= 1
        intentionalCopyBytes &+= UInt64(bytes)
    }

    public func recordWrite(bytes: Int) {
        outputBytesWritten &+= UInt64(bytes)
    }

    public func recordRead(bytes: Int) {
        sourceBytesRead &+= UInt64(bytes)
    }

    public func recordRemoteRetry(label: String, attempt: Int, detail: String) {
        remoteRangeRetries &+= 1
        remoteRetries.append(RemoteRetryRecord(label: label, attempt: attempt, detail: detail))
    }

    public func toJSONData(outputDir: String) throws -> Data {
        var filesArr: [[String: Any]] = []
        for f in outputFiles {
            filesArr.append([
                "path": f.relativePath,
                "size": f.size,
                "sha256": f.sha256
            ])
        }
        var retryArr: [[String: Any]] = []
        for retry in remoteRetries {
            retryArr.append([
                "label": retry.label,
                "attempt": retry.attempt,
                "detail": retry.detail
            ])
        }
        var dict: [String: Any] = [
            "output_dir": outputDir,
            "source_bytes_read": sourceBytesRead,
            "output_bytes_written": outputBytesWritten,
            "intentional_copy_bytes": intentionalCopyBytes,
            "byte_copy_tiles": byteCopyTiles,
            "largest_scratch_bytes": largestScratchBytes,
            "whole_file_heap_buffers": wholeFileHeapBuffers,
            "bit_width_overrides_honored": bitWidthOverridesHonored,
            "source_snapshot_sha256": sourceSnapshotSha256,
            "tensors_dropped_multimodal": tensorsDroppedMultimodal,
            "wall_time_s": wallTimeSeconds,
            "packed_expert_layout_mode": packedExpertLayoutMode,
            "remote_bytes_downloaded": remoteBytesDownloaded,
            "remote_gap_bytes_downloaded": remoteGapBytesDownloaded,
            "remote_range_requests": remoteRangeRequests,
            "remote_range_retries": remoteRangeRetries,
            "remote_range_streaming_supported": remoteRangeStreamingSupported,
            "largest_remote_transfer_bytes": largestRemoteTransferBytes,
            "remote_retries": retryArr,
            "output_files": filesArr
        ]
        if let remoteRepoID {
            dict["remote_repo_id"] = remoteRepoID
        }
        if let remoteRequestedRevision {
            dict["remote_requested_revision"] = remoteRequestedRevision
        }
        if let remoteResolvedCommit {
            dict["remote_resolved_commit"] = remoteResolvedCommit
        }
        return try JSONSerialization.data(withJSONObject: dict,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
