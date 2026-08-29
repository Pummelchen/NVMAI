import Foundation
public struct RemoteStreamingRepackOptions: Sendable {
    public let repoID: String
    public let revision: String
    public let outputDir: String
    public let token: String?
    public let requireKnownSource: Bool
    public let copyAuditPath: String?
    public let rangeChunkBytes: Int
    public let writeTileBytes: Int
    public let minFreeReserveBytes: UInt64
    public let overwrite: Bool
    public let resume: Bool
    public let dryRunSpaceCheck: Bool
    public let downloadSession: RemoteDownloadSession
    public let baseURL: URL
    public let rangeRetryAttempts: Int
    public let retryBaseDelayNs: UInt64

    public init(repoID: String,
                revision: String,
                outputDir: String,
                token: String? = nil,
                requireKnownSource: Bool = false,
                copyAuditPath: String? = nil,
                rangeChunkBytes: Int = RemoteChunkPolicy.defaultBytes,
                writeTileBytes: Int = WriterCore.tileBytes,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024,
                overwrite: Bool = false,
                resume: Bool = false,
                dryRunSpaceCheck: Bool = false,
                downloadSession: RemoteDownloadSession = RemoteDownloadSession(),
                baseURL: URL = RemoteBaseURL.huggingFace,
                rangeRetryAttempts: Int = 4,
                retryBaseDelayNs: UInt64 = 1_000_000_000) {
        self.repoID = repoID
        self.revision = revision
        self.outputDir = outputDir
        self.token = token
        self.requireKnownSource = requireKnownSource
        self.copyAuditPath = copyAuditPath
        self.rangeChunkBytes = rangeChunkBytes
        self.writeTileBytes = writeTileBytes
        self.minFreeReserveBytes = minFreeReserveBytes
        self.overwrite = overwrite
        self.resume = resume
        self.dryRunSpaceCheck = dryRunSpaceCheck
        self.downloadSession = downloadSession
        self.baseURL = baseURL
        self.rangeRetryAttempts = rangeRetryAttempts
        self.retryBaseDelayNs = retryBaseDelayNs
    }
}

public struct RemoteStreamingRepackResult: Sendable {
    public let outputDir: String
    public let resolvedCommit: String
    let plan: RepackPlan
    /// Number of ranged HTTP requests issued this run. Retried attempts are
    /// not counted here; see `remoteRetryCount`.
    public let rangeRequestCount: Int
    public let remoteBytesToDownload: UInt64
    public let remoteGapBytesDownloaded: UInt64
    public let remoteRetryCount: UInt64
    public let reusedBytes: UInt64
    /// Unique payload bytes transferred this run: each coalesced range's
    /// successful transfer is counted once, so retries are not double-counted
    /// and the value equals `remoteBytesToDownload` on a fresh install. The
    /// live progress callbacks may temporarily report attempt-bytes while a
    /// retried transfer is re-streaming, but this final value is exact.
    public let downloadedThisRunBytes: UInt64
    public let dryRun: Bool
}

public final class RemoteStreamingRepacker {
    private let options: RemoteStreamingRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: RemoteStreamingRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> RemoteStreamingRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        var hasCheckpoint = try Posix.entryKind(paths.checkpointFile) == .regular
        if !hasPartial, hasCheckpoint,
           try Posix.entryKind(paths.finalDirectory) == .directory {
            // A previous run renamed the partial directory into place but
            // crashed before deleting the checkpoint. The final directory is
            // authoritative: re-verify its completion markers and drop the
            // stale checkpoint instead of throwing installStateCorrupt.
            let manifestKind = try Posix.entryKind((paths.finalDirectory as NSString)
                .appendingPathComponent("manifest.json"))
            let receiptKind = try Posix.entryKind((paths.finalDirectory as NSString)
                .appendingPathComponent(VerifiedInstallReceiptWriter.fileName))
            guard manifestKind == .regular, receiptKind == .regular else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "final directory exists without a complete verified install")
            }
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
            try Posix.fsyncDirectory(paths.parentDirectory)
            hasCheckpoint = false
        }
        guard hasPartial == hasCheckpoint else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        if options.resume {
            guard hasPartial else {
                throw RepackError.installStateMissing(path: paths.checkpointFile)
            }
        } else if hasPartial {
            throw RepackError.installStateIncompatible(
                detail: "saved download exists; resume or discard it")
        }
        do {
            return try await runPrepared(paths: paths, progress: progress)
        } catch {
            if !hasCheckpoint,
               (try? Posix.entryKind(paths.checkpointFile)) != .regular {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            throw error
        }
    }

    public static func inspectPersistentInstall(
        outputDirectory: String,
        repoID: String,
        requestedRevision: String
    ) throws -> RemoteInstallCheckpoint? {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        let partial = try Posix.entryKind(paths.partialDirectory)
        let checkpoint = try Posix.entryKind(paths.checkpointFile)
        if partial == .absent, checkpoint == .absent { return nil }
        if partial == .absent, checkpoint == .regular,
           try Posix.entryKind(paths.finalDirectory) == .directory {
            // Crash window: the previous run renamed partial → final but
            // crashed before deleting the checkpoint. Nothing is resumable;
            // drop the stale checkpoint and report no saved state.
            try? FileManager.default.removeItem(atPath: paths.checkpointFile)
            try Posix.fsyncDirectory(paths.parentDirectory)
            return nil
        }
        guard partial == .directory, checkpoint == .regular else {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "partial directory and checkpoint must exist together")
        }
        let value = try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
        guard value.repoID == repoID, value.requestedRevision == requestedRevision else {
            throw RepackError.installStateIncompatible(
                detail: "saved download belongs to a different source")
        }
        return value
    }

    public static func discardPartial(outputDirectory: String) throws {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        defer { withExtendedLifetime(lock) {} }
        let paths = lock.paths
        let hasPartial = try Posix.entryKind(paths.partialDirectory) != .absent
        let hasCheckpoint = try Posix.entryKind(paths.checkpointFile) != .absent
        guard hasPartial || hasCheckpoint else {
            throw RepackError.installStateMissing(path: paths.checkpointFile)
        }
        if hasPartial {
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        }
        if hasCheckpoint {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
        }
        try Posix.fsyncDirectory(paths.parentDirectory)
    }

    /// lint:allow-long the install pipeline for one prepared plan: fetch
    /// ranges, verify, write, checkpoint, promote. The stages share the
    /// checkpoint, the byte budget and the progress reporter, and their order
    /// is the resumability contract -- separating them would move that
    /// contract into parameter lists.
    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> RemoteStreamingRepackResult {
        try Task.checkCancellation()
        let saved = options.resume
            ? try RemoteInstallCheckpoint.load(from: paths.checkpointFile)
            : nil
        if let saved {
            guard saved.repoID == options.repoID,
                  saved.requestedRevision == options.revision else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download belongs to a different source")
            }
        }
        let retryPolicy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                            baseDelayNs: options.retryBaseDelayNs)
        let remote = HuggingFaceRemoteSource(repoID: options.repoID,
                                             requestedRevision: options.revision,
                                             resolvedCommit: saved?.resolvedCommit,
                                             token: options.token,
                                             downloadSession: options.downloadSession,
                                             baseURL: options.baseURL,
                                             tempDirectory: paths.partialDirectory,
                                             retryPolicy: retryPolicy)
        progress(.downloadingMetadata)
        let snapshot = try await RemoteSnapshotLoader.load(remote: remote,
                                                           requireKnownSource: options.requireKnownSource,
                                                           metadataDirectory: paths.metadataDirectory,
                                                           audit: audit)
        try Task.checkCancellation()
        // Files this family carries verbatim alongside the tensor payload.
        // They are standalone files rather than index entries, so their sizes
        // come from the remote before planning; the planner then treats them
        // as ordinary resumable range copies.
        var passthroughFiles: [PassthroughFile] = []
        // The transfer layer resolves a copy's source through this map. These
        // files are not safetensors shards, so the snapshot loader never put
        // them there -- carrying the resolved info forward is what makes them
        // fetchable rather than merely planned.
        var passthroughRemoteInfo: [String: RemoteFileInfo] = [:]
        for requirement in RepackPlanner.passthroughRequirements(
            family: snapshot.arch.family) {
            let info: RemoteFileInfo
            do {
                info = try await remote.resolveFileInfo(filename: requirement.name,
                                                        audit: audit)
            } catch {
                // An absent optional file leaves a runnable install; an absent
                // required one does not, and must not be discovered later.
                if requirement.required || !isRemoteNotFound(error) { throw error }
                continue
            }
            guard info.size <= requirement.capBytes else {
                throw RepackError.remoteFileTooLarge(path: requirement.name,
                                                     size: info.size,
                                                     cap: requirement.capBytes)
            }
            passthroughFiles.append(PassthroughFile(sourceName: requirement.name,
                                                    destinationName: requirement.name,
                                                    size: info.size,
                                                    required: requirement.required))
            passthroughRemoteInfo[requirement.name] = info
        }
        let plan = try RepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: paths.partialDirectory,
                                          passthroughFiles: passthroughFiles)
        let rangePlan = try RangeCopyPlanner.plan(repackPlan: plan,
                                                  rangeChunkBytes: options.rangeChunkBytes,
                                                  layoutMode: "identity",
                                                  layoutOrderSha256: nil)
        var checkpoint = saved ?? RemoteInstallCheckpoint(
            repoID: options.repoID,
            requestedRevision: options.revision,
            resolvedCommit: snapshot.resolvedCommit,
            sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
            planFingerprint: rangePlan.canonicalFingerprint,
            totalSourceBytes: rangePlan.remoteBytesToDownload)
        if saved != nil {
            guard checkpoint.resolvedCommit == snapshot.resolvedCommit,
                  checkpoint.totalSourceBytes == rangePlan.remoteBytesToDownload,
                  checkpoint.matches(
                      repoID: options.repoID,
                      requestedRevision: options.revision,
                      sourceIndexSHA256: snapshot.metadata.indexSha256Hex,
                      planFingerprint: rangePlan.canonicalFingerprint) else {
                throw RepackError.installStateIncompatible(
                    detail: "saved download source or copy plan changed")
            }
            if try outputFilesMatch(plan: plan, rangePlan: rangePlan) {
                checkpoint.completedRanges = try Self.validatedCompletedRanges(
                    checkpoint.completedRanges,
                    copies: rangePlan.coalescedCopies,
                    partialDirectory: paths.partialDirectory)
            } else {
                checkpoint.completedRanges = []
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
                try Posix.mkdirP(paths.partialDirectory)
                try createOutputFiles(plan: plan, paths: paths)
            }
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let reusedDestinationBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.destinationBytes
        }
        let remainingOutputBytes = outputBytes > reusedDestinationBytes
            ? outputBytes - reusedDestinationBytes
            : 0
        // The extra chunk budget accounts for the `.range.tmp` staging file
        // (at most one chunk is staged at a time, including on failure paths
        // where the file is unlinked before the error propagates).
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: remainingOutputBytes + UInt64(options.rangeChunkBytes),
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.remoteRepoID = options.repoID
        audit.remoteRequestedRevision = options.revision
        audit.remoteResolvedCommit = snapshot.resolvedCommit
        audit.remoteRangeStreamingSupported = true
        audit.remoteGapBytesDownloaded = rangePlan.remoteGapBytesDownloaded
        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = snapshot.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = "identity"

        if options.dryRunSpaceCheck {
            if saved == nil {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            return RemoteStreamingRepackResult(outputDir: options.outputDir,
                                               resolvedCommit: snapshot.resolvedCommit,
                                               plan: plan,
                                               // Dry run issues no HTTP requests,
                                               // so report the planned count.
                                               rangeRequestCount: rangePlan.coalescedCopies.count,
                                               remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                                               remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                                               remoteRetryCount: audit.remoteRangeRetries,
                                               reusedBytes: checkpoint.completedRanges.reduce(0) {
                                                   $0 + $1.sourceBytes
                                               },
                                               downloadedThisRunBytes: 0,
                                               dryRun: true)
        }

        if saved == nil {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFiles(plan: plan, paths: paths)
            try checkpoint.write(
                to: paths.checkpointFile,
                parentDirectory: paths.parentDirectory)
        }

        let provider = HTTPRangeSourceByteProvider(remote: remote.pinned(commit: snapshot.resolvedCommit),
                                                   files: snapshot.remoteFiles
                                                       .merging(passthroughRemoteInfo) { shard, _ in shard },
                                                   writeTileBytes: options.writeTileBytes)
        let reusedBytes = checkpoint.completedRanges.reduce(UInt64(0)) {
            $0 + $1.sourceBytes
        }
        let payloadDownloadStart = audit.remoteBytesDownloaded
        progress(.copyingPayload(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: 0,
            totalBytes: rangePlan.remoteBytesToDownload))
        // The checkpoint is rewritten at most once per 16 coalesced ranges
        // (or 64 MiB of payload) instead of after every range. The first
        // commit is still written immediately so an early cancellation keeps
        // its completed ranges, and one final fsynced write after the batch
        // makes the last ranges durable before they are relied on. Resume
        // correctness does not depend on the checkpoint being fresh: the next
        // run re-hashes destination bytes before trusting completedRanges.
        var rangesSinceCheckpointWrite = 0
        var pendingCheckpointBytes: UInt64 = 0
        try await provider.copyBatch(
            rangePlan.coalescedCopies,
            completedRangeIDs: Set(checkpoint.completedRanges.map(\.id)),
            partialDirectory: paths.partialDirectory,
            temporaryPath: paths.rangeTemporaryFile,
            audit: audit,
            progress: { downloadedBytes in
                progress(.copyingPayload(
                    reusedBytes: reusedBytes,
                    downloadedThisRunBytes: downloadedBytes,
                    totalBytes: rangePlan.remoteBytesToDownload))
            },
            commit: { completed in
                checkpoint.completedRanges.removeAll { $0.id == completed.id }
                checkpoint.completedRanges.append(completed)
                checkpoint.completedRanges.sort { $0.id < $1.id }
                rangesSinceCheckpointWrite += 1
                pendingCheckpointBytes += completed.sourceBytes
                // The first commit is written immediately (an early
                // cancellation must keep its completed ranges); afterwards the
                // checkpoint is rewritten at most once every 16 ranges or
                // 64 MiB of payload.
                if rangesSinceCheckpointWrite == 1
                    || rangesSinceCheckpointWrite % 16 == 0
                    || pendingCheckpointBytes >= 64 * 1024 * 1024 {
                    pendingCheckpointBytes = 0
                    try checkpoint.write(
                        to: paths.checkpointFile,
                        parentDirectory: paths.parentDirectory)
                }
            })
        try checkpoint.write(
            to: paths.checkpointFile,
            parentDirectory: paths.parentDirectory)

        try recordOutputFile(relativePath: "model_weights.bin",
                             path: plan.resident.path,
                             progress: progress)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let rel = "packed_experts/" + (layer.path as NSString).lastPathComponent
            try recordOutputFile(relativePath: rel, path: layer.path, progress: progress)
        }

        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try GTurboJSON.encodeLayout(plan: plan, expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try Task.checkCancellation()
        // The MTP sidecar deliberately contains only tensors needed by the
        // draft layer. It shares tokenization, embedding and lm_head with the
        // target bundle, so copying tokenizer/config sidecars would be both
        // redundant and a misleading standalone-model contract.
        if plan.arch.family != .qwen36MTP {
            try await copyRemoteMetadataSidecars(snapshot: snapshot,
                                                 remote: remote,
                                                 partialDir: paths.partialDirectory,
                                                 progress: progress)
        }
        try? FileManager.default.removeItem(atPath: paths.rangeTemporaryFile)
        try? FileManager.default.removeItem(atPath: paths.metadataDirectory)
        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(plan: plan,
                          partialDir: paths.partialDirectory,
                          metadata: snapshot.metadata,
                          expertStride: expertStride,
                          resolvedCommit: snapshot.resolvedCommit,
                          modelIDOverride: nil)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return RemoteStreamingRepackResult(outputDir: options.outputDir,
                                           resolvedCommit: snapshot.resolvedCommit,
                                           plan: plan,
                                           // Actual ranged HTTP requests issued
                                           // this run, counted by the byte
                                           // provider (retries are separate).
                                           rangeRequestCount: Int(min(
                                               audit.remoteRangeRequests,
                                               UInt64(Int.max))),
                                           remoteBytesToDownload: rangePlan.remoteBytesToDownload,
                                           remoteGapBytesDownloaded: rangePlan.remoteGapBytesDownloaded,
                                           remoteRetryCount: audit.remoteRangeRetries,
                                           reusedBytes: reusedBytes,
                                           downloadedThisRunBytes:
                                               audit.remoteBytesDownloaded - payloadDownloadStart,
                                           dryRun: false)
    }

    private func validateOptions() throws {
        guard options.rangeChunkBytes >= RemoteChunkPolicy.minBytes,
              options.rangeChunkBytes <= RemoteChunkPolicy.maxBytes else {
            throw RepackError.configurationInvalid(
                detail: "range chunk bytes \(options.rangeChunkBytes) outside "
                    + "[\(RemoteChunkPolicy.minBytes), \(RemoteChunkPolicy.maxBytes)]")
        }
        guard options.writeTileBytes > 0,
              options.writeTileBytes <= BoundedScratch.defaultLimitBytes else {
            throw RepackError.configurationInvalid(detail: "bad write tile bytes \(options.writeTileBytes)")
        }
        guard options.rangeRetryAttempts >= 0 else {
            throw RepackError.configurationInvalid(detail:
                "bad range retry attempts \(options.rangeRetryAttempts)")
        }
    }

    private func createOutputFiles(plan: RepackPlan,
                                   paths: RemoteInstallPaths) throws {
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))
        let resident = try ResidentWriter.createAndWriteIndex(
            plan: plan.resident,
            audit: audit)
        defer { close(resident) }
        try Posix.fsync(resident, path: plan.resident.path)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            let descriptor = try Posix.openCreateRW(layer.path)
            defer { close(descriptor) }
            try Posix.ftruncate(descriptor, path: layer.path, size: layer.fileSize)
            try Posix.fsync(descriptor, path: layer.path)
        }
        try Posix.fsyncDirectory(paths.partialDirectory)
    }

    private func outputFilesMatch(plan: RepackPlan,
                                  rangePlan: RangeCopyPlan) throws -> Bool {
        for output in rangePlan.expectedOutputs {
            let path = ((plan.resident.path as NSString).deletingLastPathComponent
                as NSString).appendingPathComponent(output.relativePath)
            guard try Posix.entryKind(path) == .regular else { return false }
            let descriptor = try Posix.openReadNoFollow(path)
            defer { close(descriptor) }
            guard try Posix.fileSize(fd: descriptor, path: path) == output.size else {
                return false
            }
        }

        let expectedIndex = try ResidentWriter.encodeIndex(plan: plan.resident)
        let descriptor = try Posix.openReadNoFollow(plan.resident.path)
        defer { close(descriptor) }
        let scratch = UnsafeMutableRawBufferPointer.allocate(
            byteCount: min(WriterCore.tileBytes, max(1, expectedIndex.count)),
            alignment: 16_384)
        defer { scratch.deallocate() }
        return try expectedIndex.withUnsafeBytes { expected in
            var offset = 0
            while offset < expected.count {
                let count = min(scratch.count, expected.count - offset)
                try Posix.preadAll(
                    fd: descriptor,
                    path: plan.resident.path,
                    buf: scratch.baseAddress!,
                    count: count,
                    offset: UInt64(offset))
                guard memcmp(
                    scratch.baseAddress!,
                    expected.baseAddress!.advanced(by: offset),
                    count) == 0 else { return false }
                offset += count
            }
            return true
        }
    }

    static func validatedCompletedRanges(
        _ completed: [RemoteCompletedRange],
        copies: [CoalescedRangeCopy],
        partialDirectory: String
    ) throws -> [RemoteCompletedRange] {
        let copiesByID = Dictionary(uniqueKeysWithValues: copies.map { ($0.id, $0) })
        var valid: [RemoteCompletedRange] = []
        for range in completed {
            guard let copy = copiesByID[range.id],
                  range.sourceBytes == copy.size,
                  range.destinationBytes
                      == copy.destinations.reduce(UInt64(0), { $0 + $1.size }) else {
                throw RepackError.installStateCorrupt(
                    path: partialDirectory,
                    detail: "checkpoint contains an unknown range")
            }
            let digest = try HTTPRangeSourceByteProvider.destinationDigest(
                copy,
                partialDirectory: partialDirectory)
            if digest == range.destinationDigest {
                valid.append(range)
            }
        }
        return valid.sorted { $0.id < $1.id }
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        // O_NOFOLLOW: hashing must never follow a symlink planted inside the
        // partial directory.
        let fd = try Posix.openReadNoFollow(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    private func writeSmall(path: String, data: Data) throws {
        let directory = (path as NSString).deletingLastPathComponent
        try Posix.mkdirP(directory)
        try Posix.atomicWrite(data, to: path, durableIn: directory)
        audit.recordWrite(bytes: data.count)
    }

    private func copyRemoteMetadataSidecars(snapshot: RemoteSnapshot,
                                           remote: HuggingFaceRemoteSource,
                                           partialDir: String,
                                           progress: @Sendable (ModelInstallProgress) -> Void) async throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        let pinned = remote.pinned(commit: snapshot.resolvedCommit)
        try Posix.mkdirP(tokenizerDir)

        // config.json is a required sidecar: the source snapshot always
        // declares it (metadata loading fails without it). Prefer the copy
        // fetched this run into .remote-metadata; if a resume-mismatch wiped
        // the partial directory (and with it the fresh metadata), re-fetch it
        // from the remote instead of silently skipping it. The read doubles as
        // the existence check, and atomicWrite fsyncs both the file and its
        // directory after the rename.
        let localConfig = (snapshot.metadataDirectory as NSString)
            .appendingPathComponent("config.json")
        let dstConfig = (tokenizerDir as NSString).appendingPathComponent("config.json")
        do {
            let configData = try Posix.readBoundedData(localConfig,
                                                       maximumBytes: 1024 * 1024)
            try Posix.atomicWrite(configData, to: dstConfig, durableIn: tokenizerDir)
        } catch {
            if (try? Posix.entryKind(localConfig)) != .regular {
                let info = try await pinned.resolveFileInfo(filename: "config.json",
                                                            audit: audit)
                guard info.size <= 1024 * 1024 else {
                    throw RepackError.remoteFileTooLarge(
                        path: "config.json",
                        size: info.size,
                        cap: 1024 * 1024)
                }
                try await pinned.fetchSmallFile(filename: "config.json",
                                                info: info,
                                                capBytes: 1024 * 1024,
                                                outputPath: dstConfig,
                                                audit: audit)
            } else {
                throw error
            }
        }
        try recordOutputFile(relativePath: "tokenizer/config.json",
                             path: dstConfig,
                             progress: progress)

        let tokenizerFiles: [(name: String, cap: UInt64, required: Bool)] = [
            ("tokenizer.json", 64 * 1024 * 1024, true),
            ("tokenizer_config.json", 4 * 1024 * 1024, true),
            ("special_tokens_map.json", 1 * 1024 * 1024, false),
            ("chat_template.jinja", 4 * 1024 * 1024, false),
            ("chat_template.json", 4 * 1024 * 1024, false),
        ]
        for file in tokenizerFiles {
            try Task.checkCancellation()
            let info: RemoteFileInfo
            do {
                info = try await pinned.resolveFileInfo(filename: file.name, audit: audit)
            } catch {
                if file.required || !isRemoteNotFound(error) {
                    throw error
                }
                continue
            }
            let dst = (tokenizerDir as NSString).appendingPathComponent(file.name)
            try await pinned.fetchSmallFile(filename: file.name,
                                            info: info,
                                            capBytes: file.cap,
                                            outputPath: dst,
                                            audit: audit)
            try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                 path: dst,
                                            progress: progress)
        }
    }

    private func isRemoteNotFound(_ error: Error) -> Bool {
        if case RepackError.remoteHTTPStatus(_, 404) = error {
            return true
        }
        if case RepackError.remoteHTTPResponse(_, 404, _) = error {
            return true
        }
        return false
    }

    private func writeManifest(plan: RepackPlan,
                               partialDir: String,
                               metadata: IndexLoader.SourceMetadata,
                               expertStride: UInt64,
                               resolvedCommit: String,
                               modelIDOverride: String?) throws {
        // Determine quantization bits from actual tensor data, not hardcoded
        var bits = GTurboJSON.QuantBitWidths(
            embedding: 4,
            attention: 4,
            router: 8,
            sharedExpert: 8,
            routedExpert: 4)
        for e in plan.resident.entries {
            guard let quantSpec = e.quantSpec else { continue }
            if e.name.hasSuffix(".embed_tokens.weight") {
                bits.embedding = quantSpec.bits
            }
            if e.name.hasSuffix(".self_attn.q_proj.weight")
                || e.name.hasSuffix(".self_attn.k_proj.weight")
                || e.name.hasSuffix(".self_attn.v_proj.weight")
                || e.name.hasSuffix(".self_attn.o_proj.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_qkv.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_z.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_a.weight")
                || e.name.hasSuffix(".linear_attn.in_proj_b.weight")
                || e.name.hasSuffix(".linear_attn.out_proj.weight") {
                bits.attention = quantSpec.bits
            }
            // Router slot: the Qwen router tensor is `.mlp.gate.weight`.
            if e.name.hasSuffix(".router.proj.weight")
                || e.name.hasSuffix(".mlp.gate.weight") {
                bits.router = quantSpec.bits
            }
            // Shared-expert slot: the sigmoid-gated shared expert MLP. Routed
            // experts (`.mlp.switch_mlp.*`) are deliberately excluded here —
            // their bits land in `bits.routedExpert` below from the layer
            // sub-tensors, so no tensor feeds more than one slot.
            if e.name.hasSuffix(".mlp.shared_expert.gate_proj.weight")
                || e.name.hasSuffix(".mlp.shared_expert.up_proj.weight")
                || e.name.hasSuffix(".mlp.shared_expert.down_proj.weight") {
                bits.sharedExpert = quantSpec.bits
            }
        }
        if let layer = plan.layers.first(where: { !$0.subTensors.isEmpty }),
           let routedBits = layer.subTensors.first?.bitsForWeights {
            bits.routedExpert = routedBits
        }
        let files = audit.outputFiles.map {
            ($0.relativePath, GTurboJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: modelIDOverride ?? plan.matchedModelID ?? "unknown/snapshot",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        try Posix.fsyncDirectory(partialDir)
        let manifestSha = try Sha256Stream.hashFile(path: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: options.repoID,
            sourceRevision: resolvedCommit,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        let tmpReceiptPath = receiptPath + ".tmp"
        try writeSmall(path: tmpReceiptPath, data: receipt)
        try Posix.rename(from: tmpReceiptPath, to: receiptPath)
        try Posix.fsyncDirectory(partialDir)
    }
}

public extension RemoteStreamingRepacker {
    /// Repack an already-complete local affine safetensors snapshot through
    /// the same planner, file layout, hashing, and trusted-receipt path as a
    /// pinned remote install. Local imports intentionally do not support
    /// resume: the source is already present, so a failed attempt is removed
    /// atomically and can be restarted without network transfer.
    static func runLocalSnapshot(
        options local: LocalSnapshotRepackOptions,
        audit: RepackAudit = RepackAudit(),
        progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }
    ) async throws -> RemoteStreamingRepackResult {
        let source = try LocalSnapshotLoader.load(directory: local.inputSnapshotDir)
        let worker = RemoteStreamingRepacker(
            options: RemoteStreamingRepackOptions(
                repoID: "local/snapshot",
                revision: String(source.metadata.indexSha256Hex.prefix(40)),
                outputDir: local.outputDir,
                requireKnownSource: false,
                rangeChunkBytes: local.rangeChunkBytes,
                writeTileBytes: local.writeTileBytes,
                minFreeReserveBytes: local.minFreeReserveBytes,
                overwrite: local.overwrite),
            audit: audit)
        return try await worker.runLocalPrepared(
            source: source,
            local: local,
            progress: progress)
    }

    private func runLocalPrepared(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws -> RemoteStreamingRepackResult {
        try validateOptions()
        try validateLocalModelID(local.modelID)
        let installLock = try InstallLock.acquire(outputDirectory: local.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        try validateLocalDestination(paths: paths, overwrite: local.overwrite)
        let (plan, rangePlan, outputBytes) = try prepareLocalPlan(
            source: source, local: local, paths: paths, progress: progress)
        configureLocalAudit(source: source, plan: plan)
        try await executeLocalCopy(source: source,
                                   local: local,
                                   paths: paths,
                                   plan: plan,
                                   rangePlan: rangePlan,
                                   outputBytes: outputBytes,
                                   progress: progress)
        return RemoteStreamingRepackResult(
            outputDir: local.outputDir,
            resolvedCommit: String(source.metadata.indexSha256Hex.prefix(40)),
            plan: plan,
            rangeRequestCount: 0,
            remoteBytesToDownload: rangePlan.remoteBytesToDownload,
            remoteGapBytesDownloaded: 0,
            remoteRetryCount: 0,
            reusedBytes: 0,
            downloadedThisRunBytes: rangePlan.remoteBytesToDownload,
            dryRun: false)
    }

    private func validateLocalModelID(_ modelID: String) throws {
        guard !modelID.isEmpty,
              modelID.utf8.count <= 256,
              !modelID.contains(where: { $0.isWhitespace }) else {
            throw RepackError.configurationInvalid(
                detail: "local snapshot model ID must be non-empty and contain no whitespace")
        }
    }

    private func validateLocalDestination(paths: RemoteInstallPaths,
                                          overwrite: Bool) throws {
        if try Posix.entryKind(paths.finalDirectory) == .directory,
           !overwrite {
            throw RepackError.configurationInvalid(
                detail: "output directory already exists: \(paths.finalDirectory)")
        }
        guard try Posix.entryKind(paths.partialDirectory) == .absent,
              try Posix.entryKind(paths.checkpointFile) == .absent else {
            throw RepackError.installStateIncompatible(
                detail: "saved remote download exists; resume or discard it first")
        }
    }

    private func prepareLocalPlan(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        paths: RemoteInstallPaths,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) throws -> (RepackPlan, RangeCopyPlan, UInt64) {
        progress(.downloadingMetadata)
        // The same non-tensor files the remote path carries. Omitting them
        // here would silently produce an install missing its n-gram table and
        // PLE constants -- an install that loads and is quietly degraded,
        // which is worse than one that fails.
        var passthroughFiles: [PassthroughFile] = []
        for requirement in RepackPlanner.passthroughRequirements(
            family: source.arch.family) {
            let path = (local.inputSnapshotDir as NSString)
                .appendingPathComponent(requirement.name)
            guard let attrs = try? FileManager.default
                .attributesOfItem(atPath: path),
                  let size = (attrs[FileAttributeKey.size] as? NSNumber)?.uint64Value
            else {
                if requirement.required {
                    throw RepackError.snapshotFileMissing(
                        path: path,
                        detail: "\(requirement.name) is required by this "
                            + "architecture; the snapshot is incomplete")
                }
                continue
            }
            passthroughFiles.append(PassthroughFile(sourceName: requirement.name,
                                                    destinationName: requirement.name,
                                                    size: size,
                                                    required: requirement.required))
        }
        let plan = try RepackPlanner.plan(
            meta: source.metadata,
            arch: source.arch,
            shardHeaders: source.shardHeaders,
            outputDir: paths.partialDirectory,
            passthroughFiles: passthroughFiles)
        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan,
            rangeChunkBytes: local.rangeChunkBytes,
            layoutMode: "identity",
            layoutOrderSha256: nil)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: rangePlan.remoteBytesToDownload,
                           outputBytes: outputBytes))
        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: outputBytes,
            reserveBytes: local.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()
        return (plan, rangePlan, outputBytes)
    }

    private func configureLocalAudit(source: LocalSnapshot, plan: RepackPlan) {
        audit.remoteRepoID = "local/snapshot"
        audit.remoteRequestedRevision = source.metadata.indexSha256Hex
        audit.remoteResolvedCommit = String(source.metadata.indexSha256Hex.prefix(40))
        audit.remoteRangeStreamingSupported = false
        audit.remoteGapBytesDownloaded = 0
        audit.sourceSnapshotSha256 = source.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = source.metadata.bitsOverrides.count
        audit.tensorsDroppedMultimodal = plan.excludedMultimodalTensorNames
        audit.packedExpertLayoutMode = "identity"
    }

    private func executeLocalCopy(
        source: LocalSnapshot,
        local: LocalSnapshotRepackOptions,
        paths: RemoteInstallPaths,
        plan: RepackPlan,
        rangePlan: RangeCopyPlan,
        outputBytes: UInt64,
        progress: @escaping @Sendable (ModelInstallProgress) -> Void
    ) async throws {
        do {
            progress(.reservingOutput(bytes: outputBytes))
            try createOutputFiles(plan: plan, paths: paths)
            let provider = LocalSourceByteProvider(
                snapshotDirectory: local.inputSnapshotDir,
                writeTileBytes: local.writeTileBytes)
            progress(.copyingPayload(reusedBytes: 0,
                                     downloadedThisRunBytes: 0,
                                     totalBytes: rangePlan.remoteBytesToDownload))
            try await provider.copyBatch(
                rangePlan.coalescedCopies,
                completedRangeIDs: [],
                partialDirectory: paths.partialDirectory,
                temporaryPath: paths.rangeTemporaryFile,
                audit: audit,
                progress: { bytes in
                    progress(.copyingPayload(
                        reusedBytes: 0,
                        downloadedThisRunBytes: bytes,
                        totalBytes: rangePlan.remoteBytesToDownload))
                },
                commit: { _ in })

            try recordOutputFile(relativePath: "model_weights.bin",
                                 path: plan.resident.path,
                                 progress: progress)
            for layer in plan.layers where layer.expertsPerLayer > 0 {
                let relative = "packed_experts/"
                    + (layer.path as NSString).lastPathComponent
                try recordOutputFile(relativePath: relative,
                                     path: layer.path,
                                     progress: progress)
            }
            let layoutPath = ((paths.partialDirectory as NSString)
                .appendingPathComponent("packed_experts") as NSString)
                .appendingPathComponent("layout.json")
            let expertStride = plan.layers.first(where: {
                $0.expertsPerLayer > 0
            })?.expertStride ?? 0
            let layoutData = try GTurboJSON.encodeLayout(
                plan: plan,
                expertStride: expertStride)
            try writeSmall(path: layoutPath, data: layoutData)
            try GTurboLayoutValidator.validate(path: layoutPath, plan: plan)
            try recordOutputFile(relativePath: "packed_experts/layout.json",
                                 path: layoutPath,
                                 progress: progress)
            progress(.finalizing)
            try writeManifest(
                plan: plan,
                partialDir: paths.partialDirectory,
                metadata: source.metadata,
                expertStride: expertStride,
                resolvedCommit: String(source.metadata.indexSha256Hex.prefix(40)),
                modelIDOverride: local.modelID)

            if try Posix.entryKind(paths.finalDirectory) == .directory {
                try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
                try Posix.fsyncDirectory(paths.parentDirectory)
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            } else {
                try Posix.rename(from: paths.partialDirectory,
                                 to: paths.finalDirectory)
                try Posix.fsyncDirectory(paths.parentDirectory)
            }
        } catch {
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            throw error
        }
    }
}
