import Foundation
import Darwin
import NVMAIFormat

public struct VerifyInstallOptions: Sendable {
    public let inputGTurbo: String

    public init(inputGTurbo: String) {
        self.inputGTurbo = inputGTurbo
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    // 64 MiB: sized for Qwen 3.6's ~22 MB layout.json (40 layers x 256 experts).
    public static let metadataMaxBytes: UInt64 = 64 * 1024 * 1024
    // Hard ceiling for any single payload file named by the manifest. The
    // manifest size remains the exact per-file contract (verified below); this
    // cap only rejects absurd corrupted manifests before hashing.
    //
    // 128 GiB, sized against the largest legitimate payload: Qwen3.8-Flash-Next's
    // n-gram embedding table is 102,400,491,520 bytes (95.4 GiB) of fp16 rows.
    // The previous 64 GiB predated any file that large and rejected a correct
    // install. Kept finite, and well under any plausible disk, so a manifest
    // claiming a terabyte is still refused before hashing begins.
    public static let payloadMaxBytes: UInt64 = 128 * 1024 * 1024 * 1024

    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let access = try GTurboDirectoryAccess(rootPath: options.inputGTurbo)
        let manifestPath = "manifest.json"
        try GTurboPathValidator.validateRelativePath(
            manifestPath, field: "manifest.files.manifest.json")
        let manifestSize = try access.fileSize(manifestPath)
        let manifestSha = try access.hash(manifestPath, noCache: true)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)
        try validateQuantAgainstResident(access: access, manifest: manifest)

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            // Path validation before any filesystem operation: reject `..`,
            // absolute and non-normalized names, and duplicate filesystem keys.
            try GTurboPathValidator.validateRelativePath(
                relativePath, field: "manifest.files.\(relativePath)")
            guard entry.size <= payloadMaxBytes else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(entry.size) exceeds "
                        + "the \(payloadMaxBytes)-byte per-file cap")
            }
            // All reads go through the root-anchored openat chain with
            // O_NOFOLLOW at every level (no symlink escapes) and the
            // descriptor's type is verified after open.
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            let actualSha = try access.hash(relativePath, noCache: true)
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            let (verified, overflow) = bytesVerified.addingReportingOverflow(actualSize)
            guard !overflow else {
                throw RepackError.configurationInvalid(detail: "verified byte total overflows")
            }
            bytesVerified = verified
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: access.rootPath,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: nil,
            sourceRevision: manifest.sourceSnapshotHash,
            toolVersion: "NVMAIRepack verify-install",
            files: files)
        let receiptPath = access.rootPath
            + "/" + VerifiedInstallReceiptWriter.fileName
        try receiptData.write(to: URL(fileURLWithPath: receiptPath), options: .atomic)
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    static func validatePackedExpertLayout(inputGTurbo: String) throws {
        let access = try GTurboDirectoryAccess(rootPath: inputGTurbo)
        let manifest = try loadManifest(access: access)
        try validatePackedExpertLayout(access: access, manifest: manifest)
    }

    /// Read the resident index and hand it to the width cross-check.
    ///
    /// The index is the head of `model_weights.bin`, and that file is the model
    /// itself, so this reads it in two bounded steps -- the 24-byte header, then
    /// exactly the index it names -- and never touches the payload.
    private static func validateQuantAgainstResident(access: GTurboDirectoryAccess,
                                                     manifest: Manifest) throws {
        guard let quant = manifest.quant else {
            throw RepackError.configurationInvalid(detail: "manifest.json has no quant block")
        }
        let relativePath = "model_weights.bin"
        guard manifest.files[relativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
        }
        let headerBytes = try access.readPrefix(
            relativePath, maxBytes: UInt64(GTurboFormatV1.residentHeaderBytes))
        guard headerBytes.count == GTurboFormatV1.residentHeaderBytes else {
            throw RepackError.configurationInvalid(
                detail: "\(relativePath) is shorter than the resident index header")
        }
        let header = try headerBytes.withUnsafeBytes {
            try GTurboResidentIndexCodec.decodeHeader($0)
        }
        guard header.indexSize <= UInt64(GTurboFormatV1.residentIndexMaxBytes) else {
            throw RepackError.configurationInvalid(detail:
                "\(relativePath) index \(header.indexSize) exceeds the "
                + "\(GTurboFormatV1.residentIndexMaxBytes)-byte v1 cap")
        }
        let indexBytes = try access.readPrefix(relativePath, maxBytes: header.indexSize)
        guard indexBytes.count == Int(header.indexSize) else {
            throw RepackError.configurationInvalid(detail:
                "\(relativePath) holds \(indexBytes.count) bytes but its header "
                + "claims an index of \(header.indexSize)")
        }
        let entries = try indexBytes.withUnsafeBytes {
            try GTurboResidentIndexCodec.decodeRegion($0, header: header)
        }
        try validateQuantAgainstResident(quant: quant,
                                         expertsPerLayer: manifest.expertsPerLayer,
                                         entries: entries)
    }

    /// Cross-check the manifest's declared widths against the resident bytes.
    ///
    /// Everything else in this file hashes well-formed files. This is the only
    /// check that asks whether the manifest's *description* of the payload is
    /// true, and it exists because two bugs shipped past all the others:
    ///
    ///   - the repacker wrote only the five width slots and dropped the source
    ///     checkpoint's per-tensor widths, so a 4-bit build's 8-bit attention
    ///     K/V were dequantized as 4-bit;
    ///   - `routedExpert` started at a literal 4, so every 8-bit dense install
    ///     advertised itself as 4-bit and the catalog skipped it as a duplicate
    ///     of the real 4-bit one. It installed, verified, loaded, and could not
    ///     be selected.
    ///
    /// Both passed every existing check, including this tool's, because a
    /// manifest can be internally consistent and still lie about the bytes beside
    /// it. The width is not something the manifest has to be trusted for: a u32
    /// packed weight's byte extent determines it exactly --
    /// `sizeBytes = rows * columns * bits / 8` -- so the payload is the
    /// authority and the manifest is what gets checked.
    ///
    /// The question asked is "would this install *read* correctly", not "was it
    /// written by the current writer". For each packed tensor the check resolves
    /// the width exactly as the CPU reader does -- an explicit per-tensor entry,
    /// else the slot the reader falls back to -- and requires it to equal what
    /// the bytes say. That is why the duplication of the reader's fallback rule
    /// below is deliberate: it is the contract being verified, and it is what
    /// makes the failure name the tensors that would be dequantized wrongly
    /// rather than every tensor the writer happened not to annotate.
    static func validateQuantAgainstResident(quant: GTurboManifestQuantV1,
                                             expertsPerLayer: Int,
                                             entries: [GTurboResidentIndexEntryV1]) throws {
        // Every dtype-0 entry is a packed u32 `.weight`: the planner only marks a
        // tensor quantized when its source dtype is u32 and its name ends in
        // `.weight`, and everything else is stored bf16.
        let attentionSlot = quant.attention.weightBits
        let embeddingSlot = quant.embedding.weightBits
        var impliedCounts: [Int: Int] = [:]
        var unreadable: [String] = []

        for entry in entries where entry.dtype == GTurboFormatV1.DType.u32.rawValue {
            let implied = try impliedWidth(of: entry)
            impliedCounts[implied, default: 0] += 1
            // Overrides are keyed by stem, without the `.weight` suffix.
            let stem = entry.name.hasSuffix(".weight")
                ? String(entry.name.dropLast(".weight".count)) : entry.name

            let resolved: Int
            if let declared = quant.overrides?[stem]?.weightBits {
                resolved = declared
            } else if expertsPerLayer == 0 {
                // `AffineSnapshot.init(gturbo:)`: the embedding slot covers the
                // tied head as well, and the attention slot is the default.
                resolved = stem.hasSuffix("embed_tokens") || stem.hasSuffix("lm_head")
                    ? embeddingSlot : attentionSlot
            } else {
                // A packed-expert install keeps its routed-expert widths in
                // `packed_experts/layout.json`, and its resident tensors are the
                // GPU path's business, not the CPU reader's. Nothing to check.
                continue
            }

            if resolved != implied {
                unreadable.append("\(entry.name) would be read as \(resolved)-bit but "
                    + "is \(implied)-bit (\(entry.sizeBytes) bytes at "
                    + "\(entry.shape[0])x\(entry.shape[1]))")
            }
        }

        guard unreadable.isEmpty else {
            let shown = unreadable.prefix(3).joined(separator: "; ")
            let more = unreadable.count > 3 ? " and \(unreadable.count - 3) more" : ""
            throw RepackError.configurationInvalid(detail:
                "the manifest's widths disagree with the resident payload, so this "
                + "install would dequantize wrongly: \(shown)\(more). Repack it "
                + "from its source snapshot; the bytes are fine, the description "
                + "of them is not")
        }

        // With no routed experts the slot describes no tensor, so it is not
        // derived from anything and has to be checked on its own. It is what
        // `ManifestIdentity.weightBits` reads and what becomes the `_<bits>-Bit`
        // suffix in `/v1/models`, so a wrong value here is not cosmetic.
        //
        // The requirement is that the declared width is one the payload actually
        // uses, preferring the dominant one. Written as a *set* rather than
        // "the maximum" on purpose: resolving a tie by picking one of the tied
        // widths would make the verdict depend on dictionary iteration order,
        // which is not stable between runs. On a genuine tie any tied width is
        // accepted; with a clear majority only that width is.
        guard expertsPerLayer == 0, let most = impliedCounts.values.max() else { return }
        let dominant = Set(impliedCounts.filter { $0.value == most }.keys)
        let declared = quant.routedExpert.weightBits
        guard dominant.contains(declared) else {
            let total = impliedCounts.values.reduce(0, +)
            let histogram = impliedCounts
                .sorted { $0.key < $1.key }
                .map { "\($0.key)-bit x\($0.value)" }
                .joined(separator: ", ")
            throw RepackError.configurationInvalid(detail:
                "the install has no routed experts, but declares a routed-expert "
                + "width of \(declared) while its resident payload is "
                + "\(histogram) over \(total) quantized tensors. That value names "
                + "the model in /v1/models and decides which build the catalog "
                + "thinks this is")
        }
    }

    /// The width a packed u32 weight's own byte extent implies.
    ///
    /// A u32 word holds `32 / bits` values, so a `rows x columns` logical matrix
    /// occupies `rows * columns * bits / 8` bytes and nothing else. The shape is
    /// the *logical* width in the resident index, unlike a safetensors header,
    /// which is what makes this invertible.
    private static func impliedWidth(of entry: GTurboResidentIndexEntryV1) throws -> Int {
        let rows = UInt64(entry.shape[0])
        let columns = UInt64(entry.shape[1])
        guard rows > 0, columns > 0 else {
            throw RepackError.configurationInvalid(detail:
                "\(entry.name): packed weight with a zero dimension "
                + "\(entry.shape[0])x\(entry.shape[1])")
        }
        guard entry.sizeBytes % 4 == 0 else {
            throw RepackError.configurationInvalid(detail:
                "\(entry.name): \(entry.sizeBytes) bytes is not a whole number "
                + "of u32 words")
        }
        let (values, valuesOverflow) = (entry.sizeBytes / 4).multipliedReportingOverflow(by: 32)
        let (cells, cellsOverflow) = rows.multipliedReportingOverflow(by: columns)
        guard !valuesOverflow, !cellsOverflow else {
            throw RepackError.configurationInvalid(
                detail: "\(entry.name): dimensions overflow")
        }
        guard values % cells == 0 else {
            throw RepackError.configurationInvalid(detail:
                "\(entry.name): \(entry.sizeBytes) bytes at \(entry.shape[0])x"
                + "\(entry.shape[1]) does not divide into a whole number of "
                + "values per element")
        }
        let bits = values / cells
        guard bits == 4 || bits == 8 else {
            throw RepackError.configurationInvalid(detail:
                "\(entry.name): implied width \(bits) is not a supported "
                + "4- or 8-bit packing")
        }
        return Int(bits)
    }

    private struct ManifestFileEntry: Decodable {
        let size: UInt64
        let sha256: String
    }

    private struct Manifest: Decodable {
        let files: [String: ManifestFileEntry]
        let expertsPerLayer: Int
        let numLayers: Int
        let expertStride: UInt64
        let sourceSnapshotHash: String?
        /// Decoded through `GTurboManifestQuantV1`, whose hand-written `Codable`
        /// keeps the open set of per-tensor width keys. A synthesised decoder
        /// would drop them, which is the bug this whole check exists to catch.
        let quant: GTurboManifestQuantV1?
    }

    private struct PackedExpertsLayout: Decodable {
        let expertStride: UInt64
        let numLayers: Int
        let expertsPerLayer: Int
        let layers: [Layer]
    }

    private struct Layer: Decodable {
        let layer: Int
        let file: String
        let experts: [Expert]
    }

    private struct Expert: Decodable {
        let expert: Int?
        let offset: UInt64
        let size: UInt64
    }

    private static func loadManifest(access: GTurboDirectoryAccess) throws -> Manifest {
        do {
            let data = try loadMetadataJSON(access: access, relativePath: "manifest.json")
            return try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(access: GTurboDirectoryAccess) throws -> PackedExpertsLayout {
        do {
            let data = try loadMetadataJSON(access: access,
                                            relativePath: "packed_experts/layout.json")
            return try JSONDecoder().decode(PackedExpertsLayout.self, from: data)
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func loadMetadataJSON(access: GTurboDirectoryAccess,
                                         relativePath: String) throws -> Data {
        try GTurboPathValidator.validateRelativePath(
            relativePath, field: "metadata.\(relativePath)")
        return try access.readMetadata(relativePath, maxBytes: metadataMaxBytes)
    }

    private static func validatePackedExpertLayout(access: GTurboDirectoryAccess,
                                                   manifest: Manifest) throws {
        let layoutRelativePath = "packed_experts/layout.json"
        guard manifest.files[layoutRelativePath] != nil else {
            throw RepackError.configurationInvalid(detail: "manifest missing \(layoutRelativePath)")
        }
        let layout = try loadLayout(access: access)
        let alignment = GTurboFormatV1.alignmentBytes
        guard layout.expertStride == manifest.expertStride,
              layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer else {
            throw RepackError.configurationInvalid(detail: "packed expert layout dimensions mismatch manifest")
        }
        guard layout.expertStride % alignment == 0 else {
            throw RepackError.configurationInvalid(
                detail: "expertStride \(layout.expertStride) is not aligned to \(alignment) bytes")
        }
        guard layout.layers.count == layout.numLayers else {
            throw RepackError.configurationInvalid(detail: "packed expert layout layer count mismatch")
        }
        let expectedLayerSize = UInt64(layout.expertsPerLayer) * layout.expertStride
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < layout.numLayers else {
                throw RepackError.configurationInvalid(detail: "packed expert layer index out of range")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw RepackError.configurationInvalid(
                    detail: "packed_experts/\(layer.file) expert count mismatch")
            }
            try GTurboPathValidator.validateBasename(
                layer.file, field: "packed_experts/layout.json layers[\(layer.layer)].file")
            // A layer with no routed experts has no file, and writing one
            // empty `layer_NN.bin` per layer to satisfy this loop would be
            // worse than the check: the dense Qwen 3.5 installs are exactly
            // this shape, 24 layouts and no packed experts at all. The
            // expected size is what makes this safe rather than a hole --
            // `expectedLayerSize` is 0 only when the layer is empty, so a
            // layer that should carry bytes still fails on a missing file
            // below, with a non-zero expected size to compare against.
            if expectedLayerSize == 0 {
                continue
            }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            let actualSize = try access.fileSize(relativePath)
            guard actualSize == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedLayerSize)")
            }
            var seenExperts = Set<Int>()
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expertID >= 0 && expertID < layout.expertsPerLayer else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert id out of range")
                }
                guard seenExperts.insert(expertID).inserted else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) duplicate expert \(expertID)")
                }
                guard expert.size == layout.expertStride else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) size mismatch")
                }
                guard expert.offset % GTurboFormatV1.alignmentBytes == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) offset is not aligned to \(GTurboFormatV1.alignmentBytes) bytes")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func findUnexpectedEntries(access: GTurboDirectoryAccess,
                                              manifest: Manifest) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")

        let entries = try access.relativeEntries()
        var unexpected: [String] = []
        for rel in entries {
            // .DS_Store may appear at any depth (Finder writes it into
            // subdirectories too).
            if rel == ".DS_Store" || rel.hasSuffix("/.DS_Store") { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}
