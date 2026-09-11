import Foundation

/// The side-engine's view of a snapshot written by `tools/prepare_qwen35.py`.
///
/// One directory: a `config.json` carrying the architecture and the
/// quantization block, an index naming which shard holds what, and the
/// shards themselves. Everything the engine needs to run is here, and
/// nothing it does not.
public struct AffineSnapshot: Sendable {

    /// A quantized matrix, as the kernels want it: unsigned lanes packed
    /// low-first into `UInt32`, one BF16 scale and bias per group.
    ///
    /// The pointers are into the mapping and stay valid for the snapshot's
    /// life. Nothing is copied, which is what makes loading a two-gigabyte
    /// model instant and its resident cost the page cache's problem rather
    /// than the process's.
    ///
    /// unchecked-invariant: every pointer addresses a read-only `MAP_PRIVATE`
    /// mapping owned by the `SafeTensorsFile` this came from, which outlives
    /// the snapshot that vends it. The pages are never written, by anyone, so
    /// concurrent readers cannot race and threading the GEMV over row ranges
    /// is safe by construction.
    public struct Matrix: @unchecked Sendable {
        public let weights: UnsafeRawBufferPointer
        public let scales: UnsafePointer<UInt16>
        public let biases: UnsafePointer<UInt16>
        public let rows: Int
        public let columns: Int
        public let bits: Int
        public let groupSize: Int
    }

    public struct Configuration: Sendable, Equatable {
        public let hiddenSize: Int
        public let layers: Int
        public let heads: Int
        public let keyValueHeads: Int
        public let headDim: Int
        public let fullAttentionInterval: Int
        public let linearKeyHeads: Int
        public let linearValueHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let convKernel: Int
        public let intermediateSize: Int
        public let vocabulary: Int
        public let normEpsilon: Float
        public let ropeTheta: Float
        public let partialRotaryFactor: Double
        public let tiedEmbedding: Bool
        /// The context the checkpoint claims. A CPU engine will not want all
        /// of it -- attention here is a plain loop over the cache -- but the
        /// ceiling belongs to the model, not to the server.
        public let maxPositions: Int

        /// Dimensions of the rotation, which is partial here: 64 of 256.
        /// Rotating the whole head is the single most plausible way to get a
        /// model that runs and is quietly wrong.
        public var rotaryDim: Int { Int(Double(headDim) * partialRotaryFactor) }

        /// Whether this layer is full attention rather than Gated DeltaNet.
        /// Every fourth, counting from the end of each group.
        public func isAttention(_ layer: Int) -> Bool {
            (layer + 1) % fullAttentionInterval == 0
        }
    }

    /// Where the tensors actually live.
    ///
    /// Two shapes carry the same architecture: a plain affine safetensors
    /// snapshot, which the converter writes and the CPU engine has always
    /// read, and a `.gturbo` install, which is what every other model here
    /// is. A `.gturbo` is a byte copy of the same quantized tensors (both
    /// quantizers are group-64 affine), so this is a storage difference and
    /// not a semantic one -- which `tools/gturbo_diff_snapshot.py` checks
    /// rather than assumes.
    private enum Storage {
        case safetensors(shards: [String: SafeTensorsFile],
                         placement: [String: String])
        case gturbo(index: ResidentIndex, weights: ResidentWeights)
    }

    /// One read-only mapping of a `.gturbo`'s resident payload, held for the
    /// life of the snapshot.
    ///
    /// Mapping once matters: `matrix(_:)` is called per tensor (several
    /// hundred times a token), and mapping the 1.3 GB file on each call paged
    /// the whole payload in repeatedly -- generation went from seconds to
    /// never finishing.
    ///
    /// unchecked-invariant: the file is opened read-only and mapped with
    /// `.alwaysMapped`, no operation in this project writes it, and the
    /// mapping outlives every pointer vended from it because the `Data` is
    /// held for the snapshot's lifetime. Readers therefore only ever read
    /// immutable pages, so handing the same base address to the CPU engine's
    /// row-range reads from several threads cannot race. The same reasoning
    /// the safetensors case documents for its shard mappings.
    struct ResidentWeights: @unchecked Sendable {
        let data: Data
        var base: UnsafeRawPointer? {
            data.withUnsafeBytes { $0.baseAddress }
        }
    }

    public let directory: URL
    public let configuration: Configuration
    private let storage: Storage
    private let baseBits: Int
    private let groupSize: Int
    /// Per-tensor width overrides, keyed by stem. The 4-bit build keeps the
    /// tied embedding and the attention K/V at 8 bits, because measuring
    /// said that is where the error actually is.
    private let widths: [String: Int]

    /// The safetensors shards, for the paths that are only reachable from the
    /// snapshot initializer. Nil for a `.gturbo` install.
    private var shards: [String: SafeTensorsFile] {
        if case .safetensors(let shards, _) = storage { return shards }
        return [:]
    }
    private var placement: [String: String] {
        if case .safetensors(_, let placement) = storage { return placement }
        return [:]
    }

    public init(directory: URL) throws {
        self.directory = directory
        let config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let config = config as? [String: Any] else {
            throw SafeTensorsFile.Failure.malformed("config.json is not an object")
        }
        func integer(_ key: String, _ fallback: Int? = nil) throws -> Int {
            if let value = config[key] as? Int { return value }
            if let fallback { return fallback }
            throw SafeTensorsFile.Failure.malformed("config.json lacks \(key)")
        }
        func double(_ key: String, _ fallback: Double) -> Double {
            (config[key] as? NSNumber)?.doubleValue ?? fallback
        }
        configuration = Configuration(
            hiddenSize: try integer("hidden_size"),
            layers: try integer("num_hidden_layers"),
            heads: try integer("num_attention_heads"),
            keyValueHeads: try integer("num_key_value_heads"),
            headDim: try integer("head_dim"),
            fullAttentionInterval: try integer("full_attention_interval", 4),
            linearKeyHeads: try integer("linear_num_key_heads"),
            linearValueHeads: try integer("linear_num_value_heads"),
            linearKeyHeadDim: try integer("linear_key_head_dim"),
            linearValueHeadDim: try integer("linear_value_head_dim"),
            convKernel: try integer("linear_conv_kernel_dim", 4),
            intermediateSize: try integer("intermediate_size"),
            vocabulary: try integer("vocab_size"),
            normEpsilon: Float(double("rms_norm_eps", 1e-6)),
            ropeTheta: Float(double("rope_theta", 10_000_000)),
            partialRotaryFactor: double("partial_rotary_factor", 0.25),
            tiedEmbedding: (config["tie_word_embeddings"] as? Bool) ?? true,
            maxPositions: try integer("max_position_embeddings", 32_768))

        modelType = config["model_type"] as? String
        guard let quantization = config["quantization"] as? [String: Any],
              let bits = quantization["bits"] as? Int,
              let group = quantization["group_size"] as? Int else {
            throw SafeTensorsFile.Failure.malformed("config.json lacks a quantization block")
        }
        baseBits = bits
        groupSize = group
        var overrides: [String: Int] = [:]
        for (stem, value) in quantization {
            if let entry = value as? [String: Any], let width = entry["bits"] as? Int {
                overrides[stem] = width
            }
        }
        widths = overrides

        let index = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent(
                "model.safetensors.index.json")))
        guard let index = index as? [String: Any],
              let map = index["weight_map"] as? [String: String] else {
            throw SafeTensorsFile.Failure.malformed("index has no weight_map")
        }
        var opened: [String: SafeTensorsFile] = [:]
        for file in Set(map.values) {
            opened[file] = try SafeTensorsFile(url: directory.appendingPathComponent(file))
        }
        storage = .safetensors(shards: opened, placement: map)
    }

    /// Load a `.gturbo` install as CPU weights.
    ///
    /// The install's `manifest.json` carries the same architecture facts the
    /// converter used to write `config.json`, and its resident index carries
    /// the tensor offsets, so this needs no second architecture description --
    /// it is the snapshot's reader pointed at a different file layout.
    ///
    /// Two facts live only in the snapshot config and are re-derived here:
    /// `full_attention_interval`, from the spacing of the manifest's
    /// full-attention mask, and the norm epsilon, which the manifest does not
    /// record and which is 1e-6 for every Qwen 3.5-family model. The first is
    /// checked for regularity rather than assumed, because a wrong interval
    /// silently changes which layers use DeltaNet and which use attention.
    public init(gturbo directory: URL) throws {
        self.directory = directory
        // Refused deliberately, and loudly, until the reader is corrected.
        //
        // The repacker produces a byte-identical dense `.gturbo`
        // (`tools/gturbo_diff_snapshot.py` checks that), but this reader does
        // not yet interpret the resident index correctly: it loads, it is
        // fast, and it produces fluent nonsense. A silently wrong answer is
        // the one failure this project refuses, so the install is rejected
        // with the reason rather than served. Flip this once the equivalence
        // check passes -- same model, both paths, token-for-token.
        throw SafeTensorsFile.Failure.malformed(
            "dense .gturbo reading is not implemented yet: use the affine "
            + "snapshot install for this model (tools/install_models.sh "
            + "qwen35-2b|qwen35-4b|qwen35-9b writes one)")
        let manifest = try ManifestReader.read(directoryURL: directory)
        let arch = manifest.arch
        // The family is an identity fact, not an arch field; reading it keeps a
        // GPU install from being offered to the CPU engine by mistake.
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        guard identity.family == .qwen35Dense else {
            throw SafeTensorsFile.Failure.malformed(
                "not a dense install: manifest declares \(identity.family.rawValue)")
        }
        modelType = CPUModelFamily.qwen35Dense.rawValue

        // The DeltaNet geometry is optional in the manifest because older
        // installs predate it. A dense install must carry it -- these values
        // decide the linear-attention arithmetic -- so a missing one is a
        // refusal, not a default.
        func required(_ value: Int?, _ field: String) throws -> Int {
            guard let value else {
                throw SafeTensorsFile.Failure.malformed(
                    "manifest.arch lacks \(field), which the CPU engine needs")
            }
            return value
        }
        configuration = Configuration(
            hiddenSize: arch.hiddenSize,
            layers: arch.numLayers,
            heads: arch.numHeads,
            keyValueHeads: arch.numKVHeads,
            headDim: arch.headDim,
            fullAttentionInterval: try Self.fullAttentionInterval(of: arch),
            linearKeyHeads: try required(arch.linearNumKHeads, "linearNumKHeads"),
            linearValueHeads: try required(arch.linearNumVHeads, "linearNumVHeads"),
            linearKeyHeadDim: try required(arch.linearKeyHeadDim, "linearKeyHeadDim"),
            linearValueHeadDim: try required(arch.linearValueHeadDim, "linearValueHeadDim"),
            convKernel: try required(arch.linearConvKernelSize, "linearConvKernelSize"),
            intermediateSize: arch.ffnIntermediate,
            vocabulary: arch.vocabSize,
            normEpsilon: 1e-6,
            ropeTheta: Float(arch.ropeTheta),
            partialRotaryFactor: arch.partialRotaryFactor,
            tiedEmbedding: arch.tieWordEmbeddings,
            maxPositions: 262_144)

        guard let quant = manifest.quant else {
            throw SafeTensorsFile.Failure.malformed("manifest.quant is missing")
        }
        baseBits = quant.attention.weightBits
        groupSize = quant.attention.groupSize
        // The manifest names slots, not tensors. Every dense tensor is an
        // attention-slot tensor except the embedding and the head, and the
        // engine asks by name, so those two carry the override.
        let embeddingBits = quant.embedding.weightBits
        var widths: [String: Int] = [:]
        let weightsURL = directory.appendingPathComponent("model_weights.bin")
        let index = try ResidentIndexReader.load(fileURL: weightsURL)
        // Mapped once, held for the snapshot's life; see `ResidentWeights`.
        let weights = ResidentWeights(
            data: try Data(contentsOf: weightsURL, options: .alwaysMapped))
        for name in index.entries.keys {
            let stem = name.hasSuffix(".weight")
                ? String(name.dropLast(".weight".count)) : name
            if stem.hasSuffix("embed_tokens") || stem.hasSuffix("lm_head")
                || stem == "language_model.lm_head" {
                widths[stem] = embeddingBits
            }
        }
        if embeddingBits != baseBits {
            // `matrix(_:)` keys widths by stem, and the head is read by name.
            widths["language_model.model.embed_tokens"] = embeddingBits
            widths["language_model.lm_head"] = embeddingBits
        }
        self.widths = widths
        storage = .gturbo(index: index, weights: weights)
    }

    /// The `full_attention_interval` the manifest's mask encodes.
    ///
    /// `1` marks full attention and `2` gated DeltaNet, and every Qwen 3.5
    /// dense model puts full attention on the last layer of each group of
    /// `interval`. Deriving it is exact for a regular mask and refuses an
    /// irregular one rather than guessing, because the interval decides which
    /// layers take the arithmetic path.
    private static func fullAttentionInterval(
        of arch: ManifestArch
    ) throws -> Int {
        let mask = arch.fullAttentionLayerMask
        let full = mask.enumerated().filter { $0.element == 1 }.map(\.offset)
        guard full.count >= 2 else {
            throw SafeTensorsFile.Failure.malformed(
                "manifest's attention mask has fewer than two full-attention "
                + "layers, so no interval can be derived: full at \(full)")
        }
        // The gap between *consecutive* full-attention layers. Every Qwen 3.5
        // model puts full attention on the last layer of each group, so a
        // regular mask has one gap throughout.
        let gaps = Set(zip(full, full.dropFirst()).map { $1 - $0 })
        guard gaps.count == 1, let interval = gaps.first, interval > 1 else {
            throw SafeTensorsFile.Failure.malformed(
                "manifest's attention mask is not a regular interval: full at \(full)")
        }
        return interval
    }

    public func bits(forStem stem: String) -> Int { widths[stem] ?? baseBits }

    /// The stem of a `.weight` name, which is how widths and the index are
    /// keyed.
    private func stem(of name: String) -> String {
        name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
    }

    /// One affine matrix out of a `.gturbo` resident payload.
    ///
    /// The index already carries the packed shape, the weight extent and the
    /// scale/bias extents, so this is a mapping rather than a parse. The
    /// resident file is opened read-only and never written, so the pointers
    /// handed out live as long as the mapping and concurrent row-range reads
    /// cannot race -- the same invariant the snapshot case documents.
    private static func matrix(_ name: String,
                               index: ResidentIndex,
                               weights: ResidentWeights,
                               groupSize: Int,
                               bits: Int) throws -> Matrix {
        let stem = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count)) : name
        guard let entry = index.entries[name] else {
            throw SafeTensorsFile.Failure.missing(name)
        }
        let rows = Int(entry.shape.0)
        // The resident index stores the *logical* (unpacked) width -- the
        // repacker derives it from the scales, `lastScale * groupSize` -- where
        // a safetensors snapshot stores the packed word count and needs
        // `* lanes`. Reading it the snapshot's way made every matrix 2-4x too
        // wide, which showed up as a generation that never finished rather
        // than as an error.
        let columns = Int(entry.shape.1)
        // The companions are offsets on the weight's own entry, not entries of
        // their own: a repacked `.gturbo` carries one record per tensor and
        // points at its scale and bias spans. (A safetensors snapshot names
        // them as separate tensors, which is why the two readers differ here.)
        guard entry.sizeBytes > 0, entry.scaleSize > 0, entry.biasSize > 0 else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): empty weight, scales or biases in the resident index")
        }
        guard let base = weights.base else {
            throw SafeTensorsFile.Failure.malformed("model_weights.bin could not be mapped")
        }
        return Matrix(
            weights: UnsafeRawBufferPointer(start: base.advanced(by: Int(entry.fileOffset)),
                                            count: Int(entry.sizeBytes)),
            scales: base.advanced(by: Int(entry.scaleOffset))
                .assumingMemoryBound(to: UInt16.self),
            biases: base.advanced(by: Int(entry.biasOffset))
                .assumingMemoryBound(to: UInt16.self),
            rows: rows, columns: columns, bits: bits, groupSize: groupSize)
    }

    /// Fault every shard in, so the model is in memory rather than in the
    /// page cache's good graces. Returns the bytes made resident.
    @discardableResult
    public func makeResident() -> Int {
        switch storage {
        case .safetensors:
            return shards.values.reduce(0) { $0 + $1.makeResident() }
        case .gturbo(_, let weights):
            // A `.gturbo`'s resident payload is one file, mapped once at load.
            // Faulting it in is the same intent as faulting every snapshot
            // shard in: touch a byte per page so the pages are resident rather
            // than at the page cache's mercy.
            return Self.faultIn(weights)
        }
    }

    /// Touch a byte per page so the mapping is resident. Returns the bytes.
    private static func faultIn(_ weights: ResidentWeights) -> Int {
        var touched = 0
        weights.data.withUnsafeBytes { raw in
            // `madvise(WILLNEED)` is not exposed here and this is a load-time
            // cost paid once.
            var offset = 0
            while offset < raw.count {
                touched &+= Int(raw[offset])
                offset += 4096
            }
        }
        return weights.data.count
    }

    /// The family this snapshot's layer shape belongs to, or nil when the
    /// CPU engine does not implement it.
    public var family: CPUModelFamily? {
        CPUModelFamily.resolve(modelType: modelType)
    }

    public let modelType: String?

    private func shard(_ name: String) throws -> SafeTensorsFile {
        guard let file = placement[name], let shard = shards[file] else {
            throw SafeTensorsFile.Failure.missing(name)
        }
        return shard
    }

    public func floats(_ name: String) throws -> [Float] {
        if case .gturbo(let index, let weights) = storage {
            guard let entry = index.entries[name] else {
                throw SafeTensorsFile.Failure.missing(name)
            }
            guard let base = weights.base else {
                throw SafeTensorsFile.Failure.malformed("model_weights.bin could not be mapped")
            }
            let raw = UnsafeRawBufferPointer(
                start: base.advanced(by: Int(entry.fileOffset)),
                count: Int(entry.sizeBytes))
            // Dtype codes are the repacker's (`ietnyDtype`): 0 u32, 1 bf16,
            // 2 fp16, 3 fp32. The bf16 widening is the same bit trick the
            // safetensors reader uses -- the top sixteen bits of a float32 are
            // exactly a bfloat16.
            switch entry.dtype {
            case 3:
                return Array(raw.bindMemory(to: Float.self))
            case 1:
                return raw.bindMemory(to: UInt16.self).map {
                    Float(bitPattern: UInt32($0) << 16)
                }
            case 2:
                return raw.bindMemory(to: Float16.self).map(Float.init)
            default:
                throw SafeTensorsFile.Failure.unsupported(
                    dtype: "resident dtype \(entry.dtype)", name: name)
            }
        }
        return try shard(name).floats(name)
    }

    public func has(_ name: String) -> Bool {
        switch storage {
        case .safetensors: return placement[name] != nil
        case .gturbo(let index, _): return index.entries[name] != nil
        }
    }

    /// A quantized matrix by its `.weight` name.
    public func matrix(_ name: String) throws -> Matrix {
        if case .gturbo(let index, let weights) = storage {
            return try Self.matrix(name, index: index, weights: weights,
                                   groupSize: groupSize, bits: bits(forStem: stem(of: name)))
        }
        let stem = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count)) : name
        let shard = try shard(name)
        let entry = try shard.entry(name)
        guard entry.shape.count == 2 else {
            throw SafeTensorsFile.Failure.malformed("\(name) is not a matrix")
        }
        let width = bits(forStem: stem)
        let lanes = 32 / width
        let rows = entry.shape[0]
        let columns = entry.shape[1] * lanes
        let scales = try shard.bytes(stem + ".scales")
        let biases = try shard.bytes(stem + ".biases")
        guard scales.count == rows * (columns / groupSize) * 2,
              biases.count == scales.count else {
            throw SafeTensorsFile.Failure.malformed(
                "\(stem): scales and biases do not match \(rows)x\(columns) "
                + "at \(width) bits, group \(groupSize)")
        }
        return Matrix(weights: try shard.bytes(name),
                      scales: scales.baseAddress!.assumingMemoryBound(to: UInt16.self),
                      biases: biases.baseAddress!.assumingMemoryBound(to: UInt16.self),
                      rows: rows,
                      columns: columns,
                      bits: width,
                      groupSize: groupSize)
    }
}
