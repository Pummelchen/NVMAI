import Foundation

/// The side-engine's view of a snapshot written by `prepare_qwen35_2b.py`.
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

    public let directory: URL
    public let configuration: Configuration
    private let shards: [String: SafeTensorsFile]
    private let placement: [String: String]
    private let baseBits: Int
    private let groupSize: Int
    /// Per-tensor width overrides, keyed by stem. The 4-bit build keeps the
    /// tied embedding and the attention K/V at 8 bits, because measuring
    /// said that is where the error actually is.
    private let widths: [String: Int]

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
        placement = map
        var opened: [String: SafeTensorsFile] = [:]
        for file in Set(map.values) {
            opened[file] = try SafeTensorsFile(url: directory.appendingPathComponent(file))
        }
        shards = opened
    }

    public func bits(forStem stem: String) -> Int { widths[stem] ?? baseBits }

    /// Fault every shard in, so the model is in memory rather than in the
    /// page cache's good graces. Returns the bytes made resident.
    @discardableResult
    public func makeResident() -> Int {
        shards.values.reduce(0) { $0 + $1.makeResident() }
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
        try shard(name).floats(name)
    }

    public func has(_ name: String) -> Bool { placement[name] != nil }

    /// A quantized matrix by its `.weight` name.
    public func matrix(_ name: String) throws -> Matrix {
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
