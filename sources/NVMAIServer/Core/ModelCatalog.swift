import Foundation
import NVMAI

/// Every model a models directory holds that this server can actually serve,
/// GPU installs and CPU snapshots alike.
///
/// Identity is read, never parsed out of a directory name: a GPU install's id,
/// family and width come from its manifest through the same reader the loader
/// uses, and a CPU snapshot's from its own config.json. A directory the scan
/// cannot read is skipped with a reason rather than failing startup, because
/// one half-copied model must not take the other nine offline.
public struct ModelCatalog: Sendable {

    public enum Backend: String, Sendable {
        case gpu, cpu
    }

    public enum Kind: Sendable, Equatable {
        case gpu(ModelFamily)
        case cpu(CPUModelFamily)

        public var backend: Backend {
            switch self {
            case .gpu: .gpu
            case .cpu: .cpu
            }
        }

        /// The kind that serves the same install on `backend`, where the
        /// runtime implements the family there too.
        public func kind(forBackend backend: Backend) -> Kind? {
            switch (self, backend) {
            case (.gpu, .gpu), (.cpu, .cpu):
                return self
            case (.gpu(.qwen35Dense), .cpu), (.cpu(.qwen35Dense), .gpu):
                // The dense family is the one both engines implement, and its
                // payload is identical for either.
                return backend == .cpu ? .cpu(.qwen35Dense) : .gpu(.qwen35Dense)
            default:
                return nil
            }
        }

        public var familyName: String {
            switch self {
            case .gpu(let family): family.rawValue
            case .cpu(let family): family.rawValue
            }
        }

        public var supportedReasoningLevels: [ReasoningLevel] {
            switch self {
            case .gpu(let family): family.supportedReasoningLevels
            case .cpu(let family): family.supportedReasoningLevels
            }
        }

        public var levelWhenOn: ReasoningLevel {
            switch self {
            case .gpu(let family): family.reasoningControl.levelWhenOn
            case .cpu(let family): family.levelWhenOn
            }
        }

        public func runtimeReasoning(
            for level: ReasoningLevel
        ) throws -> (thinking: ModelThinkingMode, effort: ModelReasoningEffort?) {
            switch self {
            case .gpu(let family): try family.runtimeReasoning(for: level)
            case .cpu(let family): try family.runtimeReasoning(for: level)
            }
        }
    }

    public struct Entry: Sendable, Equatable {
        public let id: String
        public let name: String
        public let kind: Kind
        /// Weight width in bits: the routed experts for a GPU install, the
        /// base affine width for a CPU snapshot.
        public let quant: Int
        public let path: URL
        /// The sampling the model will really use for anything a request
        /// omits: the tuning profile's for a GPU install, the family's for a
        /// CPU snapshot.
        public let sampling: GenerationDefaults.Sampling
        /// The context the checkpoint claims, where the engine honours less
        /// than the server's --max-context. Nil for GPU installs, which serve
        /// whatever the server was configured for.
        public let contextLimit: Int?
        public let sizeBytes: Int64
        /// Every engine that can serve this install, most-preferred first.
        ///
        /// One for almost everything -- a MoE family is GPU-only and a snapshot
        /// is CPU-only -- but the dense Qwen 3.5 models run on both, and that is
        /// what makes `cpu` or `gpu` a request-level choice rather than a
        /// property of the model. `kind` is the default; the others are named by
        /// an `@cpu` / `@gpu` suffix on the id.
        public let engines: [Backend]

        public init(id: String, name: String, kind: Kind, quant: Int, path: URL,
                    sampling: GenerationDefaults.Sampling, contextLimit: Int? = nil,
                    sizeBytes: Int64 = 0, engines: [Backend]? = nil) {
            self.id = id
            self.name = name
            self.kind = kind
            self.quant = quant
            self.path = path
            self.sampling = sampling
            self.contextLimit = contextLimit
            self.sizeBytes = sizeBytes
            // Defaulting to the entry's own engine keeps every existing caller
            // (and every earlier catalog) meaning what it meant.
            self.engines = engines ?? [kind.backend]
        }

        public var backend: Backend { kind.backend }

        /// The same install served by `backend`, or nil when that engine cannot
        /// serve it. Only the dense Qwen 3.5 family has two engines today: its
        /// `.gturbo` payload is the same file for either, and the runtime picks
        /// the engine.
        public func served(by backend: Backend, id aliasID: String) -> Entry? {
            guard engines.contains(backend),
                  let kind = kind.kind(forBackend: backend) else { return nil }
            return Entry(id: aliasID,
                         name: "\(name) (\(backend.rawValue.uppercased()))",
                         kind: kind, quant: quant, path: path, sampling: sampling,
                         contextLimit: contextLimit, sizeBytes: sizeBytes,
                         engines: engines)
        }
    }

    public struct Skipped: Sendable, Equatable {
        public let path: URL
        public let reason: String
    }

    public let directory: URL
    public private(set) var entries: [Entry]
    public private(set) var skipped: [Skipped]

    public init(directory: URL, entries: [Entry], skipped: [Skipped] = []) {
        self.directory = directory
        self.entries = entries
        self.skipped = skipped
    }

    // MARK: - Scanning

    public static func scan(directory: URL,
                            fileManager: FileManager = .default) -> ModelCatalog {
        let root = directory.standardizedFileURL
        var catalog = ModelCatalog(directory: root, entries: [])
        let children: [URL]
        do {
            children = try fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            catalog.skipped.append(Skipped(path: root, reason: "cannot list: \(error)"))
            return catalog
        }
        for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            // A model linked in from another disk is served from where it
            // lives; the link itself does not pass as a directory.
            let target = child.resolvingSymlinksInPath()
            // Lock files and other loose files sit beside the installs; they
            // are not models and not worth a warning.
            guard (try? target.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true else {
                continue
            }
            catalog.add(probing: target)
        }
        catalog.entries.sort { lhs, rhs in
            lhs.backend == rhs.backend ? lhs.id < rhs.id : lhs.backend == .gpu
        }
        return catalog
    }

    /// Probes one directory and adds it, or records why it was skipped.
    /// Returns the entry when one was added.
    @discardableResult
    public mutating func add(probing directory: URL) -> Entry? {
        let path = directory.standardizedFileURL
        switch Self.probe(path) {
        case .success(let entry):
            if let existing = entries.first(where: { $0.id == entry.id }) {
                skipped.append(Skipped(
                    path: path,
                    reason: "duplicate id \(entry.id), already served from \(existing.path.path)"))
                return nil
            }
            entries.append(entry)
            return entry
        case .failure(let failure):
            skipped.append(Skipped(path: path, reason: failure.reason))
            return nil
        }
    }

    struct ProbeFailure: Error {
        let reason: String
    }

    static func probe(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: directory.appendingPathComponent("manifest.json").path) {
            return probeInstall(directory)
        }
        if fileManager.fileExists(atPath: directory.appendingPathComponent("config.json").path) {
            return probeSnapshot(directory)
        }
        return .failure(ProbeFailure(
            reason: "neither manifest.json (a GPU install) nor config.json (a CPU snapshot)"))
    }

    private static func probeInstall(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let identity: ManifestIdentity
        do {
            identity = try ManifestReader.peekIdentity(directoryURL: directory)
        } catch {
            return .failure(ProbeFailure(reason: "unreadable manifest.json: \(error)"))
        }
        switch identity.family {
        case .qwen36MTP, .qwen38flashMTP:
            // A draft head has no tokenizer and no layers of its own to run;
            // listing it would offer a model that fails on first use.
            return .failure(ProbeFailure(
                reason: "an MTP draft head (\(identity.family.rawValue)), served only beside its target"))
        case .qwen36, .qwen38flash, .qwen35Dense:
            break
        }
        guard GFTokenizer.tokenizerFolder(forModelDirectory: directory) != nil else {
            return .failure(ProbeFailure(reason: "no tokenizer/tokenizer.json; the install is incomplete"))
        }
        let id = ServerModelIdentity.apiModelID(manifestModelID: identity.modelID,
                                                family: identity.family,
                                                weightBits: identity.weightBits)
        let base = ServerModelIdentity.base(manifestModelID: identity.modelID,
                                            family: identity.family)
        return .success(Entry(
            id: id,
            name: displayNames[base] ?? base,
            // A dense install is served by the GPU engine by default now that
            // the family is implemented there, and by the CPU engine when the
            // request (or the launch) names it -- `engines` is what says both
            // are available, and `ModelRouter` derives the `@cpu` alias from it.
            kind: .gpu(identity.family),
            quant: identity.weightBits,
            path: directory,
            sampling: ModelProfile.resolve(identity: identity).sampling,
            sizeBytes: sizeOnDisk(directory),
            engines: identity.family == .qwen35Dense ? [.gpu, .cpu] : nil))
    }

    private static func probeSnapshot(_ directory: URL) -> Result<Entry, ProbeFailure> {
        let config: [String: Any]
        do {
            let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return .failure(ProbeFailure(reason: "config.json is not a JSON object"))
            }
            config = object
        } catch {
            return .failure(ProbeFailure(reason: "unreadable config.json: \(error)"))
        }
        let modelType = config["model_type"] as? String
        guard let family = CPUModelFamily.resolve(modelType: modelType) else {
            return .failure(ProbeFailure(reason: CPUModelFamily.refusal(modelType: modelType)))
        }
        guard let quantization = config["quantization"] as? [String: Any],
              let bits = quantization["bits"] as? Int else {
            return .failure(ProbeFailure(
                reason: "config.json has no quantization block; the CPU engine serves affine snapshots"))
        }
        // Snapshots are written in place by a converter, so a config can
        // exist before the rest of the directory does.
        for required in ["model.safetensors.index.json", "tokenizer.json"]
        where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(required).path) {
            return .failure(ProbeFailure(reason: "incomplete snapshot: no \(required)"))
        }
        let declared = (config["model_id"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        let id = declared ?? directory.lastPathComponent
        let name = (config["display_name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? id
        return .success(Entry(
            id: id,
            name: name,
            kind: .cpu(family),
            quant: bits,
            path: directory,
            sampling: family.samplingDefaults,
            contextLimit: config["max_position_embeddings"] as? Int,
            sizeBytes: sizeOnDisk(directory)))
    }

    /// A single-model CPU server's family, for fitting `--reasoning` to it
    /// before the model itself is opened.
    ///
    /// Both shapes ship: a safetensors snapshot declares its architecture in
    /// `config.json`, and a `.gturbo` install declares it in `manifest.json`.
    /// Reading only the first is how `--cpu` against an installed dense model
    /// died with a raw `NSCocoaErrorDomain` "config.json couldn't be opened"
    /// instead of naming the family -- and a `.gturbo` install is exactly the
    /// shape the three dense Qwen 3.5 models are installed as.
    static func snapshotFamily(_ directory: URL) throws -> CPUModelFamily {
        if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path) {
            let family = try ManifestReader.peekFamily(directoryURL: directory)
            guard family == .qwen35Dense else {
                throw CPUModelBackend.CPUBackendError.unsupported(
                    CPUModelFamily.refusal(modelType: family.rawValue))
            }
            return .qwen35Dense
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let modelType = config?["model_type"] as? String
        guard let family = CPUModelFamily.resolve(modelType: modelType) else {
            throw CPUModelBackend.CPUBackendError.unsupported(
                CPUModelFamily.refusal(modelType: modelType))
        }
        return family
    }

    /// Human names for the installs this project ships. A manifest carries an
    /// id, not a name; an install outside this table is listed by its id.
    static let displayNames: [String: String] = [
        "qwen3.6-35b-a3b": "Qwen 3.6 35B-A3B",
        "ornith-1.5-35b-a3b": "Ornith 1.5 35B-A3B",
        "qwen-agentworld": "Qwen AgentWorld 35B-A3B",
        "kat-coder-v2.5": "KAT-Coder-V2.5-Dev 35B-A3B",
        "qwen3.8-flash-next": "Qwen 3.8 Flash Next 125B-A6B",
        // The dense CPU models. Their snapshots carried these names in
        // `config.json -> display_name`, which only the snapshot probe reads;
        // the install probe has a manifest, and a manifest carries an id, not
        // a name. Without these entries repacking them would silently rename
        // them from "Qwen 3.5 9B" to "qwen3.5-9b" in the app and the server's
        // /v1/models listing.
        "qwen3.5-2b": "Qwen 3.5 2B",
        "qwen3.5-4b": "Qwen 3.5 4B",
        "qwen3.5-9b": "Qwen 3.5 9B",
    ]

    /// Allocated bytes, so an APFS clone or a sparse file reports what it
    /// really occupies rather than its logical length.
    static func sizeOnDisk(_ directory: URL) -> Int64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .fileSizeKey, .isRegularFileKey]
        guard let walker = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: keys) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)),
                  values.isRegularFile == true else { continue }
            total += Int64(values.totalFileAllocatedSize ?? values.fileSize ?? 0)
        }
        return total
    }

    // MARK: - Lookup

    public func entry(id: String) -> Entry? {
        entries.first { $0.id == id }
    }

    /// The entry `--model` names: a catalog id, or the path of a directory
    /// the catalog holds.
    public func entry(idOrPath value: String) -> Entry? {
        if let byID = entry(id: value) { return byID }
        let path = URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath().path
        return entries.first { $0.path.resolvingSymlinksInPath().path == path }
    }

    // MARK: - Reporting

    /// One notice for everything skipped, written once at startup.
    public func reportSkipped(to handle: FileHandle = .standardError) {
        guard !skipped.isEmpty else { return }
        var lines = ["catalog: skipped \(skipped.count) of \(skipped.count + entries.count) "
            + "directories in \(directory.path):"]
        for item in skipped {
            lines.append("  \(item.path.lastPathComponent): \(item.reason)")
        }
        handle.write(Data((lines.joined(separator: "\n") + "\n").utf8))
    }

    /// `NVMAIServer --catalog` output, which a launcher script parses.
    ///
    /// Assembled by hand because JSONEncoder does not keep key order, and the
    /// document should read in the order the launcher documents its fields.
    /// Strings still go through the encoder, so escaping is not hand-rolled.
    public func jsonData() throws -> Data {
        let models = try entries.map(Self.json(for:))
        return Data(("{\"models\":[" + models.joined(separator: ",") + "]}").utf8)
    }

    private static func json(for entry: Entry) throws -> String {
        let thinking = try entry.kind.supportedReasoningLevels.map { try quoted($0.rawValue) }
        let sampling = "{\"temperature\":\(decimal(entry.sampling.temperature, places: 4)),"
            + "\"top_p\":\(decimal(entry.sampling.topP, places: 4)),"
            + "\"top_k\":\(entry.sampling.topK)}"
        let fields: [(key: String, value: String)] = [
            ("id", try quoted(entry.id)),
            ("name", try quoted(entry.name)),
            ("family", try quoted(entry.kind.familyName)),
            ("quant", String(entry.quant)),
            ("backend", try quoted(entry.backend.rawValue)),
            ("engines", try quoted(entry.engines.map(\.rawValue).joined(separator: ","))),
            ("path", try quoted(entry.path.path)),
            ("thinking", "[" + thinking.joined(separator: ",") + "]"),
            ("sampling", sampling),
            ("size_gb", decimal(Double(entry.sizeBytes) / 1e9, places: 1)),
        ]
        return "{" + fields.map { "\"\($0.key)\":\($0.value)" }.joined(separator: ",") + "}"
    }

    private static func quoted(_ value: String) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    /// Float 0.6 widens to 0.6000000238; a launcher showing the defaults
    /// should read the value the model card states.
    private static func decimal(_ value: some BinaryFloatingPoint, places: Int) -> String {
        let scale = pow(10.0, Double(places))
        return String((Double(value) * scale).rounded() / scale)
    }
}
