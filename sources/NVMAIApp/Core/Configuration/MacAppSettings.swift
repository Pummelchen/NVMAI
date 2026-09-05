import Foundation
import NVMAI

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    /// 2 (5.0.3): `expertCacheSlots` 0 means "from the model's profile" and
    /// is the default; `samplingFollowsModel` added. A version-1 file whose
    /// slot count is the old flat default (64) migrates to automatic, and its
    /// sampling follows the model only if it still held the house values.
    static let currentVersion = 2

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.fourK.tokens
    var expertCacheSlots: Int = AppRuntimeOptions.automaticSlotCount
    var samplingFollowsModel: Bool = true
    var temperature: Double = 0.6
    var topKEnabled: Bool = true
    var topK: Int = GenerationDefaults.topK
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var conciseMode: Bool = false
    var thinkingMode: String = "off"
    var kvCacheBits: Int = 8
    var ropeScalingMode: String = "none"

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case samplingFollowsModel
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case conciseMode
        case thinkingMode
        case kvCacheBits
        case ropeScalingMode
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.fourK.tokens,
         expertCacheSlots: Int = AppRuntimeOptions.automaticSlotCount,
         // nil: follow the model unless the sampling given here is already
         // a choice of its own (anything but the house values).
         samplingFollowsModel: Bool? = nil,
         temperature: Double = 0.6,
         topKEnabled: Bool = true,
         topK: Int = GenerationDefaults.topK,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         conciseMode: Bool = false,
         thinkingMode: String = "off",
         kvCacheBits: Int = 8,
         ropeScalingMode: String = "none") {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.samplingFollowsModel = samplingFollowsModel
            ?? Self.isHouseSampling(temperature: temperature, topKEnabled: topKEnabled, topK: topK,
                                    topPEnabled: topPEnabled, topP: topP)
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.conciseMode = conciseMode
        self.thinkingMode = thinkingMode
        self.kvCacheBits = kvCacheBits
        self.ropeScalingMode = ropeScalingMode
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fileVersion = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        samplingFollowsModel = try container.decodeIfPresent(
            Bool.self, forKey: .samplingFollowsModel)
            ?? Self.isHouseSampling(temperature: temperature, topKEnabled: topKEnabled, topK: topK,
                                    topPEnabled: topPEnabled, topP: topP)
        if fileVersion < 2 && expertCacheSlots == 64 {
            expertCacheSlots = AppRuntimeOptions.automaticSlotCount
        }
        version = fileVersion < 2 ? Self.currentVersion : fileVersion
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .return
        showPromptExamples = try container.decodeIfPresent(
            Bool.self,
            forKey: .showPromptExamples) ?? true
        conciseMode = try container.decodeIfPresent(
            Bool.self,
            forKey: .conciseMode) ?? false
        thinkingMode = try container.decodeIfPresent(
            String.self,
            forKey: .thinkingMode) ?? "off"
        kvCacheBits = try container.decodeIfPresent(Int.self, forKey: .kvCacheBits) ?? 8
        ropeScalingMode = try container.decodeIfPresent(
            String.self, forKey: .ropeScalingMode) ?? "none"
    }

    /// The pre-profile defaults every settings file used to start from.
    static func isHouseSampling(temperature: Double, topKEnabled: Bool, topK: Int,
                                topPEnabled: Bool, topP: Double) -> Bool {
        temperature == 0.6 && topKEnabled && topK == GenerationDefaults.topK
            && topPEnabled && topP == 0.95
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && KVCachePrecision(rawValue: kvCacheBits) != nil
            && RuntimeRoPEScalingMode(rawValue: ropeScalingMode) != nil
            && ModelThinkingMode(rawValue: thinkingMode) != nil
            && (ropeScalingMode == "yarn"
                ? RuntimeConfiguration.supportedYaRNContextTokens.contains(contextTokens)
                : contextTokens <= RuntimeConfiguration.nativeMaximumContextTokens)
            && (expertCacheSlots == AppRuntimeOptions.automaticSlotCount
                || AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots))
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        guard fileManager.fileExists(atPath: fileURL.path) else {
            let settings = MacAppSettings()
            do {
                try save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
            } catch {
                FileHandle.standardError.write(Data(
                    ("NVMAI app settings save failed: \(error)\n").utf8))
            }
            return settings
        }
        do {
            let data = try Data(contentsOf: fileURL)
            let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
            guard settings.isValid() else { throw InvalidSettings() }
            return settings
        } catch {
            // Keep the existing file on a transient/decode error so the user's
            // settings can be recovered or corrected; fall back to defaults
            // for this session only rather than destroying the file.
            FileHandle.standardError.write(Data(
                ("NVMAI app settings load failed: \(error); using defaults\n").utf8))
            return MacAppSettings()
        }
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
