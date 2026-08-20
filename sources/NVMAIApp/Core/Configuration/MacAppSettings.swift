import Foundation
import NVMAI

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 1

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.fourK.tokens
    var expertCacheSlots: Int = 64
    var temperature: Double = 0.6
    var topKEnabled: Bool = true
    var topK: Int = GenerationDefaults.topK
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var conciseMode: Bool = false
    var kvCacheBits: Int = 8
    var ropeScalingMode: String = "none"

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case conciseMode
        case kvCacheBits
        case ropeScalingMode
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.fourK.tokens,
         expertCacheSlots: Int = 64,
         temperature: Double = 0.6,
         topKEnabled: Bool = true,
         topK: Int = GenerationDefaults.topK,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         conciseMode: Bool = false,
         kvCacheBits: Int = 8,
         ropeScalingMode: String = "none") {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.conciseMode = conciseMode
        self.kvCacheBits = kvCacheBits
        self.ropeScalingMode = ropeScalingMode
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
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
        kvCacheBits = try container.decodeIfPresent(Int.self, forKey: .kvCacheBits) ?? 8
        ropeScalingMode = try container.decodeIfPresent(
            String.self, forKey: .ropeScalingMode) ?? "none"
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && KVCachePrecision(rawValue: kvCacheBits) != nil
            && RuntimeRoPEScalingMode(rawValue: ropeScalingMode) != nil
            && (ropeScalingMode == "yarn"
                ? RuntimeConfiguration.supportedYaRNContextTokens.contains(contextTokens)
                : contextTokens <= RuntimeConfiguration.nativeMaximumContextTokens)
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
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
