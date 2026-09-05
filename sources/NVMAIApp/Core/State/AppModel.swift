import Foundation
import NVMAI
import NVMAIRepackCore
import Observation
import os

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public var modelPathText: String
    public var promptText: String = ""
    public private(set) var outputPromptText: String = ""
    public var outputText: String = ""
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    /// While true, temperature / top-k / top-p track the model's tuning
    /// profile (`ModelProfile.sampling`: the card's values for Qwen 3.8, the
    /// house values for the 35B family) and are re-applied whenever the
    /// model directory changes. Editing any of them switches this off;
    /// switching it back on re-applies the profile. Persisted per model.
    public var samplingFollowsModel: Bool = true {
        didSet {
            guard samplingFollowsModel != oldValue else { return }
            if samplingFollowsModel { applyModelSampling() }
            persistSettings()
        }
    }
    public var temperature: Double = 0.6 { didSet { noteSamplingEdit() } }
    public var topKEnabled: Bool = true { didSet { noteSamplingEdit() } }
    public var topK: Int = GenerationDefaults.topK { didSet { noteSamplingEdit() } }
    public var topPEnabled: Bool = true { didSet { noteSamplingEdit() } }
    public var topP: Double = 0.95 { didSet { noteSamplingEdit() } }
    private var applyingModelSampling = false
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var showPromptExamples: Bool = true
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false

    private let client: any AppInferenceClient
    private let installer: any AppModelInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()
    private static let modelLog = Logger(subsystem: "com.nvmai.app", category: "model")

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: any AppModelInstallerClient = RepackModelInstallerClient(descriptor: .selected),
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false) {
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        self.runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            conciseMode: settings.conciseMode,
            thinkingMode: ModelThinkingMode(rawValue: settings.thinkingMode) ?? .off,
            kvCachePrecision: KVCachePrecision(rawValue: settings.kvCacheBits) ?? .int8,
            ropeScalingMode: RuntimeRoPEScalingMode(
                rawValue: settings.ropeScalingMode) ?? .none)
        self.maxContextTokens = settings.contextTokens
        self.samplingFollowsModel = settings.samplingFollowsModel
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.newlineShortcut = settings.newlineShortcut
        self.showPromptExamples = settings.showPromptExamples
        self.installationStatus = AppModelInstallationProbe.status(at: directory)
        self.client = client
        self.installer = installer
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        if settings.samplingFollowsModel { applyModelSampling() }
        refreshInstallReadiness()
    }

    /// The sampling the model at `directory` ships with: its profile row when
    /// the manifest is readable, the house values otherwise.
    public static func modelSampling(forModelAt directory: URL) -> GenerationDefaults.Sampling {
        (try? ManifestReader.peekIdentity(directoryURL: directory))
            .map { ModelProfile.resolve(identity: $0).sampling }
            ?? GenerationDefaults.house
    }

    private func applyModelSampling() {
        let sampling = Self.modelSampling(
            forModelAt: URL(fileURLWithPath: modelPathText, isDirectory: true))
        applyingModelSampling = true
        defer { applyingModelSampling = false }
        temperature = Double(sampling.temperature)
        topKEnabled = true
        topK = sampling.topK
        topPEnabled = true
        topP = Double(sampling.topP)
    }

    private func noteSamplingEdit() {
        guard !applyingModelSampling, samplingFollowsModel else { return }
        samplingFollowsModel = false
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    /// The payload is present and matches its manifest; only the receipt is
    /// bound to the directory the model used to live in.
    public var needsReattestation: Bool {
        if case .needsReattestation = installationStatus { return true }
        return false
    }

    /// Deliberately excludes `needsReattestation`. A moved model is not a
    /// missing one — offering the installer here would put a multi-gigabyte
    /// download in front of a user whose model is already on disk, which is
    /// exactly what this state exists to prevent.
    public var requiresModelInstallation: Bool {
        !isModelInstalled && !needsReattestation
    }

    /// Re-issues the install receipt in place: re-hashes the payload against
    /// the manifest and rebinds it to the current directory. No network.
    public var canReattestModel: Bool {
        needsReattestation && !isRunning && installTask == nil
    }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && requiresModelInstallation
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        return min(addition.overflow ? UInt64.max : addition.partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    public var hasOutputTranscript: Bool {
        !outputPromptText.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    public var outputConversationPlainText: String {
        let response = outputResponsePlainText
        switch (outputPromptText.isEmpty, response.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return "You:\n\(outputPromptText)"
        case (true, false):
            return "Answer:\n\(response)"
        case (false, false):
            return "You:\n\(outputPromptText)\n\nAnswer:\n\(response)"
        }
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            lastStopReason: diagnostics?.stopReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        // D15: report a live metric only while a generation is running. When
        // idle, the reporter holds the previous run's last snapshot — a
        // generation-old value the HUD must not show.
        guard isRunning else { return nil }
        if let reporter = client as? any AppInferenceMemoryReporting,
           let bytes = reporter.currentInferenceMemoryBytes {
            return bytes
        }
        return memorySampler.sample()
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: maxContextTokens,
                            options: runtimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    public func setModelURL(_ url: URL) {
        guard !isRunning else { return }
        let path = url.standardizedFileURL.path
        guard path != modelPathText else { return }

        modelPathText = path
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installer.cancel()
        installTask = nil
        resetInstallETA()
        installState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(at: URL(fileURLWithPath: path))
        refreshInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func setNewlineShortcut(_ shortcut: AppNewlineShortcut) {
        guard newlineShortcut != shortcut else { return }
        newlineShortcut = shortcut
        persistSettings()
    }

    public func setShowPromptExamples(_ show: Bool) {
        guard showPromptExamples != show else { return }
        showPromptExamples = show
        persistSettings()
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = maxContextTokens
        let options = runtimeOptions
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: options,
                                             forceLogitsHead: forceLogitsHead)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        loadState = .loading(.validatingDirectory)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    /// Re-issue the install receipt for a model that was moved or renamed.
    ///
    /// Re-hashes the payload against its manifest and rebinds the receipt to
    /// the current directory. Local only — the alternative the app used to
    /// offer for this state was a full re-download.
    /// Test hook: has the re-attestation task finished?
    var installTaskIsIdleForTesting: Bool { installTask == nil }

    public func reattestModel() {
        guard canReattestModel else { return }
        let directory = URL(fileURLWithPath: modelPathText).standardizedFileURL
        installGeneration &+= 1
        let generation = installGeneration
        installTask = Task { [weak self, installer] in
            do {
                try await installer.reattestInstall(outputDirectory: directory)
            } catch {
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.finishInstallFailure(error, generation: generation)
                return
            }
            guard let self, generation == self.installGeneration else { return }
            self.installTask = nil
            // Re-probe rather than assuming success: the re-attestation only
            // writes a receipt if the payload still matches its manifest.
            self.installationStatus = AppModelInstallationProbe.status(at: directory)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        // D5: abort the load on the service side too; cancelling only the
        // app-side wait would leave the service loading a model the user
        // asked to cancel.
        lifecycle.cancelLoad()
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.installDefaultModel(outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    public var hasPartialModelDownload: Bool {
        let paths: RemoteInstallPaths
        do {
            paths = try RemoteInstallPaths(outputDirectory: modelPathText)
        } catch {
            // D22: treat an unreadable install layout as "no partial
            // download", but record the failure instead of swallowing it
            // silently; the user will still see a validation error if they
            // attempt an install.
            Self.modelLog.error(
                "hasPartialModelDownload: cannot resolve install paths for \(self.modelPathText): \(String(describing: error))")
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        // Accept whichever supported model is installed here; the runtime
        // detects the family from the manifest at load. Only the post-install
        // check below pins the result to the descriptor we downloaded.
        installationStatus = AppModelInstallationProbe.status(at: outputDirectory)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: installer.descriptor)
            guard installationStatus == .complete else {
                let detail: String
                if case .partial(let reason) = installationStatus {
                    detail = "completed install did not pass metadata validation: \(reason)"
                } else {
                    detail = "completed install did not pass metadata validation"
                }
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: detail),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            loadState = .notLoaded
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        runtimeOptions = AppRuntimeOptions(
            expertCacheSlots: settings.expertCacheSlots,
            prefillEnabled: settings.prefillEnabled,
            conciseMode: settings.conciseMode,
            thinkingMode: ModelThinkingMode(rawValue: settings.thinkingMode) ?? .off,
            kvCachePrecision: KVCachePrecision(rawValue: settings.kvCacheBits) ?? .int8,
            ropeScalingMode: RuntimeRoPEScalingMode(
                rawValue: settings.ropeScalingMode) ?? .none)
        maxContextTokens = settings.contextTokens
        applyingModelSampling = true
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
        applyingModelSampling = false
        samplingFollowsModel = settings.samplingFollowsModel
        if settings.samplingFollowsModel { applyModelSampling() }
        newlineShortcut = settings.newlineShortcut
        showPromptExamples = settings.showPromptExamples
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            samplingFollowsModel: samplingFollowsModel,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            newlineShortcut: newlineShortcut,
            showPromptExamples: showPromptExamples,
            conciseMode: runtimeOptions.conciseMode,
            thinkingMode: runtimeOptions.thinkingMode.rawValue,
            kvCacheBits: runtimeOptions.kvCachePrecision.rawValue,
            ropeScalingMode: runtimeOptions.ropeScalingMode.rawValue)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        do {
            try MacAppSettingsFileStore.save(
                settings,
                forModelDirectory: modelDirectory)
        } catch {
            FileHandle.standardError.write(Data(
                ("NVMAI app settings persist failed: \(error)\n").utf8))
        }
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        // Set initial state based on partial download, then refine if disk space error
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    private func applyLoadState(_ state: AppModelLoadState, generation: UInt64) {
        guard generation == loadGeneration else { return }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    public func clearOutput() {
        guard !isRunning else { return }
        outputPromptText = ""
        outputText = ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
    }

    public func run() {
        guard canRun else { return }
        let request: AppGenerationRequest
        do {
            request = try makeRequest()
        } catch let appError as AppInferenceError {
            error = appError
            return
        } catch {
            let appError = AppInferenceError.unknown("\(error)")
            self.error = appError
            return
        }
        persistSettings()

        generationTranscriptMailbox?.reset()
        outputPromptText = request.prompt
        outputText = ""
        diagnostics = nil
        error = nil
        hasHandledTerminalEvent = false
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        liveMemoryBytes = nil
        phase = .prefill
        runState = .running

        runTask = Task.detached { [weak self, client, request] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"))
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            prompt: promptText,
            maxNewTokens: maxNewTokensOverride ?? maxContextTokens,
            maxContextTokens: maxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            presencePenalty: GenerationDefaults.presencePenalty,
            repetitionPenalty: 1.0,
            runtimeOptions: runtimeOptions)
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .prefillProgress(let done, let total):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            if let reporter = client as? any AppInferenceMemoryReporting {
                liveMemoryBytes = reporter.currentInferenceMemoryBytes
            } else {
                liveMemoryBytes = memorySampler.sample()
            }
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
        case .finished(let diagnostics):
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        error = appError
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        runTask = nil
        // D14: the live snapshot belongs to the generation that just ended.
        liveMemoryBytes = nil
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }
}
