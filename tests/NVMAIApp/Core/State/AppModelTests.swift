import Foundation
import Testing
@testable import NVMAIAppCore

@Suite struct AppModelTests {
    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 0.6)
        #expect(request.topK == 20)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 4_096)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
        #expect(request.runtimeOptions.expertCacheSlots == AppRuntimeOptions.automaticSlotCount)
        #expect(request.runtimeOptions.expertCachePolicy == .lfu)
        #expect(request.runtimeOptions.rdadvisePolicy == .default)
        #expect(request.runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func runDisabledWhenPromptEmpty() {
        let model = AppModel()
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "   "
        #expect(!model.canRun)
    }

    @MainActor
    @Test func runDisabledUntilModelReady() {
        let model = AppModel()
        model.promptText = "go"
        #expect(!model.canRun)
    }

    @MainActor
    @Test func disablingTopKNeutralizesBothTruncationControls() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.topKEnabled = false
        model.topPEnabled = true

        let request = try model.makeRequest()
        #expect(request.topK == nil)
        #expect(request.topP == nil)
    }

    @MainActor
    @Test func prefillToggleSurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        model.runtimeOptions.prefillEnabled = false
        #expect(try !model.makeRequest().runtimeOptions.prefillEnabled)

        model.runtimeOptions.prefillEnabled = true
        #expect(try model.makeRequest().runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func adaptiveRDAdvicePolicySurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.runtimeOptions.rdadvisePolicy = .adaptive

        let request = try model.makeRequest()
        #expect(request.runtimeOptions.rdadvisePolicy == .adaptive)
    }

    @MainActor
    @Test func loadAffectingRuntimeChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func contextChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.maxContextTokens = AppContextLengthOption.eightK.tokens
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens

        #expect(try model.makeRequest().maxNewTokens == AppContextLengthOption.sixtyFourK.tokens)
    }

    @MainActor
    @Test func requestTimePrefillChangeDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.runtimeOptions.prefillEnabled = false

        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func newlineShortcutDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setNewlineShortcut(.shiftReturn)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func promptExamplesPreferenceDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setShowPromptExamples(false)

        #expect(!model.showPromptExamples)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.run()

        let reachedIdle = await waitUntil { !model.isRunning }
        #expect(reachedIdle, "run did not reach idle within the wait budget")
        #expect(!model.isRunning)
        #expect(model.outputText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.run()

        #expect(model.outputPromptText == "original prompt")
        #expect(model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\noriginal prompt")

        model.promptText = "edited prompt"
        let reachedIdle = await waitUntil { !model.isRunning }
        #expect(reachedIdle, "run did not reach idle within the wait budget")

        #expect(model.outputPromptText == "original prompt")
        #expect(model.outputResponsePlainText == "answer")
        #expect(model.outputConversationPlainText
            == "You:\noriginal prompt\n\nAnswer:\nanswer")
        #expect(!model.outputConversationPlainText.contains("edited prompt"))
    }

    @MainActor
    @Test func staleReadySessionDisablesGenerationUntilReload() throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("stale-runtime")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "go"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.canRun)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
        #expect(!model.canRun)
        #expect(model.canReloadModel)
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func cancelAfterPartialOutputCanBeCleared() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "stop after token"
        model.maxNewTokensOverride = 10
        model.run()

        let producedToken = await waitUntil { model.liveTokenCount > 0 }
        #expect(producedToken, "no token was produced within the wait budget")
        #expect(model.liveTokenCount > 0)
        model.cancel()
        #expect(model.isCancellationPending)
        let reachedIdle = await waitUntil { !model.isRunning }
        #expect(reachedIdle, "cancellation did not reach idle within the wait budget")

        #expect(!model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(!model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\nstop after token\n\nAnswer:\n"))

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func cancelDuringPrefillKeepsPromptSnapshotUntilClear() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 20
        let model = readyModel(client: client)
        model.promptText = "prefill prompt"
        model.run()

        let prefillStarted = await waitUntil { model.livePrefillDone > 0 }
        #expect(prefillStarted, "prefill did not start within the wait budget")
        #expect(model.outputPromptText == "prefill prompt")
        model.cancel()
        let reachedIdle = await waitUntil { !model.isRunning }
        #expect(reachedIdle, "cancellation did not reach idle within the wait budget")

        #expect(!model.isRunning)
        #expect(model.outputPromptText == "prefill prompt")
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\nprefill prompt")
        #expect(model.hasOutputTranscript)

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.run()
        let reachedIdle = await waitUntil { !model.isRunning }
        #expect(reachedIdle, "run did not reach idle within the wait budget")

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        // A mock installer, so the readiness label does not depend on the
        // host's free disk space (the real probe reports "Not enough
        // storage" on a machine that is nearly full).
        let installer = MockModelInstallerClient(requirement: AppModelInstallRequirement(
            probePath: "/volume", requiredBytes: 1, availableBytes: 2))
        let model = AppModel(client: MockInferenceClient(), installer: installer)
        let oldURL = FileManager.default.temporaryDirectory.appendingPathComponent("old.gturbo")
        let newURL = FileManager.default.temporaryDirectory.appendingPathComponent("new.gturbo")
        model.modelPathText = oldURL.path
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)
        model.diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 1,
            tokensPerSecond: 1,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        model.error = .unknown("old error")

        model.setModelURL(newURL)

        #expect(model.modelPathText == newURL.standardizedFileURL.path)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
        #expect(model.presentation.label == "Model required")
        #expect(!model.canRun)
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    /// Polls `condition` for up to the wait budget (2 s), sleeping 5 ms
    /// between checks. Returns whether the condition became true. Tests must
    /// assert on the return value so a hung run fails loudly instead of
    /// silently asserting on stale state after the poll expires.
    @MainActor
    private func waitUntil(timeoutNanos: UInt64 = 2_000_000_000,
                           _ condition: () -> Bool) async -> Bool {
        let deadline = DispatchTime.now().uptimeNanoseconds + timeoutNanos
        while !condition() && DispatchTime.now().uptimeNanoseconds < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        return condition()
    }
}

@Suite("Moved model recovery")
struct MovedModelRecoveryTests {
    /// The state must be both *reachable* and *actionable*.
    ///
    /// Excluding `.needsReattestation` from `requiresModelInstallation` stops
    /// the app offering a re-download — but on its own that only hides the
    /// wrong remedy. If nothing else is offered the user is left with a moved
    /// model, no installer, and no way forward. These two assertions are the
    /// pair: the installer is suppressed AND the recovery is available.
    @MainActor
    @Test func aMovedModelSuppressesTheInstallerAndOffersRecovery() throws {
        let original = try makeCompleteModelInstall("recovery")
        defer { try? FileManager.default.removeItem(at: original) }
        let moved = original.deletingLastPathComponent()
            .appendingPathComponent("nvmai-recovery-\(UUID().uuidString).gturbo")
        try FileManager.default.moveItem(at: original, to: moved)
        defer { try? FileManager.default.removeItem(at: moved) }

        let installer = MockModelInstallerClient()
        let model = AppModel(modelDirectory: moved, installer: installer)

        #expect(model.needsReattestation)
        #expect(!model.requiresModelInstallation,
                "a moved model must not be offered the installer")
        #expect(model.canReattestModel,
                "suppressing the installer without offering recovery strands the user")
        #expect(!model.isModelInstalled)
    }

    @MainActor
    @Test func reattestingReprobesRatherThanAssumingSuccess() async throws {
        let original = try makeCompleteModelInstall("reprobe")
        defer { try? FileManager.default.removeItem(at: original) }
        let moved = original.deletingLastPathComponent()
            .appendingPathComponent("nvmai-reprobe-\(UUID().uuidString).gturbo")
        try FileManager.default.moveItem(at: original, to: moved)
        defer { try? FileManager.default.removeItem(at: moved) }

        let installer = MockModelInstallerClient()
        let model = AppModel(modelDirectory: moved, installer: installer)
        model.reattestModel()
        try await waitUntilTrue { installer.reattestCalled }

        // The mock does not write a receipt, so the re-probe must still report
        // the model as needing attestation rather than trusting the call.
        try await waitUntilTrue { model.installTaskIsIdleForTesting }
        #expect(model.needsReattestation,
                "re-attestation must be confirmed by re-probing, not assumed")
    }

    private func waitUntilTrue(
        _ predicate: @MainActor () -> Bool,
        timeout: Duration = .seconds(5)
    ) async throws {
        let deadline = ContinuousClock.now + timeout
        while ContinuousClock.now < deadline {
            if await MainActor.run(body: predicate) { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("condition not met within \(timeout)")
    }
}
