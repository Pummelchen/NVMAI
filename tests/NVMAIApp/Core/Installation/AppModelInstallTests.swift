import Foundation
import Testing
import NVMAIRepackCore

@testable import NVMAIAppCore

@Suite struct AppModelInstallTests {

  @MainActor
  @Test func missingModelCanInstall() {
    let installer = MockModelInstallerClient()
    let directory = temporaryInstallPath("missing")
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(!model.isModelInstalled)
    #expect(model.requiresModelInstallation)
    #expect(model.canInstallModel)
  }

  @MainActor
  @Test func installedModelShowsLoadNotInstall() throws {
    let directory = try makeCompleteModelInstall("installed")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(!model.canInstallModel)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainDetectsModelInstalledAfterLaunch() throws {
    let directory = try makeCompleteModelInstall("external-install")
    let stagedDirectory = directory.deletingLastPathComponent()
      .appendingPathComponent("staged-\(UUID().uuidString).gturbo")
    try FileManager.default.moveItem(at: directory, to: stagedDirectory)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    #expect(model.requiresModelInstallation)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.moveItem(at: stagedDirectory, to: directory)

    model.refreshInstallReadiness()

    #expect(model.isModelInstalled)
    #expect(!model.requiresModelInstallation)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func checkAgainUsesCurrentModelLocation() throws {
    let initialDirectory = temporaryInstallPath("initial-location")
    let currentDirectory = try makeCompleteModelInstall("current-location")
    defer { try? FileManager.default.removeItem(at: currentDirectory) }
    let model = AppModel(
      modelDirectory: initialDirectory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient())

    model.modelPathText = currentDirectory.path
    model.recheckModelAtCurrentLocation()

    #expect(model.modelPathText == currentDirectory.standardizedFileURL.path)
    #expect(model.isModelInstalled)
    #expect(model.canLoadModel)
  }

  @MainActor
  @Test func qwen36DescriptorMatchesPinnedAudit() {
    let descriptor = AppModelInstallDescriptor.qwen36
    #expect(descriptor.displayName == "Qwen3.6 35B-A3B 4-bit")
    #expect(descriptor.repoID == "mlx-community/Qwen3.6-35B-A3B-4bit")
    #expect(descriptor.revision == "38740b847e4cb78f352aba30aa41c76e08e6eb46")
    #expect(descriptor.sourceIndexSHA256 == "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea")
    #expect(descriptor.approximateDownloadBytes == 19_529_025_048)
    #expect(descriptor.installedBytes == 19_546_491_213)
    #expect(descriptor.requiredFreeBytes
            == descriptor.installedBytes + descriptor.rangeStagingBytes + descriptor.reserveBytes)
  }

  @Test func qwenQuantDescriptorsArePinnedAndDistinct() {
    #expect(AppModelInstallDescriptor.qwen36.installDirectoryName == "qwen3.6_35B_A3B_4Bit")
    #expect(AppModelInstallDescriptor.qwen36_6bit.installDirectoryName == "qwen3.6_35B_A3B_6Bit")
    #expect(AppModelInstallDescriptor.qwen36_8bit.installDirectoryName == "qwen3.6_35B_A3B_8Bit")
    #expect(Set(AppModelInstallDescriptor.all.map(\.sourceIndexSHA256)).count
            == AppModelInstallDescriptor.all.count)
    #expect(AppModelInstallDescriptor.qwen36_6bit.repoID.hasSuffix("-6bit"))
    #expect(AppModelInstallDescriptor.qwen36_8bit.repoID.hasSuffix("-8bit"))
  }

  @Test func ornithTextDescriptorsArePinnedAndSelectable() {
    let fourBit = AppModelInstallDescriptor.ornith15
    let eightBit = AppModelInstallDescriptor.ornith15_8bit

    #expect(fourBit.installDirectoryName == "ornith-1.5_35B_A3B_4Bit")
    #expect(eightBit.installDirectoryName == "ornith-1.5_35B_A3B_8Bit")
    #expect(fourBit.repoID == "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit")
    #expect(eightBit.repoID == "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit")
    #expect(fourBit.sourceIndexSHA256
            == "c118f13c0dcb729e4ca2e3d653ab193067551eb1a6410badb5192eb426104f36")
    #expect(eightBit.sourceIndexSHA256
            == "83c641a791aa957df7d280eef1b0c8faf7a2ec9b19dd3355fb13abae8ae0ed15")
    #expect(fourBit.requiredFreeBytes > fourBit.installedBytes)
    #expect(eightBit.requiredFreeBytes > eightBit.installedBytes)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "ornith15") == fourBit)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "ornith15-8bit") == eightBit)
    #expect(AppModelInstallDescriptor.selectedDescriptor(for: "qwen36-6bit") == .qwen36)
  }

  @MainActor
  @Test func insufficientSpaceDisablesInstallAndExposesShortfall() {
    let requirement = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 100,
      availableBytes: 40)
    let installer = MockModelInstallerClient(requirement: requirement)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("space"),
      client: MockLifecycleInferenceClient(),
      installer: installer)

    #expect(model.installReadiness == .insufficientSpace(requirement))
    #expect(model.installRequirement?.shortfallBytes == 60)
    #expect(!model.canInstallModel)
  }

  @MainActor
  @Test func installProgressUpdatesStatusAndByteCounts() async throws {
    let directory = temporaryInstallPath("progress")
    let installer = MockModelInstallerClient(
      events: [
        .checking,
        .copyingPayload(
          reusedBytes: 1,
          downloadedThisRunBytes: 3,
          totalBytes: 10),
      ], holdOpen: true)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      model.installState == .copyingPayload(
        reusedBytes: 1,
        downloadedThisRunBytes: 3,
        totalBytes: 10)
    }
    #expect(model.installDownloadedBytes == 4)
    #expect(model.installTotalBytes == 10)
    #expect(model.installProgressFraction == 0.4)
    #expect(model.presentation.label == "Downloading model")
    model.cancelInstall()
    try await waitUntil { model.installState == .cancelled }
  }

  @MainActor
  @Test func installCompletionStopsUnloaded() async throws {
    let requestedDirectory = temporaryInstallPath("requested")
    let completedDirectory = try makeCompleteModelInstall("complete")
    defer { try? FileManager.default.removeItem(at: completedDirectory) }
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(events: [.installed(completedDirectory)])
    let model = AppModel(
      modelDirectory: requestedDirectory,
      client: client,
      installer: installer)

    model.installModel()
    try await waitUntil {
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL)
    }
    #expect(
      model.installState == .installed(modelDirectory: completedDirectory.standardizedFileURL))
    #expect(model.modelPathText == completedDirectory.standardizedFileURL.path)
    #expect(model.loadState == .notLoaded)
    #expect(model.canLoadModel)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func installFailureDoesNotAttemptLoad() async throws {
    struct SyntheticError: Error {}
    let client = MockLifecycleInferenceClient()
    let installer = MockModelInstallerClient(failure: SyntheticError())
    let model = AppModel(
      modelDirectory: temporaryInstallPath("failure"),
      client: client,
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }
    #expect(model.loadState == .notLoaded)
    #expect(client.ensureLoadedCallCount() == 0)
  }

  @MainActor
  @Test func networkFailureWithSavedProgressLeavesResumeEnabled() async throws {
    struct NetworkFailure: Error {}
    let directory = temporaryInstallPath("network-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: NetworkFailure()))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func diskFailureCanResumeAfterSpaceRecheck() async throws {
    let directory = temporaryInstallPath("disk-resume")
    let paths = try makeSavedDownload(at: directory)
    defer { cleanUpSavedDownload(paths) }
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let model = AppModel(
      modelDirectory: directory,
      client: MockLifecycleInferenceClient(),
      installer: MockModelInstallerClient(failure: error))

    model.installModel()
    try await waitUntil {
      if case .recoverable = model.installState { return true }
      return false
    }
    #expect(!model.canInstallModel)

    model.recheckModelAtCurrentLocation()

    #expect(model.canInstallModel)
    #expect(model.canDiscardModelDownload)
  }

  @MainActor
  @Test func invalidSavedDownloadsRemainDiscardOnly() throws {
    let descriptor = AppModelInstallDescriptor.qwen36
    for incompatible in [false, true] {
      let directory = temporaryInstallPath(
        incompatible ? "incompatible-checkpoint" : "corrupt-checkpoint")
      let paths = try makeSavedDownload(at: directory)
      defer { cleanUpSavedDownload(paths) }
      if incompatible {
        try RemoteInstallCheckpoint(
          repoID: "other/model",
          requestedRevision: descriptor.revision,
          resolvedCommit: String(repeating: "a", count: 40),
          sourceIndexSHA256: String(repeating: "b", count: 64),
          planFingerprint: String(repeating: "c", count: 64),
          totalSourceBytes: 1
        ).write(
          to: paths.checkpointFile,
          parentDirectory: paths.parentDirectory)
      } else {
        try Data("{}".utf8).write(
          to: URL(fileURLWithPath: paths.checkpointFile))
      }
      let model = AppModel(
        modelDirectory: directory,
        client: MockLifecycleInferenceClient(),
        installer: RepackModelInstallerClient(descriptor: descriptor))

      #expect(!model.canInstallModel)
      #expect(model.canDiscardModelDownload)
      guard case .failed = model.installReadiness else {
        Issue.record("invalid checkpoint did not fail readiness")
        continue
      }
    }
  }

  @MainActor
  @Test func diskFailureKeepsExactRequirementAndShortfall() async throws {
    let error = RepackError.diskSpaceInsufficient(
      path: "/volume",
      required: 120,
      available: 45)
    let installer = MockModelInstallerClient(failure: error)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("disk-failure"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()

    try await waitUntil {
      if case .failed = model.installState { return true }
      return false
    }

    let expected = AppModelInstallRequirement(
      probePath: "/volume",
      requiredBytes: 120,
      availableBytes: 45)
    #expect(model.installReadiness == .insufficientSpace(expected))
    #expect(model.installRequirement?.shortfallBytes == 75)
  }

  @MainActor
  @Test func cancelInstallWaitsForAcknowledgementAndAllowsRetry() async throws {
    let installer = MockModelInstallerClient(
      events: [.downloadingMetadata],
      holdOpen: true,
      delayCancellationAcknowledgement: true)
    let model = AppModel(
      modelDirectory: temporaryInstallPath("cancel"),
      client: MockLifecycleInferenceClient(),
      installer: installer)
    model.installModel()
    try await waitUntil { model.installState == .downloadingMetadata }

    model.cancelInstall()
    #expect(installer.cancelCalled)
    try await waitUntil { installer.cancellationAcknowledgementPending }
    #expect(model.installState == .cancelling)
    #expect(!model.canInstallModel)

    await installer.releaseCancellationAcknowledgement()
    try await waitUntil { model.installState == .cancelled }

    #expect(model.loadState == .notLoaded)
    #expect(model.canInstallModel)
  }

  private func temporaryInstallPath(_ tag: String) -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("nvmai-app-install-\(tag)-\(UUID().uuidString).gturbo")
  }

  private func makeSavedDownload(at directory: URL) throws -> RemoteInstallPaths {
    let paths = try RemoteInstallPaths(outputDirectory: directory.path)
    try FileManager.default.createDirectory(
      atPath: paths.partialDirectory,
      withIntermediateDirectories: true)
    return paths
  }

  private func cleanUpSavedDownload(_ paths: RemoteInstallPaths) {
    for path in [
      paths.finalDirectory,
      paths.partialDirectory,
      paths.checkpointFile,
      paths.lockFile,
    ] {
      try? FileManager.default.removeItem(atPath: path)
    }
  }

  @MainActor
  private func waitUntil(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<200 {
      if predicate() { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for condition")
  }

}
