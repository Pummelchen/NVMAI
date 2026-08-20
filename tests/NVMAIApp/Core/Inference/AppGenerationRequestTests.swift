import Foundation
import Testing
@testable import NVMAIAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "hello")
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 0.6)
        #expect(request.topK == 20)
        #expect(request.topP == 0.95)
        #expect(request.presencePenalty == 0)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           temperature: 0)
        #expect(request.topK == 20)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "   ")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func contextAcceptsQwenMaximumAndRejectsLargerValues() throws {
        let maximum = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "hello",
            maxContextTokens: 262_144)
        try maximum.validate()

        let tooLarge = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "hello",
            maxContextTokens: 262_145)
        #expect(throws: AppInferenceError.self) {
            try tooLarge.validate()
        }
    }

    @Test func yarnAcceptsOnlyTheTwoExtendedContextChoices() throws {
        var options = AppRuntimeOptions()
        options.ropeScalingMode = .yarn
        for context in [524_288, 1_048_576] {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "hello",
                maxContextTokens: context,
                runtimeOptions: options).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "hello",
                maxContextTokens: 262_144,
                runtimeOptions: options).validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func nonzeroPresencePenaltyRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", presencePenalty: 0.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               prompt: "hello", topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            prompt: "hello")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }
}
