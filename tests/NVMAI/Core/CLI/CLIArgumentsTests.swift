import Testing
@testable import NVMAICLICore

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.gturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 0.6)
        #expect(arguments.topK == 20)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
        #expect(arguments.prefillChunk == nil)
        #expect(arguments.kvCachePrecision == .int8)
        #expect(arguments.ropeScalingMode == .none)
        #expect(arguments.thinkingMode == .off)
    }

    @Test func kvPrecisionAndYaRNOptionsParse() throws {
        let yarn = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--kv-bits", "4", "--rope-scaling", "yarn",
        ])
        #expect(yarn.kvCachePrecision == .int4)
        #expect(yarn.ropeScalingMode == .yarn)
        #expect(yarn.maxContext == 1_048_576)
        let halfMillion = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--rope-scaling", "yarn", "--max-context", "524288",
        ])
        #expect(halfMillion.maxContext == 524_288)
        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--rope-scaling", "yarn", "--max-context", "262144",
            ])
        }
    }

    @Test func prefillChunkParsesFixedAndAutoValues() throws {
        let fixed = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk", "4096",
        ])
        #expect(fixed.prefillChunk == .fixed(4_096))

        let automatic = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk", "auto",
        ])
        #expect(automatic.prefillChunk == .auto)

        #expect(throws: ArgsError.invalidValue(flag: "--prefill-chunk", value: "8192")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--prefill-chunk", "8192",
            ])
        }
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func contextArgumentAcceptsQwenMaximumAndRejectsLargerValues() throws {
        let maximum = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--max-context", "262144",
        ])
        #expect(maximum.maxContext == 262_144)
        #expect(throws: ArgsError.invalidValue(flag: "--max-context", value: "262145")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--max-context", "262145",
            ])
        }
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func conciseFlagParsesAndDefaultsOff() throws {
        let on = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--concise",
        ])
        #expect(on.concise)
        let off = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(!off.concise)
    }

    @Test func thinkingModeParsesOnlyTheOfficialBinaryValues() throws {
        let on = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi", "--thinking", "on",
        ])
        #expect(on.thinkingMode == .on)
        #expect(throws: ArgsError.invalidValue(flag: "--thinking", value: "medium")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi", "--thinking", "medium",
            ])
        }
    }

    @Test func reasoningEffortParsesOnlyTheTemplateLevels() throws {
        let parsed = try Args.parse([
            "--model", "m.gturbo", "--prompt", "hi",
            "--thinking", "on", "--reasoning-effort", "low",
        ])
        #expect(parsed.reasoningEffort == .low)
        let unset = try Args.parse(["--model", "m.gturbo", "--prompt", "hi"])
        #expect(unset.reasoningEffort == nil)
        // The template defines low, medium, and xhigh; "high" does not exist.
        #expect(throws: ArgsError.invalidValue(flag: "--reasoning-effort",
                                               value: "high")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--thinking", "on", "--reasoning-effort", "high",
            ])
        }
    }

    @Test func reasoningEffortRequiresThinkingOn() {
        #expect(throws: ArgsError.invalidValue(
            flag: "--reasoning-effort", value: "low requires --thinking on")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--reasoning-effort", "low",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--help",
            "--rdadvise", "--expert-cache-slots", "--prefill-chunk", "--concise",
            "--kv-bits", "--rope-scaling", "--thinking", "--reasoning-effort",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.gturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.gturbo"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.gturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.gturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
}
