import Testing
@testable import NVMAIAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        #expect(mebibytes == [80, 160, 320, 640, 1_280, 2_560, 5_120])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, Default",
            "8K, +80 MB",
            "16K, +240 MB",
            "32K, +560 MB",
            "64K, +1.17 GB",
            "128K, +2.42 GB",
            "256K, +4.92 GB",
        ])
    }
}
