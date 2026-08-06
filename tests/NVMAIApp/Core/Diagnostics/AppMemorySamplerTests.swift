import Testing
@testable import NVMAIAppCore

@Suite struct AppMemorySamplerTests {
    @Test func samplingDoesNotThrowAndTracksPeak() {
        let sampler = AppMemorySampler()
        sampler.resetPeak()
        // Sampling the live process footprint must succeed; a nil result is
        // itself a failure (the footprint task failing), not something to
        // paper over.
        #expect(sampler.sample() != nil, "live process footprint sample must succeed")
        #expect(sampler.sample() != nil)
        guard let peak = sampler.peakBytes else {
            Issue.record("a successful sample must record a peak")
            return
        }
        #expect(peak > 0)
    }

    @Test func samplingReportsProcessFootprint() {
        let footprint: UInt64 = 2 * 1_024 * 1_024 * 1_024
        let sampler = AppMemorySampler(processFootprint: { footprint })

        #expect(sampler.sample() == footprint)
        #expect(sampler.peakBytes == footprint)
    }

    @Test func failedSampleDoesNotSetPeak() {
        let sampler = AppMemorySampler(processFootprint: { nil })

        #expect(sampler.sample() == nil)
        #expect(sampler.peakBytes == nil)
    }
}
