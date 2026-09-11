import Testing
import Foundation
@testable import NVMAIAppCore

/// The sweep that removes decode-service jobs a force-quit left behind. The
/// launch/kill half needs a real launchd, but the decision half -- which labels
/// are ours and which of those are orphaned -- is pure and is what these pin:
/// getting it wrong either leaves a 20 GB helper resident or kills a running
/// app's helper.
@Suite struct DecodeServiceJobSweepTests {
    private let uid: uid_t = 501

    private func list(_ rows: String...) -> String {
        (["PID\tStatus\tLabel"] + rows).joined(separator: "\n")
    }

    @Test func parsesOurJobsAndIgnoresEverythingElse() {
        let output = list(
            "-\t0\tcom.apple.Finder",
            "1234\t0\tcom.nvmai.decode.501.98765.a1b2c3d4",
            "-\t0\tcom.nvmai.decode.501.98766.deadbeef",
            "1234\t0\tcom.nvmai.decode.999.11111.cafebabe",
            "1234\t0\tcom.nvmai.decode.501.notapid.a1b2c3d4",
            "1234\t0\tcom.nvmai.decode.501.11111.",
            "not a launchctl row at all")
        let jobs = DecodeServiceJobSweep.parseLaunchctlList(output, uid: uid)
        #expect(jobs.map(\.label) == [
            "com.nvmai.decode.501.98765.a1b2c3d4",
            "com.nvmai.decode.501.98766.deadbeef",
        ])
        #expect(jobs.map(\.owner) == [98765, 98766])
        #expect(jobs.map(\.socketName) == [
            "98765.a1b2c3d4.sock",
            "98766.deadbeef.sock",
        ])
    }

    /// The socket name has to match what `launchIndependentService` creates
    /// (`<getpid()>.<token>.sock`), or the orphan's socket file is left behind
    /// and the next connect attempt can find a stale path.
    @Test func socketNameMatchesTheLaunchPath() {
        let label = "com.nvmai.decode.501.4321.feedface"
        let jobs = DecodeServiceJobSweep.parseLaunchctlList(list("-\t0\t\(label)"),
                                                           uid: uid)
        #expect(jobs.first?.socketName == "4321.feedface.sock")
    }

    @Test func onlyJobsWithADeadOwnerAreOrphans() {
        let output = list(
            "1234\t0\tcom.nvmai.decode.501.98765.a1b2c3d4",
            "-\t0\tcom.nvmai.decode.501.98766.deadbeef")
        let jobs = DecodeServiceJobSweep.parseLaunchctlList(output, uid: uid)
        let orphans = DecodeServiceJobSweep.orphans(in: jobs) { $0 == 98765 }
        #expect(orphans.map(\.label) == ["com.nvmai.decode.501.98766.deadbeef"])
    }

    /// A pid the kernel has since reused must read as alive: erring towards
    /// leaving a helper in place is recoverable, killing a stranger's is not.
    @Test func aliveOwnerIsLeftAloneEvenWhenTheHelperIsNotRunning() {
        let output = list("-\t0\tcom.nvmai.decode.501.22222.0badc0de")
        let jobs = DecodeServiceJobSweep.parseLaunchctlList(output, uid: uid)
        #expect(DecodeServiceJobSweep.orphans(in: jobs) { _ in true }.isEmpty)
    }

    @Test func processIsAliveSeesThisProcessAndNotAnUnusedPid() {
        #expect(DecodeServiceJobSweep.processIsAlive(getpid()))
        // 0 and negatives are not pids to probe; asking is a process-group
        // signal, which would report "alive" for the caller's own group.
        #expect(!DecodeServiceJobSweep.processIsAlive(0))
        #expect(!DecodeServiceJobSweep.processIsAlive(-1))
        // A pid above the maximum is never in use.
        #expect(!DecodeServiceJobSweep.processIsAlive(pid_t(Int32.max)))
    }
}
