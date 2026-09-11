//
//  CPUCommands.swift
//  NVMAIBench
//
//  The CPU side-engine commands: the whole-model continuation check, text
//  generation, batch generation, the int8 GEMV and sustained-load
//  measurements, and the model/tokenizer loading they share. Split out of
//  `main.swift`; `NVMAIBench.main` dispatches into `runCPUCommand` here
//  before it creates a Metal context, so these commands never touch the GPU.
//

import Foundation
import Metal
import NVMAI

extension NVMAIBench {
    /// What the CPU side-engine can actually read, and at how many threads.
    ///
    /// The premise of the side-engine is that NVMAI leaves the CPU idle --
    /// measured, 0.20 of one core out of eight while a 35B generates. That
    /// is true of the *cores* and says nothing about the *memory*, which is
    /// the resource decode is bound by on both sides. The GPU already reads
    /// at 74-88 GB/s during a 35B decode, near this machine's practical
    /// ceiling, and a 2B model reads about 1.9 GB per token at 8-bit. So the
    /// question this answers is not "is there a spare core" but "is there
    /// spare bandwidth, and what does taking it cost the model the person is
    /// waiting for".
    ///
    /// Run it twice -- idle, and with a generation in flight -- and the
    /// difference is the answer.
    ///
    ///     NVMAIBench cpu            # 8-bit, thread sweep
    static func runCPUGEMV(iterations: Int) {
        // Big enough that nothing is served from cache: the point is the
        // memory system, and a matrix that fits in the SLC measures the SLC.
        let rows = 8192
        let n = 8192
        let group = 64
        let weightBytes = rows * n
        let groups = rows * (n / group)
        print("cpu int8 affine gemv: \(rows)x\(n), "
              + "\(Double(weightBytes) / 1e6) MB of weights per pass, "
              + "\(iterations) passes")
        print("performance cores reported: \(Int8AffineGEMV.preferredThreads)")

        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: weightBytes)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let x = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let out = UnsafeMutablePointer<Float>.allocate(capacity: rows)
        defer {
            weights.deallocate(); scales.deallocate(); biases.deallocate()
            x.deallocate(); out.deallocate()
        }
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        func next() -> UInt64 {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return state
        }
        for i in 0..<weightBytes { weights[i] = UInt8(truncatingIfNeeded: next()) }
        // 1.0 and 0.0 as BF16 bit patterns: the arithmetic is the same
        // whatever the constants, and the measurement is of the reads.
        for i in 0..<groups { scales[i] = 0x3F80; biases[i] = 0 }
        for i in 0..<n { x[i] = Float(i % 7) * 0.125 }

        let perPass = Double(weightBytes + groups * 4)
        print("  \("threads".padding(toLength: 8, withPad: " ", startingAt: 0))"
              + "\("ms/pass".padding(toLength: 10, withPad: " ", startingAt: 0))"
              + "\("GB/s".padding(toLength: 9, withPad: " ", startingAt: 0))"
              + "2B tok/s at 8-bit")
        for threads in [1, 2, 4, 6, 8] {
            // One untimed pass so the first one's page faults are not the
            // measurement.
            Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                    x: x, rows: rows, n: n, out: out, threads: threads)
            let started = ContinuousClock.now
            for _ in 0..<iterations {
                Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                        x: x, rows: rows, n: n, out: out, threads: threads)
            }
            let elapsed = started.duration(to: .now)
            let seconds = Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
            let perIteration = seconds / Double(iterations)
            let bandwidth = perPass / perIteration / 1e9
            // A 2B at 8-bit reads about 1.9 GB per token, the tied output
            // head included -- it is read in full for every token.
            let tokens = bandwidth / 1.9
            print(String(format: "  %-8d%-10.2f%-9.1f%.1f",
                         threads, perIteration * 1e3, bandwidth, tokens))
        }
        print("  (checksum \(out[0]))")
    }


    /// Hold the memory system at the side-engine's working width for a while.
    ///
    /// The companion to `runCPUGEMV`: that one asks what the CPU can read,
    /// this one exists so the same question can be asked of the GPU while
    /// the CPU is reading. A side-engine that halves the throughput of the
    /// model the person is waiting for is not a side-engine.
    static func runCPULoad(seconds: Double, threads: Int) {
        let rows = 8192, n = 8192, group = 64
        let weightBytes = rows * n, groups = rows * (n / group)
        let weights = UnsafeMutablePointer<UInt8>.allocate(capacity: weightBytes)
        let scales = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let biases = UnsafeMutablePointer<UInt16>.allocate(capacity: groups)
        let x = UnsafeMutablePointer<Float>.allocate(capacity: n)
        let out = UnsafeMutablePointer<Float>.allocate(capacity: rows)
        defer {
            weights.deallocate(); scales.deallocate(); biases.deallocate()
            x.deallocate(); out.deallocate()
        }
        var state: UInt64 = 0x2545_F491_4F6C_DD1D
        for i in 0..<weightBytes {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            weights[i] = UInt8(truncatingIfNeeded: state)
        }
        for i in 0..<groups { scales[i] = 0x3F80; biases[i] = 0 }
        for i in 0..<n { x[i] = Float(i % 7) * 0.125 }

        print("cpu load: \(threads) threads for \(seconds)s")
        let deadline = ContinuousClock.now.advanced(by: .seconds(seconds))
        var passes = 0
        let started = ContinuousClock.now
        while ContinuousClock.now < deadline {
            Int8AffineGEMV.threaded(weights: weights, scales: scales, biases: biases,
                                    x: x, rows: rows, n: n, out: out, threads: threads)
            passes += 1
        }
        let elapsed = started.duration(to: .now)
        let taken = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        let bandwidth = Double(passes) * Double(weightBytes + groups * 4) / taken / 1e9
        print(String(format: "  %d passes in %.1fs, %.1f GB/s (checksum %.0f)",
                     passes, taken, bandwidth, out[0]))
    }


/// What the CPU commands could not load, and why both shapes were refused.
enum DenseModelError: Error, CustomStringConvertible {
    case notAModel(String)
    case unreadableVocabulary(String)

    var description: String {
        switch self {
        case .notAModel(let path):
            return "\(path) is neither a .gturbo install (no manifest.json) nor a "
                + "safetensors snapshot (no config.json)"
        case .unreadableVocabulary(let path):
            return "\(path) carries neither a vocab.json mapping nor a tokenizer, "
                + "so its continuations cannot be checked"
        }
    }
}

    /// Loads a dense CPU model in either shape it ships in.
    ///
    /// These commands took an affine safetensors snapshot only. That stopped
    /// reaching a shipped model once the dense Qwen 3.5 family became `.gturbo`
    /// installs: all six carry a `manifest.json` and no `config.json`, so the
    /// whole-model check could only run against a conversion intermediate -- and
    /// the 4B/9B intermediates were deleted to reclaim disk, leaving the 2B
    /// pair. Choosing by what is on disk makes the check usable on the models a
    /// user actually has, which is what `tools/verify_cpu_models.sh` drives.
    static func loadDenseSnapshot(_ path: String) throws -> AffineSnapshot {
        let directory = URL(fileURLWithPath: path)
        if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("manifest.json").path) {
            return try AffineSnapshot(gturbo: directory)
        }
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("config.json").path) else {
            throw DenseModelError.notAModel(path)
        }
        return try AffineSnapshot(directory: directory)
    }

    /// The string-to-id and id-to-string mapping the `cpu35` checks read.
    ///
    /// A safetensors snapshot ships `vocab.json`, the same table the numpy
    /// reference reads. A shipped `.gturbo` install ships a tokenizer
    /// directory instead and no `vocab.json`, so the engine's own tokenizer
    /// supplies the mapping there. Either way the ids are the model's own
    /// rather than a bench-local table that could drift from the reference.
    ///
    /// `labelFor` spells a check's expected word the way `textFor` spells a
    /// decoded id, so the two columns of the report read alike.
    static func denseVocabulary(
        _ path: String, directory: URL
    ) throws -> (idFor: (String) -> Int?,
                 textFor: (Int) -> String,
                 labelFor: (String) -> String) {
        let vocabularyURL = directory.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabularyURL.path) {
            let object = try JSONSerialization.jsonObject(with: Data(contentsOf: vocabularyURL))
            guard let mapping = object as? [String: Int] else {
                throw DenseModelError.unreadableVocabulary(vocabularyURL.path)
            }
            var inverse: [Int: String] = [:]
            inverse.reserveCapacity(mapping.count)
            for (text, id) in mapping { inverse[id] = text }
            return ({ (word: String) -> Int? in mapping[word] },
                    { (id: Int) -> String in inverse[id] ?? "?" },
                    { (word: String) -> String in word })
        }
        guard let tokenizer = try loadTokenizer(directory) else {
            throw DenseModelError.unreadableVocabulary(path)
        }
        return ({ (word: String) -> Int? in
            // The checks spell a leading space as U+0120, the byte-level
            // encoding of a space. A word has to be exactly one token to be
            // comparable with the reference's expectation.
            let text = word.replacingOccurrences(of: "\u{120}", with: " ")
            let ids = tokenizer.encode(text, addBOS: false)
            return ids.count == 1 ? Int(ids[0]) : nil
        }, { (id: Int) -> String in tokenizer.decode([Int32(id)]) },
           { (word: String) -> String in
               word.replacingOccurrences(of: "\u{120}", with: " ")
           })
    }

    /// Qwen3.5-2B on the CPU, checked against the continuations that define
    /// correctness for the numpy reference.
    ///
    /// The token ids come out of the model itself -- its `vocab.json` in a
    /// snapshot, its tokenizer in a `.gturbo` install -- so this cannot drift
    /// from what the reference does.
    static func runCPUQwen35(snapshot path: String, dump: URL? = nil) throws {
        let directory = URL(fileURLWithPath: path)
        let started = ContinuousClock.now
        let snapshot = try Self.loadDenseSnapshot(path)
        // Width is the side-engine's scheduling knob, so it is settable
        // here: the measurement that produced the policy is a sweep of it.
        let requested = ProcessInfo.processInfo.environment["NVMAI_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        func seconds(_ from: ContinuousClock.Instant) -> Double {
            let elapsed = from.duration(to: .now)
            return Double(elapsed.components.seconds)
                + Double(elapsed.components.attoseconds) / 1e18
        }
        print("\(path): \(snapshot.configuration.layers) layers, "
              + "hidden \(snapshot.configuration.hiddenSize), "
              + "rotary \(snapshot.configuration.rotaryDim)/"
              + "\(snapshot.configuration.headDim), "
              + "loaded in \(String(format: "%.2fs", seconds(started)))")
        print("threads: \(model.threads)")

        let vocabulary = try Self.denseVocabulary(path, directory: directory)

        let checks: [([String], String)] = [
            (["Once", "\u{120}upon", "\u{120}a"], "\u{120}time"),
            (["The", "\u{120}capital", "\u{120}of", "\u{120}France", "\u{120}is"],
             "\u{120}Paris"),
            (["The", "\u{120}quick", "\u{120}brown", "\u{120}fox", "\u{120}jumps",
              "\u{120}over", "\u{120}the", "\u{120}lazy"], "\u{120}dog"),
        ]
        var failures = 0
        for (index, (words, expected)) in checks.enumerated() {
            model.reset()
            var logits: [Float] = []
            let run = ContinuousClock.now
            for word in words {
                guard let id = vocabulary.idFor(word) else {
                    print("  no token for \(word)"); failures += 1; break
                }
                logits = try model.step(token: id)
            }
            guard !logits.isEmpty else { continue }
            var best = 0
            for index in logits.indices where logits[index] > logits[best] { best = index }
            let want = vocabulary.idFor(expected) ?? -1
            let ok = best == want
            failures += ok ? 0 : 1
            let prompt = words.map { $0.replacingOccurrences(of: "\u{120}", with: " ") }
                .joined()
            let rate = Double(words.count) / seconds(run)
            print(String(format: "  %@ %-46@ -> %@ (%.2f), wanted %@  [%.1f tok/s]",
                         ok ? "ok " : "FAIL", prompt as NSString,
                         vocabulary.textFor(best), logits[best],
                         vocabulary.labelFor(expected), rate))
            if let dump {
                try? FileManager.default.createDirectory(
                    at: dump, withIntermediateDirectories: true)
                let file = dump.appendingPathComponent("check\(index).f32")
                let payload = logits.withUnsafeBufferPointer { Data(buffer: $0) }
                try? payload.write(to: file)
            }
        }
        print(failures == 0 ? "all continuations correct"
              : "\(failures) of \(checks.count) wrong")
        // Exit non-zero on a mismatch. Without this the process returned 0
        // after printing "N of 3 wrong", so a scripted run -- and this command
        // exists to be scripted -- read a dead forward pass as a pass. The
        // whole point of `cpu35` is to be the check that says the Swift forward
        // pass matches the oracle.
        exit(failures == 0 ? 0 : 1)
    }


    /// The CPU side-engine's commands. Returns whether one ran, so `main`
    /// can dispatch them before it creates a Metal context.
    static func runCPUCommand(_ name: String, iterations: Int) throws -> Bool {
        switch name {
        case "cpuload":
            // Sustained load at one width, for measuring what the side-engine
            // costs the model the person is waiting for. `iterations` is
            // seconds here; the third argument is the thread count.
            let threads = CommandLine.arguments.count > 3
                ? Int(CommandLine.arguments[3]) ?? 4 : 4
            runCPULoad(seconds: Double(iterations), threads: threads)
        case "cpu35":
            // The side-engine's model, on the same continuations the numpy
            // reference checks itself with. Agreement here is what says the
            // Swift forward pass matches the oracle.
            let snapshot = CommandLine.arguments.count > 2
                ? CommandLine.arguments[2] : ".build/qwen35-2b-affine-8bit"
            // An optional directory to write each check's full logit vector
            // into, so parity is a number rather than an impression.
            let dump = CommandLine.arguments.count > 3
                ? URL(fileURLWithPath: CommandLine.arguments[3]) : nil
            try runCPUQwen35(snapshot: snapshot, dump: dump)
        case "cpu35gen":
            // End to end: real text in, real text out, through the engine's
            // own tokenizer. `NVMAIBench cpu35gen <snapshot> "<prompt>" [n]`
            let snapshot = CommandLine.arguments.count > 2
                ? CommandLine.arguments[2] : ".build/qwen35-2b-affine-8bit"
            let prompt = CommandLine.arguments.count > 3
                ? CommandLine.arguments[3] : "The capital of France is"
            let limit = CommandLine.arguments.count > 4
                ? Int(CommandLine.arguments[4]) ?? 32 : 32
            try runCPUQwen35Generation(snapshot: snapshot, prompt: prompt, limit: limit)
        case "cpu35batch":
            // A file of prompts in, a file of completions out, so an
            // experiment can be written in Python and still run on the real
            // engine. One JSON object per line, `{"prompt": ..., "max": n}`.
            let snapshot = CommandLine.arguments.count > 2
                ? CommandLine.arguments[2] : ".build/qwen35-2b-affine-8bit"
            guard CommandLine.arguments.count > 4 else {
                print("usage: NVMAIBench cpu35batch <snapshot> <in.jsonl> <out.jsonl>")
                return true
            }
            try runCPUQwen35Batch(snapshot: snapshot,
                                  input: URL(fileURLWithPath: CommandLine.arguments[3]),
                                  output: URL(fileURLWithPath: CommandLine.arguments[4]))
        case let other where other.hasPrefix("cpu"):
            runCPUGEMV(iterations: iterations)
        default:
            return false
        }
        return true
    }


    /// The side-engine answering in text, which is what everything above was
    /// for. The tokenizer is the engine's own, loaded straight out of the
    /// snapshot the converter wrote.
    static func runCPUQwen35Generation(snapshot path: String,
                                       prompt: String,
                                       limit: Int) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try Self.loadDenseSnapshot(path)
        let requested = ProcessInfo.processInfo.environment["NVMAI_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        guard let tokenizer = try loadTokenizer(directory) else {
            // Non-zero: a run that never checked anything is not a pass.
            FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
            exit(2)
        }
        let ids = tokenizer.encode(prompt, addBOS: false).map(Int.init)
        print("prompt: \(prompt.debugDescription) -> \(ids.count) tokens, "
              + "threads \(model.threads)")
        let started = ContinuousClock.now
        let produced = try model.generate(prompt: ids, maximumTokens: limit,
                                          stopping: [Int(tokenizer.eosID)])
        let elapsed = started.duration(to: .now)
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        print("output: " + tokenizer.decode(produced.map(Int32.init)).debugDescription)
        print(String(format: "%d prompt + %d generated in %.1fs (%.1f tok/s)",
                     ids.count, produced.count, seconds,
                     Double(ids.count + produced.count) / seconds))
    }


    /// Run a file of prompts through the side-engine.
    ///
    /// The model loads once and the session resets between prompts, which is
    /// the shape every experiment wants and the shape a resident service
    /// will have: two gigabytes mapped once, then many short jobs.
    static func runCPUQwen35Batch(snapshot path: String,
                                  input: URL,
                                  output: URL) throws {
        let directory = URL(fileURLWithPath: path)
        let snapshot = try Self.loadDenseSnapshot(path)
        let requested = ProcessInfo.processInfo.environment["NVMAI_CPU35_THREADS"]
            .flatMap(Int.init)
        let model = try CPUQwen35(snapshot: snapshot, threads: requested)
        let tokenizer = try loadTokenizer(directory)
        guard let tokenizer else {
            // Non-zero: a run that never checked anything is not a pass.
            FileHandle.standardError.write(Data("no tokenizer in \(path)\n".utf8))
            exit(2)
        }

        let lines = try String(contentsOf: input, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: true)
        var results: [String] = []
        let started = ContinuousClock.now
        var tokens = 0
        for (index, line) in lines.enumerated() {
            guard let data = line.data(using: .utf8),
                  let job = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let prompt = job["prompt"] as? String else { continue }
            let limit = (job["max"] as? Int) ?? 64
            model.reset()
            // `chat` renders the model's own template, which an
            // instruction-tuned model needs to answer rather than continue.
            // Raw continuation stays the default: the parity checks depend
            // on it.
            let rendered: String
            if (job["chat"] as? Bool) == true {
                var messages: [GFTokenizer.Message] = []
                if let system = job["system"] as? String {
                    messages.append(GFTokenizer.Message(role: .system, content: system))
                }
                messages.append(GFTokenizer.Message(role: .user, content: prompt))
                rendered = (try? tokenizer.applyChatTemplate(messages)) ?? prompt
            } else {
                rendered = prompt
            }
            let ids = tokenizer.encode(rendered, addBOS: false).map(Int.init)
            let produced = try model.generate(prompt: ids, maximumTokens: limit,
                                              stopping: [Int(tokenizer.eosID)])
            tokens += ids.count + produced.count
            var record = job
            record["completion"] = tokenizer.decode(produced.map(Int32.init))
            record["prompt_tokens"] = ids.count
            record["completion_tokens"] = produced.count
            let encoded = try JSONSerialization.data(withJSONObject: record)
            results.append(String(decoding: encoded, as: UTF8.self))
            if (index + 1) % 10 == 0 {
                FileHandle.standardError.write(Data("  \(index + 1)/\(lines.count)\n".utf8))
            }
        }
        try results.joined(separator: "\n").appending("\n").write(
            to: output, atomically: true, encoding: .utf8)
        let elapsed = started.duration(to: .now)
        let seconds = Double(elapsed.components.seconds)
            + Double(elapsed.components.attoseconds) / 1e18
        print(String(format: "%d prompts, %d tokens in %.1fs (%.1f tok/s) -> %@",
                     results.count, tokens, seconds, Double(tokens) / seconds,
                     output.path as NSString))
    }

    /// GFTokenizer loads asynchronously and these commands are one-shot
    /// tools, so they wait rather than restructuring `main` around it.
    ///
    /// The folder resolution is the shared one, so a shipped `.gturbo` install
    /// (tokenizer in a `tokenizer/` sidecar) and a flat HF snapshot
    /// (`tokenizer.json` at the top level) are both reached without a second
    /// copy of that rule living here.
    static func loadTokenizer(_ directory: URL) throws -> GFTokenizer? {
        // unchecked-invariant: written only inside the Task below and read
        // only after `semaphore.wait()` returns, which the signal orders after
        // the last write. There is no concurrent access.
        final class Box: @unchecked Sendable { var value: GFTokenizer? }
        let box = Box()
        let semaphore = DispatchSemaphore(value: 0)
        guard let folder = GFTokenizer.resolvedTokenizerFolder(forModelDirectory: directory) else {
            return nil
        }
        Task {
            box.value = try? await GFTokenizer.load(from: folder)
            semaphore.signal()
        }
        semaphore.wait()
        return box.value
    }
}
