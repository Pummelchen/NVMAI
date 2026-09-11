// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "NVMAI",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .library(name: "NVMAI", targets: ["NVMAI"]),
        .library(name: "NVMAIFormat", targets: ["NVMAIFormat"]),
        .library(name: "ContinuityCore", targets: ["ContinuityCore"]),
        .executable(name: "NVMAIRepack", targets: ["NVMAIRepack"]),
        .executable(name: "NVMAICLI", targets: ["NVMAICLI"]),
        .executable(name: "NVMAIMac", targets: ["NVMAIMac"]),
        .executable(name: "NVMAIDecodeService", targets: ["NVMAIDecodeService"]),
        .executable(name: "NVMAIServer", targets: ["NVMAIServer"]),
        .executable(name: "NVMAIBench", targets: ["NVMAIBench"]),
        .executable(name: "ContinuityDemo", targets: ["ContinuityDemo"]),
        .executable(name: "nvmai-memory", targets: ["NVMAIMemoryTool"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "NVMAIFormat",
            path: "sources/NVMAIFormat"
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced. No custom
        // flags -- -O3 measured the same as SwiftPM's release default (0.675
        // vs 0.680 ms), so it is not worth the unsafeFlags constraint.
        .target(
            name: "NVMAIKernelsC",
            path: "sources/NVMAIKernelsC"
        ),
        .target(
            name: "NVMAI",
            dependencies: [
                "NVMAIFormat",
                "NVMAIKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "sources/NVMAI",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "NVMAIRepackCore",
            dependencies: ["NVMAIFormat"],
            path: "sources/NVMAIRepack/Core"
        ),
        .executableTarget(
            name: "NVMAIRepack",
            dependencies: ["NVMAIRepackCore"],
            path: "sources/NVMAIRepack/Command"
        ),
        .target(
            name: "NVMAICLICore",
            dependencies: ["NVMAI"],
            path: "sources/NVMAICLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "NVMAICLI",
            dependencies: ["NVMAICLICore"],
            path: "sources/NVMAICLI/Command"
        ),
        .target(
            name: "NVMAIAppCore",
            dependencies: ["NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "sources/NVMAIApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "NVMAIMacPresentation",
            dependencies: ["NVMAIAppCore"],
            path: "sources/NVMAIApp/MacPresentation"
        ),
        .target(
            name: "NVMAIDecodeProtocol",
            path: "sources/NVMAIDecodeProtocol"
        ),
        .executableTarget(
            name: "NVMAIDecodeService",
            dependencies: ["NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "sources/NVMAIDecodeService"
        ),
        // Continuity: sessions, task memory and context assembly, in this
        // process. Depends on nothing at all, not even NIO, so it cannot
        // reach the network and cannot be reached from one.
        .target(
            name: "ContinuityCore",
            path: "sources/ContinuityCore",
            // Documentation that lives next to the code it describes. SwiftPM
            // treats any undeclared file under a target path as unhandled and
            // warns on every clean plan; excluding it says so explicitly and
            // leaves the file where it is. (`sources/NVMAICLICore`'s
            // `exclude: ["Command"]` is the same mechanism.)
            exclude: ["README.md"]
        ),
        // Worked examples and a scale check for ContinuityCore. Not part of
        // the server; it exists so the package's claims can be run.
        .executableTarget(
            name: "ContinuityDemo",
            dependencies: ["ContinuityCore"],
            path: "sources/ContinuityDemo"
        ),
        // Agent memory: the model-facing surface (keys, tools, prompt
        // fragment, journal filter) over ContinuityCore. Depends on nothing
        // in the engine, so the serving path can use it without the memory
        // subsystem being able to reach back into inference, and on no
        // networking, so it cannot reach off the machine.
        .target(
            name: "NVMAIMemory",
            dependencies: ["ContinuityCore"],
            path: "sources/NVMAIMemory"
        ),
        // See and correct what the server remembers: list, show, delete.
        // Reads take no lock; writes need the workspace.
        .executableTarget(
            name: "NVMAIMemoryTool",
            dependencies: ["NVMAIMemory", "ContinuityCore"],
            path: "sources/NVMAIMemoryTool"
        ),
        .target(
            name: "NVMAIServerCore",
            dependencies: [
                "NVMAI",
                "NVMAIMemory",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/NVMAIServer/Core"
        ),
        .executableTarget(
            name: "NVMAIServer",
            dependencies: ["NVMAIServerCore"],
            path: "sources/NVMAIServer/Command"
        ),
        .executableTarget(
            name: "NVMAIBench",
            dependencies: ["NVMAI"],
            path: "sources/NVMAIBench"
        ),
        .executableTarget(
            name: "NVMAIMac",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "sources/NVMAIApp/Mac",
            resources: [
                .copy("Resources/nvmai-app-icon.png"),
            ]
        ),
        .target(
            name: "NVMAIValidationSupport",
            dependencies: ["NVMAI"],
            path: "sources/NVMAIValidation/Support"
        ),
        .testTarget(
            name: "NVMAITests",
            dependencies: ["NVMAI", "NVMAIValidationSupport", "NVMAIRepackCore", "NVMAICLICore"],
            path: "tests/NVMAI",
            resources: [.copy("Tokenization/Fixtures"),
                        .copy("Runtime/qwen38_tensor_names.txt"),
                        .copy("Runtime/ple_golden.json")]
        ),
        .testTarget(
            name: "NVMAIRepackTests",
            // `NVMAIFormat` directly: the manifest and resident-index validation
            // tests assert on those types rather than on JSON dictionaries.
            dependencies: ["NVMAIRepackCore", "NVMAIFormat"],
            path: "tests/NVMAIRepack/Core",
            resources: [.copy("Support/qwen38_tensor_names.txt")]
        ),
        .testTarget(
            name: "NVMAIAppCoreTests",
            dependencies: ["NVMAIAppCore", "NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "tests/NVMAIApp/Core"
        ),
        .testTarget(
            name: "NVMAIDecodeServiceTests",
            dependencies: ["NVMAIDecodeService", "NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "tests/NVMAIDecodeService"
        ),
        .testTarget(
            name: "NVMAIMacPresentationTests",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "tests/NVMAIApp/MacPresentation"
        ),
        .testTarget(
            name: "ContinuityCoreTests",
            dependencies: ["ContinuityCore"],
            path: "tests/ContinuityCore"
        ),
        .testTarget(
            name: "NVMAIMemoryTests",
            dependencies: ["NVMAIMemory", "ContinuityCore"],
            path: "tests/NVMAIMemory"
        ),
        .testTarget(
            name: "NVMAIServerTests",
            dependencies: [
                "NVMAIServerCore",
                "NVMAIMemory",
                // `GenerationDefaults.Sampling`, so the mapper tests can pin
                // that an omitted field follows the served model's profile
                // rather than a hardcoded house default.
                "NVMAI",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/NVMAIServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
