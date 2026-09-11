// swift-tools-version: 6.3
import PackageDescription

/// The language standard this package is written to.
///
/// Swift 6 language mode is set below (`swiftLanguageModes: [.v6]`); these are
/// the upcoming features that are not yet default in that mode and that the
/// tree is clean under. Enforced here rather than documented, so a target
/// added later cannot quietly opt out. The ones deliberately *not* adopted
/// (and why, with their measured diagnostic counts) are recorded in
/// `docs/swift-language-standard.md`.
let nvmaiLanguageStandard: [SwiftSetting] = [
    .enableUpcomingFeature("InferIsolatedConformances"),
    .enableUpcomingFeature("ImmutableWeakCaptures"),
    .enableUpcomingFeature("MemberImportVisibility"),
]

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
            path: "sources/NVMAIFormat",
            swiftSettings: nvmaiLanguageStandard
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced. No custom
        // flags -- -O3 measured the same as SwiftPM's release default (0.675
        // vs 0.680 ms), so it is not worth the unsafeFlags constraint.
        .target(
            name: "NVMAIKernelsC",
            path: "sources/NVMAIKernelsC",
            swiftSettings: nvmaiLanguageStandard
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
            ],
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAIRepackCore",
            dependencies: ["NVMAIFormat"],
            path: "sources/NVMAIRepack/Core",
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAIRepack",
            dependencies: ["NVMAIRepackCore"],
            path: "sources/NVMAIRepack/Command",
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAICLICore",
            dependencies: ["NVMAI"],
            path: "sources/NVMAICLI",
            exclude: ["Command"],
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAICLI",
            dependencies: ["NVMAICLICore"],
            path: "sources/NVMAICLI/Command",
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAIAppCore",
            dependencies: ["NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "sources/NVMAIApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ],
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAIMacPresentation",
            dependencies: ["NVMAIAppCore"],
            path: "sources/NVMAIApp/MacPresentation",
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAIDecodeProtocol",
            path: "sources/NVMAIDecodeProtocol",
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAIDecodeService",
            dependencies: ["NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "sources/NVMAIDecodeService",
            swiftSettings: nvmaiLanguageStandard
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
            exclude: ["README.md"],
            swiftSettings: nvmaiLanguageStandard
        ),
        // Worked examples and a scale check for ContinuityCore. Not part of
        // the server; it exists so the package's claims can be run.
        .executableTarget(
            name: "ContinuityDemo",
            dependencies: ["ContinuityCore"],
            path: "sources/ContinuityDemo",
            swiftSettings: nvmaiLanguageStandard
        ),
        // Agent memory: the model-facing surface (keys, tools, prompt
        // fragment, journal filter) over ContinuityCore. Depends on nothing
        // in the engine, so the serving path can use it without the memory
        // subsystem being able to reach back into inference, and on no
        // networking, so it cannot reach off the machine.
        .target(
            name: "NVMAIMemory",
            dependencies: ["ContinuityCore"],
            path: "sources/NVMAIMemory",
            swiftSettings: nvmaiLanguageStandard
        ),
        // See and correct what the server remembers: list, show, delete.
        // Reads take no lock; writes need the workspace.
        .executableTarget(
            name: "NVMAIMemoryTool",
            dependencies: ["NVMAIMemory", "ContinuityCore"],
            path: "sources/NVMAIMemoryTool",
            swiftSettings: nvmaiLanguageStandard
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
            path: "sources/NVMAIServer/Core",
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAIServer",
            dependencies: ["NVMAIServerCore"],
            path: "sources/NVMAIServer/Command",
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAIBench",
            dependencies: ["NVMAI"],
            path: "sources/NVMAIBench",
            swiftSettings: nvmaiLanguageStandard
        ),
        .executableTarget(
            name: "NVMAIMac",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "sources/NVMAIApp/Mac",
            resources: [
                .copy("Resources/nvmai-app-icon.png"),
            ],
            swiftSettings: nvmaiLanguageStandard
        ),
        .target(
            name: "NVMAIValidationSupport",
            dependencies: ["NVMAI"],
            path: "sources/NVMAIValidation/Support",
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAITests",
            dependencies: ["NVMAI", "NVMAIValidationSupport", "NVMAIRepackCore", "NVMAICLICore"],
            path: "tests/NVMAI",
            resources: [.copy("Tokenization/Fixtures"),
                        .copy("Runtime/qwen38_tensor_names.txt"),
                        .copy("Runtime/ple_golden.json")],
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAIRepackTests",
            // `NVMAIFormat` directly: the manifest and resident-index validation
            // tests assert on those types rather than on JSON dictionaries.
            dependencies: ["NVMAIRepackCore", "NVMAIFormat"],
            path: "tests/NVMAIRepack/Core",
            resources: [.copy("Support/qwen38_tensor_names.txt")],
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAIAppCoreTests",
            dependencies: ["NVMAIAppCore", "NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "tests/NVMAIApp/Core",
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAIDecodeServiceTests",
            dependencies: ["NVMAIDecodeService", "NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "tests/NVMAIDecodeService",
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAIMacPresentationTests",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "tests/NVMAIApp/MacPresentation",
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "ContinuityCoreTests",
            dependencies: ["ContinuityCore"],
            path: "tests/ContinuityCore",
            swiftSettings: nvmaiLanguageStandard
        ),
        .testTarget(
            name: "NVMAIMemoryTests",
            dependencies: ["NVMAIMemory", "ContinuityCore"],
            path: "tests/NVMAIMemory",
            swiftSettings: nvmaiLanguageStandard
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
            resources: [.copy("Fixtures")],
            swiftSettings: nvmaiLanguageStandard
        ),
    ],
    swiftLanguageModes: [.v6]
)
