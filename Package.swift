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
        .executable(name: "NVMAIRepack", targets: ["NVMAIRepack"]),
        .executable(name: "NVMAICLI", targets: ["NVMAICLI"]),
        .executable(name: "NVMAIMac", targets: ["NVMAIMac"]),
        .executable(name: "NVMAIDecodeService", targets: ["NVMAIDecodeService"]),
        .executable(name: "NVMAIServer", targets: ["NVMAIServer"]),
        .executable(name: "NVMAIBench", targets: ["NVMAIBench"]),
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
        .target(
            name: "NVMAIServerCore",
            dependencies: [
                "NVMAI",
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
            name: "NVMAITestsCore",
            dependencies: ["NVMAI", "NVMAIValidationSupport", "NVMAIRepackCore", "NVMAICLICore"],
            path: "tests/NVMAI/Core",
            resources: [.copy("Tokenization/Fixtures"),
                        .copy("Runtime/qwen38_tensor_names.txt")]
        ),
        .testTarget(
            name: "NVMAIRepackTests",
            dependencies: ["NVMAIRepackCore"],
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
            name: "NVMAIServerTests",
            dependencies: [
                "NVMAIServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/NVMAIServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
