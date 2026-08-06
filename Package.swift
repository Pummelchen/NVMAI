// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "NVMAI",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
    ],
    products: [
        .library(name: "NVMAI", targets: ["NVMAI"]),
        .library(name: "NVMAIFormat", targets: ["NVMAIFormat"]),
        .executable(name: "NVMAIRepack", targets: ["NVMAIRepack"]),
        .executable(name: "NVMAICLI", targets: ["NVMAICLI"]),
        .executable(name: "NVMAIMac", targets: ["NVMAIMac"]),
        .executable(name: "NVMAIDecodeService", targets: ["NVMAIDecodeService"]),
        .executable(name: "NVMAIServer", targets: ["NVMAIServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(
            name: "NVMAIFormat",
            path: "Sources/NVMAIFormat"
        ),
        .target(
            name: "NVMAI",
            dependencies: [
                "NVMAIFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/NVMAI",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "NVMAIRepackCore",
            dependencies: ["NVMAIFormat"],
            path: "Sources/NVMAIRepack/Core"
        ),
        .executableTarget(
            name: "NVMAIRepack",
            dependencies: ["NVMAIRepackCore"],
            path: "Sources/NVMAIRepack/Command"
        ),
        .target(
            name: "NVMAICLICore",
            dependencies: ["NVMAI"],
            path: "Sources/NVMAICLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "NVMAICLI",
            dependencies: ["NVMAICLICore"],
            path: "Sources/NVMAICLI/Command"
        ),
        .target(
            name: "NVMAIAppCore",
            dependencies: ["NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "Sources/NVMAIApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "NVMAIMacPresentation",
            dependencies: ["NVMAIAppCore"],
            path: "Sources/NVMAIApp/MacPresentation"
        ),
        .target(
            name: "NVMAIDecodeProtocol",
            path: "Sources/NVMAIDecodeProtocol"
        ),
        .executableTarget(
            name: "NVMAIDecodeService",
            dependencies: ["NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "Sources/NVMAIDecodeService"
        ),
        .target(
            name: "NVMAIServerCore",
            dependencies: [
                "NVMAI",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/NVMAIServer/Core"
        ),
        .executableTarget(
            name: "NVMAIServer",
            dependencies: ["NVMAIServerCore"],
            path: "Sources/NVMAIServer/Command"
        ),
        .executableTarget(
            name: "NVMAIMac",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "Sources/NVMAIApp/Mac",
            resources: [
                .copy("Resources/nvmai-app-icon.png"),
            ]
        ),
        .target(
            name: "NVMAIValidationSupport",
            dependencies: ["NVMAI"],
            path: "Sources/NVMAIValidation/Support"
        ),
        .testTarget(
            name: "NVMAITestsCore",
            dependencies: ["NVMAI", "NVMAIValidationSupport", "NVMAIRepackCore", "NVMAICLICore"],
            path: "Tests/NVMAI/Core",
            resources: [.copy("Tokenization/Fixtures")]
        ),
        .testTarget(
            name: "NVMAIRepackTests",
            dependencies: ["NVMAIRepackCore"],
            path: "Tests/NVMAIRepack/Core"
        ),
        .testTarget(
            name: "NVMAIAppCoreTests",
            dependencies: ["NVMAIAppCore", "NVMAI", "NVMAIRepackCore", "NVMAIDecodeProtocol"],
            path: "Tests/NVMAIApp/Core"
        ),
        .testTarget(
            name: "NVMAIDecodeServiceTests",
            dependencies: ["NVMAIDecodeService", "NVMAIAppCore", "NVMAIDecodeProtocol"],
            path: "Tests/NVMAIDecodeService"
        ),
        .testTarget(
            name: "NVMAIMacPresentationTests",
            dependencies: ["NVMAIAppCore", "NVMAIMacPresentation"],
            path: "Tests/NVMAIApp/MacPresentation"
        ),
        .testTarget(
            name: "NVMAIServerTests",
            dependencies: [
                "NVMAIServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/NVMAIServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
