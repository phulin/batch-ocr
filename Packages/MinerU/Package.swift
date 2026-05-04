// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MinerU",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "MinerU", targets: ["MinerU"]),
        .executable(name: "MinerUCLI", targets: ["MinerUCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MinerU",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "MinerUCLI",
            dependencies: [
                "MinerU",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "MinerUTests",
            dependencies: ["MinerU"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
