import XCTest
@testable import MinerU

/// Bisect: feed Python's exact vision_features into the LM and see if output matches oracle.
/// If yes → residual y-axis divergence comes from vision-feature drift.
/// If no → there's an LM-side bug (M-RoPE, KV cache, etc.).
final class LMOnlyParityTests: XCTestCase {
    func testLMRunsWithOracleVisionFeatures() async throws {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let fixtureDir = projectRoot.appendingPathComponent("oracle/fixtures/page2_text")

        let featURL = fixtureDir.appendingPathComponent("vision_features.npy")
        let gridURL = fixtureDir.appendingPathComponent("image_grid_thw.npy")
        let oracleJSON = fixtureDir.appendingPathComponent("blocks.json")

        let features = try NPY.read(featURL)
        let grid = try NPY.read(gridURL)
        let t = Int(grid.floats[0]), h = Int(grid.floats[1]), w = Int(grid.floats[2])
        XCTAssertEqual(features.shape, [1369, 896])

        let pw = getpwuid(getuid())!
        let realHome = String(cString: pw.pointee.pw_dir)
        let snapshotsDir = "\(realHome)/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots"
        let snapshots = (try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir)) ?? []
        guard let snap = snapshots.first else {
            XCTFail("No cached snapshot")
            return
        }
        let modelDir = "\(snapshotsDir)/\(snap)"

        let pipeline = try await MinerUPipeline.load(from: modelDir)

        // Tokenizer sanity: encode/decode of special tokens
        let testStr = "<|im_start|>system\nhi<|im_end|>"
        let ids = (try? pipeline.tokenizer.encode(text: testStr)) ?? []
        print("encode(\"<|im_start|>system\\nhi<|im_end|>\") → \(ids.count) tokens: \(ids)")
        let decoded = (try? pipeline.tokenizer.decode(tokens: ids)) ?? ""
        print("  decoded: \(decoded)")
        // Expected: <|im_start|> = 151644, <|im_end|> = 151645
        let imStart = (try? pipeline.tokenizer.encode(text: "<|im_start|>")) ?? []
        let imEnd = (try? pipeline.tokenizer.encode(text: "<|im_end|>")) ?? []
        let imagePad = (try? pipeline.tokenizer.encode(text: "<|image_pad|>")) ?? []
        let visionStart = (try? pipeline.tokenizer.encode(text: "<|vision_start|>")) ?? []
        let visionEnd = (try? pipeline.tokenizer.encode(text: "<|vision_end|>")) ?? []
        print("  <|im_start|>=\(imStart) (expect [151644])")
        print("  <|im_end|>=\(imEnd) (expect [151645])")
        print("  <|image_pad|>=\(imagePad) (expect [151655])")
        print("  <|vision_start|>=\(visionStart) (expect [151652])")
        print("  <|vision_end|>=\(visionEnd) (expect [151653])")

        let (text, blocks) = try pipeline.detectLayoutWithFeatures(
            gridT: t, gridH: h, gridW: w,
            visionFeatures: features.floats,
            visionFeatureRows: features.shape[0],
            visionFeatureCols: features.shape[1]
        )

        print("=== LM-only raw layout text (first 600 chars) ===")
        print(String(text.prefix(600)))
        print("=== LM-only parsed blocks (\(blocks.count)) ===")
        for (i, b) in blocks.prefix(15).enumerated() {
            print("  [\(i)] \(b.type) \(b.bbox)")
        }

        struct OracleBlock: Decodable {
            let type: String
            let bbox: [Double]
        }
        let oracleData = try Data(contentsOf: oracleJSON)
        let oracle = try JSONDecoder().decode([OracleBlock].self, from: oracleData)
        XCTAssertEqual(blocks.count, oracle.count, "block count")
        for (i, (s, o)) in zip(blocks, oracle).enumerated() {
            XCTAssertEqual(s.type, o.type, "block \(i) type")
            XCTAssertEqual(s.bbox.x1, o.bbox[0], accuracy: 5e-3, "block \(i) x1")
            XCTAssertEqual(s.bbox.y1, o.bbox[1], accuracy: 5e-3, "block \(i) y1")
        }
    }
}
