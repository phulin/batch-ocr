import XCTest
import ImageIO
import MinerU

/// End-to-end integration test: run the Swift MinerU 2.5 Pro port on page2_text.png and
/// compare layout output against the Python oracle (oracle/fixtures/page2_text/).
///
/// Heavy: model load (≈1.2B params, bf16) + 1024-token decode budget. Skipped unless the
/// MINERU_MODEL_PATH env var points to a local model snapshot OR the HF cache already has
/// `opendatalab/MinerU2.5-Pro-2604-1.2B`.
final class MinerUInferenceTests: XCTestCase {
    func testLayoutMatchesOracle() async throws {
        // Use the locally cached HF snapshot to avoid sandbox/DNS during tests.
        // Bypass HOME (which sandbox redirects) via getpwuid.
        let env = ProcessInfo.processInfo.environment
        let pw = getpwuid(getuid())!
        let realHome = String(cString: pw.pointee.pw_dir)
        let cachedSnapshot = "\(realHome)/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots"
        let cacheURL = URL(fileURLWithPath: cachedSnapshot)
        let snapshots = (try? FileManager.default.contentsOfDirectory(at: cacheURL, includingPropertiesForKeys: nil)) ?? []
        guard let snapshot = snapshots.first else {
            XCTFail("No cached MinerU snapshot at \(cachedSnapshot) — run oracle/dump_oracle.py first")
            return
        }
        let modelSpec = env["MINERU_MODEL_PATH"] ?? snapshot.path

        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // BatchOCRTests/
            .deletingLastPathComponent()  // repo root
        let imageURL = projectRoot.appendingPathComponent("page2_text.png")
        let oracleJSON = projectRoot.appendingPathComponent("oracle/fixtures/page2_text/blocks.json")

        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            XCTFail("Missing fixture: \(imageURL.path)")
            return
        }
        guard FileManager.default.fileExists(atPath: oracleJSON.path) else {
            XCTFail("Missing oracle JSON: \(oracleJSON.path) — run oracle/dump_oracle.py")
            return
        }

        let pipeline = try await MinerUPipeline.load(from: modelSpec)
        let (text, blocks) = try pipeline.detectLayout(imageURL: imageURL)
        print("=== Swift raw layout text ===")
        print(text)
        print("=== Swift parsed blocks (\(blocks.count)) ===")

        struct OracleBlock: Decodable {
            let type: String
            let bbox: [Double]
            let angle: Int?
            let merge_prev: Bool?
        }
        let data = try Data(contentsOf: oracleJSON)
        let oracle = try JSONDecoder().decode([OracleBlock].self, from: data)

        XCTAssertEqual(blocks.count, oracle.count,
                       "block count mismatch (Swift=\(blocks.count), Python=\(oracle.count))")
        for (i, (s, o)) in zip(blocks, oracle).enumerated() {
            XCTAssertEqual(s.type, o.type, "block \(i) type")
            XCTAssertEqual(s.bbox.x1, o.bbox[0], accuracy: 5e-3, "block \(i) x1")
            XCTAssertEqual(s.bbox.y1, o.bbox[1], accuracy: 5e-3, "block \(i) y1")
            XCTAssertEqual(s.bbox.x2, o.bbox[2], accuracy: 5e-3, "block \(i) x2")
            XCTAssertEqual(s.bbox.y2, o.bbox[3], accuracy: 5e-3, "block \(i) y2")
        }
    }
}
