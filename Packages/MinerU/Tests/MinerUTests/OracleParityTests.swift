import XCTest
@testable import MinerU

/// Parity tests against artifacts captured by the Python oracle (`oracle/dump_oracle.py`).
/// Validates that the Swift `MinerUOutputParser` produces blocks identical to MinerU's
/// official Python parser when fed the same raw model output.
final class OracleParityTests: XCTestCase {
    func testPage2TextLayoutParity() throws {
        let bundle = Bundle.module
        guard let textURL = bundle.url(forResource: "layout_text", withExtension: "txt", subdirectory: "Fixtures"),
              let jsonURL = bundle.url(forResource: "blocks", withExtension: "json", subdirectory: "Fixtures")
        else {
            XCTFail("Oracle fixtures missing — run oracle/dump_oracle.py first.")
            return
        }
        let raw = try String(contentsOf: textURL, encoding: .utf8)
        let json = try Data(contentsOf: jsonURL)

        struct OracleBlock: Decodable {
            let type: String
            let bbox: [Double]      // [x1, y1, x2, y2]
            let angle: Int?
            let merge_prev: Bool?
        }
        let oracle = try JSONDecoder().decode([OracleBlock].self, from: json)
        let swift = MinerUOutputParser.parse(raw)

        XCTAssertEqual(swift.count, oracle.count, "block count mismatch")

        for (i, (s, o)) in zip(swift, oracle).enumerated() {
            XCTAssertEqual(s.type, o.type, "block \(i) type")
            // Python rounds to 3 decimals when serializing; allow 5e-4 slack.
            XCTAssertEqual(s.bbox.x1, o.bbox[0], accuracy: 5e-4, "block \(i) x1")
            XCTAssertEqual(s.bbox.y1, o.bbox[1], accuracy: 5e-4, "block \(i) y1")
            XCTAssertEqual(s.bbox.x2, o.bbox[2], accuracy: 5e-4, "block \(i) x2")
            XCTAssertEqual(s.bbox.y2, o.bbox[3], accuracy: 5e-4, "block \(i) y2")
            XCTAssertEqual(s.rotationDegrees ?? 0, o.angle ?? 0, "block \(i) angle")
            XCTAssertEqual(s.mergeWithPrevious, o.merge_prev ?? false, "block \(i) merge_prev")
        }
    }
}
