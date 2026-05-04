import XCTest
import ImageIO
@testable import MinerU

/// Smoke test for stage-2 recognition: layout pass + per-block content extraction.
/// Not a full parity test — just verifies blocks come back with non-empty content where
/// expected and prints the result for inspection.
final class Stage2ExtractionTest: XCTestCase {
    func testExtractPage2Text() async throws {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let imageURL = projectRoot.appendingPathComponent("page2_text.png")

        let pw = getpwuid(getuid())!
        let realHome = String(cString: pw.pointee.pw_dir)
        let snapshotsDir = "\(realHome)/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots"
        let snapshots = (try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir)) ?? []
        guard let snap = snapshots.first else {
            XCTFail("No cached snapshot")
            return
        }
        let pipeline = try await MinerUPipeline.load(from: "\(snapshotsDir)/\(snap)")

        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            XCTFail("Couldn't load image"); return
        }

        let blocks = try pipeline.extract(cg)
        print("=== Extracted \(blocks.count) blocks ===")
        for (i, b) in blocks.enumerated() {
            let bbox = String(format: "(%.3f,%.3f,%.3f,%.3f)", b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2)
            let rec = b.content.map { String($0.prefix(120)).replacingOccurrences(of: "\n", with: " ⏎ ") } ?? "(skipped)"
            print("[\(i)] \(b.type) \(bbox): \(rec)")
        }

        // Expectation: the title block should have non-empty content.
        let title = blocks.first(where: { $0.type == "title" })
        XCTAssertNotNil(title?.content, "title block should have recognized content")
        XCTAssertFalse(title?.content?.isEmpty ?? true)
    }
}
