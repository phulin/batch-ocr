import XCTest
import ImageIO
import CoreGraphics
@testable import MinerU

/// Parity test: my Swift ImageProcessor on `layout_input.png` (the pre-resized 1036x1036
/// image saved by the oracle) should produce the same pixel_values as HF's processor.
final class ImageProcessorParityTests: XCTestCase {
    func testImageProcessorMatchesHF() throws {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let layoutPNG = projectRoot.appendingPathComponent("oracle/fixtures/page2_text/layout_input.png")
        let pixelsNPY = projectRoot.appendingPathComponent("oracle/fixtures/page2_text/pixel_values.npy")

        guard FileManager.default.fileExists(atPath: layoutPNG.path),
              FileManager.default.fileExists(atPath: pixelsNPY.path) else {
            XCTFail("Missing oracle artifacts — re-run oracle/dump_oracle.py")
            return
        }

        guard let src = CGImageSourceCreateWithURL(layoutPNG as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            XCTFail("Couldn't load layout_input.png")
            return
        }

        let proc = ImageProcessor()
        let out = try proc.process(cg)
        XCTAssertEqual(out.gridTHW, [1, 74, 74])

        let oracle = try NPY.read(pixelsNPY)
        XCTAssertEqual(oracle.shape, [5476, 1176])
        XCTAssertEqual(out.pixelValues.count, oracle.floats.count)

        var maxAbsDiff: Float = 0
        var sumAbsDiff: Double = 0
        for (s, o) in zip(out.pixelValues, oracle.floats) {
            let d = abs(s - o)
            sumAbsDiff += Double(d)
            if d > maxAbsDiff { maxAbsDiff = d }
        }
        let meanAbsDiff = sumAbsDiff / Double(out.pixelValues.count)
        print(String(format: "image processor parity: max=%.6f mean=%.6f", maxAbsDiff, meanAbsDiff))

        // Also bisect: are first/last patches' values close, or is the whole image flipped?
        // First patch (oracle): row 0, all 1176 channels
        let firstSwift = Array(out.pixelValues.prefix(1176))
        let firstOracle = Array(oracle.floats.prefix(1176))
        let lastSwift = Array(out.pixelValues.suffix(1176))
        let lastOracle = Array(oracle.floats.suffix(1176))
        print("first patch swift mean: \(firstSwift.reduce(0, +) / Float(1176))")
        print("first patch oracle mean: \(firstOracle.reduce(0, +) / Float(1176))")
        print("last patch swift mean: \(lastSwift.reduce(0, +) / Float(1176))")
        print("last patch oracle mean: \(lastOracle.reduce(0, +) / Float(1176))")

        // Check vertical flip: does Swift's first patch match Oracle's last-row patch?
        // Patches are laid out (oh, ow, mh, mw); for a 74x74 patch grid with merge=2,
        // last "row" of patches starts at oh=36 in flat token index ((36*37 + 0)*4)*4 = ...
        // Simpler check: compute per-row patch means and compare.
        XCTAssertLessThan(meanAbsDiff, 0.05,
                          "ImageProcessor diverges from HF (mean abs \(meanAbsDiff))")
    }
}
