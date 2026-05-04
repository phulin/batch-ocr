import XCTest
@testable import MinerU

final class ImageProcessorTests: XCTestCase {
    func testSmartResizeIsNoopOnAlignedSquare() {
        // 1036 = 37 * 28 — already aligned to factor; pixel count within bounds.
        let (h, w) = ImageProcessor.smartResize(height: 1036, width: 1036)
        XCTAssertEqual(h, 1036)
        XCTAssertEqual(w, 1036)
    }

    func testSmartResizeRoundsToFactor() {
        // 1000 → nearest multiple of 28 is 28 * round(1000/28=35.71) = 28*36 = 1008.
        let (h, w) = ImageProcessor.smartResize(height: 1000, width: 1000)
        XCTAssertEqual(h, 1008)
        XCTAssertEqual(w, 1008)
    }

    func testSmartResizeMinPixelsClamp() {
        // Very small image — should grow to at least minPixels (3136 = 4*28*28).
        let (h, w) = ImageProcessor.smartResize(height: 30, width: 30)
        XCTAssertGreaterThanOrEqual(h * w, ImageProcessor.Constants.minPixels)
        XCTAssertEqual(h % ImageProcessor.Constants.factor, 0)
        XCTAssertEqual(w % ImageProcessor.Constants.factor, 0)
    }

    func testSmartResizeMaxPixelsClamp() {
        // A huge 20000x20000 image — should shrink to <= maxPixels.
        let (h, w) = ImageProcessor.smartResize(height: 20_000, width: 20_000)
        XCTAssertLessThanOrEqual(h * w, ImageProcessor.Constants.maxPixels)
        XCTAssertEqual(h % ImageProcessor.Constants.factor, 0)
        XCTAssertEqual(w % ImageProcessor.Constants.factor, 0)
    }

    func testProcessProducesExpectedShapes() throws {
        let cg = makeCheckerboard(width: 56, height: 56)  // 56 = 2 * 28 → grid 4x4 patches → outer 2x2
        let proc = ImageProcessor()
        let out = try proc.process(cg)
        XCTAssertEqual(out.gridTHW, [1, 4, 4])
        XCTAssertEqual(out.sequenceLength, 16)
        // 16 tokens * 3*2*14*14 values each = 16 * 1176 = 18816
        XCTAssertEqual(out.pixelValues.count, 16 * 1176)
    }

    private func makeCheckerboard(width: Int, height: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )!
        for y in 0..<height {
            for x in 0..<width {
                let on = ((x / 8) + (y / 8)) % 2 == 0
                ctx.setFillColor(CGColor(
                    red: on ? 1 : 0, green: on ? 1 : 0, blue: on ? 1 : 0, alpha: 1
                ))
                ctx.fill(CGRect(x: x, y: y, width: 1, height: 1))
            }
        }
        return ctx.makeImage()!
    }
}
