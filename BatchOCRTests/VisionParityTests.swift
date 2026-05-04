import XCTest
@testable import MinerU

/// Bisect parity test for the Swift NaViT vision tower.
/// Loads the Python oracle's pixel_values.npy + image_grid_thw.npy + vision_features.npy
/// and asserts the Swift VisionEncoder produces matching features (post patch-merger).
final class VisionParityTests: XCTestCase {
    func testVisionEncoderMatchesOracle() throws {
        let projectRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // BatchOCRTests/
            .deletingLastPathComponent()  // repo root
        let fixtureDir = projectRoot.appendingPathComponent("oracle/fixtures/page2_text")

        let pixelURL = fixtureDir.appendingPathComponent("pixel_values.npy")
        let gridURL  = fixtureDir.appendingPathComponent("image_grid_thw.npy")
        let featURL  = fixtureDir.appendingPathComponent("vision_features.npy")

        guard FileManager.default.fileExists(atPath: pixelURL.path),
              FileManager.default.fileExists(atPath: featURL.path) else {
            XCTFail("Vision oracle artifacts missing — re-run oracle/dump_oracle.py")
            return
        }

        let pixels = try NPY.read(pixelURL)
        let grid   = try NPY.read(gridURL)
        let oracleFeatures = try NPY.read(featURL)

        XCTAssertEqual(pixels.shape, [5476, 1176], "unexpected pixel_values shape")
        XCTAssertEqual(oracleFeatures.shape, [1369, 896], "unexpected vision_features shape")
        XCTAssertEqual(grid.floats.count, 3, "image_grid_thw must be [t,h,w]")
        let t = Int(grid.floats[0]), h = Int(grid.floats[1]), w = Int(grid.floats[2])

        // Load weights so the Swift VisionEncoder is initialized properly.
        let pw = getpwuid(getuid())!
        let realHome = String(cString: pw.pointee.pw_dir)
        let snapshotsDir = "\(realHome)/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots"
        let snapshots = (try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir)) ?? []
        guard let snap = snapshots.first else {
            XCTFail("No cached MinerU snapshot")
            return
        }
        let modelDir = URL(fileURLWithPath: "\(snapshotsDir)/\(snap)")
        let config = try MinerUWeightLoader.loadConfig(from: modelDir)
        let model = MinerUModel(config)
        try MinerUWeightLoader.load(model, from: modelDir)

        // Run the Swift vision tower on Python's pixel_values directly (bypassing our
        // ImageProcessor) — isolates vision-tower correctness from preprocessing.
        let (swiftFloats, outShape) = model.runVisionTower(
            pixelValues: pixels.floats,
            pixelShape: pixels.shape,
            gridT: t, gridH: h, gridW: w
        )
        XCTAssertEqual(outShape, [1369, 896], "Swift vision tower output shape")
        XCTAssertEqual(swiftFloats.count, oracleFeatures.floats.count, "feature count")

        var diffs = [(idx: Int, diff: Float)]()
        var sumAbsDiff: Double = 0
        var rowDiffSum = [Double](repeating: 0, count: 1369)
        var rowMaxDiff = [Float](repeating: 0, count: 1369)
        for (i, (s, o)) in zip(swiftFloats, oracleFeatures.floats).enumerated() {
            let d = abs(s - o)
            sumAbsDiff += Double(d)
            diffs.append((i, d))
            let row = i / 896
            rowDiffSum[row] += Double(d)
            if d > rowMaxDiff[row] { rowMaxDiff[row] = d }
        }
        diffs.sort { $0.diff > $1.diff }
        let meanAbsDiff = sumAbsDiff / Double(swiftFloats.count)
        let maxAbsDiff = diffs.first?.diff ?? 0
        let maxIdx = diffs.first?.idx ?? 0
        print(String(format: "vision parity: max abs diff = %.6f at idx %d, mean = %.6f",
                     maxAbsDiff, maxIdx, meanAbsDiff))
        print("top 10 diffs:")
        for (idx, d) in diffs.prefix(10) {
            let row = idx / 896, col = idx % 896
            print("  row=\(row) col=\(col) diff=\(d) swift=\(swiftFloats[idx]) oracle=\(oracleFeatures.floats[idx])")
        }
        // Top 5 worst rows by mean abs diff
        let rowMeans = rowDiffSum.enumerated().map { ($0, $1 / 896) }.sorted { $0.1 > $1.1 }
        print("top 5 worst rows by mean diff:")
        for (row, m) in rowMeans.prefix(5) {
            print("  row=\(row) mean=\(m) max=\(rowMaxDiff[row])")
        }

        // bf16 tolerance is generous — anything under ~0.05 means we're structurally correct.
        XCTAssertLessThan(maxAbsDiff, 0.1,
                          "Swift vision tower output diverges from Python oracle (max abs \(maxAbsDiff) at idx \(maxIdx))")
    }
}
