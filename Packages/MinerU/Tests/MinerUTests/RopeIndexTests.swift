import XCTest
@testable import MinerU

final class RopeIndexTests: XCTestCase {
    func testPureTextPositionsBroadcast() {
        let r = QwenRopeIndexer.computeRaw(
            inputIds: [10, 20, 30, 40],
            imageGridTHW: [],
            imageTokenId: 999,
            spatialMergeSize: 2
        )
        XCTAssertEqual(r.t, [0, 1, 2, 3])
        XCTAssertEqual(r.h, [0, 1, 2, 3])
        XCTAssertEqual(r.w, [0, 1, 2, 3])
        XCTAssertEqual(r.ropeDelta, 0)
    }

    func testSingleImageBlockAdvancesAxesIndependently() {
        // Layout: [text, IMG x4, text].  Image grid (t=1, h=2, w=2), spatial_merge=1 → 4 image tokens.
        let imgId = 999
        let r = QwenRopeIndexer.computeRaw(
            inputIds: [10, imgId, imgId, imgId, imgId, 11],
            imageGridTHW: [(t: 1, h: 2, w: 2)],
            imageTokenId: imgId,
            spatialMergeSize: 1
        )
        // Leading text token gets pos 0 on all axes; nextStart advances to 1.
        // Image block: T axis [1,1,1,1] (llmT=1), H [1,1,2,2], W [1,2,1,2]; nextStart += max(1,2,2)=2.
        // Trailing text token: 3 on all axes.
        XCTAssertEqual(r.t, [0, 1, 1, 1, 1, 3])
        XCTAssertEqual(r.h, [0, 1, 1, 2, 2, 3])
        XCTAssertEqual(r.w, [0, 1, 2, 1, 2, 3])
        XCTAssertEqual(r.ropeDelta, 3 + 1 - 6)  // -2
    }

    func testSpatialMergeSizeReducesGrid() {
        // Image grid (1, 4, 4) with merge=2 → llmH=2, llmW=2 → 4 LM tokens.
        let imgId = 7
        let ids = [imgId, imgId, imgId, imgId]
        let r = QwenRopeIndexer.computeRaw(
            inputIds: ids,
            imageGridTHW: [(t: 1, h: 4, w: 4)],
            imageTokenId: imgId,
            spatialMergeSize: 2
        )
        XCTAssertEqual(r.t, [0, 0, 0, 0])
        XCTAssertEqual(r.h, [0, 0, 1, 1])
        XCTAssertEqual(r.w, [0, 1, 0, 1])
    }
}
