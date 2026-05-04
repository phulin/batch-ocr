import XCTest
@testable import MinerU

final class OutputParserTests: XCTestCase {
    func testSingleTextBlock() {
        let raw = "<|box_start|>100 200 800 250<|box_end|><|ref_start|>text<|ref_end|>Hello world"
        let blocks = MinerUOutputParser.parse(raw)
        XCTAssertEqual(blocks.count, 1)
        let b = blocks[0]
        XCTAssertEqual(b.type, "text")
        XCTAssertEqual(b.bbox.x1, 0.1, accuracy: 1e-9)
        XCTAssertEqual(b.bbox.y2, 0.25, accuracy: 1e-9)
        XCTAssertNil(b.rotationDegrees)
        XCTAssertFalse(b.mergeWithPrevious)
        XCTAssertEqual(b.rawTail, "Hello world")
    }

    func testMultipleBlocksWithRotationAndMerge() {
        let raw = """
        <|box_start|>10 10 100 30<|box_end|><|ref_start|>title<|ref_end|>First
        <|box_start|>10 40 100 60<|box_end|><|ref_start|>text<|ref_end|><|rotate_right|>txt_contd_tgt rotated body
        <|box_start|>10 70 100 90<|box_end|><|ref_start|>table<|ref_end|>tbl
        """
        let blocks = MinerUOutputParser.parse(raw)
        XCTAssertEqual(blocks.count, 3)
        XCTAssertEqual(blocks[0].type, "title")
        XCTAssertEqual(blocks[1].rotationDegrees, 90)
        XCTAssertTrue(blocks[1].mergeWithPrevious)
        XCTAssertEqual(blocks[2].type, "table")
    }

    func testInvalidBBoxIsDropped() {
        // x1 == x2 → zero width → drop.
        let raw = "<|box_start|>500 100 500 200<|box_end|><|ref_start|>text<|ref_end|>nope"
        XCTAssertEqual(MinerUOutputParser.parse(raw).count, 0)
    }

    func testOutOfRangeBBoxIsDropped() {
        let raw = "<|box_start|>0 0 1200 100<|box_end|><|ref_start|>text<|ref_end|>nope"
        XCTAssertEqual(MinerUOutputParser.parse(raw).count, 0)
    }

    func testInvertedAxesAreSwapped() {
        let raw = "<|box_start|>800 250 100 200<|box_end|><|ref_start|>text<|ref_end|>flipped"
        let blocks = MinerUOutputParser.parse(raw)
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].bbox.x1, 0.1, accuracy: 1e-9)
        XCTAssertEqual(blocks[0].bbox.x2, 0.8, accuracy: 1e-9)
    }
}
