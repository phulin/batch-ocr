import XCTest
@testable import MinerU

final class MinerUTests: XCTestCase {
    func testVersion() {
        XCTAssertFalse(MinerU.version.isEmpty)
    }
}
