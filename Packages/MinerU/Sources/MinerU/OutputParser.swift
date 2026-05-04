import Foundation

/// One block of parsed model output.
public struct ContentBlock: Sendable, Equatable {
    /// `[x1, y1, x2, y2]` normalized to 0...1 (model emits 0...1000 ints).
    public var bbox: BBox
    /// Block label as emitted by the model: `text`, `title`, `table`, `equation`, `image`, `chart`, ...
    public var type: String
    /// Rotation hint, in degrees. `nil` means no `<|rotate_*|>` token was emitted.
    public var rotationDegrees: Int?
    /// True when the block has the `txt_contd_tgt` flag — should be merged with the previous block.
    public var mergeWithPrevious: Bool
    /// Trailing content captured between `<|ref_end|>` and the next `<|box_start|>` (or end).
    public var rawTail: String
    /// Recognized content (set by stage-2 recognition pass; nil after stage-1 layout only).
    public var content: String?

    public struct BBox: Sendable, Equatable {
        public var x1, y1, x2, y2: Double
        public init(_ x1: Double, _ y1: Double, _ x2: Double, _ y2: Double) {
            self.x1 = x1; self.y1 = y1; self.x2 = x2; self.y2 = y2
        }
    }
}

public enum MinerUOutputParser {
    /// Mirrors `mineru_vl_utils/mineru_client.py:_layout_re` (DOTALL).
    private static let pattern: String = #"""
    <\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?(.*?)(?=<\|box_start\|>|$)
    """#

    private static let angleByToken: [Substring: Int] = [
        "<|rotate_up|>": 0,
        "<|rotate_right|>": 90,
        "<|rotate_down|>": 180,
        "<|rotate_left|>": 270,
    ]

    /// Parses the model's stage-1 layout output into block records.
    /// Returns blocks in source order; malformed bboxes (out of range or zero-area)
    /// are dropped — same as the Python helper `_convert_bbox`.
    public static func parse(_ output: String) -> [ContentBlock] {
        let regex: NSRegularExpression
        do {
            regex = try NSRegularExpression(
                pattern: pattern.trimmingCharacters(in: .whitespacesAndNewlines),
                options: [.dotMatchesLineSeparators]
            )
        } catch {
            assertionFailure("MinerU regex failed to compile: \(error)")
            return []
        }

        let ns = output as NSString
        var blocks: [ContentBlock] = []

        regex.enumerateMatches(
            in: output,
            options: [],
            range: NSRange(location: 0, length: ns.length)
        ) { match, _, _ in
            guard let match else { return }
            guard match.numberOfRanges == 8 else { return }

            func capture(_ i: Int) -> String {
                let r = match.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }

            guard
                let x1 = Int(capture(1)),
                let y1 = Int(capture(2)),
                let x2 = Int(capture(3)),
                let y2 = Int(capture(4))
            else { return }

            guard let bbox = normalizeBBox(x1: x1, y1: y1, x2: x2, y2: y2) else { return }

            let type = capture(5)
            let rotateToken = capture(6)
            let tail = capture(7)

            let angle: Int? = rotateToken.isEmpty ? nil : angleByToken[Substring(rotateToken)]

            blocks.append(
                ContentBlock(
                    bbox: bbox,
                    type: type,
                    rotationDegrees: angle,
                    mergeWithPrevious: tail.contains("txt_contd_tgt"),
                    rawTail: tail,
                    content: nil
                )
            )
        }

        return blocks
    }

    /// Mirrors `_convert_bbox`: clamp range, swap inverted axes, normalize to 0…1.
    static func normalizeBBox(x1: Int, y1: Int, x2: Int, y2: Int) -> ContentBlock.BBox? {
        let coords = [x1, y1, x2, y2]
        if coords.contains(where: { $0 < 0 || $0 > 1000 }) { return nil }
        let nx1 = min(x1, x2), nx2 = max(x1, x2)
        let ny1 = min(y1, y2), ny2 = max(y1, y2)
        if nx1 == nx2 || ny1 == ny2 { return nil }
        return .init(
            Double(nx1) / 1000.0,
            Double(ny1) / 1000.0,
            Double(nx2) / 1000.0,
            Double(ny2) / 1000.0
        )
    }
}
