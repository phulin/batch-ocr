import ArgumentParser
import Foundation
import MinerU

@main
struct MinerUCLITool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mineru",
        abstract: "MinerU 2.5 Pro Swift port — runs the layout pass on an image and prints parsed blocks."
    )

    @Argument(help: "Image to OCR")
    var imagePath: String

    @Option(name: .shortAndLong, help: "HF id or local model path")
    var model: String = "opendatalab/MinerU2.5-Pro-2604-1.2B"

    @Flag(help: "Print raw model output instead of parsed JSON")
    var raw: Bool = false

    func run() async throws {
        FileHandle.standardError.write(Data("loading \(model)…\n".utf8))
        let pipeline = try await MinerUPipeline.load(from: model)
        let url = URL(fileURLWithPath: (imagePath as NSString).expandingTildeInPath)
        FileHandle.standardError.write(Data("running layout pass on \(url.lastPathComponent)…\n".utf8))
        let (text, blocks) = try pipeline.detectLayout(imageURL: url)

        if raw {
            print(text)
            return
        }

        struct OutputBlock: Encodable {
            let type: String
            let bbox: [Double]
            let angle: Int?
            let merge_prev: Bool
        }
        let out = blocks.map { b in
            OutputBlock(
                type: b.type,
                bbox: [b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2],
                angle: b.rotationDegrees,
                merge_prev: b.mergeWithPrevious
            )
        }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        print(String(data: try enc.encode(out), encoding: .utf8)!)
    }
}
