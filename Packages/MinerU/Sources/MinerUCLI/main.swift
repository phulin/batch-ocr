import ArgumentParser
import Foundation
import MinerU

@main
struct MinerUCLITool: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mineru",
        abstract: "MinerU 2.5 Pro Swift port — CLI driver."
    )

    @Argument(help: "Image to OCR")
    var imagePath: String?

    func run() throws {
        print("MinerU \(MinerU.version) — port WIP")
        if let imagePath {
            print("Would OCR: \(imagePath)")
        }
    }
}
