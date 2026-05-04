import CoreGraphics
import Foundation
import MinerU

struct TextLine: Sendable, Codable {
    var text: String
    var box: CGRect          // normalized 0…1 in image coords (origin top-left to match MinerU bbox)
    var type: String         // ContentBlock.type — text, title, table, etc.
}

struct OCRPage: Sendable, Codable {
    var index: Int
    var lines: [TextLine]
}

/// MinerU 2.5 Pro–backed OCR. Loads the model once via the actor and reuses it across calls.
actor OCRService {
    static let shared = OCRService()

    private var pipelineTask: Task<MinerUPipeline, Error>?

    /// Resolve the MinerU pipeline (loaded on first call, then reused).
    private func pipeline() async throws -> MinerUPipeline {
        if let task = pipelineTask {
            return try await task.value
        }
        let task = Task<MinerUPipeline, Error> {
            // Default to local HF cache; first run will fall back to download via Hub.
            let pw = getpwuid(getuid())
            let home = pw.flatMap { String(cString: $0.pointee.pw_dir) } ?? NSHomeDirectory()
            let snapshotsDir = "\(home)/.cache/huggingface/hub/models--opendatalab--MinerU2.5-Pro-2604-1.2B/snapshots"
            if let snap = (try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir))?.first {
                return try await MinerUPipeline.load(from: "\(snapshotsDir)/\(snap)")
            }
            return try await MinerUPipeline.load(from: "opendatalab/MinerU2.5-Pro-2604-1.2B")
        }
        pipelineTask = task
        return try await task.value
    }

    func recognize(_ image: CGImage) async throws -> [TextLine] {
        let p = try await pipeline()
        let blocks = try p.extract(image)
        return blocks.compactMap { block in
            guard let content = block.content, !content.isEmpty else { return nil }
            // Convert MinerU's top-left-origin normalized bbox into a CGRect.
            let rect = CGRect(
                x: block.bbox.x1,
                y: block.bbox.y1,
                width: block.bbox.x2 - block.bbox.x1,
                height: block.bbox.y2 - block.bbox.y1
            )
            return TextLine(text: content, box: rect, type: block.type)
        }
    }
}
