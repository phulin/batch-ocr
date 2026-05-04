import CoreGraphics
import Foundation
import ImageIO
import MLX
import Tokenizers

public final class MinerUPipeline {
    public let model: MinerUModel
    public let tokenizer: any Tokenizer
    public let processor: ImageProcessor
    public let generator: Generator
    public let config: MinerUConfig

    public init(model: MinerUModel, tokenizer: any Tokenizer, config: MinerUConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.processor = ImageProcessor()
        self.config = config
        self.generator = Generator(model: model, tokenizer: tokenizer)
    }

    public static func load(
        from pathOrId: String = "opendatalab/MinerU2.5-Pro-2604-1.2B"
    ) async throws -> MinerUPipeline {
        let dir = try await MinerUWeightLoader.resolveModel(pathOrId)
        let config = try MinerUWeightLoader.loadConfig(from: dir)
        let model = MinerUModel(config)
        try MinerUWeightLoader.load(model, from: dir)
        let tokenizer = try await AutoTokenizer.from(modelFolder: dir)
        return MinerUPipeline(model: model, tokenizer: tokenizer, config: config)
    }

    /// Stage-1 layout pass driven by an externally-provided vision-feature tensor (skips
    /// our VisionEncoder). Used to bisect whether residual divergence lives in vision or LM.
    public func detectLayoutWithFeatures(
        gridT: Int,
        gridH: Int,
        gridW: Int,
        visionFeatures: [Float],
        visionFeatureRows: Int,
        visionFeatureCols: Int
    ) throws -> (text: String, blocks: [ContentBlock]) {
        let mergeSq = config.vision.spatialMergeSize * config.vision.spatialMergeSize
        let imageTokenCount = (gridT * gridH * gridW) / mergeSq
        precondition(visionFeatureRows == imageTokenCount,
                     "feature row count must match image_token count")

        let inputIds = try buildLayoutPromptIds(imageTokenCount: imageTokenCount)
        let result = generator.generate(
            inputIds: inputIds,
            visionFeaturesOverride: (
                values: visionFeatures,
                rows: visionFeatureRows,
                cols: visionFeatureCols
            ),
            gridTHW: [(t: gridT, h: gridH, w: gridW)],
            maxTokens: 1024,
            noRepeatNgramSize: 100
        )
        return (result.text, MinerUOutputParser.parse(result.text))
    }

    /// Stage-1 layout pass on the resized 1036x1036 image.
    /// Returns parsed `[ContentBlock]` plus the raw layout-text the model produced.
    public func detectLayout(_ image: CGImage) throws -> (text: String, blocks: [ContentBlock]) {
        guard let resized = processor.prepareForLayout(image) else {
            throw MinerUPipelineError.preprocessingFailed
        }
        let processed = try processor.process(resized)
        let pixelTensor = MLXArray(processed.pixelValues)
            .reshaped(processed.sequenceLength, -1)  // [seq, 1176]

        // image_grid_thw is single-image: (t=1, gridH, gridW) where gridH=gridW=74 for 1036/14.
        let grid = (t: processed.gridTHW[0], h: processed.gridTHW[1], w: processed.gridTHW[2])
        // LM image-token count = (gridH/merge) * (gridW/merge) per frame.
        let mergeSq = config.vision.spatialMergeSize * config.vision.spatialMergeSize
        let imageTokenCount = (grid.t * grid.h * grid.w) / mergeSq

        let inputIds = try buildLayoutPromptIds(imageTokenCount: imageTokenCount)
        let result = generator.generate(
            inputIds: inputIds,
            pixelValues: pixelTensor,
            gridTHW: [grid],
            maxTokens: 1024,
            noRepeatNgramSize: 100
        )
        return (result.text, MinerUOutputParser.parse(result.text))
    }

    /// Build the Qwen2-VL chat prompt for the layout-detection task. We tokenize the literal
    /// chat string and splice in `imageTokenCount` `<|image_pad|>` IDs at the image position —
    /// safer than relying on a Jinja template at runtime.
    private func buildLayoutPromptIds(imageTokenCount: Int) throws -> [Int] {
        let systemPrompt = "You are a helpful assistant."
        let layoutPrompt = "\nLayout Detection:"
        // Qwen2-VL chat format. The vision_start/end markers wrap the image_pad block.
        let pre =
            "<|im_start|>system\n\(systemPrompt)<|im_end|>\n" +
            "<|im_start|>user\n<|vision_start|>"
        let post = "<|vision_end|>\(layoutPrompt)<|im_end|>\n<|im_start|>assistant\n"

        let preIds = try tokenizer.encode(text: pre)
        let postIds = try tokenizer.encode(text: post)
        let imagePadId = config.imageTokenId
        let imagePadIds = Array(repeating: imagePadId, count: imageTokenCount)
        return preIds + imagePadIds + postIds
    }
}

public enum MinerUPipelineError: Error {
    case preprocessingFailed
    case tokenizationFailed
}

public extension MinerUPipeline {
    /// Convenience: open an image file → CGImage → layout pass.
    func detectLayout(imageURL: URL) throws -> (text: String, blocks: [ContentBlock]) {
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw MinerUPipelineError.preprocessingFailed
        }
        return try detectLayout(cg)
    }

    /// Stage-2 recognition: crop each block from the source image, run the VLM with the
    /// type-specific prompt, populate `block.content`. Mutates and returns `blocks`.
    /// Honors task cancellation between blocks.
    func recognize(blocks: [ContentBlock], in source: CGImage) throws -> [ContentBlock] {
        var out = blocks
        for i in 0..<out.count {
            try Task.checkCancellation()
            let b = out[i]
            if Self.skipForRecognition(b.type) { continue }
            guard let crop = cropAndRotate(source, bbox: b.bbox, angleDegrees: b.rotationDegrees) else {
                continue
            }
            out[i].content = try recognizeContent(crop: crop, type: b.type)
        }
        return out
    }

    /// Full pipeline: layout → recognition.
    func extract(_ image: CGImage) throws -> [ContentBlock] {
        let (_, blocks) = try detectLayout(image)
        return try recognize(blocks: blocks, in: image)
    }

    // MARK: helpers

    /// Block types we never run recognition on (mirrors mineru_client `skip_list`).
    static func skipForRecognition(_ type: String) -> Bool {
        ["list", "equation_block", "image_block", "page_number", "header", "footer",
         "image", "chart", "page_footnote"].contains(type)
    }

    /// Recognition prompt per block type (mirrors mineru's DEFAULT_PROMPTS).
    private static let prompts: [String: String] = [
        "table": "\nTable Recognition:",
        "equation": "\nFormula Recognition:",
        "image": "\nImage Analysis:",
        "chart": "\nImage Analysis:",
    ]
    private static let defaultPrompt = "\nText Recognition:"

    private func recognizeContent(crop: CGImage, type: String) throws -> String {
        let processed = try processor.process(crop)
        let pixelTensor = MLXArray(processed.pixelValues).reshaped(processed.sequenceLength, -1)
        let grid = (t: processed.gridTHW[0], h: processed.gridTHW[1], w: processed.gridTHW[2])
        let mergeSq = config.vision.spatialMergeSize * config.vision.spatialMergeSize
        let imageTokenCount = (grid.t * grid.h * grid.w) / mergeSq
        let prompt = Self.prompts[type] ?? Self.defaultPrompt
        let inputIds = try buildContentPromptIds(imageTokenCount: imageTokenCount, recognitionPrompt: prompt)
        let result = generator.generate(
            inputIds: inputIds,
            pixelValues: pixelTensor,
            gridTHW: [grid],
            maxTokens: 4096,
            // Layout pass uses no_repeat=100 to avoid loops; for content extraction MinerU's
            // defaults set per-type frequency_penalty but we keep it simple with no n-gram filter
            // to allow legitimate repetition (table rows, etc).
            noRepeatNgramSize: nil
        )
        return result.text
    }

    private func buildContentPromptIds(imageTokenCount: Int, recognitionPrompt: String) throws -> [Int] {
        let systemPrompt = "You are a helpful assistant."
        let pre =
            "<|im_start|>system\n\(systemPrompt)<|im_end|>\n" +
            "<|im_start|>user\n<|vision_start|>"
        let post = "<|vision_end|>\(recognitionPrompt)<|im_end|>\n<|im_start|>assistant\n"
        let preIds = try tokenizer.encode(text: pre)
        let postIds = try tokenizer.encode(text: post)
        let imagePadIds = Array(repeating: config.imageTokenId, count: imageTokenCount)
        return preIds + imagePadIds + postIds
    }

    /// Crop `source` to the normalized bbox in original-image coordinates, then rotate by
    /// `angleDegrees` if non-nil and not 0. Returns nil for degenerate crops.
    private func cropAndRotate(_ source: CGImage, bbox: ContentBlock.BBox, angleDegrees: Int?) -> CGImage? {
        let w = source.width, h = source.height
        let x1 = max(0, Int((bbox.x1 * Double(w)).rounded()))
        let y1 = max(0, Int((bbox.y1 * Double(h)).rounded()))
        let x2 = min(w, Int((bbox.x2 * Double(w)).rounded()))
        let y2 = min(h, Int((bbox.y2 * Double(h)).rounded()))
        guard x2 > x1, y2 > y1 else { return nil }
        let rect = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        guard let cropped = source.cropping(to: rect) else { return nil }
        guard let angle = angleDegrees, angle == 90 || angle == 180 || angle == 270 else {
            return cropped
        }
        return rotate(cropped, by: angle)
    }

    private func rotate(_ image: CGImage, by degrees: Int) -> CGImage? {
        let radians = CGFloat(degrees) * .pi / 180
        let w = image.width, h = image.height
        let outW: Int, outH: Int
        switch degrees {
        case 90, 270: outW = h; outH = w
        default:      outW = w; outH = h
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: outW, height: outH,
            bitsPerComponent: 8, bytesPerRow: 0, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.translateBy(x: CGFloat(outW) / 2, y: CGFloat(outH) / 2)
        ctx.rotate(by: radians)
        ctx.translateBy(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2)
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }
}
