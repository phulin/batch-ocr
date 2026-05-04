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
}
