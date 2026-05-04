import Foundation
import MLX
import MLXNN

/// Top-level MinerU 2.5 Pro / Qwen2-VL model: vision tower + language model + the splice that
/// inserts vision features into the LM's input embeddings at image_token_id positions.
public final class MinerUModel: Module {
    @ModuleInfo(key: "vision_tower") public var visionTower: VisionEncoder
    @ModuleInfo(key: "language_model") public var languageModel: QwenLanguageModel

    public let imageTokenId: Int
    public let spatialMergeSize: Int
    public let mropeSection: [Int]

    public init(_ config: MinerUConfig) {
        self.imageTokenId = config.imageTokenId
        self.spatialMergeSize = config.vision.spatialMergeSize
        self.mropeSection = config.mropeSection
        self._visionTower.wrappedValue = VisionEncoder(config.vision)
        self._languageModel.wrappedValue = QwenLanguageModel(
            config.text,
            ropeTheta: Float(config.ropeTheta),
            mropeSection: config.mropeSection,
            tieWordEmbeddings: config.tieWordEmbeddings
        )
    }

    /// Run the LM on an input that may contain image tokens. The vision features replace the
    /// LM input embeddings at every `imageTokenId` position; positional indices are built via
    /// `QwenRopeIndexer` from the input ids and image grid.
    /// Returns logits `[1, seqLen, vocab]`.
    public func callAsFunction(
        inputIds: [Int],
        pixelValues: MLXArray? = nil,
        gridTHW: [(t: Int, h: Int, w: Int)] = [],
        caches: [QwenKVCache?]? = nil
    ) -> MLXArray {
        let idsArr = MLXArray(inputIds.map { Int32($0) }).expandedDimensions(axis: 0)  // [1, L]
        let baseEmbeds = languageModel.model.embedTokens(idsArr)  // [1, L, D]

        var inputEmbeds = baseEmbeds
        if let pixelValues, !gridTHW.isEmpty {
            let visionFeatures = visionTower(pixelValues: pixelValues, gridTHW: gridTHW)  // [Nimg, D]
            inputEmbeds = spliceImageFeatures(
                inputIds: inputIds,
                inputEmbeds: baseEmbeds,
                features: visionFeatures
            )
        }

        let (positionIds, _) = QwenRopeIndexer.compute(
            inputIds: inputIds,
            imageGridTHW: gridTHW,
            imageTokenId: imageTokenId,
            spatialMergeSize: spatialMergeSize
        )

        return languageModel(
            inputs: nil,
            inputEmbeds: inputEmbeds,
            mask: nil,
            caches: caches,
            positionIds: positionIds
        )
    }

    /// Replace embeddings at `imageTokenId` positions with the corresponding vision feature row.
    /// `inputEmbeds`: `[1, L, D]`. `features`: `[N, D]` where N == number of image tokens.
    private func spliceImageFeatures(
        inputIds: [Int],
        inputEmbeds: MLXArray,
        features: MLXArray
    ) -> MLXArray {
        // Mostly Swift-side index work — we collect the seq positions of image tokens then write
        // feature rows into a copy of the embedding tensor. MLX-swift doesn't have an easy
        // scatter assignment so we rebuild via concatenation along seq.
        let imagePositions = inputIds.enumerated().compactMap { $0.element == imageTokenId ? $0.offset : nil }
        if imagePositions.isEmpty { return inputEmbeds }
        precondition(imagePositions.count == features.dim(0),
                     "image-token count (\(imagePositions.count)) ≠ feature row count (\(features.dim(0)))")

        // Build pieces alternating: [embed slice, feature row, embed slice, feature row, ...]
        var pieces: [MLXArray] = []
        var cursor = 0
        for (idx, pos) in imagePositions.enumerated() {
            if pos > cursor {
                pieces.append(inputEmbeds[0..., cursor..<pos, 0...])
            }
            // features[idx, :] reshaped to [1, 1, D]
            pieces.append(features[idx].expandedDimensions(axes: [0, 1]))
            cursor = pos + 1
        }
        if cursor < inputEmbeds.dim(1) {
            pieces.append(inputEmbeds[0..., cursor..<inputEmbeds.dim(1), 0...])
        }
        return concatenated(pieces, axis: 1)
    }
}

/// Apply the same key transforms mlx-vlm's `Model.sanitize` and the vision encoder's
/// `sanitize` perform on raw safetensors weights:
///   - rename top-level `model.…` → `language_model.model.…`
///   - rename top-level `lm_head` → `language_model.lm_head`
///   - rename `visual.…` → `vision_tower.…`
///   - drop unused `position_ids`
///   - transpose Conv3d weight from [C_out, C_in, kT, kH, kW] (PyTorch) to
///     [C_out, kT, kH, kW, C_in] (MLX) if needed.
public enum MinerUWeightSanitizer {
    public static func transformKey(_ key: String) -> String {
        var k = key
        if !k.contains("vision_tower") {
            k = k.replacingOccurrences(of: "visual", with: "vision_tower")
        }
        if !k.contains("language_model") {
            if k.hasPrefix("model.") {
                k = "language_model." + k
            } else if k.hasPrefix("lm_head") {
                k = "language_model." + k
            }
        }
        return k
    }

    public static func sanitizeKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("position_ids") { continue }
            let newKey = transformKey(key)
            out[newKey] = value
        }
        return out
    }
}
