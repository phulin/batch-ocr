import Foundation
import MLX
import MLXNN
import Tokenizers

public struct GenerationResult: Sendable {
    public var tokens: [Int]
    public var text: String
}

/// Greedy generator (temp=0, top-k=1) with KV cache and incremental M-RoPE position tracking.
/// Optional no-repeat-ngram masking (size 100 for the layout pass, per MinerU's defaults).
public final class Generator {
    public let model: MinerUModel
    public let tokenizer: any Tokenizer
    public var stopTokenIds: Set<Int>

    public init(model: MinerUModel, tokenizer: any Tokenizer, stopTokenIds: Set<Int> = [151643, 151645]) {
        self.model = model
        self.tokenizer = tokenizer
        self.stopTokenIds = stopTokenIds
    }

    public func generate(
        inputIds: [Int],
        pixelValues: MLXArray? = nil,
        visionFeaturesOverride: (values: [Float], rows: Int, cols: Int)? = nil,
        gridTHW: [(t: Int, h: Int, w: Int)] = [],
        maxTokens: Int = 1024,
        noRepeatNgramSize: Int? = nil
    ) -> GenerationResult {
        let numLayers = model.languageModel.model.layers.count
        let caches: [QwenKVCache?] = (0..<numLayers).map { _ in QwenKVCache() }

        // Prefill — either via the vision tower, or by injecting precomputed features.
        let prefillLogits: MLXArray
        if let f = visionFeaturesOverride {
            let featTensor = MLXArray(f.values).reshaped([f.rows, f.cols])
                .asType(model.languageModel.model.embedTokens.weight.dtype)
            prefillLogits = model.callWithVisionFeatures(
                inputIds: inputIds,
                visionFeatures: featTensor,
                gridTHW: gridTHW,
                caches: caches
            )
        } else {
            prefillLogits = model(
                inputIds: inputIds,
                pixelValues: pixelValues,
                gridTHW: gridTHW,
                caches: caches
            )
        }

        // Last-token logits → first generated token.
        let lastLogits = prefillLogits[0, prefillLogits.dim(1) - 1]   // [vocab]
        var nextToken = Int(lastLogits.argMax(axis: -1).item(Int32.self))

        // Position bookkeeping: the LM has already attended to `prefillLen` tokens; subsequent
        // decode positions live one past the last text-axis position assigned during prefill.
        let raw = QwenRopeIndexer.computeRaw(
            inputIds: inputIds,
            imageGridTHW: gridTHW,
            imageTokenId: model.imageTokenId,
            spatialMergeSize: model.spatialMergeSize
        )
        var nextPos = Int(max(raw.t.last ?? 0, max(raw.h.last ?? 0, raw.w.last ?? 0))) + 1

        var generated: [Int] = []
        var ngramTracker = noRepeatNgramSize.map { NgramTracker(size: $0) }

        for _ in 0..<maxTokens {
            if Task.isCancelled { break }
            if stopTokenIds.contains(nextToken) { break }
            generated.append(nextToken)
            ngramTracker?.append(nextToken)

            // Decode step: feed one token at the next 3D position (t,h,w all = nextPos).
            let posArr = MLXArray([Int32(nextPos), Int32(nextPos), Int32(nextPos)])
                .reshaped(3, 1, 1)
            let oneToken = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)  // [1, 1]
            let embeds = model.languageModel.model.embedTokens(oneToken)             // [1, 1, D]
            let logits = model.languageModel(
                inputs: nil,
                inputEmbeds: embeds,
                mask: nil,
                caches: caches,
                positionIds: posArr
            )
            var stepLogits = logits[0, 0]  // [vocab]

            // Optional no-repeat-ngram mask
            if let banned = ngramTracker?.bannedTokens(), !banned.isEmpty {
                let mask = MLXArray.zeros([stepLogits.dim(0)], dtype: .float32)
                let bannedIdx = MLXArray(banned.map { Int32($0) })
                stepLogits = stepLogits + MLXArray(0)  // no-op to ensure float
                // Set logits at banned indices to -inf
                let neg = MLXArray(Array(repeating: Float(-Float.infinity), count: banned.count))
                stepLogits[bannedIdx] = neg
                _ = mask
            }

            nextToken = Int(stepLogits.argMax(axis: -1).item(Int32.self))
            nextPos += 1
        }

        let text = (try? tokenizer.decode(tokens: generated)) ?? ""
        return GenerationResult(tokens: generated, text: text)
    }
}

/// Tracks the most-recent generated tokens and reports tokens that would form a repeat n-gram.
final class NgramTracker {
    let size: Int
    private var history: [Int] = []
    private var seenNgrams: [[Int]: Set<Int>] = [:]   // prefix → set of follow-up tokens already seen

    init(size: Int) { self.size = size }

    func append(_ token: Int) {
        history.append(token)
        if history.count >= size {
            let start = history.count - size
            let prefix = Array(history[start..<(start + size - 1)])
            let next = history[history.count - 1]
            seenNgrams[prefix, default: []].insert(next)
        }
    }

    /// Tokens that, if generated next, would create an already-seen n-gram.
    func bannedTokens() -> Set<Int> {
        guard history.count >= size - 1 else { return [] }
        let start = history.count - (size - 1)
        let prefix = Array(history[start..<history.count])
        return seenNgrams[prefix] ?? []
    }
}
