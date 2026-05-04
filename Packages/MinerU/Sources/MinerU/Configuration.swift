import Foundation

/// Mirrors the relevant fields of MinerU 2.5 Pro's HF config.json.
/// The model is a Qwen2-VL family VLM (vision NaViT ViT-L/14 + Qwen2 decoder).
public struct MinerUConfig: Codable, Sendable {
    public var text: TextConfig
    public var vision: VisionConfig
    public var imageTokenId: Int
    public var visionStartTokenId: Int?
    public var visionEndTokenId: Int?
    public var ropeTheta: Double
    public var mropeSection: [Int]
    public var tieWordEmbeddings: Bool

    public struct TextConfig: Codable, Sendable {
        public var hiddenSize: Int        // 896
        public var numHiddenLayers: Int   // 24
        public var numAttentionHeads: Int // 14
        public var numKeyValueHeads: Int  // 2 (GQA)
        public var intermediateSize: Int  // 4864
        public var vocabSize: Int         // 151936
        public var rmsNormEps: Double
        public var maxPositionEmbeddings: Int

        public var headDim: Int { hiddenSize / numAttentionHeads } // 64
    }

    public struct VisionConfig: Codable, Sendable {
        public var depth: Int             // 32
        public var embedDim: Int          // 1280
        public var numHeads: Int          // 16
        public var mlpRatio: Double       // 4.0
        public var patchSize: Int         // 14
        public var temporalPatchSize: Int // 2
        public var spatialMergeSize: Int  // 2
        public var inChannels: Int        // 3
        public var hiddenAct: String      // "quick_gelu"
        public var hiddenSize: Int        // 896 (LM input dim, target of patch merger)
    }
}
