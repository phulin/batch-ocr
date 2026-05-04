import Foundation
import MLX
import MLXNN
import MLXFast

/// Qwen2 decoder with M-RoPE — the language half of MinerU 2.5 Pro.
/// Source-of-truth: refs/mlx-vlm/mlx_vlm/models/qwen2_vl/language.py.
///
/// Structural notes:
/// - Standard pre-norm transformer with RMSNorm, GQA (14 q-heads, 2 kv-heads), SwiGLU MLP.
/// - RoPE is **M-RoPE**: position_ids are 3D (T, H, W) for image tokens, broadcast to (T,T,T)
///   for text tokens. Each axis owns a contiguous slice of the head_dim/2 frequency pairs as
///   defined by `mrope_section` (e.g. [8,12,12] for the 1.2B Pro variant — 8+12+12 = 32 = head_dim/2).
/// - Tied embeddings: `tie_word_embeddings` is null/false in MinerU's HF config, so `lm_head`
///   should be present in the safetensors. Loader will detect-and-fallback regardless.

// MARK: KV cache

public final class QwenKVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public var offset: Int = 0

    public init() {}

    public func updateAndFetch(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let k = keys, let v = values {
            self.keys = concatenated([k, newKeys], axis: 2)
            self.values = concatenated([v, newValues], axis: 2)
        } else {
            self.keys = newKeys
            self.values = newValues
        }
        offset += newKeys.dim(2)
        return (self.keys!, self.values!)
    }

    public func reset() {
        keys = nil; values = nil; offset = 0
    }
}

// MARK: M-RoPE

/// Builds cos/sin tensors for M-RoPE given 3D position ids of shape `(3, batch, seqLen)`.
/// Returns `(cos, sin)` of shape `(batch, seqLen, headDim)`.
public final class QwenRotaryEmbedding {
    public let headDim: Int
    public let base: Float
    public let mropeSection: [Int]
    private let invFreq: MLXArray

    public init(headDim: Int, base: Float, mropeSection: [Int]) {
        self.headDim = headDim
        self.base = base
        self.mropeSection = mropeSection
        precondition(mropeSection.reduce(0, +) == headDim / 2,
                     "mrope_section must sum to head_dim/2")
        // inv_freq = 1 / base^(arange(0, dim, 2) / dim)
        let pairs = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
            .asType(.float32)
        let exponent = pairs / Float(headDim)
        self.invFreq = MLXArray(1.0) / pow(MLXArray(base), exponent)
    }

    /// `positionIds`: shape `(3, batch, seqLen)` Int32.
    /// Returns `(cos, sin)` shape `(batch, seqLen, headDim)`.
    public func compute(positionIds: MLXArray, dtype: DType) -> (MLXArray, MLXArray) {
        precondition(positionIds.ndim == 3 && positionIds.dim(0) == 3,
                     "positionIds must be (3, batch, seqLen)")
        let seqLen = positionIds.dim(2)

        // freqs[axis] = invFreq @ positionIds[axis]  →  shape (batch, dim/2, seqLen) per axis
        // then transpose last two axes → (batch, seqLen, dim/2).
        var perAxis: [MLXArray] = []
        for axis in 0..<3 {
            let pos = positionIds[axis].asType(.float32)            // (batch, seqLen)
            // Broadcast invFreq[:, None] @ pos[None, :] per batch → (batch, dim/2, seqLen)
            let invF = self.invFreq.expandedDimensions(axes: [0, 2])  // (1, dim/2, 1)
            let p = pos.expandedDimensions(axes: [1])                  // (batch, 1, seqLen)
            var freqs = invF * p                                       // (batch, dim/2, seqLen)
            freqs = freqs.swappedAxes(1, 2)                            // (batch, seqLen, dim/2)
            perAxis.append(freqs)
        }

        // Apply mrope: take T-axis as base, overlay H/W slices on top.
        var merged = perAxis[0]
        var offset = mropeSection[0]
        for (idx, length) in mropeSection.dropFirst().enumerated() {
            let axis = idx + 1
            // splice: merged[..., offset..offset+length] = perAxis[axis][..., offset..offset+length]
            let prefix = merged[0..., 0..., 0..<offset]
            let middle = perAxis[axis][0..., 0..., offset..<(offset + length)]
            let suffix = merged[0..., 0..., (offset + length)..<merged.dim(2)]
            merged = concatenated([prefix, middle, suffix], axis: 2)
            offset += length
        }

        // Duplicate freqs along last axis (head_dim/2 → head_dim) before cos/sin.
        let emb = concatenated([merged, merged], axis: 2)
        let cos = MLX.cos(emb).asType(dtype)
        let sin = MLX.sin(emb).asType(dtype)
        _ = seqLen
        return (cos, sin)
    }
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let last = x.dim(x.ndim - 1)
    let x1 = x[.ellipsis, 0..<(last / 2)]
    let x2 = x[.ellipsis, (last / 2)..<last]
    return concatenated([-x2, x1], axis: x.ndim - 1)
}

/// Apply M-RoPE to query/key. `cos`/`sin` shape `(batch, seq, headDim)`; q/k shape `(batch, heads, seq, headDim)`.
public func applyMRope(q: MLXArray, k: MLXArray, cos cosA: MLXArray, sin sinA: MLXArray) -> (MLXArray, MLXArray) {
    let cosE = cosA.expandedDimensions(axes: [1])  // (batch, 1, seq, headDim)
    let sinE = sinA.expandedDimensions(axes: [1])
    let qOut = (q * cosE) + (rotateHalf(q) * sinE)
    let kOut = (k * cosE) + (rotateHalf(k) * sinE)
    return (qOut, kOut)
}

// MARK: Attention

public final class QwenAttention: Module {
    public let nHeads: Int
    public let nKVHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public let rotary: QwenRotaryEmbedding

    public init(_ cfg: MinerUConfig.TextConfig, ropeTheta: Float, mropeSection: [Int]) {
        self.nHeads = cfg.numAttentionHeads
        self.nKVHeads = cfg.numKeyValueHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / Float(headDim).squareRoot()
        self._qProj.wrappedValue = Linear(cfg.hiddenSize, nHeads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(nHeads * headDim, cfg.hiddenSize, bias: false)
        self.rotary = QwenRotaryEmbedding(headDim: headDim, base: ropeTheta, mropeSection: mropeSection)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: QwenKVCache? = nil,
        positionIds: MLXArray
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        var q = qProj(x).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)

        let (cos, sin) = rotary.compute(positionIds: positionIds, dtype: q.dtype)
        (q, k) = applyMRope(q: q, k: k, cos: cos, sin: sin)

        var keys = k, values = v
        if let cache {
            (keys, values) = cache.updateAndFetch(keys: k, values: v)
        }

        // Use auto-causal mode for LM autoregressive attention (Python's create_attention_mask).
        // For decode steps (L=1) we still want causal behavior so future-token logits don't leak.
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .causal
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: keys,
            values: values,
            scale: scale,
            mask: maskMode
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: MLP

public final class QwenMLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(_ dim: Int, _ hidden: Int) {
        self._gateProj.wrappedValue = Linear(dim, hidden, bias: false)
        self._upProj.wrappedValue = Linear(dim, hidden, bias: false)
        self._downProj.wrappedValue = Linear(hidden, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: Decoder layer

public final class QwenDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var attn: QwenAttention
    @ModuleInfo public var mlp: QwenMLP
    @ModuleInfo(key: "input_layernorm") public var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postNorm: RMSNorm

    public init(_ cfg: MinerUConfig.TextConfig, ropeTheta: Float, mropeSection: [Int]) {
        self._attn.wrappedValue = QwenAttention(cfg, ropeTheta: ropeTheta, mropeSection: mropeSection)
        self._mlp.wrappedValue = QwenMLP(cfg.hiddenSize, cfg.intermediateSize)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: Float(cfg.rmsNormEps))
        self._postNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: Float(cfg.rmsNormEps))
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        cache: QwenKVCache? = nil,
        positionIds: MLXArray
    ) -> MLXArray {
        let h = x + attn(inputNorm(x), mask: mask, cache: cache, positionIds: positionIds)
        return h + mlp(postNorm(h))
    }
}

// MARK: Qwen2 model + LM head

public final class Qwen2Model: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [QwenDecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    public init(_ cfg: MinerUConfig.TextConfig, ropeTheta: Float, mropeSection: [Int]) {
        self._embedTokens.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self.layers = (0..<cfg.numHiddenLayers).map { _ in
            QwenDecoderLayer(cfg, ropeTheta: ropeTheta, mropeSection: mropeSection)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: Float(cfg.rmsNormEps))
    }

    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        caches: [QwenKVCache?]? = nil,
        positionIds: MLXArray
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeds {
            h = inputEmbeds
        } else if let inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Qwen2Model requires either inputs or inputEmbeds")
        }

        let cs = caches ?? Array(repeating: nil, count: layers.count)
        for (layer, c) in zip(layers, cs) {
            h = layer(h, mask: mask, cache: c, positionIds: positionIds)
        }
        return norm(h)
    }
}

public final class QwenLanguageModel: Module {
    @ModuleInfo public var model: Qwen2Model
    @ModuleInfo(key: "lm_head") public var lmHead: Linear?
    public let tieWordEmbeddings: Bool

    public init(_ cfg: MinerUConfig.TextConfig, ropeTheta: Float, mropeSection: [Int], tieWordEmbeddings: Bool) {
        self._model.wrappedValue = Qwen2Model(cfg, ropeTheta: ropeTheta, mropeSection: mropeSection)
        self.tieWordEmbeddings = tieWordEmbeddings
        if tieWordEmbeddings {
            self._lmHead.wrappedValue = nil
        } else {
            self._lmHead.wrappedValue = Linear(cfg.hiddenSize, cfg.vocabSize, bias: false)
        }
    }

    public func callAsFunction(
        inputs: MLXArray? = nil,
        inputEmbeds: MLXArray? = nil,
        mask: MLXArray? = nil,
        caches: [QwenKVCache?]? = nil,
        positionIds: MLXArray
    ) -> MLXArray {
        let h = model(inputs: inputs, inputEmbeds: inputEmbeds, mask: mask, caches: caches, positionIds: positionIds)
        if let lmHead {
            return lmHead(h)
        }
        // Tied embeddings: project through transposed embed_tokens.
        return h.matmul(model.embedTokens.weight.T)
    }
}

// MARK: get_rope_index

/// Build M-RoPE 3D position ids from input token ids and image grids.
/// CPU-side prep — runs once per prompt; output is small and fed into `compute(positionIds:)`.
///
/// For a single image scenario (MinerU layout/recognition): we have one `<|image_pad|>` run of
/// length `gridT * gridH * gridW / merge^2` where the LM sees image tokens. Position assignment:
///   - text tokens before image: position_ids[T,H,W] = [k, k, k] with k incrementing.
///   - image tokens: nested grid coordinates, each axis incrementing within its dimension.
///   - text tokens after image: each axis incrementing past the image's max coordinate.
///
/// Returns `(positionIds, ropeDelta)` where positionIds is `(3, 1, seqLen)` and ropeDelta is the
/// position offset for incremental decoding (caller stores this and adds to subsequent positions).
public enum QwenRopeIndexer {
    /// Pure-Swift core. Returns three flat `[Int32]` axes (length seqLen each) plus the rope
    /// delta — the offset to add to subsequent decode positions so KV cache positions match.
    public static func computeRaw(
        inputIds: [Int],
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        imageTokenId: Int,
        spatialMergeSize: Int
    ) -> (t: [Int32], h: [Int32], w: [Int32], ropeDelta: Int) {
        let seqLen = inputIds.count
        if imageGridTHW.isEmpty {
            let pos = (0..<seqLen).map { Int32($0) }
            return (pos, pos, pos, 0)
        }
        var t = [Int32](); var h = [Int32](); var w = [Int32]()
        t.reserveCapacity(seqLen); h.reserveCapacity(seqLen); w.reserveCapacity(seqLen)
        var imgIdx = 0, i = 0
        var nextStart: Int32 = 0
        while i < seqLen {
            if inputIds[i] == imageTokenId, imgIdx < imageGridTHW.count {
                let g = imageGridTHW[imgIdx]
                let llmH = g.h / spatialMergeSize
                let llmW = g.w / spatialMergeSize
                let llmT = g.t
                let count = llmT * llmH * llmW
                for tt in 0..<llmT {
                    for hh in 0..<llmH {
                        for ww in 0..<llmW {
                            t.append(nextStart + Int32(tt))
                            h.append(nextStart + Int32(hh))
                            w.append(nextStart + Int32(ww))
                        }
                    }
                }
                nextStart += Int32(max(llmT, max(llmH, llmW)))
                i += count
                imgIdx += 1
            } else {
                t.append(nextStart); h.append(nextStart); w.append(nextStart)
                nextStart += 1
                i += 1
            }
        }
        let maxPos = max(t.last ?? 0, max(h.last ?? 0, w.last ?? 0))
        let ropeDelta = Int(maxPos) + 1 - seqLen
        return (t, h, w, ropeDelta)
    }

    /// MLX-backed wrapper. Position ids shape `(3, 1, seqLen)` Int32.
    public static func compute(
        inputIds: [Int],
        imageGridTHW: [(t: Int, h: Int, w: Int)],
        imageTokenId: Int,
        spatialMergeSize: Int
    ) -> (positionIds: MLXArray, ropeDelta: Int) {
        let raw = computeRaw(
            inputIds: inputIds,
            imageGridTHW: imageGridTHW,
            imageTokenId: imageTokenId,
            spatialMergeSize: spatialMergeSize
        )
        let arrT = MLXArray(raw.t).expandedDimensions(axes: [0])
        let arrH = MLXArray(raw.h).expandedDimensions(axes: [0])
        let arrW = MLXArray(raw.w).expandedDimensions(axes: [0])
        return (stacked([arrT, arrH, arrW], axis: 0), raw.ropeDelta)
    }
}
