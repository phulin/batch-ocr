import Foundation
import MLX
import MLXNN
import MLXFast

/// NaViT-style ViT-L/14 vision encoder for MinerU 2.5 Pro / Qwen2-VL.
/// Source-of-truth: refs/mlx-vlm/mlx_vlm/models/qwen2_vl/vision.py.
///
/// Input  : flattened patches `[seq, 3*2*14*14] = [seq, 1176]` from `ImageProcessor.process`,
///          plus `image_grid_thw [t, h, w]`.
/// Output : `[seq/4, hiddenSize=896]` LM-ready image tokens after PatchMerger
///          (4× spatial token reduction via merge_size=2).

// MARK: 2D rotary pos emb

func rotateHalfVision(_ x: MLXArray) -> MLXArray {
    let last = x.dim(x.ndim - 1)
    let x1 = x[.ellipsis, 0..<(last / 2)]
    let x2 = x[.ellipsis, (last / 2)..<last]
    return concatenated([-x2, x1], axis: x.ndim - 1)
}

/// Apply 2D vision rotary embedding to q or k of shape `[seq, heads, headDim]`.
/// `freqs` shape `[seq, headDim/2]`. Mirrors `apply_rotary_pos_emb_vision`.
func applyVisionRoPE(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
    let dtype = tensor.dtype
    let cos = MLX.cos(freqs)
    let sin = MLX.sin(freqs)
    // freqs: [seq, dim/2] → cos/sin tiled to [seq, dim] then expanded to [1, seq, 1, dim]
    let cosTiled = tiled(cos.expandedDimensions(axis: 1), repetitions: [1, 1, 2])
        .expandedDimensions(axis: 0)
    let sinTiled = tiled(sin.expandedDimensions(axis: 1), repetitions: [1, 1, 2])
        .expandedDimensions(axis: 0)
    let out = (tensor * cosTiled) + (rotateHalfVision(tensor) * sinTiled)
    return out.asType(dtype)
}

final class VisionRotaryEmbedding {
    let dim: Int
    let theta: Float
    init(dim: Int, theta: Float = 10_000) {
        self.dim = dim; self.theta = theta
    }
    /// Returns `[seqlen, dim/2]` raw frequency table indexed by spatial position.
    func compute(seqlen: Int) -> MLXArray {
        let pairs = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
            .asType(.float32)
        let invFreq = MLXArray(1.0) / pow(MLXArray(theta), pairs / Float(dim))
        let seq = MLXArray((0..<seqlen).map { Float($0) })
        return outer(seq, invFreq)
    }
}

// MARK: PatchEmbed

final class PatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let embedDim: Int

    init(_ cfg: MinerUConfig.VisionConfig) {
        self.patchSize = cfg.patchSize
        self.temporalPatchSize = cfg.temporalPatchSize
        self.inChannels = cfg.inChannels
        self.embedDim = cfg.embedDim
        self._proj.wrappedValue = Conv3d(
            inputChannels: cfg.inChannels,
            outputChannels: cfg.embedDim,
            kernelSize: IntOrTriple((cfg.temporalPatchSize, cfg.patchSize, cfg.patchSize)),
            stride: IntOrTriple((cfg.temporalPatchSize, cfg.patchSize, cfg.patchSize)),
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: [seq, C * T * H * W] from the image processor.
        // Reshape to NDHWC for Conv3d with N=seq, D=T, H=H, W=W, C=C, then squeeze the
        // 1×1×1 spatial output back to [seq, embedDim].
        let r = x.reshaped(-1, inChannels, temporalPatchSize, patchSize, patchSize)
            .movedAxis(source: 1, destination: 4)  // → [seq, T, H, W, C]
        return proj(r).reshaped(-1, embedDim)
    }
}

// MARK: PatchMerger (vision → LM dim)

final class PatchMerger: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "ln_q") var lnQ: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: [Module]   // [Linear, GELU(act-only), Linear]

    init(dim: Int, contextDim: Int, spatialMergeSize: Int) {
        self.hiddenSize = contextDim * spatialMergeSize * spatialMergeSize
        self._lnQ.wrappedValue = LayerNorm(dimensions: contextDim, eps: 1e-6)
        self._mlp.wrappedValue = [
            Linear(self.hiddenSize, self.hiddenSize),
            GELU(),
            Linear(self.hiddenSize, dim),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input [seq, contextDim] → LN → reshape so adjacent merge×merge tokens become channels →
        // [seq/(merge^2), contextDim*merge^2] → MLP → [seq/(merge^2), dim].
        var h = lnQ(x).reshaped(-1, hiddenSize)
        for layer in mlp {
            if let lin = layer as? Linear { h = lin(h) }
            else if let gelu = layer as? GELU { h = gelu(h) }
            else { fatalError("PatchMerger.mlp slot of unexpected type") }
        }
        return h
    }
}

// MARK: Vision attention (variable-length via cu_seqlens)

final class VisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(self.headDim).squareRoot()
        self._qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, cuSeqlens: [Int], rotaryFreqs: MLXArray) -> MLXArray {
        let seqLen = x.dim(0)
        // qkv: [seq, dim] → [seq, 3, heads, headDim] → [3, seq, heads, headDim]
        let merged = qkv(x)
            .reshaped(seqLen, 3, numHeads, headDim)
            .transposed(1, 0, 2, 3)
        let parts = merged.split(parts: 3, axis: 0)
        var q = parts[0]   // [1, seq, heads, headDim] → squeeze axis 0
        var k = parts[1]
        var v = parts[2]
        q = q.squeezed(axis: 0)
        k = k.squeezed(axis: 0)
        v = v.squeezed(axis: 0)

        // Apply 2D vision RoPE to q and k. Adds a leading batch axis, applies, removes it.
        q = applyVisionRoPE(q.expandedDimensions(axis: 0), freqs: rotaryFreqs).squeezed(axis: 0)
        k = applyVisionRoPE(k.expandedDimensions(axis: 0), freqs: rotaryFreqs).squeezed(axis: 0)

        // Permute to [heads, seq, headDim], add batch axis to [1, heads, seq, headDim] for SDPA.
        q = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
        k = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
        v = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

        // Variable-length attention: split along seq axis at cu_seqlens interior boundaries,
        // run SDPA per segment, concat. mlx-swift SDPA does not accept cu_seqlens directly.
        let interior = Array(cuSeqlens.dropFirst().dropLast())
        let qSplits = MLX.split(q, indices: interior, axis: 2)
        let kSplits = MLX.split(k, indices: interior, axis: 2)
        let vSplits = MLX.split(v, indices: interior, axis: 2)

        var outs: [MLXArray] = []
        outs.reserveCapacity(qSplits.count)
        for (qi, ki, vi) in zip(qSplits, zip(kSplits, vSplits)).map({ ($0.0, $0.1.0, $0.1.1) }) {
            outs.append(MLXFast.scaledDotProductAttention(
                queries: qi, keys: ki, values: vi, scale: scale, mask: nil
            ))
        }
        // Concatenate back along seq → [1, heads, seq, headDim] → [seq, heads*headDim].
        let cat = concatenated(outs, axis: 2)
        let merged2 = cat.transposed(0, 2, 1, 3).reshaped(seqLen, -1)
        return proj(merged2)
    }
}

// MARK: Vision MLP (fast GELU)

final class VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dim: Int, hidden: Int) {
        self._fc1.wrappedValue = Linear(dim, hidden, bias: true)
        self._fc2.wrappedValue = Linear(hidden, dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluFastApproximate(fc1(x)))
    }
}

// MARK: Block + top-level encoder

final class VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP

    init(_ cfg: MinerUConfig.VisionConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: cfg.embedDim, eps: 1e-6)
        self._norm2.wrappedValue = LayerNorm(dimensions: cfg.embedDim, eps: 1e-6)
        self._attn.wrappedValue = VisionAttention(dim: cfg.embedDim, numHeads: cfg.numHeads)
        self._mlp.wrappedValue = VisionMLP(dim: cfg.embedDim, hidden: Int(Double(cfg.embedDim) * cfg.mlpRatio))
    }

    func callAsFunction(_ x: MLXArray, cuSeqlens: [Int], rotaryFreqs: MLXArray) -> MLXArray {
        var h = x + attn(norm1(x), cuSeqlens: cuSeqlens, rotaryFreqs: rotaryFreqs)
        h = h + mlp(norm2(h))
        return h
    }
}

public final class VisionEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    let blocks: [VisionBlock]
    @ModuleInfo(key: "merger") var merger: PatchMerger
    let cfg: MinerUConfig.VisionConfig
    private let rotary: VisionRotaryEmbedding

    public init(_ cfg: MinerUConfig.VisionConfig) {
        self.cfg = cfg
        self._patchEmbed.wrappedValue = PatchEmbed(cfg)
        self.blocks = (0..<cfg.depth).map { _ in VisionBlock(cfg) }
        self._merger.wrappedValue = PatchMerger(
            dim: cfg.hiddenSize,
            contextDim: cfg.embedDim,
            spatialMergeSize: cfg.spatialMergeSize
        )
        self.rotary = VisionRotaryEmbedding(dim: (cfg.embedDim / cfg.numHeads) / 2)
    }

    /// Build per-token (h_pos, w_pos) lookup indices and gather rotary freqs.
    /// Returns `[totalSeq, headDim/2]` already-expanded freqs ready for `applyVisionRoPE`.
    public func rotaryPosEmb(gridTHW: [(t: Int, h: Int, w: Int)]) -> MLXArray {
        let merge = cfg.spatialMergeSize
        var allHPos: [Int32] = []
        var allWPos: [Int32] = []
        var maxGrid = 0

        for (t, h, w) in gridTHW {
            // Build merge-aware spatial position arrays. The reshape+transpose pattern in
            // the Python source effectively reorders so that within each merge×merge block
            // the four positions land contiguously, aligning with the patch merger.
            // Equivalent to producing the (h, w) pair sequence in token-major order.
            let outerH = h / merge
            let outerW = w / merge
            for oh in 0..<outerH {
                for ow in 0..<outerW {
                    for mh in 0..<merge {
                        for mw in 0..<merge {
                            let hp = oh * merge + mh
                            let wp = ow * merge + mw
                            // Repeat each (h_pos, w_pos) `t` times along time.
                            for _ in 0..<t {
                                allHPos.append(Int32(hp))
                                allWPos.append(Int32(wp))
                            }
                        }
                    }
                }
            }
            maxGrid = max(maxGrid, max(h, w))
        }

        let freqs = rotary.compute(seqlen: maxGrid)  // [maxGrid, dim/2]
        let hIdx = MLXArray(allHPos)
        let wIdx = MLXArray(allWPos)
        let hFreqs = freqs[hIdx]   // [seq, dim/2]
        let wFreqs = freqs[wIdx]
        // Stack along last axis to pair (h, w) freqs at adjacent indices, mirroring Python's
        //   stacked_pos_ids = mx.stack([hpos, wpos], axis=-1) → freqs[pos_ids] → reshape(-1)
        let stackedFreqs = stacked([hFreqs, wFreqs], axis: -1)  // [seq, dim/2, 2]
        return stackedFreqs.reshaped(stackedFreqs.dim(0), -1)   // [seq, dim]
    }

    /// Build `cu_seqlens` cumulative offsets including 0 prefix and final total.
    public static func cumulativeSeqlens(gridTHW: [(t: Int, h: Int, w: Int)]) -> [Int] {
        var out = [0]
        var total = 0
        for (t, h, w) in gridTHW {
            let perFrame = h * w
            for _ in 0..<t {
                total += perFrame
                out.append(total)
            }
        }
        return out
    }

    /// Forward pass.
    /// - Parameter pixelValues: `[seq, C*T*patch*patch]` Float32, output of `ImageProcessor.process`.
    /// - Parameter gridTHW: per-image `(t, h, w)` patch grids.
    /// - Returns: `[seq/(merge*merge), hiddenSize]` LM image tokens.
    public func callAsFunction(
        pixelValues: MLXArray,
        gridTHW: [(t: Int, h: Int, w: Int)]
    ) -> MLXArray {
        var h = patchEmbed(pixelValues)
        let rotaryFreqs = rotaryPosEmb(gridTHW: gridTHW)
        let cuSeqlens = Self.cumulativeSeqlens(gridTHW: gridTHW)

        for block in blocks {
            h = block(h, cuSeqlens: cuSeqlens, rotaryFreqs: rotaryFreqs)
        }
        return merger(h)
    }
}
