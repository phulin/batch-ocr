import Accelerate
import CoreGraphics
import Foundation
import ImageIO

/// Preprocesses images for MinerU 2.5 Pro (Qwen2-VL family).
///
/// Two entry points:
///   - `prepareForLayout(_:)`  — stage-1: bicubic resize to 1036×1036, no normalization.
///   - `process(_:)`           — full Qwen2-VL preprocessing: smart_resize → CHW Float32 →
///                               normalize with OpenAI CLIP mean/std → tile to
///                               `[grid_h*grid_w, 3*2*14*14]` patches plus `image_grid_thw`.
public struct ImageProcessor: Sendable {
    public struct Constants {
        public static let patchSize = 14
        public static let mergeSize = 2
        public static let temporalPatchSize = 2
        public static let factor = patchSize * mergeSize  // 28
        public static let minPixels = 4 * factor * factor       // 3136
        public static let maxPixels = 16384 * factor * factor   // ~12.8M
        public static let layoutSize = 1036                     // MinerU stage-1
        // OpenAI CLIP normalization (Qwen2-VL inherits these).
        public static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
        public static let std:  [Float] = [0.26862954, 0.26130258, 0.27577711]
    }

    public init() {}

    public struct Processed: Sendable {
        /// Flattened patches `[gridH * gridW, 3 * 2 * 14 * 14]` in CHW-by-patch layout.
        public var pixelValues: [Float]
        /// `[t, h, w]` patch grid (single image: t = 1).
        public var gridTHW: [Int]

        public var sequenceLength: Int { gridTHW[0] * gridTHW[1] * gridTHW[2] }
    }

    // MARK: smart_resize

    /// Mirrors HF transformers `Qwen2VLImageProcessor.smart_resize`.
    /// Rounds to multiples of `factor` and clamps by min/max pixels (preserving aspect).
    public static func smartResize(
        height: Int,
        width: Int,
        factor: Int = Constants.factor,
        minPixels: Int = Constants.minPixels,
        maxPixels: Int = Constants.maxPixels
    ) -> (height: Int, width: Int) {
        precondition(height > 0 && width > 0, "smartResize requires positive dimensions")
        let edgeRatio = Double(max(height, width)) / Double(min(height, width))
        precondition(edgeRatio <= 200, "smart_resize aspect ratio too extreme: \(edgeRatio)")

        var hBar = max(factor, Int((Double(height) / Double(factor)).rounded()) * factor)
        var wBar = max(factor, Int((Double(width)  / Double(factor)).rounded()) * factor)

        if hBar * wBar > maxPixels {
            let beta = (Double(height) * Double(width) / Double(maxPixels)).squareRoot()
            hBar = Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor
            wBar = Int((Double(width)  / beta / Double(factor)).rounded(.down)) * factor
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (Double(height) * Double(width))).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width)  * beta / Double(factor)).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    // MARK: stage-1 layout preparation

    /// Resize `image` to the fixed layout-pass size with bicubic-ish interpolation.
    /// NOTE: CoreGraphics interpolation differs subtly from PIL BICUBIC; Phase C verifies parity.
    public func prepareForLayout(_ image: CGImage) -> CGImage? {
        return resize(image, to: CGSize(width: Constants.layoutSize, height: Constants.layoutSize))
    }

    // MARK: full processing

    /// Pipeline: smart_resize → render to RGB Float32 → normalize → tile into patches.
    public func process(_ image: CGImage) throws -> Processed {
        let (h, w) = Self.smartResize(height: image.height, width: image.width)
        guard let resized = resize(image, to: CGSize(width: w, height: h)) else {
            throw ProcessorError.resizeFailed
        }
        let chw = try renderRGBFloat32CHW(resized)            // [3, h, w] in 0…1
        let normalized = normalize(chw, height: h, width: w)  // CLIP mean/std
        let (patches, gridTHW) = tile(chw: normalized, height: h, width: w)
        return Processed(pixelValues: patches, gridTHW: gridTHW)
    }

    public enum ProcessorError: Error {
        case resizeFailed
        case renderFailed
    }

    // MARK: implementation helpers

    private func resize(_ image: CGImage, to size: CGSize) -> CGImage? {
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(origin: .zero, size: size))
        return ctx.makeImage()
    }

    /// Rasterize to planar Float32 `[3, h, w]` in `0…1`.
    private func renderRGBFloat32CHW(_ image: CGImage) throws -> [Float] {
        let h = image.height, w = image.width
        let cs = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = w * 4
        var bytes = [UInt8](repeating: 0, count: h * bytesPerRow)
        guard let ctx = bytes.withUnsafeMutableBufferPointer({ buf -> CGContext? in
            CGContext(
                data: buf.baseAddress,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                    | CGBitmapInfo.byteOrder32Big.rawValue
            )
        }) else { throw ProcessorError.renderFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        var out = [Float](repeating: 0, count: 3 * h * w)
        let inv: Float = 1.0 / 255.0
        for y in 0..<h {
            for x in 0..<w {
                let p = y * bytesPerRow + x * 4
                let r = Float(bytes[p]) * inv
                let g = Float(bytes[p + 1]) * inv
                let b = Float(bytes[p + 2]) * inv
                let i = y * w + x
                out[0 * h * w + i] = r
                out[1 * h * w + i] = g
                out[2 * h * w + i] = b
            }
        }
        return out
    }

    /// In-place style normalize: `(x - mean) / std` per channel, returns a new buffer.
    private func normalize(_ chw: [Float], height: Int, width: Int) -> [Float] {
        var out = chw
        let plane = height * width
        for c in 0..<3 {
            var negMean = -Constants.mean[c]
            var invStd = 1.0 / Constants.std[c]
            out.withUnsafeMutableBufferPointer { buf in
                let base = buf.baseAddress!.advanced(by: c * plane)
                vDSP_vsadd(base, 1, &negMean, base, 1, vDSP_Length(plane))
                vDSP_vsmul(base, 1, &invStd, base, 1, vDSP_Length(plane))
            }
        }
        return out
    }

    /// Tile CHW pixels into Qwen2-VL patches.
    /// Input:  `[3, h, w]` Float32. Output: `[gridH*gridW, 3 * 2 * 14 * 14]` (single-image t=1).
    /// Layout matches the HF processor's reshape+transpose chain, with the temporal patch
    /// dimension achieved by replicating the single frame.
    private func tile(chw: [Float], height h: Int, width w: Int) -> (patches: [Float], grid: [Int]) {
        let patch = Constants.patchSize
        let merge = Constants.mergeSize
        let temporal = Constants.temporalPatchSize
        precondition(h % patch == 0 && w % patch == 0, "smart_resize should make h,w divisible by 14")
        let gridH = h / patch
        let gridW = w / patch
        precondition(gridH % merge == 0 && gridW % merge == 0,
                     "smart_resize should make h,w divisible by 28")

        // HF reshape:
        //   [grid_t, temporal_patch, C,
        //    grid_h/merge, merge, patch,
        //    grid_w/merge, merge, patch]
        // transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        // → flatten to [grid_t*grid_h*grid_w, C*temporal_patch*patch*patch]
        //
        // grid_t = 1 (single image with T=2 frames consumed by temporal_patch=2).
        let outerH = gridH / merge
        let outerW = gridW / merge
        let perPatchValues = 3 * temporal * patch * patch        // 1176
        let numTokens = gridH * gridW
        var out = [Float](repeating: 0, count: numTokens * perPatchValues)
        let plane = h * w

        // For each LM image token (post-merge order):
        for ohIdx in 0..<outerH {
            for owIdx in 0..<outerW {
                for mhIdx in 0..<merge {
                    for mwIdx in 0..<merge {
                        let gh = ohIdx * merge + mhIdx
                        let gw = owIdx * merge + mwIdx
                        // The "row" in the post-merge flattening:
                        //   token = ((oh*outerW + ow) * merge + mh) * merge + mw
                        // and after flatten with grid_t*grid_h*grid_w that == gh*gridW+gw
                        // when traversed in (oh, ow, mh, mw) order? Actually the HF transpose
                        // chain produces token-major order over (grid_t, oh, ow, mh, mw).
                        // For grid_t = 1, that's (oh, ow, mh, mw). Compute index that way:
                        let tokenIdx = ((ohIdx * outerW + owIdx) * merge + mhIdx) * merge + mwIdx
                        let dstBase = tokenIdx * perPatchValues
                        // Inner stride: C, temporal_patch, patch, patch
                        for c in 0..<3 {
                            for t in 0..<temporal {
                                _ = t // single image: temporal frames are identical
                                for py in 0..<patch {
                                    for px in 0..<patch {
                                        let y = gh * patch + py
                                        let x = gw * patch + px
                                        let srcIdx = c * plane + y * w + x
                                        let innerOffset = ((c * temporal + t) * patch + py) * patch + px
                                        out[dstBase + innerOffset] = chw[srcIdx]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return (out, [1, gridH, gridW])
    }
}
