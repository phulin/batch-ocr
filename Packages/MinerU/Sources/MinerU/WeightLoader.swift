import Foundation
import MLX
import MLXNN
import Hub

public enum WeightLoaderError: Error {
    case noSafetensors(URL)
    case configMissing(URL)
    case configMalformed(URL, Error)
}

/// Hub-aware weight loader for MinerU 2.5 Pro (Qwen2-VL safetensors).
public enum MinerUWeightLoader {
    /// Resolve a model path: a local directory or a HuggingFace repo id.
    /// Downloads matching files (config + safetensors + tokenizer) into the user's HF cache.
    public static func resolveModel(_ pathOrId: String) async throws -> URL {
        let url = URL(fileURLWithPath: (pathOrId as NSString).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let hub = HubApi.shared
        let repo = Hub.Repo(id: pathOrId)
        return try await hub.snapshot(
            from: repo,
            matching: ["*.json", "*.safetensors", "tokenizer.*"]
        )
    }

    /// Parse the HF `config.json` (with `text_config` flattened to root).
    public static func loadConfig(from directory: URL) throws -> MinerUConfig {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else {
            throw WeightLoaderError.configMissing(url)
        }
        do {
            return try ConfigJSON.parse(data)
        } catch {
            throw WeightLoaderError.configMalformed(url, error)
        }
    }

    /// Load all safetensors files in `directory`, sanitize keys + Conv3d layout, and apply to `model`.
    public static func load(_ model: MinerUModel, from directory: URL) throws {
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw WeightLoaderError.noSafetensors(directory)
        }

        var raw: [String: MLXArray] = [:]
        for file in files {
            let part = try MLX.loadArrays(url: file)
            for (k, v) in part { raw[k] = v }
        }

        // Stage 1: rename keys (model.* → language_model.model.*, lm_head → language_model.lm_head,
        // visual → vision_tower; drop position_ids).
        let renamed = MinerUWeightSanitizer.sanitizeKeys(raw)

        // Stage 2: transpose Conv3d weight from PyTorch [out, in, kT, kH, kW] to MLX
        // [out, kT, kH, kW, in] when needed.
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in renamed {
            if key.hasSuffix("patch_embed.proj.weight") && value.ndim == 5 {
                let s = value.shape
                // PyTorch layout if axes 1 (C_in) is *not* last and the last three (kH, kW, ?) ≠ kernel match.
                // Simple heuristic mirrored from mlx-vlm: if shape[1] is small (== inChannels) and
                // shape[4] is not, transpose (0, 2, 3, 4, 1).
                if s[1] <= 4 && s[4] > 4 {
                    sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    continue
                }
            }
            sanitized[key] = value
        }

        // Stage 3: tied embeddings — if lm_head.weight is missing, mirror embed_tokens.weight.
        let lmHeadKey = "language_model.lm_head.weight"
        let embedKey = "language_model.model.embed_tokens.weight"
        if !model.languageModel.tieWordEmbeddings,
           sanitized[lmHeadKey] == nil,
           let embed = sanitized[embedKey] {
            sanitized[lmHeadKey] = embed
        }

        let params = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: params, verify: .noUnusedKeys)
    }
}

/// Tiny config.json parser that flattens `text_config.*` to root and tolerates extra fields.
enum ConfigJSON {
    static func parse(_ data: Data) throws -> MinerUConfig {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "MinerU", code: -1, userInfo: [NSLocalizedDescriptionKey: "config.json root not an object"])
        }

        // Merge text_config to root (text_config wins on conflict, mirroring mineru's mlx_compat).
        var flat = root
        if let textCfg = root["text_config"] as? [String: Any] {
            for (k, v) in textCfg { flat[k] = v }
        }

        func intF(_ key: String, default fallback: Int? = nil) -> Int {
            if let v = flat[key] as? Int { return v }
            if let v = flat[key] as? Double { return Int(v) }
            if let f = fallback { return f }
            preconditionFailure("config.json missing required Int field: \(key)")
        }
        func doubleF(_ key: String, default fallback: Double? = nil) -> Double {
            if let v = flat[key] as? Double { return v }
            if let v = flat[key] as? Int { return Double(v) }
            if let f = fallback { return f }
            preconditionFailure("config.json missing required Double field: \(key)")
        }

        let visionRaw = root["vision_config"] as? [String: Any] ?? [:]
        func vInt(_ k: String, default d: Int? = nil) -> Int {
            if let v = visionRaw[k] as? Int { return v }
            if let v = visionRaw[k] as? Double { return Int(v) }
            if let d { return d }
            preconditionFailure("vision_config missing required Int field: \(k)")
        }
        func vStr(_ k: String, default d: String) -> String {
            (visionRaw[k] as? String) ?? d
        }

        let mrope = ((flat["rope_scaling"] as? [String: Any])?["mrope_section"] as? [Any])?
            .compactMap { ($0 as? Int) ?? Int(($0 as? Double) ?? 0) } ?? [8, 12, 12]

        let text = MinerUConfig.TextConfig(
            hiddenSize: intF("hidden_size"),
            numHiddenLayers: intF("num_hidden_layers"),
            numAttentionHeads: intF("num_attention_heads"),
            numKeyValueHeads: intF("num_key_value_heads"),
            intermediateSize: intF("intermediate_size"),
            vocabSize: intF("vocab_size"),
            rmsNormEps: doubleF("rms_norm_eps", default: 1e-6),
            maxPositionEmbeddings: intF("max_position_embeddings", default: 32768)
        )
        let vision = MinerUConfig.VisionConfig(
            depth: vInt("depth"),
            embedDim: vInt("embed_dim"),
            numHeads: vInt("num_heads"),
            mlpRatio: (visionRaw["mlp_ratio"] as? Double) ?? 4.0,
            patchSize: vInt("patch_size"),
            temporalPatchSize: vInt("temporal_patch_size"),
            spatialMergeSize: vInt("spatial_merge_size"),
            inChannels: vInt("in_channels", default: 3),
            hiddenAct: vStr("hidden_act", default: "quick_gelu"),
            hiddenSize: text.hiddenSize  // patch merger maps vision → LM hidden_size.
        )
        return MinerUConfig(
            text: text,
            vision: vision,
            imageTokenId: intF("image_token_id"),
            visionStartTokenId: flat["vision_start_token_id"] as? Int,
            visionEndTokenId: flat["vision_end_token_id"] as? Int,
            ropeTheta: doubleF("rope_theta", default: 1_000_000),
            mropeSection: mrope,
            tieWordEmbeddings: (flat["tie_word_embeddings"] as? Bool) ?? false
        )
    }
}
