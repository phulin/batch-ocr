# batch-ocr — Swift+MLX port of MinerU 2.5 Pro

Native macOS PDF OCR using a Swift port of [MinerU 2.5 Pro](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B). Single 1.2B Qwen2-VL model that does **layout + recognition** in two prompted passes — produces per-block bboxes natively, unlike PaddleOCR-VL.

## Why this model

- Apache-2.0, 1.2B params, SOTA 95.69 on OmniDocBench v1.6.
- **Native bboxes** via the layout pass — no separate detector needed.
- Output format: `<|box_start|>x y w h<|box_end|><|ref_start|>type<|ref_end|>...content...` repeated. Coords are `[0,1000]`.
- Two-step inference: (1) whole page → layout JSON, (2) crop each block → content recognition.

## Model spec (from HF config.json)

- **LM (Qwen2 decoder, ~0.5B)**: hidden_size 896, 24 layers, 14 heads, **GQA KV=2**, intermediate 4864, RoPE θ=1e6, head_dim 64, **M-RoPE** mrope_section=[8,12,12], vocab 151936.
- **Vision (NaViT ViT-L/14)**: depth 32, embed_dim 1280, 16 heads, patch 14, temporal_patch 2, spatial_merge 2, mlp_ratio 4, quick-gelu.
- Special tokens: `image_token_id=151655`, `<|box_start|>`, `<|box_end|>`, `<|ref_start|>`, `<|ref_end|>`, `<|rotate_{up,right,down,left}|>`, `<|image_pad|>`.

## Repo layout

- `BatchOCR/` — SwiftUI app (existing).
- `Packages/MinerU/` — standalone Swift package for the model port.
- `oracle/` — Python harness that runs MinerU 2.5 Pro and dumps per-stage tensors + parsed output. Used as ground truth for Swift unit tests.
- `refs/` — gitignored; clones of mlx-vlm, mineru-vl-utils, paddleocr-vl.swift, mlx-swift-examples, PaddleOCR.

## Porting plan (file-level)

### Phase A — infrastructure
- [x] Clone Python references: `mlx-vlm`, `mineru-vl-utils`.
- [x] Verify model = `qwen2_vl`, capture config.
- [ ] Swift package skeleton (`Packages/MinerU/Package.swift`) depending on `mlx-swift` + `swift-transformers`.
- [ ] Python oracle: `uv` venv with `mlx-vlm` + `mineru-vl-utils`, script that runs MinerU on a fixture and dumps:
  - tokenizer ids for the prompt + first 50 generated tokens (greedy)
  - vision encoder output tensor (per-image-patch features)
  - language model logits at step 0
  - parsed `[ContentBlock]` JSON
- [ ] Test fixture: a single-page PDF + rendered PNG at the size MinerU expects.

### Phase B — port (in dependency order)

Source-of-truth Python paths in parens.

1. **Configuration.swift** — `Config`, `VisionConfig`, `TextConfig` mirroring HF JSON. Port `mlx_compat._build_mlx_compatible_config` (flatten `text_config` to root). (`mlx_vlm/models/qwen2_vl/config.py`, `mineru-vl-utils/.../mlx_compat.py`)
2. **Tokenizer wiring** — use `swift-transformers` `AutoTokenizer.from(modelFolder:)`. Verify special tokens load from `added_tokens.json`.
3. **ImageProcessor.swift** — port HF `Qwen2VLImageProcessor`: `smart_resize` (round H,W to multiples of patch×merge=28; clamp by min/max pixels), normalize (OpenAI CLIP mean/std), tile to `[seq, channels·temporal·patch·patch]`, return `image_grid_thw [1,t,h,w]`. CoreImage BICUBIC differs from PIL — render via vImage or CGContext for parity. (`mlx_vlm/models/qwen2_vl/processing_qwen2_vl.py` + HF processor)
4. **VisionEncoder.swift** — NaViT ViT: `PatchEmbed` (Conv3d 14×14 + temporal=2), 32 transformer blocks with **2D rotary pos emb** + **variable-length attention via cu_seqlens** (split q/k/v at segment boundaries, per-segment SDPA — mlx-swift SDPA doesn't accept cu_seqlens), `PatchMerger` (LN + 2-layer MLP, 4× token reduction). (`mlx_vlm/models/qwen2_vl/vision.py`)
5. **LanguageModel.swift** — Qwen2 decoder with **M-RoPE**: implement `apply_mrope` slicing freqs along [T,H,W] axes per `mrope_section`; `get_rope_index` building 3D position_ids from `image_grid_thw` + image-token spans. KV cache must store `_rope_deltas` for incremental decode. GQA. (`mlx_vlm/models/qwen2_vl/language.py`)
6. **Model.swift** — top-level glue: vision encoder → patch merger → splice features at `image_token_id` positions in the LM input embeddings. (`mlx_vlm/models/qwen2_vl/qwen2_vl.py`)
7. **Generator.swift** — greedy (temp=0, top_k=1), KV cache, `noRepeatNgram(size:100)` mask helper for the layout pass. EOS list. Adapt scaffolding from `paddleocr-vl.swift/Sources/PaddleOCRVL/Generator.swift`.
8. **OutputParser.swift** — port `_layout_re` from `mineru_client.py` to `NSRegularExpression`; emit `[ContentBlock { type, bbox: CGRect, angle, text }]`.
9. **Pipeline.swift** — orchestrate two passes: (a) layout on resized page → parse blocks → (b) per-block crop+rotate → content prompt per type → assemble final document.
10. **WeightLoader.swift** — Hub snapshot via swift-transformers; tied-embedding fixup (mineru's `mlx_compat` logic).

### Phase C — accuracy harness
- [ ] Swift CLI `MinerUCLI` that takes an image, runs the model, prints raw tokens + parsed JSON.
- [ ] Compare against the Python oracle on the same fixture:
  - exact prompt-token equality (tokenizer parity)
  - vision encoder output tensor — max abs diff < 1e-2 in bf16
  - first 50 generated token IDs match exactly (greedy)
  - parsed bboxes match within ±2 in [0,1000]
- [ ] Iterate until parity, then expand to 5 fixtures spanning text/table/formula/chart/rotated.

### Phase D — wire into BatchOCR app
- [ ] Replace Vision-based OCR pipeline with MinerU.
- [ ] Author searchable PDF using parsed bboxes (already-built `SearchablePDFWriter` adapts).
- [ ] First-run model download UI.

## Risks (ranked)

1. **M-RoPE correctness** — three-axis RoPE with stateful `_rope_deltas` across decode steps is subtle. First unit test against Python golden cos/sin.
2. **Variable-length vision attention** — mlx-swift SDPA doesn't accept cu_seqlens; we loop per segment. Performance acceptable for one image at a time.
3. **smart_resize / image preprocessing parity** — PIL BICUBIC ≠ CoreImage BICUBIC. Use vImage or pre-rendered CGContext to match.
4. **Tied embeddings** — 1.2B model likely ties `lm_head` to `embed_tokens`; safetensors may omit `lm_head.weight`. Detect and fall back.
5. **Tokenizer edge cases** — the `<|placeholder|>` 2-pass replace dance (`processing_qwen2_vl.py:91-103`) must run on the raw chat string before tokenization.
6. **Conv3d weight layout** — mlx-swift Conv3d weight order may differ from Python; verify with shape-comparison after `model.load_weights`.

## Hardest 3 components

1. M-RoPE in `LanguageModel.swift` (~150 LOC of math, easy to silently break).
2. Variable-length attention loop in `VisionEncoder.swift`.
3. `smart_resize` + tiling in `ImageProcessor.swift`.

## References (cloned to `refs/`, gitignored)

- `refs/mlx-vlm/mlx_vlm/models/qwen2_vl/{vision,language,config,processing_qwen2_vl,qwen2_vl}.py` — primary porting source.
- `refs/mineru-vl-utils/mineru_vl_utils/{mineru_client.py,mlx_compat.py,vlm_client/mlx_client.py}` — pipeline + parser.
- `refs/paddleocr-vl.swift/Sources/PaddleOCRVL/` — structural template (NaViT VLM in Swift+MLX).
- `refs/mlx-swift-examples/` — mostly stripped upstream; not useful as reference here.
