"""
Run MinerU 2.5 Pro on a fixture image, dump every artifact the Swift port needs
to compare against:

  - prompt token IDs
  - first N generated token IDs (greedy)
  - vision encoder output tensor (.npy)
  - language model step-0 logits (.npy)
  - parsed [ContentBlock] JSON

Usage:
    uv run python dump_oracle.py <image> --out fixtures/<name>/

The Swift test target loads these artifacts and asserts parity at each stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model",
        default="opendatalab/MinerU2.5-Pro-2604-1.2B",
        help="HF id; will be downloaded to ~/.cache/huggingface",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Imports deferred so `--help` works without the heavy deps installed.
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.utils import load_config
    from mineru_vl_utils import MinerUClient

    print(f"loading {args.model}")
    model, processor = load(args.model)
    config = load_config(args.model)

    client = MinerUClient(model=model, processor=processor, config=config)

    # Layout pass — single image, MinerU's stage-1 prompt.
    print("layout pass")
    layout_text, layout_tokens = client.layout_with_tokens(args.image)
    (args.out / "layout_text.txt").write_text(layout_text)
    (args.out / "layout_tokens.json").write_text(json.dumps(layout_tokens))

    # Parsed blocks
    blocks = client.parse_layout(layout_text)
    (args.out / "blocks.json").write_text(
        json.dumps([b.__dict__ for b in blocks], indent=2)
    )

    # Vision encoder output for the resized page
    print("vision encoder dump")
    pixel_values, image_grid_thw = client.preprocess(args.image)
    np.save(args.out / "pixel_values.npy", np.array(pixel_values, copy=False))
    np.save(args.out / "image_grid_thw.npy", np.array(image_grid_thw, copy=False))

    vision_out = model.vision_tower(pixel_values, image_grid_thw)
    np.save(args.out / "vision_features.npy", np.array(vision_out, copy=False))

    print(f"oracle dumped to {args.out}")


if __name__ == "__main__":
    main()
