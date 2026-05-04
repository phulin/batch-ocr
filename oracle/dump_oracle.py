"""
Run MinerU 2.5 Pro via the official Python API and dump artifacts the Swift
port can compare against.

Stage 1 (this script): user-facing API only — captures the layout-pass output
text, parsed blocks JSON, and the resized 1036x1036 layout image.

Stage 2 (a follow-up script will add): direct vision_tower / LM-step-0 logits
dumps by bypassing the high-level client. Needed once Swift port has
component-level tests.

Usage:
    cd oracle
    uv sync
    uv run python dump_oracle.py ../page2_text.png --out fixtures/page2_text/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--model-path",
        default="opendatalab/MinerU2.5-Pro-2604-1.2B",
        help="HF id or local path; downloads to ~/.cache/huggingface on first run",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Imports deferred so --help works without heavy deps installed.
    from mineru_vl_utils import MinerUClient

    print(f"loading {args.model_path} (mlx-engine backend)")
    client = MinerUClient(backend="mlx-engine", model_path=args.model_path)

    image = Image.open(args.image).convert("RGB")
    print(f"input image: {image.size}")

    # 1. Save the resized 1036x1036 layout-pass image — Swift parity check for resize.
    layout_image = client.helper.prepare_for_layout(image)
    if isinstance(layout_image, Image.Image):
        layout_image.save(args.out / "layout_input.png")

    # 2. Run layout pass; capture raw text + parsed blocks.
    print("layout pass")
    blocks = list(client.layout_detect(image))

    # Re-call underlying _predict to capture the raw model text — high-level API hides it.
    layout_prompt = (
        client.helper.prompts.get("[layout]") or client.helper.prompts["[default]"]
    )
    layout_params = (
        client.helper.sampling_params.get("[layout]")
        or client.helper.sampling_params.get("[default]")
    )
    raw_output = client._predict(layout_image, layout_prompt, layout_params, None, None)
    (args.out / "layout_text.txt").write_text(raw_output.text)

    # 3. Serialize blocks. ContentBlock is a dict subclass.
    (args.out / "blocks.json").write_text(json.dumps([dict(b) for b in blocks], indent=2))

    print(f"oracle dumped to {args.out}: {len(blocks)} blocks")


if __name__ == "__main__":
    main()
