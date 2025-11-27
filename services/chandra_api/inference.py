from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from transformers import AutoModel, AutoProcessor

from chandra.model.hf import generate_hf
from chandra.model.schema import BatchInputItem
from chandra.output import parse_markdown

# Load the model once so repeated invocations reuse the same weights.
model = AutoModel.from_pretrained("datalab-to/chandra").cuda()
model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")


def load_image(path: str | Path) -> Image.Image:
    """Open an image from disk and normalize it to RGB."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as img:
        return img.convert("RGB")


def run_inference(image_path: str, prompt_type: str = "ocr_layout") -> str:
    """Run Chandra on the provided image and return the markdown output."""
    batch = [
    BatchInputItem(
        image=load_image(image_path),
        prompt_type="ocr_layout"
        )
    ]
    result = generate_hf(batch, model)[0]
    markdown = parse_markdown(result.raw)
    return parse_markdown(result.raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chandra inference on an image.")
    parser.add_argument("--image_path", help="Path to the image file.")
    parser.add_argument(
        "--prompt-type",
        default="ocr_layout",
        help="Prompt type to use for inference (default: ocr_layout).",
    )
    args = parser.parse_args()

    markdown = run_inference(args.image_path, args.prompt_type)
    print(markdown)


if __name__ == "__main__":
    main()
