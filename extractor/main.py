#!/usr/bin/env python3
"""Rotate a PDF via PaddleOCR orientation classification and save layout crops."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF
from paddleocr import DocImgOrientationClassification

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from extractor.layout_processor import LayoutProcessor
from extractor.ocr_client import OCRServiceClient
from extractor.preprocess import build_orientation_map, regularize_pdf

DEFAULT_ORIENTATION_MODEL = "PP-LCNet_x1_0_doc_ori"
DEFAULT_OCR_BASE_URL = "http://localhost:9667"
ORIENTATION_BATCH_SIZE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rotate a PDF, then run layout detection and save the cropped elements."
    )
    parser.add_argument("pdf_path", type=Path, help="Input PDF to process.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the rotated PDF, rendered pages, and layout crops are stored.",
    )
    return parser.parse_args()


def render_pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Render a PDF to PNG images suitable for layout detection."""

    pdf_path = pdf_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []

    with fitz.open(pdf_path) as doc:
        for page in doc:
            image_path = output_dir / f"{pdf_path.stem}_{page.number + 1:03d}.png"
            pix = page.get_pixmap(alpha=False)
            pix_width, pix_height = pix.width, pix.height
            max_dim = max(pix_width, pix_height)
            scale = 1800 / max_dim if max_dim else 1.0
            matrix = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(image_path))
            image_paths.append(image_path)
    return image_paths


def main() -> None:
    args = parse_args()
    input_pdf = args.pdf_path.expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rotated_pdf = output_dir / f"{input_pdf.stem}_regularized.pdf"

    classifier = DocImgOrientationClassification(model_name=DEFAULT_ORIENTATION_MODEL)
    results = classifier.predict(str(input_pdf), batch_size=ORIENTATION_BATCH_SIZE)
    orientation_by_page = build_orientation_map(results)

    stats = regularize_pdf(input_pdf, rotated_pdf, orientation_by_page)
    print(
        f"Wrote {rotated_pdf} | rotated {stats['rotated_pages']} "
        f"of {stats['total_pages']} pages."
    )

    pages_dir = output_dir / f"{rotated_pdf.stem}_pages"
    images = render_pdf_to_images(rotated_pdf, pages_dir)
    if not images:
        print(f"No pages rendered from {rotated_pdf}; skipping layout extraction.")
        return

    # client = OCRServiceClient(base_url=DEFAULT_OCR_BASE_URL)
    # processor = LayoutProcessor(client=client, output_dir=output_dir, verbose=True)
    # processor.process_folder(pages_dir)
    # output_root = (processor.output_dir or output_dir) / pages_dir.name
    # print(f"Layout crops written under {output_root}.")


if __name__ == "__main__":
    main()
