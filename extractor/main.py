#!/usr/bin/env python3
"""Regularize PDF orientation, then run layout extraction via DotsOCR."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from paddleocr import DocImgOrientationClassification

import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from extractor.preprocess import build_orientation_map, regularize_pdf
from extractor.dotocr import build_client, build_processor
from extractor.extract_text import DEFAULT_SKIP_CATEGORIES, extract_markdown
from extractor.remove_stamp import remove_stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regularize a PDF by rotating pages so that orientation is 0°, then run DotsOCR "
            "layout extraction on the curated pages."
        )
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF that should be regularized.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Destination PDF path. Defaults to curated_<input_name>.pdf in the same folder.",
    )
    parser.add_argument(
        "--model-name",
        default="PP-LCNet_x1_0_doc_ori",
        help="PaddleOCR Doc orientation model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used when running orientation classification.",
    )
    parser.add_argument(
        "--skip-dotocr",
        action="store_true",
        help="Only regularize the PDF; skip running layout extraction.",
    )
    parser.add_argument(
        "--remove-stamp",
        action="store_true",
        help="Remove stamps from pages before running DotsOCR.",
    )
    parser.add_argument(
        "--pages-dir",
        type=Path,
        help="Directory where rendered page images are stored before running DotsOCR.",
    )
    parser.add_argument(
        "--dotocr-output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory to store DotsOCR outputs (annotated pages and crops).",
    )
    parser.add_argument(
        "--dotocr-base-url",
        default="http://localhost:9667",
        help="Base URL for the Surya OCR service.",
    )
    parser.add_argument(
        "--dotocr-table-url",
        default="http://localhost:9670/chandra/extract",
        help="Full URL for the table recognition API (empty string to disable).",
    )
    parser.add_argument(
        "--dotocr-quiet",
        action="store_true",
        help="Silence layout extraction logs.",
    )
    parser.add_argument(
        "--page-zoom",
        type=float,
        default=2.0,
        help="Scale factor when rendering PDF pages to images for DotsOCR.",
    )
    parser.add_argument(
        "--extract-markdown",
        action="store_true",
        help="Run Marker on crop images to build per-element files and a merged markdown document.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Destination for merged markdown when --extract-markdown is used. Defaults to document.md under the DotsOCR output root.",
    )
    return parser.parse_args()


def render_pdf_to_images(
    pdf_path: Path, output_dir: Path, zoom: float = 2.0, remove_stamps: bool = False
) -> list[Path]:
    """Render each page of a PDF to PNG images."""

    pdf_path = pdf_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for page in doc:
            image_path = output_dir / f"{pdf_path.stem}_{page.number + 1:03d}.png"
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            if remove_stamps:
                img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_no_stamp = remove_stamp(img_bgr)
                cv2.imwrite(str(image_path), img_no_stamp)
            else:
                pix.save(str(image_path))

            image_paths.append(image_path)
    return image_paths


def main() -> None:
    args = parse_args()
    input_pdf = args.pdf_path.expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")

    if args.output:
        output_pdf = args.output.expanduser().resolve()
    else:
        output_pdf = input_pdf.with_name(f"curated_{input_pdf.name}")

    classifier = DocImgOrientationClassification(model_name=args.model_name)
    results = classifier.predict(str(input_pdf), batch_size=args.batch_size)
    orientation_by_page = build_orientation_map(results)

    stats = regularize_pdf(input_pdf, output_pdf, orientation_by_page)
    print(
        f"Wrote {output_pdf} | rotated {stats['rotated_pages']} "
        f"of {stats['total_pages']} pages."
    )

    if args.skip_dotocr:
        if args.extract_markdown:
            raise SystemExit("--extract-markdown requires DotsOCR outputs; remove --skip-dotocr.")
        return

    pages_dir = (args.pages_dir or output_pdf.with_name(f"{output_pdf.stem}_pages")).resolve()
    images = render_pdf_to_images(
        output_pdf, pages_dir, zoom=args.page_zoom, remove_stamps=args.remove_stamp
    )
    if not images:
        print(f"No pages rendered from {output_pdf}; skipping DotsOCR.")
        return

    dotocr_output_dir = args.dotocr_output_dir.expanduser().resolve()
    if dotocr_output_dir.exists() and not dotocr_output_dir.is_dir():
        raise NotADirectoryError(
            f"DotsOCR output base exists and is not a directory: {dotocr_output_dir}"
        )

    client = build_client(args.dotocr_base_url, args.dotocr_table_url)
    processor = build_processor(client, dotocr_output_dir, args.dotocr_quiet)
    processor.process_folder(pages_dir)
    output_root = (processor.output_dir or dotocr_output_dir) / pages_dir.name
    print(f"DotsOCR results written under {output_root}.")

    if args.extract_markdown:
        markdown_output = args.markdown_output or (output_root / "document.md")
        extract_markdown(
            layout_root=output_root,
            output_path=markdown_output,
            skip_categories=DEFAULT_SKIP_CATEGORIES,
        )
        print(f"Merged markdown written to {markdown_output}")


if __name__ == "__main__":
    main()
