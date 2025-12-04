#!/usr/bin/env python3
"""CLI entrypoint for running layout extraction via a generic OCR service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from extractor.layout_processor import DEFAULT_TEMP_OUTPUT_DIR, LayoutProcessor
    from extractor.ocr_client import OCRServiceClient
except ModuleNotFoundError:  # pragma: no cover - script execution fallback
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from extractor.layout_processor import DEFAULT_TEMP_OUTPUT_DIR, LayoutProcessor
    from extractor.ocr_client import OCRServiceClient

__all__ = ["OCRServiceClient", "LayoutProcessor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run layout inference and table recognition via Surya OCR"
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:7877",
        help="Base URL for the OCR service.",
    )
    parser.add_argument(
        "--table_url",
        default="http://localhost:9675/get-table",
        help="Full URL for the table recognition API. Provide an empty string to disable.",
    )
    parser.add_argument(
        "--output_dir",
        help=(
            "Directory to store page-wise results (crops + annotated page). Defaults "
            f"to {DEFAULT_TEMP_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence progress logging while processing.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single image file")
    group.add_argument("--input_dir", help="Directory containing multiple images")
    return parser.parse_args()


def build_client(base_url: str, table_url: Optional[str]) -> OCRServiceClient:
    clean_table_url = table_url.strip() if table_url else None
    return OCRServiceClient(base_url=base_url, table_url=clean_table_url)


def build_processor(client: OCRServiceClient, output_dir: Optional[str], quiet: bool) -> LayoutProcessor:
    resolved_output = Path(output_dir).expanduser().resolve() if output_dir else None
    return LayoutProcessor(client=client, output_dir=resolved_output, verbose=not quiet)


def main() -> None:
    args = parse_args()
    client = build_client(args.base_url, args.table_url)
    processor = build_processor(client, args.output_dir, args.quiet)

    if args.image:
        processor.process_image(Path(args.image).expanduser().resolve())
    else:
        processor.process_folder(Path(args.input_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
