#!/usr/bin/env python3
"""Submit a PDF to the submission API and print the returned markdown."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests

DEFAULT_API_URL = os.environ.get("SUBMISSION_API_URL", "http://localhost:8080/process")


def call_api(pdf_path: Path, api_url: str) -> Dict[str, Any]:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with pdf_path.open("rb") as handle:
        files = {"file": (pdf_path.name, handle, "application/pdf")}
        response = requests.post(api_url, files=files, timeout=600)
    response.raise_for_status()

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {exc}") from exc

    if payload.get("status") != "success":
        raise RuntimeError(f"API returned error status: {payload}")
    if "markdown" not in payload:
        raise RuntimeError("Response is missing 'markdown' field.")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the submission API with a PDF file.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file.")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Submission API endpoint (default: {DEFAULT_API_URL}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = call_api(args.pdf, args.api_url)
    except Exception as exc:  # pragma: no cover - network IO
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

    print(result["markdown"])


if __name__ == "__main__":
    main()
