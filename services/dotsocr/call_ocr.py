#!/usr/bin/env python3
"""Small helper to call the dots.ocr /get-text endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import requests

DEFAULT_URL = "http://localhost:9667/get-text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send an image to dots.ocr and print the text.")
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="dots.ocr /get-text endpoint URL (default: http://localhost:9667/get-text).",
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Also print the raw JSON response.",
    )
    return parser.parse_args()


def extract_text(payload: Any) -> str:
    """Best-effort extraction of text content from the API response."""

    if not isinstance(payload, dict):
        return str(payload).strip()

    result = payload.get("result")
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "content", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value.strip()
        if "lines" in result and isinstance(result["lines"], list):
            lines = [str(item).strip() for item in result["lines"] if str(item).strip()]
            if lines:
                return "\n".join(lines)
    if isinstance(payload.get("results"), list):
        texts = [extract_text({"result": item}) for item in payload["results"]]
        return "\n".join(filter(None, texts))

    return ""


def call_ocr(url: str, image_path: Path) -> Tuple[str, dict]:
    """Send image to dots.ocr and return (text, full_response)."""

    image_path = image_path.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        response = requests.post(url, files=files)
    response.raise_for_status()
    data = response.json()
    text = extract_text(data)
    return text, data


def main() -> None:
    args = parse_args()
    text, raw = call_ocr(args.url, args.image)
    print(text)
    if args.show_json:
        print("\n--- raw response ---")
        print(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
