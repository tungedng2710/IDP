#!/usr/bin/env bash
set -euo pipefail

PDF_PATH=${1:-${PDF_PATH:-/root/tungn197/IDP/test_samples/curated_mwg2025_pages_pdf/curated_mwg2025_020.pdf}}
MARKDOWN_OUTPUT=${2:-${MARKDOWN_OUTPUT:-/root/tungn197/IDP/outputs_test/final_document.md}}
DOT_OCR_BASE_URL=${DOT_OCR_BASE_URL:-http://localhost:9667}
DOT_OCR_TABLE_URL=${DOT_OCR_TABLE_URL:-http://localhost:9670/chandra/extract}

mkdir -p "$(dirname "$MARKDOWN_OUTPUT")"

python extractor/main.py "$PDF_PATH" \
  --dotocr-base-url "$DOT_OCR_BASE_URL" \
  --dotocr-table-url "$DOT_OCR_TABLE_URL" \
  --extract-markdown \
  --remove-stamp \
  --markdown-output "$MARKDOWN_OUTPUT"
