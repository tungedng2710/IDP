#!/usr/bin/env bash
set -euo pipefail

PDF_PATH=${1:-${PDF_PATH:-test_samples/mwg2025.pdf}}
OUTPUT_DIR=${2:-${OUTPUT_DIR:-outputs}}

python extractor/main.py "$PDF_PATH" "$OUTPUT_DIR"