#!/usr/bin/env bash
set -euo pipefail

PDF_PATH=${1:-${PDF_PATH:-test_samples/mwg2025.pdf}}
MARKDOWN_OUTPUT=${2:-${MARKDOWN_OUTPUT:-}}
REMOVE_STAMP=${REMOVE_STAMP:-1}

cmd=(python extractor/main.py "$PDF_PATH" --extract-markdown)

if [[ "$REMOVE_STAMP" == "1" ]]; then
  cmd+=(--remove-stamp)
fi

if [[ -n "$MARKDOWN_OUTPUT" ]]; then
  mkdir -p "$(dirname "$MARKDOWN_OUTPUT")"
  cmd+=(--markdown-output "$MARKDOWN_OUTPUT")
fi

"${cmd[@]}"
