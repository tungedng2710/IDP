## Extractor Utilities

The `extractor` package bundles several small tools that sit on top of Surya OCR
and PaddleOCR. The refactored layout stack is split across a few focused modules:

- `ocr_client.py` — thin HTTP client for Surya OCR + the standalone table API.
- `layout_processor.py` — converts layout detections into annotated pages and per-element crops.
- `dotocr.py` — CLI entrypoint that wires the pieces together.

This structure makes it easy to script against the pipeline or just invoke it from the command line.

### CLI Usage

```bash
python extractor/dotocr.py --image /path/to/page.png --output_dir /desired/output

python extractor/dotocr.py --input_dir /path/to/pages \
    --base_url http://localhost:9667 \
    --table_url http://localhost:9671/get-table
```

Key options:

- `--image` processes a single page.
- `--input_dir` iterates all supported image files inside a folder.
- `--output_dir` controls where annotated pages + crops are stored (defaults to `/tmp/dotocr_pages`).
- `--table_url` toggles table recognition; provide an empty string to skip HTML generation.
- `--quiet` suppresses progress logging.

### Programmatic Usage

```python
from pathlib import Path
from extractor import DotsOCRClient, LayoutProcessor

client = DotsOCRClient(base_url="http://localhost:9667")
processor = LayoutProcessor(client=client, output_dir=Path("./outputs"))
processor.process_folder(Path("./scans"))
```

### Output Structure

For each input document a parent folder is created at:

```
<output_dir>/<document_name>/pages/
```

Every page gets its own numbered subdirectory:

```
page_01/
    annotated.png
    crop_elements/
        text/
            page-01_0001.png
        table/
            page-01_0002.png
            page-01_0002.html
        ...
```

- `annotated.png` is the original page with layout boxes and category labels rendered via OpenCV.
- `crop_elements/<category>/` stores crops grouped by normalized category names, preserving multiple instances per class.
- Table crops trigger an extra recognition pass (via `--table_url`) and emit an `.html` file beside the corresponding PNG.

### Cropping Rules

- Non-table regions receive a fixed 10 px expansion before cropping; overflow areas are padded back with white pixels.
- Tables skip padding to match the detected bounds while still passing the crop through table recognition.
- Crop expansion settings can be customized by instantiating `LayoutProcessor` directly.

### End-to-end PDF → Markdown

`extractor/main.py` handles PDF rotation, page rendering, DotsOCR layout, and optional Markdown extraction (via Marker):

```bash
# Regularize orientation, render pages, run DotsOCR, then build a merged markdown
python extractor/main.py /path/to/input.pdf \
  --dotocr-base-url http://localhost:7877 \
  --dotocr-table-url http://localhost:9675/get-table \
  --extract-markdown \
  --markdown-output outputs/curated_input_pages/document.md
```

Key flags:
- `--skip-dotocr` stops after rotation (no crops/markdown).
- `--pages-dir` reuses an existing rendered pages folder instead of re-rendering.
- `--dotocr-output-dir` controls where the `pages/` tree is written.
- `--extract-markdown` converts every crop to Markdown beside the PNG and merges everything (tables use the emitted `.html`).
- `--markdown-output` sets the merged Markdown path (defaults to `document.md` under the DotsOCR output root).

### Standalone Markdown Extraction

If you already have DotsOCR crops, run the extractor directly:

```bash
python extractor/extract_text.py outputs/curated_mwg2025_pages/pages \
  --skip-category page_footer \
  --skip-category picture \
  --output outputs/curated_mwg2025_pages/pages/document.md
```

Behavior:
- Skips `page_footer` and `picture` by default; add `--skip-category <name>` to ignore more.
- For tables, copies the `.html` into the merged Markdown and does not OCR.
- For all other categories, runs Marker on each crop, writes `<crop>.md` next to the PNG, and merges content in page/element order.
