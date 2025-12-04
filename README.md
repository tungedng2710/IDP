## Document Layout Extractor

This repository contains a stripped-down document-processing pipeline that performs three steps:

1. Predict page orientation with PaddleOCR’s `DocImgOrientationClassification`.
2. Rotate the PDF so that every page is upright and render each page to PNG.
3. Send the rendered pages to a Surya layout server, draw the detected regions, and save a crop for every element.

The goal is to keep the system minimal: you specify **only** the input PDF and an output directory, and the repo handles the orientation correction, layout classification, and crop export. Text OCR, table readers, and stamp removal have been removed so you can add those pieces manually if needed.

---

### Components

| Path | Purpose |
| --- | --- |
| `extractor/main.py` | Main pipeline orchestration (PDF → rotation → rendered pages → layout crops). |
| `extractor/layout_processor.py` | Converts layout detections into annotated pages and per-category crop folders. |
| `extractor/ocr_client.py` | Minimal HTTP client for the Surya layout endpoint (`/get-layout`). |
| `extractor/ocr_service.py` | Utility CLI to hit the layout service directly with raw images. |
| `run_pipeline.sh` | Convenience wrapper that calls `extractor/main.py` with environment overrides. |

---

### Requirements

* Python 3.10+
* `pip install -r requirements.txt`
* A running Surya layout service (defaults to `http://localhost:9667/get-layout`)
* GPU is optional but recommended for PaddleOCR if you process large PDFs

Make sure the Surya server is reachable from the machine running this repo; otherwise `OCRServiceClient` will raise on its POST requests.

---

### Quick Start

```bash
# Rotate a PDF and emit layout crops under ./outputs
python extractor/main.py test_samples/mwg2025.pdf outputs

# Equivalent helper
./run_pipeline.sh test_samples/mwg2025.pdf outputs
```

`extractor/main.py` always writes files inside the provided output directory:

```
outputs/
├── mwg2025_regularized.pdf                # rotated PDF
└── mwg2025_regularized_pages/             # rendered pages + layout crops
    ├── page_01/
    │   ├── annotated.png                  # page with boxes + labels
    │   └── crop_elements/
    │       ├── title/page_01_0001.png
    │       ├── table/page_01_0002.png
    │       └── ...
    └── page_02/
        └── ...
```

- Each page folder is zero-padded (`page_01`, `page_02`, …).
- Crops are organized by normalized category (`crop_elements/<category>/`).
- `annotated.png` uses OpenCV to draw red rectangles and category labels on the original page.

---

### Pipeline Details

1. **Orientation classification** — PaddleOCR predicts per-page angles. The defaults are baked into `extractor/main.py`:
   - Model: `PP-LCNet_x1_0_doc_ori`
   - Batch size: `1`
2. **Rotation** — `extractor.preprocess.regularize_pdf` rewrites the PDF so every page is upright, producing `<input>_regularized.pdf`.
3. **Rendering** — PyMuPDF renders PNG pages at a dynamic scale so the longest side is ~1800 px (consistent context for Surya).
4. **Layout detection** — `OCRServiceClient.layout` POSTs each PNG to the Surya `/get-layout` endpoint.
5. **Crop saving** — `LayoutProcessor` draws boxes, expands each bbox slightly (30 px x/y for non-tables, 5 px for tables), pads overflow with white, and saves each crop plus the annotated page.

If a detection’s bbox cannot be normalized or yields an empty crop, it is skipped. All log statements stream to stdout.

---

### Layout-only CLI

When you have pre-rendered PNGs and only want the layout portion, use `extractor/ocr_service.py`:

```bash
python extractor/ocr_service.py \
  --input_dir outputs/mwg2025_regularized_pages \
  --output_dir outputs/layout_only \
  --base_url http://localhost:9667 \
  --quiet   # optional

python extractor/ocr_service.py \
  --image test_samples/bol1.png \
  --output_dir outputs/single_page
```

Options:

- `--image` or `--input_dir` (mutually exclusive) point to supported image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`).
- `--output_dir` overrides the crop destination (defaults to the temp dir defined in `layout_processor.py`).
- `--base_url` sets the Surya service base URL.
- `--quiet` suppresses progress logs.

---

### Programmatic Usage

```python
from pathlib import Path
from extractor.ocr_client import OCRServiceClient
from extractor.layout_processor import LayoutProcessor

client = OCRServiceClient(base_url="http://localhost:9667")
processor = LayoutProcessor(client=client, output_dir=Path("outputs"))
processor.process_folder(Path("outputs/mwg2025_regularized_pages"))
```

The crop expansion can be customized by passing `crop_expand_px=(w, h)` and `table_crop_expand_px=(w, h)` when creating `LayoutProcessor`.

---

### Troubleshooting

- **Surya service unavailable**: `requests.exceptions.ConnectionError` means the layout endpoint is down or misconfigured. Update `DEFAULT_OCR_BASE_URL` in `extractor/main.py` or pass `--base_url` to `extractor/ocr_service.py`.
- **Unsupported images**: only the extensions listed in `layout_processor.VALID_EXTENSIONS` are processed.
- **Empty outputs**: make sure PaddleOCR’s orientation model can read the PDF (corrupted PDFs or encrypted files will fail when opened by PyMuPDF).

That’s it—feed in a PDF and tell the pipeline where to write results. Extend or swap in any additional OCR / table / NLP logic on top of the generated crops.
