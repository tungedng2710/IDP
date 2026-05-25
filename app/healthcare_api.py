from __future__ import annotations

import asyncio
import copy
import functools
import json
import os
import tempfile
from pathlib import Path

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scripts.classify_doc import MODEL_NAME as DEFAULT_CLASSIFIER_MODEL
from scripts.classify_doc import OLLAMA_BASE_URL as DEFAULT_OLLAMA_URL
from scripts.classify_doc import classify_document
from scripts.healthcare_ollama_pipeline import DOCUMENT_TYPES
from scripts.healthcare_ollama_pipeline import EXTRACTOR_MODEL as DEFAULT_EXTRACTOR_MODEL
from scripts.healthcare_ollama_pipeline import (
    HealthcareOllamaPipeline,
    extract_information,
    load_templates,
    process_document,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT_DIR / "data" / "prompts"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "output")))
TEMPLATES_DIR = Path(os.getenv("TEMPLATES_DIR", str(PROMPTS_DIR)))
DEFAULT_PROMPT_TEMPLATE = os.getenv(
    "DEFAULT_PROMPT_TEMPLATE",
    "healthcare_types/prompt_with_template.txt",
)
DEFAULT_JSON_TEMPLATE = os.getenv(
    "DEFAULT_JSON_TEMPLATE",
    "healthcare_types/page-04-template.json",
)
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/tiff",
    "image/bmp",
    "application/pdf",
}

classify_app = FastAPI(title="Healthcare Classify API", version="1.0.0")
extraction_app = FastAPI(title="Healthcare Extraction API", version="1.0.0")
healthcare_pipeline_app = FastAPI(title="Healthcare Ollama Pipeline API", version="1.0.0")


def _ensure_supported_upload(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Expected an image or PDF upload, "
                f"got {content_type or 'unknown content type'}"
            ),
        )
    return content_type


def _resolve_template_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = TEMPLATES_DIR / candidate

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=400, detail=f"Template file not found: {resolved}")
    return resolved


def _write_temp_upload(file_bytes: bytes, filename: str, suffix: str) -> Path:
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=safe_suffix, prefix="idp-") as temp_file:
        temp_file.write(file_bytes)
        return Path(temp_file.name)


def _prepare_upload_image(file: UploadFile, file_bytes: bytes, content_type: str, page: int = 1) -> Path:
    if content_type == "application/pdf":
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                if not doc:
                    raise HTTPException(status_code=400, detail="PDF has no pages")
                page_idx = max(0, page - 1)
                if page_idx >= len(doc):
                    raise HTTPException(status_code=400, detail=f"Page {page} out of range (document has {len(doc)} pages)")
                pix = doc[page_idx].get_pixmap(dpi=PDF_RENDER_DPI)
                image_bytes = pix.tobytes("png")
            return _write_temp_upload(image_bytes, file.filename or "upload", ".png")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    return _write_temp_upload(file_bytes, file.filename or "upload", suffix)


def _render_pdf_pages(file_bytes: bytes, page_start: int, page_end: int) -> list[Path]:
    """Render PDF pages page_start..page_end (1-based, inclusive) to temp PNG files.
    page_end=0 means the last page of the document.
    """
    temp_paths: list[Path] = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            total = len(doc)
            if not total:
                raise HTTPException(status_code=400, detail="PDF has no pages")
            start_idx = max(0, page_start - 1)
            end_idx = total - 1 if page_end == 0 else min(total - 1, page_end - 1)
            if start_idx > end_idx:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid page range {page_start}-{page_end} (document has {total} pages)",
                )
            for idx in range(start_idx, end_idx + 1):
                pix = doc[idx].get_pixmap(dpi=PDF_RENDER_DPI)
                temp_paths.append(_write_temp_upload(pix.tobytes("png"), f"page-{idx + 1}", ".png"))
    except HTTPException:
        for p in temp_paths:
            p.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for p in temp_paths:
            p.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc
    return temp_paths


def _is_empty_value(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _merge_extracted_values(current: object, new_value: object) -> object:
    if _is_empty_value(current):
        return copy.deepcopy(new_value)
    if _is_empty_value(new_value):
        return copy.deepcopy(current)

    if isinstance(current, dict) and isinstance(new_value, dict):
        merged = copy.deepcopy(current)
        for key, value in new_value.items():
            if key in merged:
                merged[key] = _merge_extracted_values(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    if isinstance(current, list) and isinstance(new_value, list):
        return [*copy.deepcopy(current), *copy.deepcopy(new_value)]

    if current == new_value:
        return copy.deepcopy(current)

    return copy.deepcopy(current)


def _merge_page_extracted_data(page_results: list[dict[str, object]]) -> dict[str, object]:
    extracted_pages = [
        result["extracted_data"]
        for result in page_results
        if isinstance(result.get("extracted_data"), dict)
    ]
    if not extracted_pages:
        return {}

    merged = copy.deepcopy(extracted_pages[0])
    for page_data in extracted_pages[1:]:
        page_dict = page_data if isinstance(page_data, dict) else {}
        if (
            isinstance(merged.get("content"), dict)
            and "content" not in page_dict
            and not any(key in page_dict for key in ("id", "hospital", "form"))
        ):
            merged["content"] = _merge_extracted_values(merged["content"], page_dict)
        else:
            merged = _merge_extracted_values(merged, page_dict)
    return merged


async def _run_blocking(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(None, call)


@classify_app.get("/")
def classify_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "classify",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "classifier_model": os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
    }


@classify_app.post("/classify")
async def classify_endpoint(
    file: UploadFile = File(...),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
) -> dict[str, str | None]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)

    try:
        document_type, template_filename = await _run_blocking(
            classify_document,
            str(temp_path),
            base_url=ollama_url,
            model=model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if not document_type:
        raise HTTPException(status_code=502, detail="Document classification failed")

    return {
        "status": "success",
        "document_type": document_type,
        "template_filename": template_filename,
    }


@extraction_app.get("/")
def extraction_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "extraction",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "extractor_model": os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
        "templates_dir": str(TEMPLATES_DIR),
    }


@extraction_app.post("/extract")
async def extraction_endpoint(
    file: UploadFile = File(...),
    prompt_template_path: str = Form(default=DEFAULT_PROMPT_TEMPLATE),
    json_template_path: str = Form(default=DEFAULT_JSON_TEMPLATE),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
    page_start: int = Form(default=1),
    page_end: int | None = Form(default=None),
    batch_size: int = Form(default=1),
) -> dict[str, object]:
    if not 1 <= batch_size <= 3:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 3")

    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    prompt_template = _resolve_template_path(prompt_template_path).read_text(encoding="utf-8")
    with _resolve_template_path(json_template_path).open(encoding="utf-8") as template_file:
        json_template = json.load(template_file)

    multi_page = content_type == "application/pdf" and page_end is not None and (page_end == 0 or page_end != page_start)
    if multi_page:
        temp_paths = _render_pdf_pages(file_bytes, page_start, page_end)
    else:
        temp_paths = [_prepare_upload_image(file, file_bytes, content_type, page=page_start)]

    loop = asyncio.get_running_loop()

    async def _run_extract(page_num: int, tp: Path) -> dict[str, object]:
        fn = functools.partial(
            extract_information,
            image_path=str(tp),
            prompt_template=prompt_template,
            json_template=json_template,
            base_url=ollama_url,
            model=model,
        )
        data = await loop.run_in_executor(None, fn)
        return {"page": page_num, "extracted_data": data}

    page_results: list[dict[str, object]] = []
    try:
        for batch_start in range(0, len(temp_paths), batch_size):
            batch_slice = temp_paths[batch_start: batch_start + batch_size]
            batch_page_nums = [
                (page_start + batch_start + j) if multi_page else page_start
                for j in range(len(batch_slice))
            ]
            batch_results = await asyncio.gather(
                *[_run_extract(pn, tp) for pn, tp in zip(batch_page_nums, batch_slice)]
            )
            for res in batch_results:
                if res["extracted_data"] is None:
                    raise HTTPException(status_code=502, detail=f"Information extraction failed on page {res['page']}")
                page_results.append(res)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

    if multi_page:
        return _merge_page_extracted_data(page_results)
    return page_results[0]["extracted_data"]


@healthcare_pipeline_app.get("/")
def healthcare_pipeline_health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "healthcare_ollama_pipeline",
        "ollama_url": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        "classifier_model": os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL),
        "extractor_model": os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL),
        "templates_dir": str(TEMPLATES_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


@healthcare_pipeline_app.post("/process")
async def healthcare_pipeline_endpoint(
    file: UploadFile = File(...),
    output_subdir: str = Form(default="api"),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    classifier_model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
    page_start: int = Form(default=1),
    page_end: int | None = Form(default=None),
) -> dict[str, object]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename or "upload").stem

    multi_page = content_type == "application/pdf" and page_end is not None and (page_end == 0 or page_end != page_start)
    if multi_page:
        temp_paths = _render_pdf_pages(file_bytes, page_start, page_end)
    else:
        temp_paths = [_prepare_upload_image(file, file_bytes, content_type, page=page_start)]

    page_results = []
    try:
        for i, temp_path in enumerate(temp_paths):
            page_num = (page_start + i) if multi_page else page_start
            result = await _run_blocking(
                process_document,
                image_path=str(temp_path),
                output_dir=str(output_dir),
                templates_base_dir=templates_dir,
                ollama_base_url=ollama_url,
                classifier_model=classifier_model,
                extractor_model=extractor_model,
            )
            if result.get("status") != "success":
                raise HTTPException(status_code=502, detail={**result, "page": page_num})
            if multi_page:
                # rename output file to include page number
                out_src = result.get("output_path")
                if out_src:
                    out_dst = output_dir / f"{stem}_page{page_num}.json"
                    Path(out_src).rename(out_dst)
            page_results.append({"page": page_num, "extracted_data": result["extracted_data"]})
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

    if multi_page:
        return _merge_page_extracted_data(page_results)
    return page_results[0]["extracted_data"]


@healthcare_pipeline_app.post("/classify")
async def healthcare_pipeline_classify_endpoint(
    file: UploadFile = File(...),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    classifier_model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> dict[str, object]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    temp_path = _prepare_upload_image(file, file_bytes, content_type)

    pipeline = HealthcareOllamaPipeline(
        templates_base_dir=templates_dir,
        ollama_base_url=ollama_url,
        classifier_model=classifier_model,
        extractor_model=extractor_model,
    )

    try:
        result = await _run_blocking(pipeline.classify, str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    if result.get("status") != "success":
        raise HTTPException(status_code=502, detail=result)

    return result


@healthcare_pipeline_app.post("/extraction")
async def healthcare_pipeline_extraction_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form(default="PHIẾU KHÁM BỆNH VÀO VIỆN"),
    output_subdir: str = Form(default="api"),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
    page_start: int = Form(default=1),
    page_end: int | None = Form(default=None),
    batch_size: int = Form(default=1),
) -> dict[str, object]:
    if not 1 <= batch_size <= 3:
        raise HTTPException(status_code=400, detail="batch_size must be between 1 and 3")

    if document_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown document_type '{document_type}'. Valid values: {list(DOCUMENT_TYPES.keys())}",
        )

    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    template_info = DOCUMENT_TYPES[document_type]
    templates_base = Path(templates_dir)
    prompt_template, json_template = load_templates(template_info, templates_base)
    if not prompt_template or not json_template:
        raise HTTPException(status_code=502, detail="Failed to load templates for the given document type")

    multi_page = content_type == "application/pdf" and page_end is not None and (page_end == 0 or page_end != page_start)
    if multi_page:
        temp_paths = _render_pdf_pages(file_bytes, page_start, page_end)
    else:
        temp_paths = [_prepare_upload_image(file, file_bytes, content_type, page=page_start)]

    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename or "upload").stem

    async def _run_extract_pipeline(page_num: int, tp: Path) -> dict[str, object]:
        data = await _run_blocking(
            extract_information,
            image_path=str(tp),
            prompt_template=prompt_template,
            json_template=json_template,
            base_url=ollama_url,
            model=extractor_model,
        )
        return {"page": page_num, "extracted_data": data}

    page_results: list[dict[str, object]] = []
    try:
        for batch_start in range(0, len(temp_paths), batch_size):
            batch_slice = temp_paths[batch_start: batch_start + batch_size]
            batch_page_nums = [
                (page_start + batch_start + j) if multi_page else page_start
                for j in range(len(batch_slice))
            ]
            batch_results = await asyncio.gather(
                *[_run_extract_pipeline(pn, tp) for pn, tp in zip(batch_page_nums, batch_slice)]
            )
            for res in batch_results:
                page_num = res["page"]
                extracted_data = res["extracted_data"]
                if extracted_data is None:
                    raise HTTPException(status_code=502, detail=f"Information extraction failed on page {page_num}")
                out_name = f"{stem}_page{page_num}.json" if multi_page else f"{stem}.json"
                out_path = output_dir / out_name
                out_path.write_text(json.dumps(extracted_data, ensure_ascii=False, indent=2))
                page_results.append({"page": page_num, "extracted_data": extracted_data, "output_path": str(out_path)})
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

    if multi_page:
        return _merge_page_extracted_data(page_results)
    return page_results[0]["extracted_data"]


@healthcare_pipeline_app.post("/classify_batch")
async def healthcare_pipeline_classify_batch_endpoint(
    file: UploadFile = File(...),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    classifier_model: str = Form(default=os.getenv("CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)),
    page_start: int = Form(default=1),
    page_end: int | None = Form(default=None),
) -> list[dict[str, object]]:
    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Non-PDF: single image treated as one page
    if content_type != "application/pdf":
        temp_path = _prepare_upload_image(file, file_bytes, content_type)
        try:
            doc_type, _ = await _run_blocking(
                classify_document,
                str(temp_path),
                base_url=ollama_url,
                model=classifier_model,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return [{"pages": [1], "document_type": doc_type or "UNKNOWN"}]

    # Determine total pages to resolve effective_end
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc
    if not total_pages:
        raise HTTPException(status_code=400, detail="PDF has no pages")

    effective_end = total_pages if (page_end is None or page_end == 0) else page_end
    temp_paths = _render_pdf_pages(file_bytes, page_start, effective_end)

    page_classifications: list[tuple[int, str]] = []
    try:
        for i, temp_path in enumerate(temp_paths):
            page_num = page_start + i
            doc_type, _ = await _run_blocking(
                classify_document,
                str(temp_path),
                base_url=ollama_url,
                model=classifier_model,
            )
            page_classifications.append((page_num, doc_type or "UNKNOWN"))
    finally:
        for tp in temp_paths:
            tp.unlink(missing_ok=True)

    # Group consecutive pages sharing the same document type
    groups: list[dict[str, object]] = []
    for page_num, doc_type in page_classifications:
        if groups and groups[-1]["document_type"] == doc_type:
            groups[-1]["pages"].append(page_num)  # type: ignore[union-attr]
        else:
            groups.append({"pages": [page_num], "document_type": doc_type})

    return groups


@healthcare_pipeline_app.post("/extraction_batch")
async def healthcare_pipeline_extraction_batch_endpoint(
    file: UploadFile = File(...),
    blocks: str = Form(...),
    output_subdir: str = Form(default="api"),
    templates_dir: str = Form(default=str(TEMPLATES_DIR)),
    ollama_url: str = Form(default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL)),
    extractor_model: str = Form(default=os.getenv("EXTRACTOR_MODEL", DEFAULT_EXTRACTOR_MODEL)),
) -> list[dict[str, object]]:
    """Extract information for each block produced by /classify_batch.

    ``blocks`` is the JSON array returned by /classify_batch:
    [{"pages": [1, 2], "document_type": "..."}, ...]

    Each block is processed as a group: all pages in the block are rendered and
    extracted individually, then collected under a single result entry.
    """
    # Parse blocks
    try:
        block_list: list[dict] = json.loads(blocks)
        if not isinstance(block_list, list):
            raise ValueError("blocks must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid blocks JSON: {exc}") from exc

    content_type = _ensure_supported_upload(file)
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    output_dir = OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename or "upload").stem
    templates_base = Path(templates_dir)

    results: list[dict[str, object]] = []

    for block in block_list:
        document_type = block.get("document_type", "")
        pages: list[int] = block.get("pages", [])

        if not pages:
            results.append({"document_type": document_type, "pages": [], "error": "No pages specified"})
            continue

        if document_type not in DOCUMENT_TYPES:
            results.append({
                "document_type": document_type,
                "pages": pages,
                "error": f"Unknown document_type '{document_type}'. Valid values: {list(DOCUMENT_TYPES.keys())}",
            })
            continue

        template_info = DOCUMENT_TYPES[document_type]
        prompt_template, json_template = load_templates(template_info, templates_base)
        if not prompt_template or not json_template:
            results.append({
                "document_type": document_type,
                "pages": pages,
                "error": "Failed to load templates for the given document type",
            })
            continue

        # Render pages for this block
        if content_type == "application/pdf" and len(pages) > 1:
            temp_paths = _render_pdf_pages(file_bytes, min(pages), max(pages))
            rendered_page_nums = list(range(min(pages), max(pages) + 1))
        else:
            first_page = pages[0]
            temp_paths = [_prepare_upload_image(file, file_bytes, content_type, page=first_page)]
            rendered_page_nums = [first_page]

        page_results: list[dict[str, object]] = []
        try:
            for temp_path, page_num in zip(temp_paths, rendered_page_nums):
                extracted_data = await _run_blocking(
                    extract_information,
                    image_path=str(temp_path),
                    prompt_template=prompt_template,
                    json_template=json_template,
                    base_url=ollama_url,
                    model=extractor_model,
                )
                if extracted_data is None:
                    page_results.append({"page": page_num, "error": "Extraction failed"})
                    continue
                out_name = f"{stem}_page{page_num}.json"
                out_path = output_dir / out_name
                out_path.write_text(json.dumps(extracted_data, ensure_ascii=False, indent=2))
                page_results.append({"page": page_num, "extracted_data": extracted_data, "output_path": str(out_path)})
        finally:
            for tp in temp_paths:
                tp.unlink(missing_ok=True)

        results.append({
            "document_type": document_type,
            "pages": pages,
            "template_filename": template_info.get("template"),
            "page_results": page_results,
        })

    return results


HEALTHCARE_TYPES_DIR = PROMPTS_DIR / "healthcare_types"


@healthcare_pipeline_app.post("/upload-schema")
async def upload_schema_endpoint(
    file: UploadFile = File(..., description="JSON template file"),
    version: str = Form(..., description="Template version, e.g. '1' or '1.0'"),
    document_type: str = Form(..., description="Document type name, e.g. 'PHIẾU KHÁM BỆNH VÀO VIỆN'"),
) -> dict[str, object]:
    """Upload a JSON schema template for a given document type.

    Saves the file to data/prompts/healthcare_types/templates_v1/. If the
    document type already exists in doctype_map.json, only that mapped template
    file is updated. The map is updated only for new document types. The
    version input is accepted for compatibility but does not change the
    destination folder.
    """
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a JSON file (.json)")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        json.loads(file_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    version_str = str(version).strip()
    templates_dir = HEALTHCARE_TYPES_DIR / "templates_v1"
    templates_dir.mkdir(parents=True, exist_ok=True)

    doctype_map_path = templates_dir / "doctype_map.json"
    if doctype_map_path.exists():
        with doctype_map_path.open(encoding="utf-8") as f:
            doctype_map: dict[str, str] = json.load(f)
    else:
        doctype_map = {}

    uploaded_filename = file.filename
    existing_template_filename = doctype_map.get(document_type)
    template_filename = existing_template_filename or uploaded_filename
    template_path = templates_dir / template_filename
    template_path.write_bytes(file_bytes)

    doctype_map_updated = existing_template_filename is None
    if doctype_map_updated:
        doctype_map[document_type] = template_filename
        doctype_map_path.write_text(json.dumps(doctype_map, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "version": version_str,
        "effective_version": "1",
        "document_type": document_type,
        "uploaded_filename": uploaded_filename,
        "template_filename": template_filename,
        "template_path": str(template_path),
        "doctype_map_path": str(doctype_map_path),
        "doctype_map_updated": doctype_map_updated,
    }
