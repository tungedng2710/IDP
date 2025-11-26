#!/usr/bin/env python3
"""FastAPI service that runs the DotsOCR pipeline on uploaded PDFs."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_SCRIPT = REPO_ROOT / "run_pipeline.sh"

if not PIPELINE_SCRIPT.exists():
    raise RuntimeError(f"run_pipeline.sh not found at {PIPELINE_SCRIPT}")

PDF_MIME_TYPES = {"application/pdf"}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
}

app = FastAPI(title="Submission Pipeline API", version="1.0.0")


async def _write_upload_to_path(upload: UploadFile, destination: Path) -> None:
    """Stream an uploaded file to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    await upload.seek(0)
    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            buffer.write(chunk)


def _run_pipeline(pdf_path: Path, markdown_output: Path) -> None:
    """Invoke run_pipeline.sh with the provided paths."""
    cmd = ["bash", str(PIPELINE_SCRIPT), str(pdf_path), str(markdown_output)]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or "Pipeline execution failed."
        raise RuntimeError(message)


def _looks_like_pdf(path: Path) -> bool:
    """Check the first bytes to determine if the file is already a PDF."""
    try:
        with path.open("rb") as handle:
            signature = handle.read(5)
        return signature.startswith(b"%PDF-")
    except OSError:
        return False


def _convert_image_to_pdf(source_path: Path, target_path: Path) -> Path:
    """Convert common image formats into a single-page PDF."""
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - import failure
        raise RuntimeError("Pillow is required to convert images to PDF.") from exc

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image.save(target_path, "PDF")
    return target_path


def _convert_with_libreoffice(source_path: Path, desired_target: Path) -> Path:
    """Use LibreOffice (soffice) to convert office formats to PDF."""
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError("LibreOffice (soffice) executable not found for conversion.")

    output_dir = desired_target.parent
    cmd = [executable, "--headless", "--convert-to", "pdf", str(source_path), "--outdir", str(output_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(stderr or stdout or "LibreOffice conversion failed.")

    primary_candidate = output_dir / f"{source_path.stem}.pdf"
    if primary_candidate.exists():
        if primary_candidate != desired_target:
            primary_candidate.rename(desired_target)
        return desired_target

    matches = sorted(output_dir.glob(f"{source_path.stem}*.pdf"))
    if matches:
        if matches[0] != desired_target:
            matches[0].rename(desired_target)
        return desired_target

    if desired_target.exists():
        return desired_target
    raise RuntimeError("Conversion reported success but no PDF was produced.")


def _ensure_pdf_path(source_path: Path, content_type: str | None) -> Path:
    """Return a path to a PDF version of the uploaded file, converting if needed."""
    if _looks_like_pdf(source_path):
        if source_path.suffix.lower() == ".pdf":
            return source_path
        target = source_path.with_suffix(".pdf")
        source_path.rename(target)
        return target

    is_image = source_path.suffix.lower() in IMAGE_EXTENSIONS
    if not is_image and content_type:
        is_image = content_type.startswith("image/")

    target_name = (
        f"{source_path.stem}_converted.pdf"
        if source_path.suffix.lower() == ".pdf"
        else f"{source_path.stem}.pdf"
    )
    target_path = source_path.with_name(target_name)

    if is_image:
        return _convert_image_to_pdf(source_path, target_path)

    try:
        return _convert_with_libreoffice(source_path, target_path)
    except RuntimeError:
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/process")
async def process_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an upload, ensure it is a PDF (converting if needed), and return merged markdown."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)
            filename = Path(file.filename).name if file.filename else "input"
            upload_path = tmp_dir / filename
            await _write_upload_to_path(file, upload_path)
            try:
                pdf_path = await asyncio.to_thread(_ensure_pdf_path, upload_path, file.content_type)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=f"Unable to convert upload to PDF: {exc}") from exc
            markdown_path = tmp_dir / "document.md"

            try:
                await asyncio.to_thread(_run_pipeline, pdf_path, markdown_path)
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

            if not markdown_path.exists():
                raise HTTPException(status_code=500, detail="Markdown output not found.")

            markdown_text = markdown_path.read_text(encoding="utf-8")
            return {"status": "success", "markdown": markdown_text}
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "submission_api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        reload=os.environ.get("RELOAD", "0") == "1",
    )
