from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
import tqdm
import fitz  # PyMuPDF

_CLASS_ID_TO_DEGREES = {
    0: 0,
    1: 90,
    2: 180,
    3: 270,
}


def _extract_angle(result: Mapping[str, object]) -> int:
    """Return the predicted orientation in degrees."""
    labels: Sequence[str] | None = result.get("label_names")  # type: ignore[assignment]
    if labels:
        try:
            return int(labels[0]) % 360
        except (TypeError, ValueError):
            pass

    class_ids: Sequence[int] | None = result.get("class_ids")  # type: ignore[assignment]
    if class_ids:
        class_id = int(class_ids[0])
        if class_id in _CLASS_ID_TO_DEGREES:
            return _CLASS_ID_TO_DEGREES[class_id]

    raise ValueError(f"Could not extract orientation angle from result: {result}")


def build_orientation_map(results: Iterable[Mapping[str, object]]) -> dict[int, int]:
    """Convert PaddleOCR predictions into a page-indexed angle map."""
    orientation_by_page: dict[int, int] = {}
    for idx, result in enumerate(results):
        page_index_raw = result.get("page_index", idx)
        try:
            page_index = int(page_index_raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid page index {page_index_raw!r}") from exc

        orientation_by_page[page_index] = _extract_angle(result)

    return orientation_by_page


def regularize_pdf(
    input_pdf: Path, output_pdf: Path, orientation_by_page: Mapping[int, int]
) -> dict[str, int]:
    """Rotate PDF pages to an upright orientation based on classifier output."""
    input_pdf = Path(input_pdf)
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    rotated_pages = 0
    with fitz.open(input_pdf) as doc:
        total_pages = doc.page_count
        for page in doc:
            predicted_angle = orientation_by_page.get(page.number)
            if predicted_angle is None:
                continue

            rotation_delta = (-int(predicted_angle)) % 360
            if rotation_delta == 0:
                continue

            # PyMuPDF rotations are clockwise and cumulative.
            page.set_rotation((page.rotation + rotation_delta) % 360)
            rotated_pages += 1

        doc.save(output_pdf)

    return {"total_pages": total_pages, "rotated_pages": rotated_pages}
