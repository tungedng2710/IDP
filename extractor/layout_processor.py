"""High-level helpers for running layout extraction and persisting results."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ocr_client import DotsOCRClient

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_TEMP_OUTPUT_DIR = Path(tempfile.gettempdir()) / "dotocr_pages"
PAGE_FOLDER_PREFIX = "page_"
CROP_ELEMENTS_DIRNAME = "crop_elements"
ANNOTATED_FILENAME = "annotated.png"
PAGES_DIRNAME = "pages"


@dataclass
class LayoutProcessor:
    """Reusable pipeline for extracting layouts and saving crops."""

    client: DotsOCRClient
    output_dir: Optional[Path] = None
    crop_expand_px: Tuple[int, int] = (30, 30)
    table_crop_expand_px: Tuple[int, int] = (5, 5)
    verbose: bool = True

    def process_image(self, image_path: Path) -> Path:
        """Process a single image and return the annotated output path."""

        detections = self._parse_result_items(self.client.layout(str(image_path)))
        doc_name = image_path.stem
        pages_root = self._get_pages_root(doc_name)
        return self._save_page_result(image_path, detections, pages_root, page_index=1)

    def process_folder(self, folder_path: Path) -> List[Path]:
        """Iterate over a directory, processing each supported image."""

        image_files = self._iter_image_files(folder_path)
        if not image_files:
            raise FileNotFoundError(f"No supported images found in {folder_path}")

        doc_name = folder_path.name
        pages_root = self._get_pages_root(doc_name)
        annotated_paths: List[Path] = []
        for idx, image_path in enumerate(image_files, start=1):
            try:
                detections = self._parse_result_items(self.client.layout(str(image_path)))
            except Exception as exc:  # pragma: no cover - network/IO heavy
                if self.verbose:
                    print(f"Failed to request layout for {image_path}: {exc}")
                continue
            try:
                annotated = self._save_page_result(
                    image_path,
                    detections,
                    pages_root,
                    page_index=idx,
                )
                annotated_paths.append(annotated)
            except Exception as exc:  # pragma: no cover - disk IO
                if self.verbose:
                    print(f"Failed to save results for {image_path}: {exc}")
        return annotated_paths

    # --- Internal helpers -------------------------------------------------

    def _get_pages_root(self, document_name: str) -> Path:
        base_dir = self.output_dir or DEFAULT_TEMP_OUTPUT_DIR
        if base_dir.exists() and not base_dir.is_dir():
            raise NotADirectoryError(
                f"DotOCR output base exists and is not a directory: {base_dir}"
            )

        doc_dir = base_dir / document_name
        if doc_dir.exists() and not doc_dir.is_dir():
            raise NotADirectoryError(
                f"DotOCR document path exists and is not a directory: {doc_dir}"
            )

        pages_root = doc_dir / PAGES_DIRNAME
        pages_root.mkdir(parents=True, exist_ok=True)
        return pages_root

    def _save_page_result(
        self,
        image_path: Path,
        detections: List[Dict[str, Any]],
        pages_root: Path,
        page_index: int,
    ) -> Path:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Unable to read image at {image_path}")

        page_dir = pages_root / f"{PAGE_FOLDER_PREFIX}{page_index:02d}"
        crop_root = page_dir / CROP_ELEMENTS_DIRNAME
        page_dir.mkdir(parents=True, exist_ok=True)
        crop_root.mkdir(parents=True, exist_ok=True)

        overlay = image.copy()
        img_height, img_width = overlay.shape[:2]

        for idx, item in enumerate(detections, start=1):
            bbox = self._normalize_bbox(item.get("bbox"), img_width, img_height)
            if not bbox:
                continue
            crop = self._extract_crop(image, bbox, item)
            if crop is None or crop.size == 0:
                continue

            category = str(item.get("category", "unknown"))
            category_norm = category.strip().lower()
            rect_bottom_right = (
                min(bbox[2] - 1, img_width - 1),
                min(bbox[3] - 1, img_height - 1),
            )
            cv2.rectangle(overlay, (bbox[0], bbox[1]), rect_bottom_right, (0, 0, 255), 2)
            cv2.putText(
                overlay,
                category,
                (bbox[0], max(0, bbox[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            cat_dir = crop_root / self._sanitize_category(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{image_path.stem}_{idx:04d}.png"
            crop_output_path = cat_dir / crop_name
            cv2.imwrite(str(crop_output_path), crop)

            if category_norm == "table":
                self._save_table_html(crop_output_path, crop)

        annotated_path = page_dir / ANNOTATED_FILENAME
        cv2.imwrite(str(annotated_path), overlay)
        if self.verbose:
            print(f"Saved page results to {page_dir}")
        return annotated_path

    def _extract_crop(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        detection: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        category = str(detection.get("category", "")).strip().lower()
        if category == "table":
            expand = self.table_crop_expand_px
            white_pad_axes = (False, False)
        else:
            expand = self.crop_expand_px
            white_pad_axes = (expand[0] > 2, expand[1] > 2)
        return self._extract_expanded_crop(image, bbox, expand, white_pad_axes)

    def _save_table_html(self, crop_output_path: Path, crop: np.ndarray) -> None:
        table_html = None
        try:
            table_html = self.client.recognize_table(crop)
        except Exception as exc:  # pragma: no cover - network/IO heavy
            if self.verbose:
                print(f"Table recognition failed for {crop_output_path.name}: {exc}")
        if table_html:
            html_path = crop_output_path.with_suffix(".html")
            html_path.write_text(table_html, encoding="utf-8")

    @staticmethod
    def _iter_image_files(folder_path: Path) -> List[Path]:
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ]
        )

    @staticmethod
    def _parse_result_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        items = result.get("result", [])
        if isinstance(items, str):
            return json.loads(items)
        return items

    @staticmethod
    def _sanitize_category(value: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
        return sanitized.strip("_") or "uncategorized"

    @staticmethod
    def _normalize_bbox(
        bbox: Sequence[float],
        img_width: int,
        img_height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not bbox or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = map(float, bbox[:4])
        except (TypeError, ValueError):
            return None

        x1 = int(round(max(0.0, min(x1, img_width - 1))))
        y1 = int(round(max(0.0, min(y1, img_height - 1))))
        x2 = int(round(max(0.0, min(x2, img_width))))
        y2 = int(round(max(0.0, min(y2, img_height))))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _extract_expanded_crop(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        expand_px: Tuple[int, int],
        white_pad_axes: Tuple[bool, bool] = (False, False),
    ) -> Optional[np.ndarray]:
        pad_horizontal, pad_vertical = white_pad_axes
        if pad_horizontal or pad_vertical:
            base_expand = (
                0 if pad_horizontal else expand_px[0],
                0 if pad_vertical else expand_px[1],
            )
            crop = LayoutProcessor._extract_expanded_crop(
                image,
                bbox,
                base_expand,
                (False, False),
            )
            if crop is None:
                return None
            return LayoutProcessor._pad_crop_with_white(
                crop,
                pad_horizontal,
                pad_vertical,
                expand_px,
            )

        expand_w, expand_h = expand_px
        if expand_w <= 0 and expand_h <= 0:
            x1, y1, x2, y2 = bbox
            crop = image[y1:y2, x1:x2]
            return crop if crop.size else None

        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox

        exp_x1 = x1 - expand_w
        exp_y1 = y1 - expand_h
        exp_x2 = x2 + expand_w
        exp_y2 = y2 + expand_h

        crop_x1 = max(0, floor(exp_x1))
        crop_y1 = max(0, floor(exp_y1))
        crop_x2 = min(width, ceil(exp_x2))
        crop_y2 = min(height, ceil(exp_y2))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None

        crop = image[int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)]
        if crop.size == 0:
            return None

        pad_left = max(0, int(ceil(0 - exp_x1)))
        pad_top = max(0, int(ceil(0 - exp_y1)))
        pad_right = max(0, int(ceil(exp_x2 - width)))
        pad_bottom = max(0, int(ceil(exp_y2 - height)))

        if pad_left:
            pad_left = max(1, pad_left // 2)
        if pad_right:
            pad_right = max(1, pad_right // 2)
        if pad_top:
            pad_top = max(1, pad_top // 2)
        if pad_bottom:
            pad_bottom = max(1, pad_bottom // 2)

        target_h = crop.shape[0] + pad_top + pad_bottom
        target_w = crop.shape[1] + pad_left + pad_right
        channels = 1 if crop.ndim == 2 else crop.shape[2]
        white_value = 255
        if channels == 1:
            canvas = np.full((target_h, target_w), white_value, dtype=image.dtype)
        else:
            canvas = np.full((target_h, target_w, channels), white_value, dtype=image.dtype)

        canvas[
            pad_top : pad_top + crop.shape[0],
            pad_left : pad_left + crop.shape[1],
        ] = crop
        return canvas

    @staticmethod
    def _pad_crop_with_white(
        crop: np.ndarray,
        pad_horizontal: bool,
        pad_vertical: bool,
        expand_px: Tuple[int, int],
    ) -> np.ndarray:
        pad_left = pad_right = expand_px[0] if pad_horizontal else 0
        pad_top = pad_bottom = expand_px[1] if pad_vertical else 0
        if not any((pad_left, pad_right, pad_top, pad_bottom)):
            return crop

        white_value = 255
        if crop.ndim == 2:
            border_value: Tuple[int, ...] | int = white_value
        else:
            channels = crop.shape[2]
            border_value = tuple([white_value] * channels)

        padded = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_value,
        )
        return padded
