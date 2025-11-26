"""HTTP client helpers for Surya OCR and table recognition endpoints."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import requests
from PIL import Image

ImageInput = Union[np.ndarray, Image.Image, str]


@dataclass
class DotsOCRClient:
    """Convenience wrapper around the Surya OCR REST API."""

    base_url: str = "http://localhost:9667"
    table_url: Optional[str] = "http://localhost:9671/get-table"
    session: requests.Session = field(default_factory=requests.Session)

    def detect(self, image: Union[ImageInput, Sequence[ImageInput]]) -> Dict[str, Any]:
        """Call the get-lines endpoint for a single image or batch."""

        return self._post_to_base("get-lines", image)

    def layout(self, image: Union[ImageInput, Sequence[ImageInput]]) -> Dict[str, Any]:
        """Call the get-layout endpoint for a single image or batch."""

        return self._post_to_base("get-layout", image)

    def recognize(self, image: Union[ImageInput, Sequence[ImageInput]]) -> Dict[str, Any]:
        """Call the get-text endpoint for a single image or batch."""

        return self._post_to_base("get-text", image)

    def recognize_text(
        self, image: Union[ImageInput, Sequence[ImageInput]]
    ) -> Union[str, List[str]]:
        """
        Run recognition and return plain text.

        A single image input yields a single string; batches return a list of strings in
        the same order as the input images.
        """

        response = self.recognize(image)
        texts = self._parse_text_response(response)
        if isinstance(image, (list, tuple)):
            return texts
        return texts[0] if texts else ""

    def recognize_table(self, image: ImageInput) -> Optional[str]:
        """Call the table recognition endpoint and normalize the HTML output."""

        if not self.table_url:
            return None

        payload = {"image": self._to_base64(image)}
        response = self.session.post(self.table_url, data=payload)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            fallback = response.text.strip()
            return fallback or None

        html_text = self._extract_table_html_from_payload(data)
        if html_text:
            return html_text

        fallback = response.text.strip()
        return fallback or None

    def _post_to_base(
        self,
        endpoint: str,
        image: Union[ImageInput, Sequence[ImageInput]],
    ) -> Dict[str, Any]:
        payload = self._build_payload(image)
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = self.session.post(url, data=payload)
        response.raise_for_status()
        return response.json()

    def _build_payload(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
    ) -> Dict[str, Any]:
        if isinstance(image, (list, tuple)):
            images_b64 = [self._to_base64(img) for img in image]
            return {"images": json.dumps(images_b64)}
        return {"image": self._to_base64(image)}

    @staticmethod
    def _to_base64(image: ImageInput) -> str:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                arr = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                arr = image
            if arr.ndim == 2:
                img = Image.fromarray(arr, mode="L")
            elif arr.ndim == 3:
                channels = arr.shape[2]
                if channels == 3:
                    img = Image.fromarray(arr, mode="RGB")
                elif channels == 4:
                    img = Image.fromarray(arr, mode="RGBA")
                else:
                    raise ValueError(f"Unsupported array shape: {arr.shape}")
            else:
                raise ValueError(f"Unsupported array shape: {arr.shape}")
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, str):
            img = Image.open(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @staticmethod
    def _extract_table_html_from_payload(payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if isinstance(payload, str):
            html_text = payload.strip()
            return html_text or None
        if isinstance(payload, (list, tuple)):
            for item in payload:
                html_text = DotsOCRClient._extract_table_html_from_payload(item)
                if html_text:
                    return html_text
            return None
        if isinstance(payload, dict):
            preferred_keys = ("html", "result", "data", "table", "content", "value")
            for key in preferred_keys:
                if key in payload:
                    html_text = DotsOCRClient._extract_table_html_from_payload(payload[key])
                    if html_text:
                        return html_text
            for value in payload.values():
                html_text = DotsOCRClient._extract_table_html_from_payload(value)
                if html_text:
                    return html_text
        return None

    @staticmethod
    def _parse_text_response(payload: Dict[str, Any]) -> List[str]:
        """Normalize recognition responses into a list of newline-joined strings."""

        if not payload:
            return []
        if not isinstance(payload, dict):
            return []

        if "results" in payload and isinstance(payload["results"], list):
            return [
                DotsOCRClient._collect_line_text(item) for item in payload["results"]
            ]
        if "result" in payload:
            return [DotsOCRClient._collect_line_text(payload["result"])]
        if "text_lines" in payload:
            return [DotsOCRClient._collect_line_text(payload)]
        return []

    @staticmethod
    def _collect_line_text(block: Any) -> str:
        """Join the text_line entries from a recognition result."""

        if not isinstance(block, dict):
            return ""
        lines: List[str] = []
        for line in block.get("text_lines") or []:
            text_val = ""
            if isinstance(line, dict):
                text_val = line.get("text", "")
            else:
                text_val = getattr(line, "text", "")
            text = str(text_val).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()
