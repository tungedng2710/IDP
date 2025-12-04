"""HTTP client helpers for the OCR layout service."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Union

import numpy as np
import requests
from PIL import Image

ImageInput = Union[np.ndarray, Image.Image, str]


@dataclass
class OCRServiceClient:
    """Convenience wrapper around a generic OCR REST API."""

    base_url: str = "http://localhost:9667"
    session: requests.Session = field(default_factory=requests.Session)

    def detect(self, image: Union[ImageInput, Sequence[ImageInput]]) -> Dict[str, Any]:
        """Call the get-lines endpoint for a single image or batch."""

        return self._post_to_base("get-lines", image)

    def layout(self, image: Union[ImageInput, Sequence[ImageInput]]) -> Dict[str, Any]:
        """Call the get-layout endpoint for a single image or batch."""

        return self._post_to_base("get-layout", image)

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
