from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


def rgb_to_cmyk(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Convert a BGR OpenCV image to its CMYK channels (uint8)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    rgb_normalized = image.astype(np.float32) / 255.0
    # OpenCV uses BGR ordering, so extract manually to preserve intent.
    r = rgb_normalized[:, :, 2]
    g = rgb_normalized[:, :, 1]
    b = rgb_normalized[:, :, 0]

    k = 1.0 - np.max(rgb_normalized, axis=2)
    k_inv = 1.0 - k
    k_inv[k_inv == 0] = 1e-10

    c = (1.0 - r - k) / k_inv
    m = (1.0 - g - k) / k_inv
    y = (1.0 - b - k) / k_inv

    return {
        "C": (c * 255).clip(0, 255).astype(np.uint8),
        "M": (m * 255).clip(0, 255).astype(np.uint8),
        "Y": (y * 255).clip(0, 255).astype(np.uint8),
        "K": (k * 255).clip(0, 255).astype(np.uint8),
    }


def detect_stamp_regions(
    image: np.ndarray,
    saturation_threshold: int = 40,
    value_min: int = 50,
) -> np.ndarray:
    """Return a binary mask where red/pink stamp-like pixels are 255."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, saturation_threshold, value_min])
    upper_red = np.array([30, 255, 255])

    lower_pink = np.array([150, saturation_threshold, value_min])
    upper_pink = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_pink, upper_pink)
    red_mask = cv2.bitwise_or(mask1, mask2)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    not_too_dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]
    red_mask = cv2.bitwise_and(red_mask, not_too_dark)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel_expand = np.ones((10, 10), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel_expand, iterations=1)

    return red_mask


def remove_stamp(
    image: np.ndarray,
    saturation_threshold: int = 40,
    blend_border: bool = True,
    border_size: int = 3,
) -> np.ndarray:
    """Remove red/pink stamps from an OpenCV image and return a new image."""
    stamp_regions = detect_stamp_regions(
        image, saturation_threshold=saturation_threshold
    )

    if cv2.countNonZero(stamp_regions) == 0:
        return image.copy()

    cmyk = rgb_to_cmyk(image)
    black_channel = 255 - cmyk["K"]
    black_enhanced = cv2.convertScaleAbs(black_channel, alpha=1.5, beta=30)
    _, binary = cv2.threshold(
        black_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if not blend_border or border_size <= 0:
        mask = stamp_regions[:, :, np.newaxis]
        return np.where(mask == 255, processed, image)

    stamp_regions_float = stamp_regions.astype(np.float32) / 255.0
    blurred_mask = cv2.GaussianBlur(
        stamp_regions_float, (border_size * 2 + 1, border_size * 2 + 1), 0
    )
    blurred_mask = blurred_mask[:, :, np.newaxis]
    blended = (image * (1.0 - blurred_mask) + processed * blurred_mask).astype(
        np.uint8
    )
    return blended


def remove_stamp_from_path(
    image_path: Path | str,
    saturation_threshold: int = 40,
    blend_border: bool = True,
    border_size: int = 3,
) -> np.ndarray:
    """Load an image from disk, remove the stamp, and return the OpenCV image."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    return remove_stamp(
        image,
        saturation_threshold=saturation_threshold,
        blend_border=blend_border,
        border_size=border_size,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove red stamps from document images."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to an input image readable by OpenCV.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output path. Defaults to <input>_stamp_removed.<ext>.",
    )
    parser.add_argument(
        "--saturation-threshold",
        type=int,
        default=40,
        help="Color saturation threshold (lower is more sensitive).",
    )
    parser.add_argument(
        "--border-size",
        type=int,
        default=3,
        help="Border size in pixels for blending processed regions.",
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Disable border blending; replacements are hard-edged.",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_stamp_removed{input_path.suffix}")


def main() -> None:
    args = _parse_args()
    output_path = args.output or _default_output_path(args.image_path)

    result = remove_stamp_from_path(
        args.image_path,
        saturation_threshold=args.saturation_threshold,
        blend_border=not args.no_blend,
        border_size=args.border_size,
    )

    cv2.imwrite(str(output_path), result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
