#!/usr/bin/env python3
"""Convert OCR crop outputs into per-element Markdown and a merged document."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import requests


def remove_repetition(text: str, 
                      min_pattern_length: int = 1,
                      min_repetitions: int = 2,
                      context_chars: int = 50) -> str:
    """
    Remove repetitive patterns from text while preserving legitimate repetitions.
    
    Args:
        text: Input text to process
        min_pattern_length: Minimum length of pattern to consider (default: 3)
        min_repetitions: Minimum number of repetitions to remove (default: 2)
        context_chars: Characters to check around pattern for context (default: 50)
    
    Returns:
        Text with repetitive patterns removed
    """
    
    def is_legitimate_repetition(pattern: str, context_before: str, context_after: str) -> bool:
        """Check if a repetition should be preserved."""
        
        # Preserve single characters (punctuation, symbols)
        if len(pattern.strip()) <= 1:
            return True
        
        # Preserve patterns that are mostly digits (phone numbers, IDs, etc.)
        if re.match(r'^[\d\s\-\(\)\+\.]+$', pattern) and len(re.findall(r'\d', pattern)) >= 3:
            return True
        
        # Preserve common abbreviations and acronyms (2-5 uppercase letters)
        if re.match(r'^[A-Z]{2,5}$', pattern.strip()):
            return True
        
        # Preserve patterns with special formatting (code, URLs, emails)
        if any(char in pattern for char in ['@', '://', '\\', '{', '}', '[', ']', '<', '>']):
            return True
        
        # Preserve mathematical or technical notation
        if re.match(r'^[\w\d]+[\+\-\*/=\^]+[\w\d]+$', pattern.strip()):
            return True
        
        # Preserve currency and units
        if re.match(r'^[$€£¥₹]\s*[\d,\.]+|[\d,\.]+\s*[kmgtKMGT]?[bB]?$', pattern.strip()):
            return True
        
        # Check if it's part of a list or enumeration (e.g., "1. ", "a) ", "- ")
        if re.match(r'^[\d\w]+[\.\)\]\-]\s*$', pattern) or re.match(r'^[\-\*\+]\s+$', pattern):
            return True
        
        return False
    
    def find_repetitive_patterns(text: str) -> List[Tuple[int, int, str]]:
        """Find repetitive patterns and their positions."""
        patterns_to_remove = []
        
        # Normalize whitespace for pattern detection but keep track of original positions
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            # Look for consecutive repetitions within a line
            i = 0
            while i < len(line):
                # Try different pattern lengths
                for pattern_len in range(min_pattern_length, min(len(line) - i, 100) + 1):
                    pattern = line[i:i + pattern_len]
                    
                    # Count consecutive repetitions
                    reps = 1
                    pos = i + pattern_len
                    
                    while pos + pattern_len <= len(line):
                        next_segment = line[pos:pos + pattern_len]
                        if next_segment == pattern:
                            reps += 1
                            pos += pattern_len
                        else:
                            break
                    
                    # If we found repetitions
                    if reps >= min_repetitions:
                        context_before = line[max(0, i - context_chars):i]
                        context_after = line[pos:min(len(line), pos + context_chars)]
                        
                        if not is_legitimate_repetition(pattern, context_before, context_after):
                            # Calculate absolute position in original text
                            abs_start = sum(len(lines[j]) + 1 for j in range(line_idx)) + i
                            abs_end = sum(len(lines[j]) + 1 for j in range(line_idx)) + pos
                            patterns_to_remove.append((abs_start, abs_end, pattern))
                            i = pos  # Skip past this repetition
                            break
                else:
                    i += 1
                    continue
                break
        
        # Look for patterns that repeat across lines
        normalized = re.sub(r'\s+', ' ', text)
        words = normalized.split()
        
        for n in range(min_pattern_length, min(len(words), 20)):
            for i in range(len(words) - n * min_repetitions):
                pattern = ' '.join(words[i:i + n])
                
                # Check for repetitions
                reps = 1
                j = i + n
                
                while j + n <= len(words):
                    next_pattern = ' '.join(words[j:j + n])
                    if next_pattern == pattern:
                        reps += 1
                        j += n
                    else:
                        break
                
                if reps >= min_repetitions:
                    # Find position in original text
                    try:
                        start_pos = text.find(' '.join(words[i:i + n]))
                        end_pos = text.find(' '.join(words[j-n:j])) + len(' '.join(words[j-n:j]))
                        
                        if start_pos != -1 and end_pos != -1:
                            context_before = text[max(0, start_pos - context_chars):start_pos]
                            context_after = text[end_pos:min(len(text), end_pos + context_chars)]
                            
                            if not is_legitimate_repetition(pattern, context_before, context_after):
                                patterns_to_remove.append((start_pos, end_pos, pattern))
                    except:
                        pass
        
        # Sort by position and remove overlapping patterns (keep longest)
        patterns_to_remove.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered = []
        for pattern in patterns_to_remove:
            if not any(p[0] <= pattern[0] < p[1] or p[0] < pattern[1] <= p[1] 
                      for p in filtered):
                filtered.append(pattern)
        
        return filtered
    
    # Find and remove patterns
    patterns = find_repetitive_patterns(text)
    
    if not patterns:
        return text
    
    # Build result by removing repetitions (keep one instance)
    result = []
    last_end = 0
    
    for start, end, pattern in sorted(patterns, key=lambda x: x[0]):
        # Add text before this pattern
        result.append(text[last_end:start])
        
        # Add one instance of the pattern
        result.append(pattern)
        
        last_end = end
    
    # Add remaining text
    result.append(text[last_end:])
    
    return ''.join(result)



def standardize_deepseekocr_extract_label(data):
    # Pattern to extract text after each <|det|> block until the next tag or end
    pattern = r'<\|det\|>.*?<\|/det\|>\s*([\s\S]*?)(?=<\|ref\|>|$)'
    matches = re.findall(pattern, data, re.DOTALL)

    cleaned_lines = [m.strip() for m in matches if m.strip()]
    return remove_repetition("\n".join(cleaned_lines))

DEFAULT_API_URL = "http://localhost:9666/extract"

DEFAULT_SKIP_CATEGORIES = {"picture"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _collect_ocr_line_text(block: Any) -> List[str]:
    if not isinstance(block, dict):
        return []
    lines: List[str] = []
    for line in block.get("text_lines") or []:
        text = ""
        if isinstance(line, dict):
            text = str(line.get("text", "")).strip()
        else:
            text = str(line).strip()
        if text:
            lines.append(text)
    return lines


def extract_ocr_text(payload: Any) -> List[str]:
    """Best-effort extraction of text lines from the OCR response."""

    if not isinstance(payload, dict):
        return [str(payload).strip()] if payload else []

    texts: List[str] = []

    if isinstance(payload.get("results"), list):
        for item in payload["results"]:
            texts.extend(extract_ocr_text(item))

    result = payload.get("result")
    if isinstance(result, dict):
        texts.extend(_collect_ocr_line_text(result))
    elif isinstance(result, list):
        for item in result:
            texts.extend(_collect_ocr_line_text(item))
    elif result:
        texts.append(str(result).strip())

    if not texts and isinstance(payload.get("text_lines"), list):
        texts.extend(_collect_ocr_line_text(payload))

    return texts


def call_ocr_api(api_url: str, image_path: Path) -> dict:
    resolved = image_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")

    with resolved.open("rb") as handle:
        files = {"file": (resolved.name, handle, "application/octet-stream")}
        response = requests.post(api_url, files=files)

    response.raise_for_status()
    return response.json()


@dataclass
class CropElement:
    category: str
    index: int
    image_path: Optional[Path] = None
    html_path: Optional[Path] = None

    @property
    def markdown_path(self) -> Optional[Path]:
        if not self.image_path:
            return None
        return self.image_path.with_suffix(".md")


def parse_page_index(page_dir: Path) -> int:
    """Extract numeric page index from a page directory name."""
    match = re.search(r"page[_-]?(\d+)", page_dir.name, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def parse_element_index(stem: str) -> Optional[int]:
    """Extract numeric element index from a crop filename stem."""
    match = re.search(r"_(\d+)$", stem)
    return int(match.group(1)) if match else None


def resolve_pages_dir(layout_root: Path) -> Path:
    """Return the directory that directly contains page_* folders."""
    layout_root = layout_root.expanduser().resolve()
    candidate = layout_root / "pages"
    if candidate.is_dir():
        return candidate
    if layout_root.is_dir():
        return layout_root
    raise FileNotFoundError(f"Could not find pages directory under {layout_root}")


def collect_page_elements(page_dir: Path) -> List[CropElement]:
    """Gather crop images and HTML per page, grouped by element index."""
    crop_root = page_dir / "crop_elements"
    if not crop_root.is_dir():
        return []

    elements: dict[tuple[str, int], CropElement] = {}
    for category_dir in sorted(crop_root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for path in sorted(category_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix != ".html":
                continue
            element_index = parse_element_index(path.stem)
            if element_index is None:
                continue
            key = (category, element_index)
            entry = elements.get(key)
            if entry is None:
                entry = CropElement(category=category, index=element_index)
                elements[key] = entry
            if suffix == ".html":
                entry.html_path = path
            else:
                entry.image_path = path

    return sorted(
        elements.values(),
        key=lambda item: (item.index, item.category.lower()),
    )


def image_to_markdown(image_path: Path, api_url: str) -> str:
    """Run dots.ocr on a single crop image and return markdown text."""
    response = call_ocr_api(api_url, image_path)
    text_lines = extract_ocr_text(response)
    if isinstance(text_lines, list):
        content = "\n".join(line.strip() for line in text_lines if str(line).strip()).strip()
        return standardize_deepseekocr_extract_label(content)
    return str(text_lines).strip()


def write_markdown_file(path: Path, content: str) -> None:
    """Write markdown text to disk, always creating the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if content and not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def extract_page_markdown(
    page_dir: Path,
    api_url: str,
    skip_categories: Sequence[str],
) -> List[str]:
    """Create per-element markdown files for a page and return their contents."""
    parts: List[str] = []
    skip = {cat.lower() for cat in skip_categories}

    for element in collect_page_elements(page_dir):
        category = element.category.lower()
        if category in skip:
            continue

        if category == "table":
            if element.html_path and element.html_path.exists():
                html_text = element.html_path.read_text(encoding="utf-8").strip()
                if html_text:
                    parts.append(html_text)
            else:
                print(f"[warn] Table {element.index:04d} missing HTML in {page_dir}", file=sys.stderr)
            continue

        if not element.image_path:
            print(f"[warn] Missing image for {element.category} {element.index:04d} in {page_dir}", file=sys.stderr)
            continue

        markdown = image_to_markdown(element.image_path, api_url)

        md_path = element.markdown_path
        if md_path:
            write_markdown_file(md_path, markdown)

        cleaned = markdown.strip()
        if category == "page_footer" and len(cleaned) < 3 and cleaned.isdigit():
            continue

        if cleaned:
            parts.append(cleaned)

    return parts


def extract_markdown(
    layout_root: Path,
    output_path: Optional[Path] = None,
    skip_categories: Iterable[str] = DEFAULT_SKIP_CATEGORIES,
    api_url: str = DEFAULT_API_URL,
) -> str:
    """Extract markdown from every crop under layout_root and optionally write a merged file."""
    pages_dir = resolve_pages_dir(layout_root)

    merged: List[str] = []
    page_dirs = sorted(
        [item for item in pages_dir.iterdir() if item.is_dir()],
        key=parse_page_index,
    )

    for page_dir in page_dirs:
        merged.extend(
            extract_page_markdown(
                page_dir=page_dir,
                api_url=api_url,
                skip_categories=skip_categories,
            )
        )

    final_markdown = "\n\n".join(part.strip() for part in merged if part.strip())
    if output_path:
        output_path = output_path.expanduser().resolve()
        write_markdown_file(output_path, final_markdown)
    return final_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-crop markdown files and merge them into a single document."
    )
    parser.add_argument(
        "layout_root",
        type=Path,
        help="OCR output directory that contains the pages/ folder (or the pages folder itself).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the merged markdown. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Full OCR /get-text endpoint (default: http://localhost:7877/get-text).",
    )
    parser.add_argument(
        "--skip-category",
        action="append",
        default=list(DEFAULT_SKIP_CATEGORIES),
        help="Element category to skip entirely. Defaults to picture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown = extract_markdown(
        layout_root=args.layout_root,
        output_path=args.output,
        api_url=args.api_url,
        skip_categories=args.skip_category,
    )
    if args.output:
        print(f"Wrote merged markdown to {args.output}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
