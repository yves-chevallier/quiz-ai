"""
Extract question regions from a PDF annotated with hyperref anchors and return
a robust Pydantic schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import fitz  # PyMuPDF
from pydantic import (
    BaseModel,
    Field,
    model_validator,  # Pydantic v2
)

PT_TO_MM: float = 25.4 / 72.0
INCLUDE_BOTTOM_SEGMENT_DEFAULT: bool = True


class Anchor(BaseModel):
    """Single anchor (PDF bottom-left coordinate system, millimeters)."""

    x_mm: float = Field(..., ge=0, description="X in mm from the left edge.")
    y_mm: float = Field(..., ge=0, description="Y in mm from the bottom edge.")
    qnum: int = Field(..., ge=0, description="Question number extracted from the anchor name.")


class Region(BaseModel):
    """Vertical region of interest (horizontal band between two Y coordinates)."""

    qnum: int = Field(..., ge=0, description="Question number associated with this region.")
    y_start_mm: float = Field(..., ge=0, description="Lower bound (inclusive) in mm.")
    y_end_mm: float = Field(..., ge=0, description="Upper bound (inclusive) in mm.")

    @model_validator(mode="after")
    def _check_bounds(self) -> "Region":
        if self.y_end_mm <= self.y_start_mm:
            raise ValueError("y_end_mm must be strictly greater than y_start_mm")
        return self


class PageAnchors(BaseModel):
    """Anchors and regions extracted for a single page."""

    page_index: int = Field(..., ge=0, description="Zero-based page index.")
    page_height_mm: float = Field(..., gt=0, description="Page height in mm.")
    anchors_mm: List[Anchor] = Field(default_factory=list, description="Anchors (top to bottom).")
    regions_mm: List[Region] = Field(default_factory=list, description="Regions between anchors.")

    @model_validator(mode="after")
    def _sort_anchors_desc(self) -> "PageAnchors":
        # Ensure anchors are sorted from top to bottom (descending y)
        self.anchors_mm = sorted(self.anchors_mm, key=lambda a: a.y_mm, reverse=True)
        return self


class Anchors(BaseModel):
    """Top-level result schema."""

    pages: List[PageAnchors] = Field(default_factory=list, description="Per-page anchors and regions.")

    def json_pretty(self) -> str:
        """Convenience pretty JSON."""
        return json.dumps(self.model_dump(), ensure_ascii=False, indent=2)


def load_anchors(path: Path) -> Anchors:
    """
    Load anchors from a JSON file produced by :func:`extract_anchors`.
    """
    return Anchors.model_validate_json(path.read_text(encoding="utf-8"))


def save_anchors(anchors: Anchors, path: Path) -> None:
    """
    Persist anchors to disk as pretty-printed JSON.
    """
    path.write_text(anchors.json_pretty(), encoding="utf-8")


def _anchors_for_page(
    doc: fitz.Document,
    pno: int,
    r_tag: re.Pattern[str],
) -> List[Tuple[float, float, int]]:
    """
    Collect anchors for a given page as tuples: (x_mm_from_left, y_mm_from_bottom, qnum)
    from hyperref named destinations.
    """
    out: List[Tuple[float, float, int]] = []

    resolver = getattr(doc, "resolve_names", None)
    if not resolver:
        return out

    page = doc[pno]
    crop = page.cropbox

    for name, dest in resolver().items():
        m = r_tag.match(name)
        if not m or dest.get("page") != pno:
            continue

        x_pt, y_pt = dest.get("to", (None, None))
        if x_pt is None or y_pt is None:
            continue

        # Adjust by CropBox
        x_pt -= crop.x0
        y_pt -= crop.y0

        out.append((x_pt * PT_TO_MM, y_pt * PT_TO_MM, int(m.group(1))))

    # Sort top → bottom (descending Y)
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _regions_from_anchors(
    page_h_mm: float,
    anchors_mm: List[Tuple[float, float, int]],
    overlap_mm: float,
    include_bottom_segment: bool = INCLUDE_BOTTOM_SEGMENT_DEFAULT,
) -> List[Dict[str, float]]:
    """
    Build regions [y_start_mm, y_end_mm] between successive anchors with overlap.
    Y increases bottom → top (PDF coordinates). Regions are clamped to [0, page_h_mm].
    Optionally includes the bottom-of-page → last-anchor segment.
    """
    regions: List[Dict[str, float]] = []
    if not anchors_mm:
        return regions

    for j in range(len(anchors_mm) - 1):
        _, y_top, qnum = anchors_mm[j]
        _, y_bottom, _ = anchors_mm[j + 1]

        y_start = max(0.0, y_bottom - overlap_mm)
        y_end = min(page_h_mm, y_top + overlap_mm)

        if y_end - y_start > 0.1:
            regions.append({"qnum": qnum, "y_start_mm": y_start, "y_end_mm": y_end})

    if include_bottom_segment:
        _, y_last, q_last = anchors_mm[-1]
        y_start = 0.0
        y_end = min(page_h_mm, y_last + overlap_mm)
        if y_end - y_start > 0.1:
            regions.append({"qnum": q_last, "y_start_mm": y_start, "y_end_mm": y_end})

    return regions


def extract_anchors(
    file: Union[str, Path],
    overlap: float = 3.0,
    anchor_pattern: str = r"^Q(\d+)-anchor$",
    include_bottom_segment: bool = INCLUDE_BOTTOM_SEGMENT_DEFAULT,
) -> Anchors:
    """
    Extract anchors/regions from a PDF and return an Anchors (Pydantic) object.
    """
    pdf_path = Path(file)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if overlap < 0:
        raise ValueError("`overlap` must be >= 0")

    r_tag = re.compile(anchor_pattern)

    doc = fitz.open(str(pdf_path))
    try:
        pages_out: List[PageAnchors] = []
        # Use explicit indexing to avoid Pylance complaining about Document being non-iterable
        for pno in range(len(doc)):  # pylint: disable=consider-using-enumerate
            page = doc[pno]
            height_mm = page.rect.height * PT_TO_MM

            raw_anchors = _anchors_for_page(doc, pno, r_tag)
            anchors_models = [Anchor(x_mm=x, y_mm=y, qnum=q) for (x, y, q) in raw_anchors]

            raw_regions = _regions_from_anchors(
                page_h_mm=height_mm,
                anchors_mm=raw_anchors,
                overlap_mm=overlap,
                include_bottom_segment=include_bottom_segment,
            )
            regions_models = [Region(**r) for r in raw_regions]

            pages_out.append(
                PageAnchors(
                    page_index=pno,
                    page_height_mm=height_mm,
                    anchors_mm=anchors_models,
                    regions_mm=regions_models,
                )
            )

        return Anchors(pages=pages_out)
    finally:
        doc.close()
