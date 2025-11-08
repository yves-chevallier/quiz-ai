"""
Extract question regions from a PDF annotated with hyperref anchors and return
a robust Pydantic schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal, DefaultDict

import fitz  # PyMuPDF
from pydantic import (
    BaseModel,
    Field,
    model_validator,  # Pydantic v2
)

PT_TO_MM: float = 25.4 / 72.0
INCLUDE_BOTTOM_SEGMENT_DEFAULT: bool = True


QUESTION_ANCHOR_RE = re.compile(r"^Q(\d+)(?:-([A-Za-z0-9_-]+))?-anchor$")
PART_ANCHOR_RE = re.compile(r"^part@(\d+)@(\d+).*?-anchor$")
SUBPART_ANCHOR_RE = re.compile(r"^subpart@(\d+)@(\d+)@(\d+).*?-anchor$")
BOX_CORNER_RE = re.compile(r"^(?P<name>.+)-(tl|tr|bl|br)$")


QUESTION_END_SUFFIXES = {"end", "fin", "finish", "bottom", "stop", "tail"}
QUESTION_BREAK_SUFFIXES = {"break", "split", "mid", "page", "pagebreak", "pb", "continue"}


class Anchor(BaseModel):
    """Single anchor (PDF bottom-left coordinate system, millimeters)."""

    name: str = Field(..., description="Raw anchor name as found in the PDF.")
    kind: Literal["question", "question-end", "question-break", "part", "subpart", "other"] = Field(
        "other",
        description="Anchor classification used to derive regions.",
    )
    x_mm: float = Field(..., ge=0, description="X in mm from the left edge.")
    y_mm: float = Field(..., ge=0, description="Y in mm from the bottom edge.")
    qnum: int = Field(..., ge=0, description="Question number extracted from the anchor name.")
    part: Optional[int] = Field(None, ge=1, description="Part index if applicable.")
    subpart: Optional[int] = Field(None, ge=1, description="Subpart index if applicable.")


class Region(BaseModel):
    """Vertical region of interest (horizontal band between two Y coordinates)."""

    qnum: int = Field(..., ge=0, description="Question number associated with this region.")
    y_start_mm: float = Field(..., ge=0, description="Lower bound (inclusive) in mm.")
    y_end_mm: float = Field(..., ge=0, description="Upper bound (inclusive) in mm.")
    part: Optional[int] = Field(None, ge=1, description="Part index if the region targets a part.")
    subpart: Optional[int] = Field(
        None,
        ge=1,
        description="Subpart index if the region targets a specific subpart.",
    )

    @model_validator(mode="after")
    def _check_bounds(self) -> "Region":
        if self.y_end_mm <= self.y_start_mm:
            raise ValueError("y_end_mm must be strictly greater than y_start_mm")
        return self


class BoxRegion(BaseModel):
    """Named rectangular box defined by four anchors."""

    name: str = Field(..., description="Base name of the box (without -tl etc.).")
    x_min_mm: float = Field(..., ge=0, description="Left edge in mm from the left.")
    x_max_mm: float = Field(..., ge=0, description="Right edge in mm from the left.")
    y_min_mm: float = Field(..., ge=0, description="Bottom edge in mm from the bottom.")
    y_max_mm: float = Field(..., ge=0, description="Top edge in mm from the bottom.")

    @model_validator(mode="after")
    def _check_coords(self) -> "BoxRegion":
        if self.x_max_mm <= self.x_min_mm:
            raise ValueError("x_max_mm must be greater than x_min_mm")
        if self.y_max_mm <= self.y_min_mm:
            raise ValueError("y_max_mm must be greater than y_min_mm")
        return self


class PageAnchors(BaseModel):
    """Anchors and regions extracted for a single page."""

    page_index: int = Field(..., ge=0, description="Zero-based page index.")
    page_height_mm: float = Field(..., gt=0, description="Page height in mm.")
    page_width_mm: float = Field(..., gt=0, description="Page width in mm.")
    anchors_mm: List[Anchor] = Field(default_factory=list, description="Anchors (top to bottom).")
    regions_mm: List[Region] = Field(default_factory=list, description="Regions between anchors.")
    boxes_mm: List[BoxRegion] = Field(default_factory=list, description="Named rectangular boxes.")

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


def _classify_anchor(name: str) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Map an anchor name to a semantic kind plus optional part/subpart indices.
    """
    part_match = PART_ANCHOR_RE.match(name)
    if part_match:
        try:
            return "part", int(part_match.group(2)), None
        except (IndexError, ValueError):
            return "part", None, None

    subpart_match = SUBPART_ANCHOR_RE.match(name)
    if subpart_match:
        try:
            return "subpart", int(subpart_match.group(2)), int(subpart_match.group(3))
        except (IndexError, ValueError):
            return "subpart", None, None

    question_match = QUESTION_ANCHOR_RE.match(name)
    if question_match:
        suffix = question_match.group(2)
        if suffix:
            suffix_clean = suffix.strip().lower()
            if suffix_clean in QUESTION_END_SUFFIXES:
                return "question-end", None, None
            if suffix_clean in QUESTION_BREAK_SUFFIXES:
                return "question-break", None, None
            return "other", None, None
        return "question", None, None

    return "other", None, None


def _anchors_for_page(
    doc: fitz.Document,
    pno: int,
    r_tag: re.Pattern[str],
) -> Tuple[List[Anchor], List[Tuple[str, float, float]]]:
    """
    Collect anchors for a given page as Anchor objects with coordinates.
    """
    out: List[Anchor] = []
    raw_entries: List[Tuple[str, float, float]] = []

    resolver = getattr(doc, "resolve_names", None)
    if not resolver:
        return out

    page = doc[pno]
    crop = page.cropbox

    for name, dest in resolver().items():
        if dest.get("page") != pno:
            continue

        x_pt, y_pt = dest.get("to", (None, None))
        if x_pt is None or y_pt is None:
            continue

        # Adjust by CropBox
        x_pt -= crop.x0
        y_pt -= crop.y0

        x_mm = x_pt * PT_TO_MM
        y_mm = y_pt * PT_TO_MM
        raw_entries.append((name, x_mm, y_mm))

        m = r_tag.match(name)
        if not m:
            continue

        kind, part_idx, subpart_idx = _classify_anchor(name)

        out.append(
            Anchor(
                name=name,
                kind=kind,
                x_mm=x_mm,
                y_mm=y_mm,
                qnum=int(m.group(1)),
                part=part_idx,
                subpart=subpart_idx,
            )
        )

    # Sort top → bottom (descending Y)
    out.sort(key=lambda anchor: anchor.y_mm, reverse=True)
    return out, raw_entries


def _question_split_levels(pages: List[PageAnchors]) -> Dict[int, int]:
    """
    Determine whether each question should be split by question (0), part (1), or subpart (2).
    """
    levels: Dict[int, int] = {}
    for page in pages:
        for anchor in page.anchors_mm:
            q = anchor.qnum
            if anchor.kind == "part":
                levels[q] = 1
            elif anchor.kind == "subpart":
                if levels.get(q, 0) < 1:
                    levels[q] = max(levels.get(q, 0), 2)
            else:
                levels.setdefault(q, 0)
    return levels


def _anchor_starts_region(anchor: Anchor, split_level: int) -> bool:
    if split_level == 1:
        return anchor.kind == "part"
    if split_level == 2:
        return anchor.kind == "subpart"
    # Default: question-level extraction
    return anchor.kind == "question"


def _is_boundary(
    current: Anchor,
    candidate: Anchor,
    split_level: int,
) -> bool:
    if split_level == 1:
        if candidate.kind == "part" and candidate.qnum == current.qnum:
            return True
        if candidate.kind in {"question-end", "question-break"} and candidate.qnum == current.qnum:
            return True
        if candidate.kind == "question" and candidate.qnum != current.qnum:
            return True
        if candidate.kind == "part" and candidate.qnum != current.qnum:
            return True
        return False

    if split_level == 2:
        if candidate.kind == "subpart" and candidate.qnum == current.qnum:
            return True
        if candidate.kind in {"question-end", "question-break"} and candidate.qnum == current.qnum:
            return True
        if candidate.kind == "question" and candidate.qnum != current.qnum:
            return True
        return False

    # Question-level (default)
    if candidate.kind in {"question-end", "question-break"} and candidate.qnum == current.qnum:
        return True
    if candidate.kind == "question" and candidate.qnum != current.qnum:
        return True
    return False


def _regions_for_page(
    page: PageAnchors,
    split_levels: Dict[int, int],
    overlap_mm: float,
    include_bottom_segment: bool = INCLUDE_BOTTOM_SEGMENT_DEFAULT,
) -> List[Dict[str, float]]:
    """
    Build vertical regions for a single page based on anchors and question split levels.
    """
    regions: List[Dict[str, float]] = []
    anchors_mm = page.anchors_mm
    if not anchors_mm:
        return regions

    for idx, anchor in enumerate(anchors_mm):
        split_level = split_levels.get(anchor.qnum, 0)
        if not _anchor_starts_region(anchor, split_level):
            continue

        boundary_anchor: Optional[Anchor] = None
        for candidate in anchors_mm[idx + 1 :]:
            if _is_boundary(anchor, candidate, split_level):
                boundary_anchor = candidate
                break

        if boundary_anchor is None and not include_bottom_segment:
            continue

        y_top = anchor.y_mm
        y_bottom = boundary_anchor.y_mm if boundary_anchor else 0.0
        y_start = max(0.0, y_bottom - overlap_mm)
        y_end = min(page.page_height_mm, y_top + overlap_mm)

        if y_end - y_start > 0.1:
            reg: Dict[str, Union[int, float]] = {
                "qnum": anchor.qnum,
                "y_start_mm": y_start,
                "y_end_mm": y_end,
            }
            if anchor.part:
                reg["part"] = anchor.part
            if anchor.subpart:
                reg["subpart"] = anchor.subpart
            regions.append(reg)

    return regions


def _boxes_from_entries(
    entries: List[Tuple[str, float, float]],
) -> List[BoxRegion]:
    grouped: DefaultDict[str, Dict[str, Tuple[float, float]]] = DefaultDict(dict)
    for name, x_mm, y_mm in entries:
        match = BOX_CORNER_RE.match(name)
        if not match:
            continue
        corner = name.rsplit("-", 1)[-1]
        base = match.group("name")
        grouped[base][corner] = (x_mm, y_mm)

    boxes: List[BoxRegion] = []
    required = {"tl", "tr", "bl", "br"}
    for base_name, corners in grouped.items():
        if not required.issubset(corners):
            continue
        xs = [corners[c][0] for c in required]
        ys = [corners[c][1] for c in required]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        boxes.append(
            BoxRegion(
                name=base_name,
                x_min_mm=x_min,
                x_max_mm=x_max,
                y_min_mm=y_min,
                y_max_mm=y_max,
            )
        )
    return boxes


def extract_anchors(
    file: Union[str, Path],
    overlap: float = 3.0,
    anchor_pattern: str = r"^(?:Q|part@|subpart@)(\d+).*?-anchor$",
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
        raw_entries_per_page: Dict[int, List[Tuple[str, float, float]]] = {}
        # Use explicit indexing to avoid Pylance complaining about Document being non-iterable
        for pno in range(len(doc)):  # pylint: disable=consider-using-enumerate
            page = doc[pno]
            height_mm = page.rect.height * PT_TO_MM
            width_mm = page.rect.width * PT_TO_MM

            anchors_models, raw_entries = _anchors_for_page(doc, pno, r_tag)
            raw_entries_per_page[pno] = raw_entries
            pages_out.append(
                PageAnchors(
                    page_index=pno,
                    page_height_mm=height_mm,
                    page_width_mm=width_mm,
                    anchors_mm=anchors_models,
                    regions_mm=[],
                    boxes_mm=[],
                )
            )

        split_levels = _question_split_levels(pages_out)
        for page in pages_out:
            raw_regions = _regions_for_page(
                page=page,
                split_levels=split_levels,
                overlap_mm=overlap,
                include_bottom_segment=include_bottom_segment,
            )
            page.regions_mm = [Region(**region_dict) for region_dict in raw_regions]
            raw_entries = raw_entries_per_page.get(page.page_index, [])
            page.boxes_mm = _boxes_from_entries(raw_entries)

        return Anchors(pages=pages_out)
    finally:
        doc.close()
