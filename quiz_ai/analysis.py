"""
High-level orchestrator to run the analysis step on scanned quiz responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from .anchors import Anchors, extract_anchors, load_anchors
from .decompose import PdfCutter, RegionCrop
from .llm import DEFAULT_VISION_MODEL, build_openai_client, call_vision, image_file_to_data_url
from .utils import ensure_directory, write_json


PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "analysis.prompt.md"


@dataclass(frozen=True)
class AnalysisItem:
    """Single analyzed question region."""

    question_id: int
    page_index: int
    region_index: int
    image_path: Path
    raw_response: str
    parsed_json: Optional[object]


def load_or_extract_anchors(
    *,
    anchors_path: Optional[Path],
    source_pdf: Optional[Path],
    anchor_overlap_mm: float = 3.0,
) -> Anchors:
    """
    Load anchors from a JSON file or extract them from the source PDF if needed.
    """
    if anchors_path:
        return load_anchors(anchors_path)
    if not source_pdf:
        raise ValueError("Either anchors_path or source_pdf must be provided.")
    return extract_anchors(source_pdf, overlap=anchor_overlap_mm)


def _parse_response_json(raw_text: str) -> Optional[object]:
    """
    Attempt to parse the raw text returned by the model as JSON.
    """
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def run_analysis(
    *,
    responses_pdf: Path,
    anchors: Anchors,
    output_dir: Path,
    client: Optional[OpenAI] = None,
    prompt_path: Path = PROMPT_PATH,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = 220,
    user_label: Optional[str] = None,
) -> List[AnalysisItem]:
    """
    Run the end-to-end analysis on `responses_pdf` and return the list of items.
    """
    if not responses_pdf.exists():
        raise FileNotFoundError(f"Responses PDF does not exist: {responses_pdf}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    client = client or build_openai_client()

    images_dir = ensure_directory(output_dir / "images")
    cutter = PdfCutter(dpi=dpi)
    region_crops: List[RegionCrop] = cutter.crop_regions_for_pdf(
        pdf_path=responses_pdf,
        anchors=anchors,
        out_dir=images_dir,
    )

    items: List[AnalysisItem] = []
    aggregated = {
        "source_pdf": str(responses_pdf),
        "model": model,
        "prompt_path": str(prompt_path),
        "items": [],
    }

    for crop in region_crops:
        image_data_url = image_file_to_data_url(crop.path)
        response = call_vision(
            client,
            prompt=prompt_text,
            image_data_url=image_data_url,
            model=model,
            user=user_label,
        )
        raw_text = getattr(response, "output_text", None) or ""
        parsed = _parse_response_json(raw_text)

        item = AnalysisItem(
            question_id=crop.question_id,
            page_index=crop.page_index,
            region_index=crop.region_index,
            image_path=crop.path,
            raw_response=raw_text,
            parsed_json=parsed,
        )
        items.append(item)
        aggregated["items"].append(
            {
                "question_id": item.question_id,
                "page_index": item.page_index,
                "region_index": item.region_index,
                "image": str(item.image_path.relative_to(output_dir)),
                "raw_response": item.raw_response,
                "json": item.parsed_json,
            }
        )

    write_json(output_dir / "analysis.json", aggregated)
    return items

