"""
High-level orchestrator to run the analysis step on scanned quiz responses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI

from .anchors import Anchors, extract_anchors, load_anchors
from .decompose import PdfCutter, RegionCrop
from .llm import (
    DEFAULT_VISION_MODEL,
    build_openai_client,
    call_vision,
    image_file_to_data_url,
)
from .utils import ensure_directory, write_json

PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "analysis.prompt.md"
TITLE_PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "title_page.prompt.md"

ProgressCallback = Callable[[str, Dict[str, Any]], None]


@dataclass(frozen=True)
class AnalysisItem:
    """Single analyzed question region."""

    question_id: int
    page_index: int
    region_index: int
    image_path: Path
    raw_response: str
    parsed_json: Optional[object]
    question_kind: Optional[str] = None
    summary: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class UsageSummary:
    """Aggregated token usage for the whole run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class AnalysisResult:
    """Rich result returned by :func:`run_analysis`."""

    items: List[AnalysisItem]
    usage: UsageSummary
    elapsed_seconds: float
    total_pages: int
    pages_with_regions: int
    pages_processed: int
    total_regions_expected: int
    questions_processed: int
    question_ids: List[int]
    expected_question_ids: List[int]
    missing_question_ids: List[int]
    ambiguous_question_ids: List[int]

    def __iter__(self) -> Iterable[AnalysisItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> AnalysisItem:
        return self.items[index]


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


def _utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 with second precision."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _extract_usage(response: Any) -> Dict[str, int]:
    """Normalize token usage information from the OpenAI response object."""
    usage_obj = getattr(response, "usage", None)
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    if usage_obj is None:
        return totals

    def _pluck(obj: Any, *candidates: str) -> int:
        for name in candidates:
            value = getattr(obj, name, None) if not isinstance(obj, dict) else obj.get(name)
            if isinstance(value, int):
                return value
        return 0

    prompt_tokens = _pluck(usage_obj, "input_tokens", "prompt_tokens")
    completion_tokens = _pluck(usage_obj, "output_tokens", "completion_tokens")
    total_tokens = _pluck(usage_obj, "total_tokens")
    totals["input_tokens"] = prompt_tokens
    totals["output_tokens"] = completion_tokens
    totals["total_tokens"] = total_tokens or (prompt_tokens + completion_tokens)
    return totals


def _extract_question_kind(parsed: Optional[object]) -> Optional[str]:
    """Try to identify the question type from the parsed JSON payload."""
    if parsed is None:
        return None
    nodes: Sequence[object]
    if isinstance(parsed, list):
        nodes = parsed
    else:
        nodes = [parsed]
    for node in nodes:
        if isinstance(node, dict):
            kind = node.get("kind")
            if isinstance(kind, str) and kind.strip():
                return kind.strip()
    return None


def _extract_summary_text(parsed: Optional[object], raw_text: str, max_len: int = 160) -> str:
    """Build a short human-readable summary from the parsed JSON or raw text."""
    candidates: List[str] = []
    nodes: List[object]
    if isinstance(parsed, list):
        nodes = parsed
    elif parsed is None:
        nodes = []
    else:
        nodes = [parsed]

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key in ("analysis", "summary"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
                break
        if candidates:
            break
        handwriting = node.get("handwriting")
        if isinstance(handwriting, str) and handwriting.strip():
            candidates.append(handwriting.strip())
            break

    if not candidates:
        text = raw_text.strip()
    else:
        text = candidates[0]

    text = " ".join(text.split())
    if not text:
        return "No summary available."
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def _detect_ambiguity(parsed: Optional[object]) -> bool:
    """Heuristic to detect ambiguous markings in the parsed payload."""
    markers = ("ambig", "uncertain", "doute", "indet", "dubious", "incertain")
    nodes: List[object]
    if isinstance(parsed, list):
        nodes = parsed
    elif parsed is None:
        nodes = []
    else:
        nodes = [parsed]

    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key in ("analysis", "summary", "comment", "status"):
            value = node.get(key)
            if isinstance(value, str) and any(marker in value.lower() for marker in markers):
                return True
        choices = node.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                for key in ("mark", "analysis", "comment"):
                    value = choice.get(key)
                    if isinstance(value, str) and any(marker in value.lower() for marker in markers):
                        return True
    return False


def _question_range(ids: Iterable[int]) -> Optional[Dict[str, int]]:
    values = sorted(set(ids))
    if not values:
        return None
    return {"min": values[0], "max": values[-1]}


def _relative_image_path(image_path: Path, base_dir: Path) -> str:
    try:
        return str(image_path.relative_to(base_dir))
    except ValueError:
        return str(image_path)


def _extract_title_metadata(
    client: OpenAI,
    *,
    image_path: Path,
    prompt_path: Path,
    model: str,
    user_label: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, int], str]:
    """
    Run a dedicated vision prompt on the title page to extract student metadata.
    """
    prompt_text = prompt_path.read_text(encoding="utf-8")
    data_url = image_file_to_data_url(image_path)
    response = call_vision(
        client,
        prompt=prompt_text,
        image_data_url=data_url,
        model=model,
        user=user_label,
    )
    raw_text = getattr(response, "output_text", "") or ""
    parsed = _parse_response_json(raw_text)
    metadata: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        metadata = parsed
    usage = _extract_usage(response)
    return metadata, usage, raw_text


def run_analysis(
    *,
    responses_pdf: Path,
    anchors: Anchors,
    output_dir: Path,
    client: Optional[OpenAI] = None,
    prompt_path: Path = PROMPT_PATH,
    title_prompt_path: Optional[Path] = None,
    model: str = DEFAULT_VISION_MODEL,
    dpi: int = 220,
    user_label: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    """
    Run the end-to-end analysis on `responses_pdf` and return an :class:`AnalysisResult`.
    """
    if not responses_pdf.exists():
        raise FileNotFoundError(f"Responses PDF does not exist: {responses_pdf}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    effective_title_prompt = title_prompt_path or TITLE_PROMPT_PATH
    client = client or build_openai_client()
    images_dir = ensure_directory(output_dir / "images")

    expected_question_ids = sorted({region.qnum for page in anchors.pages for region in page.regions_mm})
    expected_question_id_set = set(expected_question_ids)
    total_regions_expected = sum(len(page.regions_mm) for page in anchors.pages)
    pages_with_regions = sum(1 for page in anchors.pages if page.regions_mm)

    def emit(event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if progress is None:
            return
        progress(event, payload or {})

    start_perf = time.perf_counter()
    started_at = _utc_now_iso()

    aggregated: Dict[str, Any] = {
        "source_pdf": str(responses_pdf),
        "model": model,
        "prompt_path": str(prompt_path),
        "title_prompt_path": str(effective_title_prompt) if effective_title_prompt else None,
        "dpi": dpi,
        "user_label": user_label,
        "started_at": started_at,
        "completed_at": None,
        "duration_seconds": None,
        "stats": {
            "total_pages": 0,
            "pages_with_regions": pages_with_regions,
            "pages_processed": 0,
            "total_regions_expected": total_regions_expected,
            "questions_processed": 0,
            "question_ids": [],
            "question_range": None,
            "missing_question_ids": [],
            "ambiguous_question_ids": [],
        },
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "metadata": {
            "student_name": "",
            "student_name_confidence": "",
            "student_name_raw": "",
            "notes": "",
            "grading_date": started_at.split("T")[0],
            "title_page_image": None,
            "raw_response": "",
        },
        "items": [],
    }
    analysis_path = output_dir / "analysis.json"
    write_json(analysis_path, aggregated)

    emit(
        "run:init",
        {
            "total_regions_expected": total_regions_expected,
            "pages_with_regions": pages_with_regions,
        },
    )

    cutter = PdfCutter(dpi=dpi)
    page_images = cutter.render_pdf_to_images(responses_pdf, images_dir)
    total_pages = len(page_images)
    aggregated["stats"]["total_pages"] = total_pages
    title_metadata_notes: List[str] = []
    if page_images and effective_title_prompt and effective_title_prompt.exists():
        try:
            title_metadata, title_usage, raw_title_text = _extract_title_metadata(
                client,
                image_path=page_images[0].path,
                prompt_path=effective_title_prompt,
                model=model,
                user_label=user_label,
            )
            metadata_block = aggregated["metadata"]
            metadata_block["title_page_image"] = _relative_image_path(page_images[0].path, output_dir)
            metadata_block["raw_response"] = raw_title_text
            if title_metadata:
                student_name = title_metadata.get("student_name") or ""
                metadata_block["student_name"] = str(student_name).strip()
                confidence = title_metadata.get("student_name_confidence") or ""
                metadata_block["student_name_confidence"] = str(confidence).strip()
                raw_name = title_metadata.get("student_name_raw") or student_name
                metadata_block["student_name_raw"] = str(raw_name).strip()
                notes = title_metadata.get("notes")
                if isinstance(notes, str) and notes.strip():
                    metadata_block["notes"] = notes.strip()
            for key, value in title_usage.items():
                if key in aggregated["usage"]:
                    aggregated["usage"][key] += int(value)
        except Exception as exc:  # pragma: no cover - defensive
            title_metadata_notes.append(f"Erreur d'extraction page titre: {exc}")
    if title_metadata_notes:
        existing_notes = aggregated["metadata"].get("notes")
        note_parts = []
        if isinstance(existing_notes, str) and existing_notes.strip():
            note_parts.append(existing_notes.strip())
        note_parts.extend(title_metadata_notes)
        aggregated["metadata"]["notes"] = "; ".join(note_parts)

    write_json(analysis_path, aggregated)
    emit(
        "run:pages_ready",
        {
            "total_pages": total_pages,
        },
    )

    pages_by_index = {page.page_index: page for page in anchors.pages}
    crops_per_page: Dict[int, List[RegionCrop]] = {}

    for page_image in page_images:
        page_idx = page_image.page_index
        emit(
            "page:start",
            {
                "page_index": page_idx,
                "page_number": page_idx + 1,
            },
        )

        anchor_entry = pages_by_index.get(page_idx)
        emit(
            "page:rendered",
            {
                "page_index": page_idx,
                "page_number": page_idx + 1,
                "image_path": str(page_image.path),
                "width": page_image.width,
                "height": page_image.height,
            },
        )

        if anchor_entry is None:
            emit(
                "page:skip",
                {
                    "page_index": page_idx,
                    "page_number": page_idx + 1,
                    "reason": "no-anchors",
                },
            )
            continue

        if not anchor_entry.regions_mm:
            emit(
                "page:skip",
                {
                    "page_index": page_idx,
                    "page_number": page_idx + 1,
                    "reason": "no-regions",
                },
            )
            continue

        emit(
            "page:regions",
            {
                "page_index": page_idx,
                "page_number": page_idx + 1,
                "count": len(anchor_entry.regions_mm),
            },
        )

        page_height_mm = float(anchor_entry.page_height_mm or 0.0)
        if page_height_mm <= 0:
            emit(
                "page:skip",
                {
                    "page_index": page_idx,
                    "page_number": page_idx + 1,
                    "reason": "invalid-page-height",
                },
            )
            continue

        region_dicts = [region.model_dump() for region in anchor_entry.regions_mm]
        crops = cutter.crop_page_image(
            image_path=page_image.path,
            regions_mm=region_dicts,
            page_height_mm=page_height_mm,
            output_dir=images_dir,
            page_index=page_idx,
            base_output_stem=page_image.path.stem,
        )
        crops_per_page[page_idx] = crops
        emit(
            "page:crops-ready",
            {
                "page_index": page_idx,
                "page_number": page_idx + 1,
                "count": len(crops),
            },
        )

    items: List[AnalysisItem] = []
    aggregate_usage = dict(aggregated["usage"])
    processed_pages: set[int] = set()
    question_ids_seen: set[int] = set()
    ambiguous_questions: set[int] = set()
    total_questions = sum(len(crops) for crops in crops_per_page.values()) or total_regions_expected
    question_counter = 0

    for page_idx in sorted(crops_per_page.keys()):
        crops = crops_per_page[page_idx]
        if not crops:
            continue

        emit(
            "page:process",
            {
                "page_index": page_idx,
                "page_number": page_idx + 1,
                "question_count": len(crops),
            },
        )
        processed_pages.add(page_idx)

        for crop in crops:
            question_counter += 1
            emit(
                "question:start",
                {
                    "page_index": crop.page_index,
                    "page_number": crop.page_index + 1,
                    "question_id": crop.question_id,
                    "region_index": crop.region_index,
                    "position": question_counter,
                    "total": total_questions,
                },
            )

            image_data_url = image_file_to_data_url(crop.path)
            emit(
                "question:request",
                {
                    "question_id": crop.question_id,
                    "page_number": crop.page_index + 1,
                    "position": question_counter,
                    "total": total_questions,
                },
            )
            response = call_vision(
                client,
                prompt=prompt_text,
                image_data_url=image_data_url,
                model=model,
                user=user_label,
            )
            raw_text = getattr(response, "output_text", None) or ""
            parsed = _parse_response_json(raw_text)
            question_kind = _extract_question_kind(parsed)
            summary_text = _extract_summary_text(parsed, raw_text)
            usage = _extract_usage(response)

            if _detect_ambiguity(parsed):
                ambiguous_questions.add(crop.question_id)

            item = AnalysisItem(
                question_id=crop.question_id,
                page_index=crop.page_index,
                region_index=crop.region_index,
                image_path=crop.path,
                raw_response=raw_text,
                parsed_json=parsed,
                question_kind=question_kind,
                summary=summary_text,
                usage=usage,
            )
            items.append(item)

            aggregate_usage["input_tokens"] += usage.get("input_tokens", 0)
            aggregate_usage["output_tokens"] += usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens")
            if total_tokens is None:
                total_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            aggregate_usage["total_tokens"] += total_tokens

            question_ids_seen.add(crop.question_id)

            aggregated["items"].append(
                {
                    "question_id": item.question_id,
                    "page_index": item.page_index,
                    "region_index": item.region_index,
                    "image": _relative_image_path(item.image_path, output_dir),
                    "raw_response": item.raw_response,
                    "json": item.parsed_json,
                    "question_kind": item.question_kind,
                    "summary": item.summary,
                    "usage": item.usage,
                }
            )

            stats = aggregated["stats"]
            stats["questions_processed"] = len(items)
            stats["pages_processed"] = len(processed_pages)
            stats["question_ids"] = sorted(question_ids_seen)
            stats["question_range"] = _question_range(question_ids_seen)
            if expected_question_id_set:
                stats["missing_question_ids"] = sorted(expected_question_id_set - question_ids_seen)
            else:
                stats["missing_question_ids"] = []
            stats["ambiguous_question_ids"] = sorted(ambiguous_questions)
            aggregated["usage"] = aggregate_usage

            write_json(analysis_path, aggregated)

            emit(
                "question:result",
                {
                    "question_id": crop.question_id,
                    "page_number": crop.page_index + 1,
                    "question_kind": question_kind,
                    "summary": summary_text,
                    "usage": usage,
                    "position": question_counter,
                    "total": total_questions,
                },
            )

    elapsed_seconds = time.perf_counter() - start_perf
    completed_at = _utc_now_iso()

    aggregated["completed_at"] = completed_at
    aggregated["duration_seconds"] = elapsed_seconds
    aggregated["usage"] = aggregate_usage
    stats = aggregated["stats"]
    stats["questions_processed"] = len(items)
    stats["pages_processed"] = len(processed_pages)
    stats["question_ids"] = sorted(question_ids_seen)
    stats["question_range"] = _question_range(question_ids_seen)
    if expected_question_id_set:
        stats["missing_question_ids"] = sorted(expected_question_id_set - question_ids_seen)
    stats["ambiguous_question_ids"] = sorted(ambiguous_questions)
    write_json(analysis_path, aggregated)

    emit(
        "run:complete",
        {
            "elapsed_seconds": elapsed_seconds,
            "questions_processed": len(items),
            "total_pages": total_pages,
        },
    )

    usage_summary = UsageSummary(
        input_tokens=aggregate_usage["input_tokens"],
        output_tokens=aggregate_usage["output_tokens"],
        total_tokens=aggregate_usage["total_tokens"],
    )

    return AnalysisResult(
        items=items,
        usage=usage_summary,
        elapsed_seconds=elapsed_seconds,
        total_pages=total_pages,
        pages_with_regions=pages_with_regions,
        pages_processed=len(processed_pages),
        total_regions_expected=total_regions_expected,
        questions_processed=len(items),
        question_ids=sorted(question_ids_seen),
        expected_question_ids=expected_question_ids,
        missing_question_ids=stats["missing_question_ids"],
        ambiguous_question_ids=sorted(ambiguous_questions),
    )


def analysis_output_schema() -> Dict[str, Any]:
    """Return the JSON Schema describing the analysis output file."""
    usage_schema = {
        "type": "object",
        "required": ["input_tokens", "output_tokens", "total_tokens"],
        "properties": {
            "input_tokens": {"type": "integer", "minimum": 0},
            "output_tokens": {"type": "integer", "minimum": 0},
            "total_tokens": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "QuizAIAnalysis",
        "type": "object",
        "required": [
            "source_pdf",
            "model",
            "prompt_path",
            "dpi",
            "items",
            "stats",
            "usage",
            "metadata",
        ],
        "properties": {
            "source_pdf": {"type": "string"},
            "model": {"type": "string"},
            "prompt_path": {"type": "string"},
            "title_prompt_path": {"type": ["string", "null"]},
            "dpi": {"type": "integer", "minimum": 60},
            "user_label": {"type": ["string", "null"]},
            "started_at": {"type": "string", "format": "date-time"},
            "completed_at": {"type": ["string", "null"], "format": "date-time"},
            "duration_seconds": {"type": ["number", "null"], "minimum": 0},
            "metadata": {
                "type": "object",
                "required": [
                    "student_name",
                    "student_name_confidence",
                    "student_name_raw",
                    "notes",
                    "grading_date",
                    "title_page_image",
                    "raw_response",
                ],
                "properties": {
                    "student_name": {"type": "string"},
                    "student_name_confidence": {"type": "string"},
                    "student_name_raw": {"type": "string"},
                    "notes": {"type": "string"},
                    "grading_date": {"type": "string"},
                    "title_page_image": {"type": ["string", "null"]},
                    "raw_response": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "stats": {
                "type": "object",
                "required": [
                    "total_pages",
                    "pages_with_regions",
                    "pages_processed",
                    "total_regions_expected",
                    "questions_processed",
                    "question_ids",
                    "question_range",
                    "missing_question_ids",
                    "ambiguous_question_ids",
                ],
                "properties": {
                    "total_pages": {"type": "integer", "minimum": 0},
                    "pages_with_regions": {"type": "integer", "minimum": 0},
                    "pages_processed": {"type": "integer", "minimum": 0},
                    "total_regions_expected": {"type": "integer", "minimum": 0},
                    "questions_processed": {"type": "integer", "minimum": 0},
                    "question_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "uniqueItems": True,
                    },
                    "question_range": {
                        "type": ["object", "null"],
                        "required": ["min", "max"],
                        "properties": {
                            "min": {"type": "integer"},
                            "max": {"type": "integer"},
                        },
                        "additionalProperties": False,
                    },
                    "missing_question_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "uniqueItems": True,
                    },
                    "ambiguous_question_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "uniqueItems": True,
                    },
                },
                "additionalProperties": False,
            },
            "usage": usage_schema,
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "question_id",
                        "page_index",
                        "region_index",
                        "image",
                        "raw_response",
                        "json",
                        "question_kind",
                        "summary",
                        "usage",
                    ],
                    "properties": {
                        "question_id": {"type": "integer"},
                        "page_index": {"type": "integer"},
                        "region_index": {"type": "integer"},
                        "image": {"type": "string"},
                        "raw_response": {"type": "string"},
                        "json": {
                            "type": [
                                "object",
                                "array",
                                "string",
                                "number",
                                "boolean",
                                "null",
                            ],
                        },
                        "question_kind": {"type": ["string", "null"]},
                        "summary": {"type": ["string", "null"]},
                        "usage": usage_schema,
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
