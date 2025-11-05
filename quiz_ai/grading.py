"""
Helpers for grading analysis JSON files against the official solution using an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from .llm import build_openai_client
from .utils import read_yaml

PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "grading.prompt.md"
DEFAULT_GRADING_MODEL = "gpt-4o"


@dataclass(frozen=True)
class Solution:
    """Wrapper around the solution YAML specification."""

    raw: Dict[str, Any] = field(repr=False)
    path: Optional[Path] = None
    source_text: Optional[str] = field(default=None, repr=False)

    @property
    def questions(self) -> Dict[int, Dict[str, Any]]:
        return {qid: entry for qid, entry in self.iter_questions()}

    @property
    def total_points(self) -> float:
        if isinstance(self.raw, dict) and "total_points" in self.raw:
            try:
                return float(self.raw["total_points"])
            except (TypeError, ValueError):
                pass
        return sum(float(entry.get("points", 1.0)) for _, entry in self.iter_questions())

    def iter_questions(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """
        Yield (question_id, entry) pairs with normalised integer identifiers.
        """
        raw_questions = self.raw.get("questions", []) if isinstance(self.raw, dict) else []
        if not isinstance(raw_questions, list):
            return []

        seen_ids: set[int] = set()
        for index, raw_entry in enumerate(raw_questions, start=1):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            normalised_id = _normalise_question_id(entry.get("id"), fallback=index)
            while normalised_id in seen_ids:
                normalised_id += 1
            seen_ids.add(normalised_id)
            entry.setdefault("_original_id", entry.get("id"))
            entry.setdefault("label", _question_label(entry, normalised_id))
            entry["_normalised_id"] = normalised_id
            yield normalised_id, entry


def load_solution(path: Path) -> Solution:
    """
    Load the solution YAML file and normalise question identifiers to integers.
    """
    text = path.read_text(encoding="utf-8")
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Solution YAML should contain a top-level mapping.")
    if "questions" not in data:
        raise ValueError("Solution YAML must define a 'questions' list.")
    return Solution(raw=data, path=path, source_text=text)


def solution_points_map(solution: Solution) -> Dict[int, float]:
    """
    Return a mapping question_id -> point_value.
    """
    out: Dict[int, float] = {}
    for qid, entry in solution.iter_questions():
        try:
            out[qid] = float(entry.get("points", 1.0))
        except (TypeError, ValueError):
            out[qid] = 1.0
    return out


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _build_grading_dataset(analysis: Dict[str, Any], solution: Solution) -> Dict[str, Any]:
    metadata = analysis.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    quiz_meta = solution.raw.get("meta") if isinstance(solution.raw, dict) else {}
    if not isinstance(quiz_meta, dict):
        quiz_meta = {}

    items = analysis.get("items")
    if not isinstance(items, list):
        items = []

    items_by_qid: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        qid = _coerce_int(item.get("question_id"))
        if qid is None:
            continue
        items_by_qid.setdefault(qid, []).append(item)

    question_entries: List[Dict[str, Any]] = []
    for qid, entry in solution.iter_questions():
        entry_copy = dict(entry)
        choices = entry_copy.get("choices") or []
        if not isinstance(choices, list):
            choices = []
        correct_choices_raw = entry_copy.get("correct_choices") or []
        if not isinstance(correct_choices_raw, list):
            correct_choices_raw = []
        correct_indices = {
            idx
            for idx in (
                _coerce_int(value)
                for value in correct_choices_raw
            )
            if idx is not None
        }

        solution_choices: List[Dict[str, Any]] = []
        for index, choice_text in enumerate(choices, start=1):
            solution_choices.append(
                {
                    "index": index,
                    "text": str(choice_text),
                    "correct": index in correct_indices,
                }
            )

        analysis_records: List[Dict[str, Any]] = []
        for seq_index, item in enumerate(items_by_qid.get(qid, []), start=1):
            analysis_records.append(
                {
                    "sequence": seq_index,
                    "image": item.get("image"),
                    "raw_response": item.get("raw_response"),
                    "structured": item.get("json"),
                    "summary": item.get("summary"),
                    "question_kind": item.get("question_kind"),
                    "usage": item.get("usage"),
                    "processed_at": item.get("processed_at"),
                }
            )

        question_entries.append(
            {
                "id": qid,
                "label": entry_copy.get("label"),
                "max_points": entry_copy.get("points"),
                "settings": {
                    "allow_multiple": entry_copy.get("allow_multiple", False),
                    "partial_credit": entry_copy.get("partial_credit"),
                    "negative_points": entry_copy.get("negative_points"),
                    "default_points": entry_copy.get("points"),
                },
                "prompt_text": entry_copy.get("question"),
                "solution": {
                    "type": entry_copy.get("type"),
                    "choices": solution_choices,
                    "extra": {k: v for k, v in entry_copy.items() if k not in {
                        "id",
                        "_normalised_id",
                        "_original_id",
                        "label",
                        "question",
                        "choices",
                        "correct_choices",
                        "type",
                        "points",
                        "allow_multiple",
                        "partial_credit",
                        "negative_points",
                    }},
                },
                "analysis": analysis_records,
            }
        )

    dataset = {
        "student": {
            "name": metadata.get("student_name"),
            "name_raw": metadata.get("student_name_raw"),
            "roster_name": metadata.get("student_name_roster"),
            "roster_first_name": metadata.get("student_name_roster_first_name"),
            "roster_last_name": metadata.get("student_name_roster_last_name"),
            "analysis_metadata": metadata,
        },
        "quiz": {
            "title": quiz_meta.get("title"),
            "code": quiz_meta.get("code"),
            "subtitle": quiz_meta.get("subtitle"),
            "total_questions": len(question_entries),
            "total_points": solution.total_points,
        },
        "analysis_overview": {
            "source_pdf": analysis.get("source_pdf"),
            "started_at": analysis.get("started_at"),
            "completed_at": analysis.get("completed_at"),
            "usage": analysis.get("usage"),
            "stats": analysis.get("stats"),
        },
        "questions": question_entries,
    }

    unmapped = [item for qid, items_list in items_by_qid.items() if qid not in solution.questions for item in items_list]
    if unmapped:
        dataset["unmatched_analysis"] = unmapped

    return dataset


def run_grading(
    analysis: Dict[str, Any],
    solution: Solution,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_GRADING_MODEL,
    user_label: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Invoke the grading LLM to evaluate the student's answers.
    """
    client = client or build_openai_client()
    prompt_file = prompt_path or PROMPT_PATH
    prompt_text = prompt_file.read_text(encoding="utf-8")
    dataset = _build_grading_dataset(analysis, solution)
    payload = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": prompt_text}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(dataset, ensure_ascii=False),
                    },
                ],
            },
        ],
        user=user_label,
    )
    raw_text = getattr(payload, "output_text", None)
    if not raw_text:
        raise RuntimeError("Grading model returned an empty response.")
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove optional Markdown code fences such as ```json ... ```
        fence_end = text.find("\n")
        closing = text.rfind("```")
        if fence_end != -1 and closing != -1 and closing > fence_end:
            text = text[fence_end + 1 : closing].strip()
    return json.loads(text)


def compute_points_from_grades(
    grades: Dict[str, Any],
    solution: Solution,
) -> Tuple[float, float]:
    """
    Convert 'granted_ratio' values to actual points using the solution weighting.
    """
    mapping = solution_points_map(solution)
    got = 0.0
    for idx, q in enumerate(grades.get("questions", []), start=1):
        if not isinstance(q, dict):
            continue
        qid = _normalise_question_id(q.get("id"), fallback=idx)
        max_points = float(mapping.get(qid, 1.0))
        ratio_value = q.get("awarded_ratio")
        if ratio_value is None:
            ratio_value = q.get("granted_ratio")
        ratio = 0.0
        if ratio_value is not None:
            try:
                ratio = max(0.0, min(1.0, float(ratio_value)))
            except (TypeError, ValueError):
                ratio = 0.0
        elif "awarded_points" in q:
            try:
                ratio = float(q.get("awarded_points", 0.0)) / max_points if max_points else 0.0
            except (TypeError, ValueError):
                ratio = 0.0
        got += ratio * max_points
    return got, solution.total_points


def _normalise_question_id(raw_id: Any, *, fallback: int) -> int:
    """
    Convert various id representations (strings like 'q5') into an integer identifier.
    """
    if raw_id is None:
        return fallback
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float) and raw_id.is_integer():
        return int(raw_id)
    if isinstance(raw_id, str):
        stripped = raw_id.strip()
        if stripped.isdigit():
            return int(stripped)
        match = re.search(r"(\d+)", stripped)
        if match:
            return int(match.group(1))
    return fallback


def _question_label(entry: Dict[str, Any], normalised_id: int) -> str:
    """
    Provide a human-readable label for a quiz question.
    """
    for key in ("label", "title", "question", "prompt", "_original_id", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"Question {normalised_id}"
