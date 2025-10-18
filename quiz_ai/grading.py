"""
Helpers for grading analysis JSON files against the official solution using an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from openai import OpenAI

from .llm import DEFAULT_VISION_MODEL, build_openai_client
from .utils import read_yaml

PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "grading.prompt.md"


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


def run_grading(
    analysis: Dict[str, Any],
    solution: Solution,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_VISION_MODEL,
    user_label: Optional[str] = None,
    prompt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Invoke the grading LLM to evaluate the student's answers.
    """
    client = client or build_openai_client()
    prompt_file = prompt_path or PROMPT_PATH
    prompt_text = prompt_file.read_text(encoding="utf-8")
    solution_text = solution.source_text or json.dumps(solution.raw, ensure_ascii=False)
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
                    {"type": "input_text", "text": "Student analysis JSON:"},
                    {
                        "type": "input_text",
                        "text": json.dumps(analysis, ensure_ascii=False),
                    },
                    {"type": "input_text", "text": "Official quiz YAML source:"},
                    {
                        "type": "input_text",
                        "text": solution_text,
                    },
                ],
            },
        ],
        user=user_label,
    )
    raw_text = getattr(payload, "output_text", None)
    if not raw_text:
        raise RuntimeError("Grading model returned an empty response.")
    return json.loads(raw_text)


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
