"""
Helpers for grading analysis JSON files against the official solution using an LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from .llm import DEFAULT_VISION_MODEL, build_openai_client
from .utils import read_yaml


GRADE_PROMPT = """You are an impartial and precise grader.

Input #1 is the student's quiz analysis JSON (multi-page). Input #2 is the official solution JSON.
Your job: produce a SINGLE strict JSON object with per-question evaluation.

Output schema:
{
  "name": "student name",
  "date": "YYYY-MM-DD or empty",
  "title": "exam title from solution if present or empty",
  "total_questions": int,
  "questions": [
    {
      "id": int,
      "answered": bool,
      "correct": bool,
      "granted_ratio": float,
      "remark": "string",
      "confidence": float
    }
  ]
}

Rules:
- Maintain a neutral, concise remark.
- For multi-part questions (multi-select), distribute partial credit proportionally via granted_ratio in [0,1].
- If the student left it blank, answered=false, correct=false, granted_ratio=0.0.
- Output ONLY valid JSON."""


@dataclass(frozen=True)
class Solution:
    """Wrapper around the solution YAML specification."""

    raw: Dict[str, Any]

    @property
    def questions(self) -> Dict[int, Dict[str, Any]]:
        return {int(q["id"]): q for q in self.raw.get("questions", []) if "id" in q}

    @property
    def total_points(self) -> float:
        if "total_points" in self.raw:
            return float(self.raw["total_points"])
        return sum(float(q.get("points", 1.0)) for q in self.raw.get("questions", []))


def load_solution(path: Path) -> Solution:
    """
    Load the solution YAML file and normalise question identifiers to integers.
    """
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Solution YAML should contain a top-level mapping.")
    if "questions" not in data:
        raise ValueError("Solution YAML must define a 'questions' list.")
    return Solution(raw=data)


def solution_points_map(solution: Solution) -> Dict[int, float]:
    """
    Return a mapping question_id -> point_value.
    """
    out: Dict[int, float] = {}
    for qid, entry in solution.questions.items():
        out[qid] = float(entry.get("points", 1.0))
    return out


def run_grading(
    analysis: Dict[str, Any],
    solution: Solution,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_VISION_MODEL,
    user_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invoke the grading LLM to evaluate the student's answers.
    """
    client = client or build_openai_client()
    payload = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": GRADE_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Student analysis JSON:"},
                    {
                        "type": "input_text",
                        "text": json.dumps(analysis, ensure_ascii=False),
                    },
                    {"type": "input_text", "text": "Official solution JSON:"},
                    {
                        "type": "input_text",
                        "text": json.dumps(solution.raw, ensure_ascii=False),
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
    for q in grades.get("questions", []):
        qid = int(q.get("id"))
        ratio = float(q.get("granted_ratio", 0.0))
        got += ratio * float(mapping.get(qid, 1.0))
    return got, solution.total_points

