"""
Helpers to craft personalised feedback emails from grading JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from .llm import (
    DEFAULT_VISION_MODEL,
    build_openai_client,
)
from .utils import read_json, write_text

FEEDBACK_PROMPT_PATH = Path(__file__).resolve().parent / "assets" / "prompts" / "feedback.prompt.md"


@dataclass(frozen=True)
class FeedbackInputs:
    """Container gathering all data required to generate the email."""

    student_name: str
    score_points: float
    score_total: float
    score_percentage: float
    quiz_title: str
    final_report: str
    positive_topics: Sequence[str]
    improvement_topics: Sequence[str]


def load_grading_file(path: Path) -> Dict[str, Any]:
    """
    Load and return the grading JSON content.
    """
    return read_json(path)


def resolve_student_name(
    _grade_path: Path,
    grades: Dict[str, Any],
) -> str:
    """
    Return the most reliable student name found in `grades`, based solely on analysis metadata.
    """
    metadata = grades.get("_analysis_metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    student_block = grades.get("student")
    if not isinstance(student_block, dict):
        student_block = {}

    candidates: List[str] = []
    direct_name = str(student_block.get("name") or "").strip()
    if direct_name:
        candidates.append(direct_name)
    meta_name = str(metadata.get("student_name") or "").strip()
    if meta_name:
        candidates.append(meta_name)
    roster_name = str(metadata.get("student_name_roster") or "").strip()
    if roster_name:
        if metadata.get("student_name_verified"):
            candidates.insert(0, roster_name)
        else:
            candidates.append(roster_name)
    raw_name = str(metadata.get("student_name_raw") or "").strip()
    if raw_name:
        candidates.append(raw_name)

    cleaned_candidates = [_normalise_name(value) for value in candidates if value]

    final_name = ""
    if cleaned_candidates:
        seen: set[str] = set()
        ordered: List[str] = []
        for candidate in cleaned_candidates:
            key = candidate.lower()
            if candidate and key not in seen:
                ordered.append(candidate)
                seen.add(key)
        if ordered:
            ordered.sort(key=len, reverse=True)
            final_name = ordered[0]

    return final_name


def build_feedback_inputs(data: Dict[str, Any]) -> FeedbackInputs:
    """
    Convert raw grading data into a structured `FeedbackInputs` instance.
    """
    student_block = data.get("student") if isinstance(data, dict) else {}
    score_block = data.get("score") if isinstance(data, dict) else {}
    quiz_block = data.get("quiz") if isinstance(data, dict) else {}
    questions = data.get("questions") if isinstance(data, dict) else []

    student_name = ""
    if isinstance(student_block, dict):
        student_name = str(student_block.get("name") or "").strip()

    points_obtained = 0.0
    points_total = 0.0
    percentage = 0.0
    if isinstance(score_block, dict):
        points_obtained = _safe_float(score_block.get("points_obtained"), default=points_obtained)
        points_total = _safe_float(score_block.get("points_total"), default=points_total)
        percentage = _safe_float(score_block.get("percentage"), default=percentage)

    quiz_title = ""
    if isinstance(quiz_block, dict):
        quiz_title = str(quiz_block.get("title") or "").strip()

    final_report = data.get("final_report") if isinstance(data, dict) else ""
    if not isinstance(final_report, str):
        final_report = ""

    positive_topics = _select_topics(questions, status_whitelist={"correct"}, limit=5)
    improvement_topics = _select_topics(questions, status_whitelist={"incorrect", "partial"}, limit=5)

    return FeedbackInputs(
        student_name=student_name,
        score_points=points_obtained,
        score_total=points_total,
        score_percentage=percentage,
        quiz_title=quiz_title,
        final_report=final_report,
        positive_topics=positive_topics,
        improvement_topics=improvement_topics,
    )


def generate_feedback_email(
    payload: FeedbackInputs,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_VISION_MODEL,
    prompt_path: Optional[Path] = None,
    user_label: Optional[str] = None,
) -> str:
    """
    Use an LLM prompt to create a feedback email from the grading summary.
    """
    prompt_file = prompt_path or FEEDBACK_PROMPT_PATH
    prompt_text = prompt_file.read_text(encoding="utf-8")

    context = {
        "student_name": payload.student_name,
        "score": {
            "points_obtained": round(payload.score_points, 2),
            "points_total": round(payload.score_total, 2),
            "percentage": round(payload.score_percentage, 2),
        },
        "quiz_title": payload.quiz_title,
        "final_report": payload.final_report,
        "positive_topics": list(payload.positive_topics),
        "improvement_topics": list(payload.improvement_topics),
    }

    client = client or build_openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(context, ensure_ascii=False, indent=2),
                    }
                ],
            },
        ],
        user=user_label,
    )

    text = getattr(response, "output_text", None) or ""
    email = text.strip()
    if not email:
        raise RuntimeError("The model did not produce any feedback text.")

    signature = "L'assistant artificiel de votre professeur"
    if signature not in email:
        email = email.rstrip() + "\n\n" + signature

    return email.strip() + "\n"


def write_feedback_file(path: Path, content: str) -> None:
    """
    Write the generated feedback email to `path`.
    """
    write_text(path, content)


# ---------------------------------------------------------------------------
# Internal helpers


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _select_topics(
    questions: Any,
    *,
    status_whitelist: Sequence[str],
    limit: int,
) -> List[str]:
    topics: List[str] = []
    if not isinstance(questions, list):
        return topics
    whitelist = {status.lower() for status in status_whitelist}
    for question in questions:
        if not isinstance(question, dict):
            continue
        status = str(question.get("status") or "").strip().lower()
        if status not in whitelist:
            continue
        label = str(question.get("label") or "").strip()
        if not label:
            continue
        topics.append(label)
        if len(topics) >= limit:
            break
    return topics


def _normalise_name(raw: str) -> str:
    tokens = [token for token in raw.replace("_", " ").split() if token]
    if not tokens:
        return ""
    return " ".join(token.capitalize() if len(token) > 1 else token.upper() for token in tokens)
