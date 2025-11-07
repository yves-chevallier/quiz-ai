"""
Helpers for grading analysis JSON files against the official solution using an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_response_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        closing = text.rfind("```")
        if first_newline != -1 and closing != -1 and closing > first_newline:
            text = text[first_newline + 1 : closing].strip()
    if not text:
        raise RuntimeError("Grading model returned an empty response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        snippet = text[max(0, exc.pos - 120) : exc.pos + 120]
        raise RuntimeError(
            f"Grading model returned invalid JSON at position {exc.pos}: {snippet!r}"
        ) from exc


def _compact_analysis_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for seq_index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "sequence": seq_index,
                "question_kind": item.get("question_kind"),
                "summary": item.get("summary"),
                "structured": item.get("json"),
                "handwriting": item.get("handwriting"),
                "drawings": item.get("drawings"),
                "processed_at": item.get("processed_at"),
            }
        )
    return compact


def _prepare_question_payloads(
    analysis: Dict[str, Any],
    solution: Solution,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
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

    solution_entries: List[Tuple[int, Dict[str, Any]]] = list(solution.iter_questions())
    solution_qids = [qid for qid, _ in solution_entries]

    payloads: List[Dict[str, Any]] = []
    for idx, (qid, entry) in enumerate(solution_entries):
        entry_copy = dict(entry)
        choices = entry_copy.get("choices")
        if not isinstance(choices, list):
            choices = []
        correct_raw = entry_copy.get("correct_choices")
        if not isinstance(correct_raw, list):
            correct_raw = []
        correct_indices = {
            idx for idx in (_coerce_int(value) for value in correct_raw) if idx is not None
        }

        solution_choices = [
            {
                "index": index,
                "text": str(choice_text),
                "correct": index in correct_indices,
            }
            for index, choice_text in enumerate(choices, start=1)
        ]

        current_records = items_by_qid.get(qid, [])
        prev_qid = solution_qids[idx - 1] if idx > 0 else None
        next_qid = solution_qids[idx + 1] if idx + 1 < len(solution_qids) else None

        payloads.append(
            {
                "id": qid,
                "label": entry_copy.get("label"),
                "prompt_text": entry_copy.get("question"),
                "type": entry_copy.get("type"),
                "max_points": entry_copy.get("points"),
                "settings": {
                    "allow_multiple": bool(entry_copy.get("allow_multiple", False)),
                    "partial_credit": entry_copy.get("partial_credit"),
                    "negative_points": entry_copy.get("negative_points"),
                },
                "solution_choices": solution_choices,
                "analysis_current": _compact_analysis_records(current_records),
                "analysis_previous": _compact_analysis_records(items_by_qid.get(prev_qid, []))
                if prev_qid is not None
                else [],
                "analysis_next": _compact_analysis_records(items_by_qid.get(next_qid, []))
                if next_qid is not None
                else [],
            }
        )

    unmatched = [
        item
        for qid, records in items_by_qid.items()
        if qid not in solution.questions
        for item in records
    ]

    return metadata, quiz_meta, payloads, unmatched


def run_grading(
    analysis: Dict[str, Any],
    solution: Solution,
    *,
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_GRADING_MODEL,
    user_label: Optional[str] = None,
    prompt_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, Optional[Dict[str, Any]]], None]] = None,
) -> Dict[str, Any]:
    """
    Invoke the grading LLM to evaluate the student's answers.
    """
    client = client or build_openai_client()
    prompt_file = prompt_path or PROMPT_PATH
    prompt_text = prompt_file.read_text(encoding="utf-8")
    metadata, quiz_meta, question_payloads, unmatched = _prepare_question_payloads(analysis, solution)

    question_results: Dict[int, Dict[str, Any]] = {}
    for payload in question_payloads:
        if progress_callback:
            progress_callback(payload["id"], len(question_payloads), {"stage": "start"})
        user_payload: Dict[str, Any] = {
            "question": {
                "id": payload["id"],
                "label": payload["label"],
                "prompt_text": payload["prompt_text"],
                "type": payload["type"],
                "max_points": payload["max_points"],
                "settings": payload["settings"],
                "solution_choices": payload["solution_choices"],
            },
            "analysis": {
                "current": payload["analysis_current"],
            },
        }
        if payload["analysis_previous"]:
            user_payload["analysis"]["previous"] = payload["analysis_previous"]
        if payload["analysis_next"]:
            user_payload["analysis"]["next"] = payload["analysis_next"]

        response = client.responses.create(
            model=model,
            temperature=0.0,
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
                            "text": json.dumps(user_payload, ensure_ascii=False),
                        }
                    ],
                },
            ],
            user=user_label,
        )
        parsed = _parse_response_json(getattr(response, "output_text", "") or "")
        question_results[payload["id"]] = parsed
        if progress_callback:
            progress_callback(
                payload["id"],
                len(question_payloads),
                {
                    "stage": "complete",
                    "status": parsed.get("status"),
                    "awarded_ratio": parsed.get("awarded_ratio"),
                    "confidence": parsed.get("confidence"),
                },
            )

    return _assemble_grading_output(
        analysis=analysis,
        metadata=metadata,
        quiz_meta=quiz_meta,
        question_payloads=question_payloads,
        question_results=question_results,
        unmatched=unmatched,
        solution=solution,
    )


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


def _assemble_grading_output(
    *,
    analysis: Dict[str, Any],
    metadata: Dict[str, Any],
    quiz_meta: Dict[str, Any],
    question_payloads: List[Dict[str, Any]],
    question_results: Dict[int, Dict[str, Any]],
    unmatched: List[Dict[str, Any]],
    solution: Solution,
) -> Dict[str, Any]:
    points_map = solution_points_map(solution)

    question_entries: List[Dict[str, Any]] = []
    total_points = 0.0
    points_obtained = 0.0

    for payload in question_payloads:
        qid = payload["id"]
        result = question_results.get(qid, {})

        ratio_pct = _coerce_float(result.get("awarded_ratio"))
        if ratio_pct is None:
            ratio_pct = 0.0
        ratio = max(0.0, min(1.0, ratio_pct / 100.0))

        status = str(result.get("status") or "").lower()
        if status not in {"correct", "partial", "incorrect", "missing", "invalid"}:
            status = "invalid"

        max_points = payload["max_points"]
        if max_points is None:
            max_points = points_map.get(qid, 1.0)
        try:
            max_points = float(max_points)
        except (TypeError, ValueError):
            max_points = float(points_map.get(qid, 1.0))

        awarded_points = ratio * max_points
        total_points += max_points
        points_obtained += awarded_points

        answer_summary = " ".join(
            filter(
                None,
                (record.get("summary") for record in payload["analysis_current"]),
            )
        ).strip()

        rationale = str(
            result.get("model_rational")
            or result.get("model_rationale")
            or ""
        )
        student_feedback = str(result.get("student_feedback") or "").strip()
        if status == "correct":
            student_feedback = ""
        elif status in {"incorrect", "partial", "missing", "invalid"}:
            if len(student_feedback) > 80:
                student_feedback = student_feedback[:80].rstrip()
            if not student_feedback:
                student_feedback = "Révise cette notion."
        else:
            student_feedback = ""

        confidence = str(result.get("confidence") or "").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        flags: List[str] = []
        if ratio >= 0.999 and status != "correct":
            flags.append("awarded_ratio suggests full credit but status differs.")
        if ratio <= 0.001 and status == "correct":
            flags.append("Status 'correct' with zero ratio detected.")

        question_entries.append(
            {
                "id": qid,
                "label": payload["label"] or payload["prompt_text"] or f"Question {qid}",
                "max_points": max_points,
                "awarded_ratio": ratio,
                "awarded_points": awarded_points,
                "status": status,
                "answer_summary": answer_summary,
                "justification": rationale,
                "remarks": student_feedback,
                "flags": flags,
                "confidence": confidence,
            }
        )

    percentage = (points_obtained / total_points * 100.0) if total_points else 0.0

    incorrect_labels = [
        entry["label"]
        for entry in question_entries
        if entry["status"] in {"incorrect", "invalid"}
    ]
    partial_labels = [
        entry["label"]
        for entry in question_entries
        if entry["status"] == "partial"
    ]

    if not incorrect_labels and not partial_labels:
        final_report = (
            f"Score {points_obtained:.2f}/{total_points:.2f} "
            f"({percentage:.1f}%). Excellent work."
        )
    else:
        parts = [
            f"Score {points_obtained:.2f}/{total_points:.2f} ({percentage:.1f}%)."
        ]
        if incorrect_labels:
            parts.append("Revisit: " + ", ".join(incorrect_labels) + ".")
        if partial_labels:
            parts.append("Partially correct: " + ", ".join(partial_labels) + ".")
        final_report = " ".join(parts).strip()

    student_name = str(metadata.get("student_name") or "").strip()
    student_date = str(metadata.get("grading_date") or "").strip()

    grade_output: Dict[str, Any] = {
        "student": {
            "name": student_name,
            "identifier": "",
            "date": student_date,
        },
        "quiz": {
            "title": quiz_meta.get("title") or "",
            "source_reference": quiz_meta.get("code")
            or quiz_meta.get("title")
            or "",
            "total_questions": len(question_entries),
        },
        "questions": question_entries,
        "score": {
            "points_obtained": points_obtained,
            "points_total": total_points,
            "percentage": percentage,
        },
        "final_report": final_report,
    }

    if unmatched:
        grade_output["_unmatched_analysis"] = unmatched

    return grade_output


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
