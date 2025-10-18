"""
Utility helpers to generate CSV/Markdown summaries from grading JSON files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .grading import Solution, compute_points_from_grades
from .utils import read_json, write_text


@dataclass(frozen=True)
class GradeSummary:
    """Summary of a single graded student."""

    path: Path
    data: dict
    points_obtained: float
    points_total: float
    note: Optional[float]


def load_grade(path: Path) -> dict:
    """
    Load a grading JSON file.
    """
    return read_json(path)


def summarise_grade(path: Path, solution: Solution) -> GradeSummary:
    """
    Compute point totals for a single grading JSON file.
    """
    data = load_grade(path)
    got, total = compute_points_from_grades(data, solution)
    note = round(((got / total) * 5.0 + 1.0), 1) if total else None
    return GradeSummary(
        path=path,
        data=data,
        points_obtained=got,
        points_total=total,
        note=note,
    )


def write_summary_csv(
    grade_paths: Sequence[Path],
    solution: Solution,
    out_csv: Path,
) -> List[GradeSummary]:
    """
    Generate a CSV summary for the provided grading files and return the summaries.
    """
    summaries = [summarise_grade(path, solution) for path in grade_paths]
    fieldnames = ["file", "points_obtenus", "points_total", "note"]

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "file": summary.data.get("_source_pdf", summary.path.name),
                    "points_obtenus": f"{summary.points_obtained:.2f}",
                    "points_total": f"{summary.points_total:.2f}",
                    "note": "" if summary.note is None else f"{summary.note:.1f}",
                }
            )

    return summaries


def build_markdown_report(
    summaries: Iterable[GradeSummary],
    out_markdown: Path,
) -> None:
    """
    Produce a simple Markdown report aggregating per-student notes.
    """
    lines = ["# Rapport de notation", ""]
    for summary in summaries:
        data = summary.data
        student = data.get("name") or summary.path.stem
        note_display = "N/A" if summary.note is None else f"{summary.note:.1f}"
        lines.append(f"## {student}")
        lines.append(f"- Fichier source : `{summary.data.get('_source_pdf', summary.path.name)}`")
        lines.append(f"- Points obtenus : {summary.points_obtained:.2f} / {summary.points_total:.2f}")
        lines.append(f"- Note (6) : {note_display}")
        lines.append("")

    write_text(out_markdown, "\n".join(lines).strip() + "\n")
