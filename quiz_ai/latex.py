"""
Simple converter from the repository YAML schema to a LaTeX document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from .utils import read_yaml, write_text

LATEX_HEADER = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{amsmath}
\geometry{margin=2cm}
\setlist[enumerate,1]{label=\alph*)}
\begin{document}
"""

LATEX_FOOTER = r"\end{document}"


def latex_escape(text: str) -> str:
    """
    Escape basic LaTeX meta characters.
    """
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _render_meta(meta: Dict[str, object]) -> List[str]:
    lines: List[str] = []
    title = latex_escape(str(meta.get("title", "Quiz")))
    subtitle = latex_escape(str(meta.get("subtitle", "")))
    lines.append(f"\\section*{{{title}}}")
    if subtitle:
        lines.append(f"\\textbf{{{subtitle}}}\\\\[1em]")

    directives = meta.get("directives") or []
    if directives:
        lines.append("\\textbf{Directives}\\\\")
        lines.append("\\begin{itemize}")
        for directive in directives:
            lines.append(f"  \\item {latex_escape(str(directive))}")
        lines.append("\\end{itemize}")

    lines.append("")
    return lines


def _render_multiple_choice(question: Dict[str, object]) -> List[str]:
    lines = ["\\begin{enumerate}"]
    for choice in question.get("choices", []):
        lines.append(f"  \\item {latex_escape(str(choice))}")
    lines.append("\\end{enumerate}")
    return lines


def _render_fill(question: Dict[str, object]) -> List[str]:
    items = question.get("items") or []
    width = int(question.get("fill_width", 25))
    lines = ["\\begin{itemize}"]
    for item in items:
        prompt = latex_escape(str(item.get("prompt", "")))
        lines.append(f"  \\item {prompt} \\rule{{{max(width, 5)}mm}}{{0.4pt}}")
    lines.append("\\end{itemize}")
    return lines


def _render_open(question: Dict[str, object]) -> List[str]:
    width = int(question.get("lines", 4))
    return [f"\\vspace{{{width}em}}"]


QUESTION_RENDERERS = {
    "multiple_choice": _render_multiple_choice,
    "multiple_fill_in": _render_fill,
    "fill_in_the_blank": _render_fill,
    "short_answer": _render_open,
    "open": _render_open,
}


def render_question(index: int, question: Dict[str, object]) -> List[str]:
    """
    Render a single question to LaTeX.
    """
    title = latex_escape(str(question.get("question", f"Question {index}")))
    lines = [f"\\subsection*{{Question {index}}}", title, ""]
    renderer = QUESTION_RENDERERS.get(question.get("type"))
    if renderer:
        lines.extend(renderer(question))
    else:
        lines.append("\\vspace{3em}")
    lines.append("")
    return lines


def render_quiz_yaml(path: Path) -> str:
    """
    Convert the quiz YAML located at `path` to a LaTeX string.
    """
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Quiz YAML must contain a mapping at the root.")
    questions = data.get("questions")
    if not isinstance(questions, list):
        raise ValueError("'questions' must be a list in the quiz YAML.")

    parts: List[str] = [LATEX_HEADER]
    if meta := data.get("meta"):
        if isinstance(meta, dict):
            parts.extend(_render_meta(meta))
    for idx, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        parts.extend(render_question(idx, question))
    parts.append(LATEX_FOOTER)
    return "\n".join(parts)


def write_latex_from_yaml(yaml_path: Path, tex_path: Path) -> None:
    """
    Render the YAML quiz file at `yaml_path` and write the LaTeX to `tex_path`.
    """
    tex = render_quiz_yaml(yaml_path)
    write_text(tex_path, tex)
