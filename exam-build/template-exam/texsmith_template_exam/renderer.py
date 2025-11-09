"""Renderer hooks handling exam-specific placeholders."""

from __future__ import annotations

from typing import Any

from bs4 import NavigableString, Tag

from texsmith.adapters.handlers._helpers import mark_processed
from texsmith.adapters.latex.utils import escape_latex_chars
from texsmith.core.context import RenderContext
from texsmith.core.rules import RenderPhase, renders


SUPPORTED_TEMPLATES = {"exam"}


def _is_active(context: RenderContext) -> bool:
    return context.runtime.get("template") in SUPPORTED_TEMPLATES


def _estimate_width(value: str) -> str:
    length = max(len(value.strip()), 4)
    return f"{length}em"


@renders(
    "exam-fill",
    phase=RenderPhase.POST,
    priority=60,
    name="exam_fill_blocks",
)
def render_exam_fill(element: Tag, context: RenderContext) -> None:
    """Convert <exam-fill> nodes into \\fillwith… commands."""
    if not _is_active(context):
        return

    data_count = element.get("data-count")
    try:
        count = max(int(data_count or 0), 1)
    except (TypeError, ValueError):
        return

    kind = (element.get("data-kind") or "lines").lower()
    if kind == "grid":
        command = "\\fillwithgrid"
        length = f"\\dimexpr {count}\\gridsize\\relax"
    else:
        command = "\\fillwithdottedlines"
        length = f"\\dimexpr {count}\\linefillheight\\relax"
    latex = f"{command}{{{length}}}"

    replacement = mark_processed(NavigableString(latex))
    context.mark_processed(element)
    context.suppress_children(element)
    element.replace_with(replacement)


@renders(
    "exam-fillin",
    phase=RenderPhase.POST,
    priority=60,
    name="exam_fillin_inline",
)
def render_exam_fillin(element: Tag, context: RenderContext) -> None:
    """Convert inline <exam-fillin> nodes into \\fillin commands."""
    if not _is_active(context):
        return

    value = (element.get("data-value") or "").strip()
    if not value:
        element.decompose()
        return

    width = element.get("data-width") or _estimate_width(value)
    latex_value = escape_latex_chars(value)
    latex = f"\\fillin[{latex_value}][{width}]"

    replacement = mark_processed(NavigableString(latex))
    context.mark_processed(element)
    context.suppress_children(element)
    element.replace_with(replacement)


def register(renderer: Any) -> None:
    """Register renderer hooks on the provided renderer."""
    register_callable = getattr(renderer, "register", None)
    if not callable(register_callable):  # pragma: no cover
        raise TypeError("Renderer does not expose a 'register' method.")
    register_callable(render_exam_fill)
    register_callable(render_exam_fillin)


__all__ = ["register", "render_exam_fill", "render_exam_fillin"]
