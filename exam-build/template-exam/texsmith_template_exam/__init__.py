"""Exam template integration for TeXSmith."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from texsmith.adapters import markdown as md  # noqa: F401 - side-effected import
from texsmith.core.templates import TemplateError, WrappableTemplate


_PACKAGE_ROOT = Path(__file__).parent.resolve()
_EXTENSION_ID = "texsmith_template_exam.markdown:ExamSpecialsExtension"
DEFAULT_MARKDOWN_EXTENSIONS = getattr(md, "DEFAULT_MARKDOWN_EXTENSIONS", None)
if isinstance(DEFAULT_MARKDOWN_EXTENSIONS, list) and _EXTENSION_ID not in DEFAULT_MARKDOWN_EXTENSIONS:
    DEFAULT_MARKDOWN_EXTENSIONS.append(_EXTENSION_ID)

LINE_PATTERN = re.compile(
    r"(?P<prefix>\s*)(?:—|---)\s*(?:(grid)\s+)?(?P<count>\d+)\s*(?:—|---)(?P<suffix>\s*)"
)
FILLIN_PATTERN = re.compile(r"\\\{\\\{\s*([^{}]+?)\s*\\\}\\\}")


class Template(WrappableTemplate):
    """Expose the exam template as a wrappable instance."""

    _METADATA_FIELDS: tuple[tuple[str, str], ...] = (
        ("Cours", "course"),
        ("Session", "session"),
        ("Durée", "duration"),
        ("Salle", "room"),
        ("Seed", "seed"),
        ("Version", "version"),
    )

    def __init__(self) -> None:
        try:
            super().__init__(_PACKAGE_ROOT)
        except TemplateError as exc:  # pragma: no cover - defensive
            raise TemplateError(f"Failed to initialise exam template: {exc}") from exc

    def prepare_context(
        self,
        latex_body: str,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = super().prepare_context(latex_body, overrides=overrides)
        context.pop("press", None)

        language = self._coerce_string(context.get("language"))
        if not language:
            language = self._coerce_string(self.info.get_attribute_default("language")) or "french"
        context["language"] = language

        paper = self._coerce_string(context.get("paper"))
        if not paper:
            paper = self._coerce_string(self.info.get_attribute_default("paper")) or "a4paper"

        pointsize = self._coerce_string(context.get("pointsize"))
        if not pointsize:
            pointsize = self._coerce_string(self.info.get_attribute_default("pointsize")) or "11pt"

        doc_options = [language, paper, "addpoints", pointsize]
        filtered = [option for option in doc_options if option]
        context["documentclass_options"] = f"[{','.join(filtered)}]" if filtered else ""

        metadata_pairs = self._build_metadata(context)
        context["exam_metadata"] = metadata_pairs

        directives = self._coerce_list(context.get("directives"))
        context["directives"] = directives

        show_gradetable = context.get("show_gradetable")
        context["show_gradetable"] = bool(show_gradetable if show_gradetable is not None else True)

        cover_notice = self._coerce_string(context.get("cover_notice"))
        context["cover_notice"] = cover_notice

        for key in ("title", "subtitle", "author", "department", "school", "date"):
            value = self._coerce_string(context.get(key))
            if value is not None:
                context[key] = value

        return context

    def wrap_document(
        self,
        latex_body: str,
        *,
        overrides: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        processed = self._replace_placeholders(latex_body)
        return super().wrap_document(processed, overrides=overrides, context=context)

    def wrap_document(
        self,
        latex_body: str,
        *,
        overrides: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        processed = self._replace_placeholders(latex_body)
        return super().wrap_document(processed, overrides=overrides, context=context)

    def _build_metadata(self, context: Mapping[str, Any]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for label, key in self._METADATA_FIELDS:
            value = self._coerce_string(context.get(key))
            if value:
                pairs.append((label, value))
        return pairs

    def _coerce_string(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            candidate = value.strip()
        else:
            candidate = str(value).strip()
        return candidate or None

    def _coerce_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items: list[str] = []
            for entry in value:
                text = self._coerce_string(entry)
                if text:
                    items.append(text)
            return items
        text = self._coerce_string(value)
        return [text] if text else []

    def _replace_placeholders(self, latex_body: str) -> str:
        def replace_line(match: re.Match[str]) -> str:
            try:
                count = max(int(match.group("count")), 1)
            except (TypeError, ValueError):
                return match.group(0)
            if match.group(2):
                command = "\\fillwithgrid"
                length = f"\\dimexpr {count}\\gridsize\\relax"
            else:
                command = "\\fillwithdottedlines"
                length = f"\\dimexpr {count}\\linefillheight\\relax"
            return f"{match.group('prefix')}{command}{{{length}}}{match.group('suffix')}"

        def replace_fillin(match: re.Match[str]) -> str:
            value = match.group(1).strip()
            if not value:
                return ""
            width = max(len(value), 4)
            return f"\\fillin[{value}][{width}em]"

        updated = LINE_PATTERN.sub(replace_line, latex_body)
        updated = FILLIN_PATTERN.sub(replace_fillin, updated)
        return updated

__all__ = ["Template"]
