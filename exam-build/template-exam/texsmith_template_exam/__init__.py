"""Exam template integration for TeXSmith."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from texsmith.adapters import markdown as md  # noqa: F401 - side-effected import
from texsmith.core.templates import TemplateError, WrappableTemplate

from .constants import (
    SOLUTION_LINES_SENTINEL_PREFIX,
    SOLUTION_LINES_SENTINEL_SUFFIX,
)


_PACKAGE_ROOT = Path(__file__).parent.resolve()
_EXTENSION_ID = "texsmith_template_exam.markdown:ExamSpecialsExtension"
DEFAULT_MARKDOWN_EXTENSIONS = getattr(md, "DEFAULT_MARKDOWN_EXTENSIONS", None)
if isinstance(DEFAULT_MARKDOWN_EXTENSIONS, list) and _EXTENSION_ID not in DEFAULT_MARKDOWN_EXTENSIONS:
    DEFAULT_MARKDOWN_EXTENSIONS.append(_EXTENSION_ID)

LINE_PATTERN = re.compile(
    r"(?P<prefix>\s*)(?:—|---)\s*(?:(grid)\s+)?(?P<count>\d+)\s*(?:—|---)(?P<suffix>\s*)"
)
FILLIN_PATTERN = re.compile(r"\\\{\\\{\s*([^{}]+?)\s*\\\}\\\}")
SOLUTION_SENTINEL_PATTERN = re.compile(
    r"\s*"
    + re.escape(SOLUTION_LINES_SENTINEL_PREFIX)
    + r"(?P<value>\d+)"
    + re.escape(SOLUTION_LINES_SENTINEL_SUFFIX)
    + r"\s*(?:\r?\n)?",
    re.MULTILINE,
)
SOLUTION_BLOCK_PATTERN = re.compile(r"\\begin{solution}(?P<body>.*?)\\end{solution}", re.DOTALL)


class Template(WrappableTemplate):
    """Expose the exam template as a wrappable instance."""

    _LEVELLED_COMMANDS: tuple[tuple[str, str], ...] = (
        ("\\part", "parts"),
        ("\\subpart", "subparts"),
        ("\\subsubpart", "subsubparts"),
    )
    _COMMAND_TO_LEVEL = {cmd: idx for idx, (cmd, _env) in enumerate(_LEVELLED_COMMANDS)}
    _LEVEL_TO_ENV = {idx: env for idx, (_cmd, env) in enumerate(_LEVELLED_COMMANDS)}
    _ENV_TO_LEVEL = {env: idx for idx, env in _LEVEL_TO_ENV.items()}
    _QUESTION_STARTS: tuple[str, ...] = (
        "\\titlequestion",
        "\\bonustitledquestion",
        "\\question",
        "\\bonusquestion",
    )
    _QUESTION_ENDS: tuple[str, ...] = ("\\end{questions}",)
    _QUESTION_STARTS: tuple[str, ...] = (
        "\\titlequestion",
        "\\bonustitledquestion",
        "\\question",
        "\\bonusquestion",
    )
    _QUESTION_ENDS: tuple[str, ...] = ("\\end{questions}",)

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
        updated = self._ensure_parts_environments(updated)
        updated = self._convert_solution_environments(updated)
        return updated

    def _ensure_parts_environments(self, latex_body: str) -> str:
        lines = latex_body.splitlines()
        output: list[str] = []
        open_stack: list[tuple[int, bool]] = []  # (level, auto_opened)

        def has_level(level: int) -> bool:
            return any(entry[0] == level for entry in open_stack)

        def auto_open(level: int) -> None:
            env = self._LEVEL_TO_ENV[level]
            output.append(f"\\begin{{{env}}}")
            open_stack.append((level, True))

        def close_auto_levels(min_level: int) -> None:
            nonlocal open_stack
            while open_stack and open_stack[-1][0] >= min_level and open_stack[-1][1]:
                level, _ = open_stack.pop()
                env = self._LEVEL_TO_ENV[level]
                output.append(f"\\end{{{env}}}")

        def close_all_auto() -> None:
            close_auto_levels(0)

        for raw_line in lines:

            stripped = raw_line.lstrip()
            manual_begin = None
            manual_end = None
            for env_name in self._ENV_TO_LEVEL:
                if stripped.startswith(f"\\begin{{{env_name}}}"):
                    manual_begin = env_name
                    break
                if stripped.startswith(f"\\end{{{env_name}}}"):
                    manual_end = env_name
                    break

            if manual_begin:
                level = self._ENV_TO_LEVEL[manual_begin]
                open_stack.append((level, False))
                output.append(raw_line)
                continue

            if manual_end:
                level = self._ENV_TO_LEVEL[manual_end]
                # Pop until corresponding level is removed
                while open_stack:
                    lvl, auto = open_stack.pop()
                    if auto:
                        env = self._LEVEL_TO_ENV[lvl]
                        output.append(f"\\end{{{env}}}")
                    if lvl == level:
                        break
                output.append(raw_line)
                continue

            if self._starts_with_any(stripped, self._QUESTION_STARTS) or stripped.startswith(
                self._QUESTION_ENDS
            ):
                close_all_auto()

            command_level = None
            for level_cmd in self._COMMAND_TO_LEVEL:
                if stripped.startswith(level_cmd):
                    command_level = self._COMMAND_TO_LEVEL[level_cmd]
                    break

            if command_level is not None:
                close_auto_levels(command_level + 1)
                for required_level in range(0, command_level + 1):
                    if not has_level(required_level):
                        auto_open(required_level)

            output.append(raw_line)

        close_all_auto()
        return "\n".join(output)

    def _convert_solution_environments(self, latex_body: str) -> str:
        def replace_block(match: re.Match[str]) -> str:
            body = match.group("body")
            sentinel = SOLUTION_SENTINEL_PATTERN.search(body)
            if not sentinel:
                return match.group(0)
            value = sentinel.group("value")
            stripped_body = SOLUTION_SENTINEL_PATTERN.sub("", body, count=1)
            stripped_body = stripped_body.lstrip("\n")
            filler = f"\\dimexpr {value}\\linefillheight\\relax"
            prefix = "\n" if stripped_body and not stripped_body.startswith("\n") else ""
            return f"\\begin{{solutionordottedlines}}[{filler}]{prefix}{stripped_body}\\end{{solutionordottedlines}}"

        return SOLUTION_BLOCK_PATTERN.sub(replace_block, latex_body)

    @staticmethod
    def _starts_with_any(text: str, commands: tuple[str, ...]) -> bool:
        for command in commands:
            if not text.startswith(command):
                continue
            if len(text) == len(command):
                return True
            next_char = text[len(command)]
            if not next_char.isalpha():
                return True
        return False

__all__ = ["Template"]
