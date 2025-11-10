"""Markdown extension enabling exam-specific shorthand syntax."""

from __future__ import annotations

import re
from xml.etree import ElementTree as etree

from markdown import Extension
from markdown.preprocessors import Preprocessor

from .constants import (
    SOLUTION_LINES_SENTINEL_PREFIX,
    SOLUTION_LINES_SENTINEL_SUFFIX,
)


class ExamSolutionMetadata(Preprocessor):
    """Capture solution callout metadata such as dotted line counts."""

    pattern = re.compile(r"^(?P<indent>\s*)!!!\s+(?P<kind>[A-Za-z0-9_-]+)(?P<rest>.*)$")
    lines_pattern = re.compile(r"lines\s*=\s*(?P<value>\d+)", re.IGNORECASE)

    def run(self, lines: list[str]) -> list[str]:
        output: list[str] = []
        for line in lines:
            match = self.pattern.match(line)
            if not match or match.group("kind").lower() != "solution":
                output.append(line)
                continue

            rest = match.group("rest") or ""
            value_match = self.lines_pattern.search(rest)
            if not value_match:
                output.append(line)
                continue

            value = value_match.group("value")
            rest_without = self.lines_pattern.sub("", rest, count=1)
            rest_without = re.sub(r"\{\s*\}", "", rest_without)
            rest_without = rest_without.rstrip()

            normalized = f"{match.group('indent')}!!! {match.group('kind')}{rest_without}"
            output.append(normalized)

            sentinel = f"{SOLUTION_LINES_SENTINEL_PREFIX}{value}{SOLUTION_LINES_SENTINEL_SUFFIX}"
            output.append(f"{match.group('indent')}    {sentinel}")
        return output


class ExamSpecialsExtension(Extension):
    """Register the Markdown processors enabling the shorthand syntax."""

    def extendMarkdown(self, md):  # type: ignore[override]
        md.preprocessors.register(ExamSolutionMetadata(md), "exam_solution_metadata", 25)


def makeExtension(**_kwargs):
    return ExamSpecialsExtension()


__all__ = ["ExamSpecialsExtension", "makeExtension"]
