"""Markdown extension enabling exam-specific shorthand syntax."""

from __future__ import annotations

import re
from xml.etree import ElementTree as etree

from markdown import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.preprocessors import Preprocessor


class ExamFillPreprocessor(Preprocessor):
    """Replace `--- n ---` shorthands by explicit markers."""

    pattern = re.compile(r"^\s*---\s*(?:(grid)\s+)?(\d+)\s*---\s*$", re.IGNORECASE)

    def run(self, lines: list[str]) -> list[str]:
        output: list[str] = []
        for line in lines:
            match = self.pattern.match(line)
            if not match:
                output.append(line)
                continue
            kind = "grid" if match.group(1) else "lines"
            count = match.group(2)
            output.append(f'<exam-fill data-kind="{kind}" data-count="{count}"></exam-fill>')
        return output


class ExamFillinPattern(InlineProcessor):
    """Convert `{{answer}}` tokens into inline markers."""

    def handleMatch(  # type: ignore[override]
        self,
        m: re.Match[str],
        data: str,
    ) -> tuple[etree.Element | None, int | None, int | None]:
        value = (m.group(1) or "").strip()
        if not value:
            return None, None, None
        node = etree.Element("exam-fillin")
        node.set("data-value", value)
        return node, m.start(0), m.end(0)


class ExamSpecialsExtension(Extension):
    """Register the Markdown processors enabling the shorthand syntax."""

    fillin_pattern = r"\{\{([^{}]+)\}\}"

    def extendMarkdown(self, md):  # type: ignore[override]
        md.preprocessors.register(ExamFillPreprocessor(md), "exam_fill_blocks", 26)
        md.inlinePatterns.register(ExamFillinPattern(self.fillin_pattern, md), "exam_fillin", 175)


def makeExtension(**_kwargs):
    return ExamSpecialsExtension()


__all__ = ["ExamSpecialsExtension", "makeExtension"]
