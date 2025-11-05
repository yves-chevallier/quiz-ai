"""
Utilities to split a merged responses PDF into per-student documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from .utils import ensure_directory


@dataclass(frozen=True)
class SplitResult:
    """Summary returned after splitting a responses PDF."""

    responses_pdf: Path
    template_pdf: Path
    total_pages: int
    template_pages: int
    packets_written: int
    packet_paths: List[Path]
    leftovers: int = 0


class SplitError(RuntimeError):
    """Raised when a responses PDF cannot be split."""


def split_responses_pdf(
    responses_pdf: Path,
    template_pdf: Path,
    output_dir: Path,
    *,
    prefix: str | None = None,
    start_index: int = 1,
) -> SplitResult:
    """
    Split `responses_pdf` into per-student packets using `template_pdf` page count.

    Parameters
    ----------
    responses_pdf:
        Scanned PDF containing all student responses in order.
    template_pdf:
        Original exam PDF used to determine per-student page count.
    output_dir:
        Directory where the split PDFs will be written.
    prefix:
        Optional filename prefix. Defaults to the responses PDF stem.
    start_index:
        Starting numeric suffix for generated files (defaults to 1).
    """
    if not responses_pdf.exists():
        raise SplitError(f"Responses PDF not found: {responses_pdf}")
    if not template_pdf.exists():
        raise SplitError(f"Template exam PDF not found: {template_pdf}")

    prefix = prefix or responses_pdf.stem
    output_dir = ensure_directory(output_dir)

    with fitz.open(responses_pdf) as responses_doc:
        total_pages = responses_doc.page_count
        if total_pages == 0:
            raise SplitError(f"Responses PDF contains no pages: {responses_pdf}")

        with fitz.open(template_pdf) as template_doc:
            template_pages = template_doc.page_count
        if template_pages <= 0:
            raise SplitError(f"Unable to determine template page count from: {template_pdf}")

        packets, leftovers = divmod(total_pages, template_pages)
        if packets == 0:
            raise SplitError(
                f"Responses PDF has fewer pages ({total_pages}) than the template ({template_pages}). "
                "Please double-check the input files."
            )

        pad_width = max(2, len(str(start_index + packets - 1)))
        written: List[Path] = []

        for index in range(packets):
            packet_doc = fitz.open()
            start_page = index * template_pages
            end_page = start_page + template_pages - 1
            packet_doc.insert_pdf(responses_doc, from_page=start_page, to_page=end_page)
            out_path = output_dir / f"{prefix}-{start_index + index:0{pad_width}d}.pdf"
            packet_doc.save(out_path)
            packet_doc.close()
            written.append(out_path)

    return SplitResult(
        responses_pdf=responses_pdf,
        template_pdf=template_pdf,
        total_pages=total_pages,
        template_pages=template_pages,
        packets_written=len(written),
        packet_paths=written,
        leftovers=leftovers,
    )

