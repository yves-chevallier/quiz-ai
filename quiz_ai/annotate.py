"""Annotate a PDF with feedback icons and comments at question anchors."""
from __future__ import annotations

import io
import re
from pathlib import Path
from fractions import Fraction
from typing import Dict, Iterable, List, Tuple, Optional, Union, Literal

import fitz  # PyMuPDF
from dataclasses import dataclass
# ReportLab (move these to top-level to avoid import-outside-toplevel)
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.platypus import Frame, Paragraph, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import Color
from reportlab import rl_config
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# svglib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing


from .anchors import Anchors  # type: ignore[reportMissingImports]

SCRIPT_PATH = Path(__file__).parent.resolve()
FONT_HAND_PATH = SCRIPT_PATH / "assets/fonts/IndieFlower-Regular.ttf"

# Geometry/appearance constants
PT_TO_MM = 25.4 / 72.0
LEFT_MARGIN_MM = 1.0
RIGHT_MARGIN_MM = 0.0
GAP_ABOVE_NEXT_ANCHOR_MM = 5.0
BOTTOM_MARGIN_MM = 10.0
ICON_SIZE_MM = 3.0
ICON_OFFSET_X_MM = 4.0
ICON_OFFSET_Y_MM = 5.0
TEXT_PADDING_MM = 0.0
PEN_COLOR = (0.85, 0.10, 0.10)
ICON_EXTRA_SHIFT_X_MM = 5.0
ANNOTATION_EXTRA_WIDTH_MM = 10.0  # Extend comment boxes by 1cm

@dataclass(frozen=True)
class MarkItem:
    """Represents a mark (text stamp) on the PDF."""
    kind: Literal["mark"]
    x_mm: float
    y_mm: float
    text: str
    fontsize: int
    align: Literal["left", "center", "right"] = "center"

@dataclass(frozen=True)
class SvgItem:
    """
    Represents an SVG image on the PDF.
    """
    kind: Literal["svg"]
    path: Path
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    color: Tuple[float, float, float]

@dataclass(frozen=True)
class TextBoxItem:
    """
    Represents a text box on the PDF.
    """
    kind: Literal["textbox"]
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    pad_mm: float
    text: str
    fontSize: int
    align: int  # TA_LEFT/TA_RIGHT/TA_CENTER
    overflow: Literal["shrink", "truncate"]  # whatever you use

OverlayItem = Union[MarkItem, SvgItem, TextBoxItem]


@dataclass(frozen=True)
class Feedback:
    """Single feedback item mapped to a question id (anchor qnum)."""
    id: int
    status: Literal["correct", "incorrect", "partial", "unknown"]
    comment: str
    awarded_points: Optional[float] = None
    max_points: Optional[float] = None
    awarded_ratio: Optional[float] = None
    flags: Tuple[str, ...] = ()

    def is_correct(self) -> bool:
        return self.status == "correct"

    def has_partial_credit(self) -> bool:
        ratio = self._ratio()
        return ratio is not None and 0.0 < ratio < 1.0

    def score_display(self) -> Optional[str]:
        """Return an x/y style score for partial credit, if available."""
        if not self.has_partial_credit():
            return None

        if self.awarded_points is not None and self.max_points not in (None, 0):
            awarded = _format_points(self.awarded_points)
            maximum = _format_points(self.max_points or 0.0)
            return f"{awarded}/{maximum}"

        ratio = self._ratio()
        if ratio is None:
            return None
        frac = Fraction(ratio).limit_denominator(20)
        return f"{frac.numerator}/{frac.denominator}"

    def _ratio(self) -> Optional[float]:
        if self.awarded_points is not None and self.max_points not in (None, 0):
            if self.max_points:
                return self.awarded_points / self.max_points
        if self.awarded_ratio is not None:
            return self.awarded_ratio
        return None

    def has_icon(self) -> bool:
        # Only show icons for full correctness or incorrectness.
        return self.status in {"correct", "incorrect"}


def _format_points(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _resolve_asset_path(path: Path) -> Path:
    """Return an absolute path for bundled assets while respecting user overrides."""
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = SCRIPT_PATH / path
    if candidate.exists():
        return candidate
    return path


def _resolve_font_path(font_path: Path | None) -> Path:
    candidate: Optional[Path] = None
    if font_path is not None:
        candidate = _resolve_asset_path(font_path)
        if candidate.exists():
            return candidate
    if FONT_HAND_PATH.exists():
        return FONT_HAND_PATH
    if candidate is not None:
        return candidate
    return FONT_HAND_PATH


def _maybe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _get_value(data: object, key: str) -> object:
    if isinstance(data, dict):
        return data.get(key)
    return getattr(data, key, None)


def _collect_flags(payload: object) -> Tuple[str, ...]:
    raw = _get_value(payload, "flags")
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple, set)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    text = str(raw).strip()
    return (text,) if text else ()


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    selected = " ".join(sentences[:max_sentences]).strip()
    return selected or cleaned


def _truncate_text(text: str, max_length: int = 240) -> str:
    if len(text) <= max_length:
        return text
    truncated = text[: max_length - 1].rstrip()
    return truncated + "…"


def _summarise_comment(
    *,
    status: str,
    comment_text: object,
    remarks: object,
    justification: object,
    flags: Tuple[str, ...],
) -> str:
    status = (status or "").lower()

    def _to_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    candidates = []
    # Prioritise remarks, then comment, then justification.
    for source in (_to_text(remarks), _to_text(comment_text), _to_text(justification)):
        if source:
            candidates.append(source)

    summary = ""
    for candidate in candidates:
        summary = _first_sentences(candidate, max_sentences=2)
        if summary:
            break

    if not summary and flags:
        summary = _first_sentences("; ".join(flags), max_sentences=1)

    if status == "correct" and not flags:
        return ""

    if not summary:
        return ""

    return _truncate_text(summary, max_length=200)


def _grade_stamp_item(
    page_w_mm: float,
    page_h_mm: float,
    overall_points: Tuple[float, float],
) -> Optional[MarkItem]:
    obtained, total = overall_points
    try:
        total = float(total)
        obtained = float(obtained)
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    ratio = max(0.0, min(1.0, obtained / total))
    note = round((ratio * 5.0) + 1.0, 1)
    x_mm = max(0.0, page_w_mm - 70.0)
    y_mm = max(0.0, page_h_mm - 50.0)
    return MarkItem(
        kind="mark",
        x_mm=x_mm,
        y_mm=y_mm,
        text=f"{note:.1f}",
        fontsize=45,
        align="left",
    )


def _register_hand_font(c: canvas.Canvas, font_path: Path | None = None) -> None:
    """Register a handwriting font if available; fallback to Helvetica."""
    font_to_use = _resolve_font_path(font_path)
    try:
        if font_to_use.exists():
            pdfmetrics.registerFont(TTFont("HandFont", str(font_to_use)))
            c.setFont("HandFont", 14)
        else:
            c.setFont("Helvetica", 12)
    except (OSError, RuntimeError, ValueError):
        c.setFont("Helvetica", 12)



def _recolor_drawing(drawing, rgb: Tuple[float, float, float]) -> None:
    """Recursively apply fill/stroke colors to an svglib drawing."""
    if hasattr(drawing, "contents"):
        for elem in drawing.contents:
            _recolor_drawing(elem, rgb)
    if hasattr(drawing, "fillColor"):
        drawing.fillColor = colors.Color(*rgb)
    if hasattr(drawing, "strokeColor"):
        drawing.strokeColor = colors.Color(*rgb)


def _overlay_pdf(
    page_size_pts: Tuple[float, float],
    items: List[OverlayItem],
    *,
    font_path: Path | None = None,
) -> bytes:
    """Build a one-page overlay PDF (bottom-left origin)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size_pts)

    rl_config.warnOnMissingFontGlyphs = False
    _register_hand_font(c, font_path=font_path)

    base_style = ParagraphStyle(
        name="base",
        fontName="HandFont",  # ReportLab will fallback if not registered
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
        textColor=Color(*PEN_COLOR),
    )

    def to_xy(x_mm: float, y_mm: float) -> Tuple[float, float]:
        return (x_mm * mm, y_mm * mm)

    for it in items:
        if isinstance(it, MarkItem):
            x, y = to_xy(it.x_mm, it.y_mm)
            c.setFillColorRGB(*PEN_COLOR)
            try:
                c.setFont("HandFont", it.fontsize)
            except (ValueError, RuntimeError):
                c.setFont("Helvetica-Bold", it.fontsize)
            if it.align == "left":
                c.drawString(x, y, it.text)
            elif it.align == "right":
                c.drawRightString(x, y, it.text)
            else:
                c.drawCentredString(x, y, it.text)

        elif isinstance(it, SvgItem):
            x, y = to_xy(it.x_mm, it.y_mm)
            drawing: Optional[Drawing] = svg2rlg(str(it.path))
            if drawing is None:
                # Skip silently if the SVG cannot be parsed
                continue

            bw = max(drawing.minWidth(), 1e-6)
            bh = max(drawing.height, 1e-6)
            sx = (it.w_mm * mm) / bw
            sy = (it.h_mm * mm) / bh
            drawing.scale(sx, sy)

            # Recolor recursively
            def recolor(d):
                if hasattr(d, "contents"):
                    for e in d.contents:
                        recolor(e)
                if hasattr(d, "fillColor"):
                    d.fillColor = colors.Color(*it.color)
                if hasattr(d, "strokeColor"):
                    d.strokeColor = colors.Color(*it.color)

            recolor(drawing)
            renderPDF.draw(drawing, c, x, y)

        elif isinstance(it, TextBoxItem):
            x, y = to_xy(it.x_mm, it.y_mm)
            w = it.w_mm * mm
            h = it.h_mm * mm
            pad = it.pad_mm * mm

            px = x + pad
            py = y + pad
            pw = max(0.0, w - 2 * pad)
            ph = max(0.0, h - 2 * pad)

            st = ParagraphStyle(
                name="box",
                parent=base_style,
                fontSize=it.fontSize,
                leading=int(1.2 * it.fontSize),
                alignment=it.align,
            )

            story = [Paragraph(it.text, st)]
            kif = KeepInFrame(pw, ph, story, mode=it.overflow)
            Frame(px, py, pw, ph, showBoundary=0).addFromList([kif], c)

    c.showPage()
    c.save()
    return buf.getvalue()



def _items_from_anchors_for_page(
    anchors_mm: List[Dict[str, float]],
    feedback_by_id: Dict[int, Feedback],
    check_svg: Path,
    cross_svg: Path,
    *,
    page_w_mm: float | None = None,
    page_h_mm: float | None = None,
) -> List[OverlayItem]:
    """Build overlay items for one page from anchors and feedback mapping."""
    items: List[OverlayItem] = []
    if not anchors_mm:
        return items

    for j, a in enumerate(anchors_mm):
        qnum = int(a["qnum"])  # type: ignore[index]
        fb = feedback_by_id.get(qnum)
        if not fb:
            continue

        x_mm = float(a["x_mm"])  # type: ignore[index]
        y_mm = float(a["y_mm"])  # type: ignore[index]

        if j + 1 < len(anchors_mm):
            y_next = float(anchors_mm[j + 1]["y_mm"])  # type: ignore[index]
        else:
            y_next = BOTTOM_MARGIN_MM

        top_y = y_mm
        bottom_y = max(0.0, y_next + GAP_ABOVE_NEXT_ANCHOR_MM)
        if bottom_y >= top_y:
            continue

        h = top_y - bottom_y
        base_width = max(0.0, max(LEFT_MARGIN_MM, x_mm - RIGHT_MARGIN_MM) - LEFT_MARGIN_MM)
        w = base_width + ANNOTATION_EXTRA_WIDTH_MM
        if page_w_mm is not None:
            w = min(w, max(0.0, page_w_mm - LEFT_MARGIN_MM))
        if w <= 0.1 or h <= 0.1:
            continue

        icon_path = check_svg if fb.is_correct() else cross_svg
        icon_x = x_mm - ICON_OFFSET_X_MM + ICON_EXTRA_SHIFT_X_MM
        icon_y = top_y - ICON_OFFSET_Y_MM

        if fb.has_icon():
            items.append(
                SvgItem(
                    kind="svg",
                    path=icon_path,
                    x_mm=icon_x,
                    y_mm=icon_y,
                    w_mm=ICON_SIZE_MM,
                    h_mm=ICON_SIZE_MM,
                    color=PEN_COLOR,
                )
            )

        score_text = fb.score_display()
        comment_text = fb.comment.strip()
        if fb.has_partial_credit() and score_text:
            # Show the score close to the anchor, using a smaller font size.
            if fb.has_icon():
                score_center_x = icon_x + (ICON_SIZE_MM / 2.0)
                score_y = icon_y - (ICON_SIZE_MM * 0.65)
            else:
                score_center_x = x_mm
                score_y = top_y - (ICON_OFFSET_Y_MM * 0.6)
            items.append(
                MarkItem(
                    kind="mark",
                    x_mm=score_center_x,
                    y_mm=score_y,
                    text=score_text,
                    fontsize=9,
                )
            )
            if comment_text:
                comment_text = f"{score_text}\n\n{comment_text}"
            else:
                comment_text = score_text

        if comment_text:
            items.append(
                TextBoxItem(
                    kind="textbox",
                    x_mm=LEFT_MARGIN_MM,
                    y_mm=bottom_y,
                    w_mm=w,
                    h_mm=max(0.0, h - 3.0),
                    pad_mm=0.0,
                    text=comment_text,
                    fontSize=10,
                    align=TA_LEFT,
                    overflow="shrink",
                )
            )

    return items


def annotate_pdf(
    pdf_input: Path,
    pdf_output: Path,
    anchors: Anchors,
    feedback: Iterable[Dict[str, object]] | Iterable[Feedback],
    *,
    overall_points: Optional[Tuple[float, float]] = None,
    font_path: Optional[Path] = None,
    check_icon: Path = Path("assets/glyphs/check.svg"),
    cross_icon: Path = Path("assets/glyphs/cross.svg"),
) -> None:
    """
    Annotate `pdf_input` with icons/comments based on `anchors` and `feedback`, and save to `pdf_output`.

    Parameters
    ----------
    pdf_input : Path
        Source PDF to annotate.
    pdf_output : Path
        Destination PDF file path.
    anchors : Anchors
        Pydantic schema produced by the extraction step.
    feedback : iterable of dicts or Feedback
        Each element must provide at least an ``id`` and ``status`` (``correct``,
        ``incorrect`` or ``partial``) plus optional ``awarded_points``,
        ``max_points``, ``awarded_ratio`` and textual feedback (``comment``,
        ``remarks``, ``justification``).
    overall_points : Optional[Tuple[float, float]]
        Tuple ``(points_obtenus, points_totaux)`` to display the hybrid grade on the first page.
    font_path : Optional[Path]
        Optional path to a TTF font used for text rendering. Falls back to the bundled handwriting font.
    check_icon : Path
        Path to the SVG icon for correct answers.
    cross_icon : Path
        Path to the SVG icon for incorrect answers.
    """
    font_path_resolved = _resolve_font_path(font_path)
    check_icon = _resolve_asset_path(check_icon)
    cross_icon = _resolve_asset_path(cross_icon)

    # Normalize feedback into a mapping by question id
    fb_map: Dict[int, Feedback] = {}
    for f in feedback:
        if isinstance(f, Feedback):
            fb_map[f.id] = f
            continue

        fid_raw = _get_value(f, "id")
        if fid_raw is None:
            raise ValueError("Feedback entry is missing required field 'id'.")
        try:
            fid = int(fid_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid feedback id: {fid_raw!r}") from exc
        status = str(_get_value(f, "status") or "").lower()
        if status not in {"correct", "incorrect", "partial"}:
            # Fall back to legacy boolean flags
            legacy_correct = _get_value(f, "correct")
            if isinstance(legacy_correct, bool):
                status = "correct" if legacy_correct else "incorrect"
            else:
                status = "unknown"

        awarded_points = _maybe_float(_get_value(f, "awarded_points"))
        max_points = _maybe_float(_get_value(f, "max_points"))
        awarded_ratio = _maybe_float(_get_value(f, "awarded_ratio"))

        flags = _collect_flags(f)

        comment = _summarise_comment(
            status=status,
            comment_text=_get_value(f, "comment"),
            remarks=_get_value(f, "remarks"),
            justification=_get_value(f, "justification"),
            flags=flags,
        )

        if status == "correct" and not flags:
            # Follow requirement: keep correct answers silent unless a special flag applies.
            comment = ""

        fb_map[fid] = Feedback(
            id=fid,
            status=status,  # type: ignore[arg-type]
            comment=comment,
            awarded_points=awarded_points,
            max_points=max_points,
            awarded_ratio=awarded_ratio,
            flags=flags,
        )

    doc = fitz.open(str(pdf_input))
    try:
        # Loop by explicit index to avoid Pylance complaints about fitz.Document iterability
        for pno in range(len(doc)):
            page = doc[pno]
            page_w_mm = page.rect.width * PT_TO_MM
            page_h_mm = page.rect.height * PT_TO_MM
            extra_items: List[OverlayItem] = []
            if pno == 0 and overall_points:
                stamp = _grade_stamp_item(page_w_mm, page_h_mm, overall_points)
                if stamp:
                    extra_items.append(stamp)

            # Find the matching page entry in anchors (O(n) is fine; page count is small)
            page_entry = next((p for p in anchors.pages if p.page_index == pno), None)
            if not page_entry or not page_entry.anchors_mm:
                if extra_items:
                    ov_pdf = _overlay_pdf(
                        (page.rect.width, page.rect.height),
                        items=extra_items,
                        font_path=font_path_resolved,
                    )
                    ov = fitz.open("pdf", ov_pdf)
                    page.show_pdf_page(page.rect, ov, 0)
                    ov.close()
                continue

            # anchors_mm are Pydantic models; convert to dicts for generic handling
            anchors_mm = [a.model_dump() for a in page_entry.anchors_mm]

            items = _items_from_anchors_for_page(
                page_w_mm=page_w_mm,
                page_h_mm=page_h_mm,
                anchors_mm=anchors_mm,
                feedback_by_id=fb_map,
                check_svg=check_icon,
                cross_svg=cross_icon,
            )
            if extra_items:
                items = extra_items + items
            if not items:
                continue

            ov_pdf = _overlay_pdf(
                (page.rect.width, page.rect.height),
                items=items,
                font_path=font_path_resolved,
            )
            ov = fitz.open("pdf", ov_pdf)
            page.show_pdf_page(page.rect, ov, 0)
            ov.close()

        doc.save(str(pdf_output))
    finally:
        doc.close()
