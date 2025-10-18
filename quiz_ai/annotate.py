"""
Annotate a PDF with feedback icons and comments at question anchors.
"""
from __future__ import annotations

from pathlib import Path
import io
from typing import Dict, Iterable, List, Tuple, Optional, Union, Literal, TypedDict

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
FONT_HAND_PATH = SCRIPT_PATH / "assets/fonts/Licorice-Regular.ttf"

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

@dataclass(frozen=True)
class MarkItem:
    """
    Represents a mark (e.g. a tick or cross) on the PDF.
    """
    kind: Literal["mark"]
    x_mm: float
    y_mm: float
    text: str
    fontsize: int

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
    correct: bool
    comment: str


def _register_hand_font(c: canvas.Canvas, font_path: Path | None = None) -> None:
    """Register a handwriting font if available; fallback to Helvetica."""
    font_to_use = font_path or FONT_HAND_PATH
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
        if it["kind"] == "mark":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            c.setFillColorRGB(*PEN_COLOR)
            c.setFont("Helvetica-Bold", it["fontsize"])
            c.drawCentredString(x, y, it["text"])

        elif it["kind"] == "svg":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            drawing: Optional[Drawing] = svg2rlg(str(it["path"]))
            if drawing is None:
                # Skip silently if the SVG cannot be parsed
                continue

            bw = max(drawing.minWidth(), 1e-6)
            bh = max(drawing.height, 1e-6)
            sx = (it["w_mm"] * mm) / bw
            sy = (it["h_mm"] * mm) / bh
            drawing.scale(sx, sy)

            # Recolor recursively
            def recolor(d):
                if hasattr(d, "contents"):
                    for e in d.contents:
                        recolor(e)
                if hasattr(d, "fillColor"):
                    d.fillColor = colors.Color(*it["color"])
                if hasattr(d, "strokeColor"):
                    d.strokeColor = colors.Color(*it["color"])

            recolor(drawing)
            renderPDF.draw(drawing, c, x, y)

        elif it["kind"] == "textbox":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            w = it["w_mm"] * mm
            h = it["h_mm"] * mm
            pad = it.get("pad_mm", 0.0) * mm

            px = x + pad
            py = y + pad
            pw = max(0.0, w - 2 * pad)
            ph = max(0.0, h - 2 * pad)

            st = ParagraphStyle(
                name="box",
                parent=base_style,
                fontSize=it.get("fontSize", base_style.fontSize),  # type: ignore[arg-type]
                leading=int(1.2 * it.get("fontSize", base_style.fontSize)),  # type: ignore[arg-type]
                alignment=it.get("align", TA_LEFT),
            )

            story = [Paragraph(it["text"], st)]
            kif = KeepInFrame(pw, ph, story, mode=it.get("overflow", "shrink"))
            Frame(px, py, pw, ph, showBoundary=0).addFromList([kif], c)

    c.showPage()
    c.save()
    return buf.getvalue()



def _items_from_anchors_for_page(
    anchors_mm: List[Dict[str, float]],
    feedback_by_id: Dict[int, Dict[str, Union[bool, str]]],
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
        w = max(0.0, max(LEFT_MARGIN_MM, x_mm - RIGHT_MARGIN_MM) - LEFT_MARGIN_MM)
        if w <= 0.1 or h <= 0.1:
            continue

        icon_path = check_svg if bool(fb["correct"]) else cross_svg

        items.append(
            SvgItem(
                kind="svg",
                path=icon_path,
                x_mm=x_mm - ICON_OFFSET_X_MM,
                y_mm=top_y - ICON_OFFSET_Y_MM,
                w_mm=ICON_SIZE_MM,
                h_mm=ICON_SIZE_MM,
                color=PEN_COLOR,
            )
        )

        items.append(
            TextBoxItem(
                kind="textbox",
                x_mm=LEFT_MARGIN_MM,
                y_mm=bottom_y,
                w_mm=w,
                h_mm=h - 3.0,
                pad_mm=0.0,
                text=str(fb["comment"]),
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
    font_path: Path = Path("assets/fonts/Licorice-Regular.ttf"),
    check_icon: Path = Path("assets/icons/check.svg"),
    cross_icon: Path = Path("assets/icons/cross.svg"),
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
        Each element must provide: id (int), correct (bool), comment (str).
    font_path : Path
        Path to a TTF font used for text rendering. Falls back to Helvetica if missing.
    check_icon : Path
        Path to the SVG icon for correct answers.
    cross_icon : Path
        Path to the SVG icon for incorrect answers.
    """
    # Normalize feedback into a mapping by question id
    fb_map: Dict[int, Feedback] = {}
    for f in feedback:
        if isinstance(f, Feedback):
            fb_map[f.id] = f
        else:
            fid = int(f["id"])           # type: ignore[index]
            ok = bool(f["correct"])      # type: ignore[index]
            txt = str(f["comment"])      # type: ignore[index]
            fb_map[fid] = Feedback(id=fid, correct=ok, comment=txt)

    doc = fitz.open(str(pdf_input))
    try:
        # Loop by explicit index to avoid Pylance complaints about fitz.Document iterability
        for pno in range(len(doc)):
            page = doc[pno]
            page_w_mm = page.rect.width * PT_TO_MM
            page_h_mm = page.rect.height * PT_TO_MM

            # Find the matching page entry in anchors (O(n) is fine; page count is small)
            page_entry = next((p for p in anchors.pages if p.page_index == pno), None)
            if not page_entry or not page_entry.anchors_mm:
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
            if not items:
                continue

            ov_pdf = _overlay_pdf(
                (page.rect.width, page.rect.height),
                items=items,
                font_path=font_path,
            )
            ov = fitz.open("pdf", ov_pdf)
            page.show_pdf_page(page.rect, ov, 0)
            ov.close()

        doc.save(str(pdf_output))
    finally:
        doc.close()
