"""
Annotate PDF with feedback at specified anchors.
"""

from __future__ import annotations

from pathlib import Path
import io
import random
import re
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF

# ReportLab
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.platypus import Frame, Paragraph, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import Color
from reportlab import rl_config

# svglib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib import colors


# =========================
# === Constantes & I/O ====
# =========================
PDF_INPUT = Path("exam2/exam.pdf")
PDF_OUTPUT = Path("fapouille.pdf")

PT_TO_MM = 0.352777778  # 25.4 / 72
LEFT_MARGIN_MM = 1.0
RIGHT_MARGIN_MM = 0.0
GAP_ABOVE_NEXT_ANCHOR_MM = 5.0
BOTTOM_MARGIN_MM = 10.0
LEFT_OF_ANCHOR_TEXT_MM = 5.0  # (gardé si besoin d'évolution)
PEN_COLOR = (0.85, 0.10, 0.10)

R_TAG = re.compile(r"^Q(\d+)-anchor$")
FONT_HAND_PATH = Path("fonts/Licorice-Regular.ttf")

FORTUNES_FR = [
    "Bien joué !",
    "Attention à la syntaxe...",
    "Excellent raisonnement !",
    "Presque parfait… encore un petit effort.",
    "C’est mieux que la dernière fois !",
    "Tu peux le faire !",
    "Une belle tentative.",
    "Je vois du progrès ici.",
    "Attention aux détails.",
    "Très bonne idée, mais mal appliquée.",
]


# =========================
# ======= Helpers =========
# =========================
def phrase_aleatoire() -> str:
    return random.choice(FORTUNES_FR)


def recolor_drawing(drawing, color: Tuple[float, float, float]) -> None:
    """Applique une couleur (fill/stroke) récursivement à un SVG 'drawing'."""
    if hasattr(drawing, "contents"):
        for elem in drawing.contents:
            recolor_drawing(elem, color)
    if hasattr(drawing, "fillColor"):
        drawing.fillColor = colors.Color(*color)
    if hasattr(drawing, "strokeColor"):
        drawing.strokeColor = colors.Color(*color)


def anchors_for_page(doc: fitz.Document, pno: int) -> List[Tuple[float, float, int]]:
    """
    Renvoie la liste des ancres présentes sur la page 'pno' sous forme de tuples:
      (x_mm_from_bottom, y_mm_from_bottom, qnum)
    en se basant sur les destinations nommées (hyperref).
    """
    out: List[Tuple[float, float, int]] = []
    resolver = getattr(doc, "resolve_names", None)
    if not resolver:
        return out

    page = doc[pno]
    crop = page.cropbox

    for name, dest in resolver().items():
        m = R_TAG.match(name)
        if not m or dest.get("page") != pno:
            continue
        x_pt, y_pt = dest.get("to", (None, None))
        if x_pt is None or y_pt is None:
            continue
        # Corrige un éventuel décalage de CropBox
        x_pt -= crop.x0
        y_pt -= crop.y0
        out.append((x_pt * PT_TO_MM, y_pt * PT_TO_MM, int(m.group(1))))

    # Tri du haut vers le bas (repère bas-gauche -> y haut > y bas)
    out.sort(key=lambda t: t[1], reverse=True)
    return out


# =========================
# ===== Overlay PDF =======
# =========================
def _register_hand_font(c: canvas.Canvas) -> None:
    """
    Enregistre la police manuscrite si disponible, sinon laisse Helvetica.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        if FONT_HAND_PATH.exists():
            pdfmetrics.registerFont(TTFont("HandFont", str(FONT_HAND_PATH)))
            c.setFont("HandFont", 14)
        else:
            c.setFont("Helvetica", 12)
    except Exception:
        # Fallback silencieux
        c.setFont("Helvetica", 12)


def _base_paragraph_style() -> ParagraphStyle:
    return ParagraphStyle(
        name="base",
        fontName="HandFont",  # si non dispo, ReportLab utilisera Helvetica enregistrée
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
        textColor=Color(0.9, 0.3, 0.1),
    )


def overlay_pdf(
    page_size_pts: Tuple[float, float], items: List[Dict[str, Any]]
) -> bytes:
    """
    Construit une page d’overlay (repère bas-gauche). Supporte:
      - kind="mark": texte centré (petite annotation)
      - kind="rect": rectangle (non utilisé ici, laissé pour extension)
      - kind="svg": rendu d’un SVG (avec redimension et recolorisation)
      - kind="textbox": texte multi-lignes dans un cadre avec word-wrap
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size_pts)

    # Légèrement plus robuste côté fonts
    rl_config.warnOnMissingFontGlyphs = False
    _register_hand_font(c)
    base_style = _base_paragraph_style()

    def to_xy(x_mm: float, y_mm: float) -> Tuple[float, float]:
        return (x_mm * mm, y_mm * mm)

    for it in items:
        kind = it["kind"]

        if kind == "mark":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            c.setFillColorRGB(*PEN_COLOR)
            c.setFont("Helvetica-Bold", it.get("fontsize", 16))
            c.drawCentredString(x, y, it["text"])

        elif kind == "rect":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            c.setStrokeColor(Color(1, 0, 1))  # magenta
            c.setLineWidth(1)
            c.rect(x, y, it["w_mm"] * mm, it["h_mm"] * mm, fill=0)

        elif kind == "svg":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            drawing = svg2rlg(str(it["path"]))
            w = it.get("w_mm")
            h = it.get("h_mm")
            if w and h:
                bw = max(drawing.minWidth(), 1e-6)
                bh = max(drawing.height, 1e-6)
                sx = (w * mm) / bw
                sy = (h * mm) / bh
                drawing.scale(sx, sy)

            if "color" in it:
                recolor_drawing(drawing, tuple(it["color"]))

            renderPDF.draw(drawing, c, x, y)

        elif kind == "textbox":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            w = it["w_mm"] * mm
            h = it["h_mm"] * mm

            pad_mm_val = it.get("pad_mm", 2.0)
            px = x + pad_mm_val * mm
            py = y + pad_mm_val * mm
            pw = max(0.0, w - 2 * pad_mm_val * mm)
            ph = max(0.0, h - 2 * pad_mm_val * mm)

            st = ParagraphStyle(
                name="box",
                parent=base_style,
                fontSize=it.get("fontSize", base_style.fontSize),
                leading=it.get("leading", None)
                or int(1.2 * it.get("fontSize", base_style.fontSize)),
                alignment=it.get("align", TA_LEFT),
            )

            story = [Paragraph(it["text"], st)]
            kif = KeepInFrame(pw, ph, story, mode=it.get("overflow", "shrink"))
            Frame(px, py, pw, ph, showBoundary=0).addFromList([kif], c)

        # Autres 'kind' ignorés volontairement

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================
# === Génération items ====
# =========================
def grade_box_items_for_first_page(
    page: fitz.Page, grade: float = 4.2
) -> List[Dict[str, Any]]:
    """Boîte de note sur la première page (en haut-droite)."""
    page_w_mm = page.rect.width * PT_TO_MM
    page_h_mm = page.rect.height * PT_TO_MM

    box_w_mm = 50.0
    box_h_mm = 50.0

    x_box_mm = max(0.0, page_w_mm - 60.0 - box_w_mm)
    y_box_mm = max(0.0, page_h_mm - 30.0 - box_h_mm)

    return [
        {
            "kind": "textbox",
            "x_mm": x_box_mm,
            "y_mm": y_box_mm,
            "w_mm": box_w_mm,
            "h_mm": box_h_mm,
            "pad_mm": 1.5,
            "text": str(grade),
            "fontSize": 55,
            "align": TA_CENTER,
            "overflow": "shrink",
        }
    ]


def feedback_items_from_anchors(
    anchors: List[Tuple[float, float, int]],
) -> List[Dict[str, Any]]:
    """
    Entre chaque paire d’ancres successives (du haut vers le bas), place:
      - une icône SVG (check/cross) au niveau de l’ancre courante
      - un encart de texte à gauche (aléa parmi FORTUNES_FR)
    """
    items: List[Dict[str, Any]] = []

    for j, (x_mm, y_mm, _qnum) in enumerate(anchors):
        if j + 1 < len(anchors):
            _, y_next, _ = anchors[j + 1]
        else:
            y_next = BOTTOM_MARGIN_MM  # bas de la page pour la dernière ancre

        top_y = y_mm
        bottom_y = max(0.0, y_next + GAP_ABOVE_NEXT_ANCHOR_MM)

        if bottom_y >= top_y:
            continue

        h = top_y - bottom_y
        w = max(0.0, max(LEFT_MARGIN_MM, x_mm - RIGHT_MARGIN_MM) - LEFT_MARGIN_MM)
        if w <= 0.1 or h <= 0.1:
            continue

        glyph = random.choice(["check", "cross"])

        # Icône près de l’ancre
        items.append(
            {
                "kind": "svg",
                "path": Path(f"{glyph}.svg"),
                "x_mm": x_mm - 4.0,
                "y_mm": top_y - 5.0,
                "w_mm": 3.0,
                "h_mm": 3.0,
                "color": (0.9, 0.3, 0.1),
            }
        )

        # Commentaire à gauche
        items.append(
            {
                "kind": "textbox",
                "x_mm": LEFT_MARGIN_MM,
                "y_mm": bottom_y,
                "w_mm": w,
                "h_mm": h - 3.0,
                "pad_mm": 0.0,
                "text": phrase_aleatoire(),
                "fontSize": 10,
                "align": TA_RIGHT,
                "overflow": "shrink",
            }
        )

    return items


# =========================
# ========= Main ==========
# =========================
def main() -> None:
    if not PDF_INPUT.exists():
        raise FileNotFoundError(f"Fichier introuvable: {PDF_INPUT}")

    doc = fitz.open(str(PDF_INPUT))
    try:
        for pno, page in enumerate(doc):
            # Page 1 : boîte de note puis on passe à la suivante (comportement d’origine)
            if pno == 0:
                items = grade_box_items_for_first_page(page, grade=4.2)
                ov = fitz.open(
                    "pdf", overlay_pdf((page.rect.width, page.rect.height), items)
                )
                page.show_pdf_page(page.rect, ov, 0)
                ov.close()
                continue

            anchors = anchors_for_page(doc, pno)
            if not anchors:
                continue

            items = feedback_items_from_anchors(anchors)
            if not items:
                continue

            ov = fitz.open(
                "pdf", overlay_pdf((page.rect.width, page.rect.height), items)
            )
            page.show_pdf_page(page.rect, ov, 0)
            ov.close()

        doc.save(str(PDF_OUTPUT))
    finally:
        doc.close()


if __name__ == "__main__":
    main()
