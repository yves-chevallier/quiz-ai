#!/usr/bin/env python3
from pathlib import Path
import io, re, random
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta
from reportlab.platypus import Frame, Paragraph, KeepInFrame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import Color
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from reportlab.lib import colors

PDF_INPUT = Path("exam2/exam.pdf")
PDF_OUTPUT = Path("fapouille.pdf")

PT_TO_MM = 0.352777778  # 25.4 / 72
LEFT_MARGIN_MM = 1.0
RIGHT_MARGIN_MM = 0.0
LEFT_OF_ANCHOR_TEXT_MM = 5.0
GAP_ABOVE_NEXT_ANCHOR_MM = 5.0  # marge au-dessus de l’ancre suivante (en mm)
PEN_COLOR = (0.85, 0.10, 0.10)

R_TAG = re.compile(r"^Q(\d+)-anchor$")
FONT_HAND_PATH = Path("fonts/Licorice-Regular.ttf")#HomemadeApple-Regular.ttf")

check_mark = svg2rlg("check.svg")
check_mark.scale(0.7, 0.7)


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

def phrase_aleatoire() -> str:
    return random.choice(FORTUNES_FR)

def recolor_drawing(drawing, color):
    """Force la couleur de tous les éléments du dessin SVG."""
    if hasattr(drawing, "contents"):
        for elem in drawing.contents:
            recolor_drawing(elem, color)
    if hasattr(drawing, "fillColor"):
        drawing.fillColor = color
    if hasattr(drawing, "strokeColor"):
        drawing.strokeColor = color

def anchors_for_page(doc: fitz.Document, pno: int) -> list[tuple[float, float, int]]:
    """
    Renvoie les ancres sous forme [(x_mm_from_bottom, y_mm_from_bottom, qnum), ...]
    en se basant sur les destinations nommées hyperref.
    """
    out: list[tuple[float, float, int]] = []
    resolver = getattr(doc, "resolve_names", None)
    if not resolver:
        return out

    crop = doc[pno].cropbox  # si jamais il y avait un offset de crop

    for name, dest in resolver().items():  # name -> {'page': int, 'to': (x,y), ...}
        m = R_TAG.match(name)
        if not m or dest.get("page") != pno:
            continue
        x_pt, y_pt = dest.get("to", (None, None))
        if x_pt is None or y_pt is None:
            continue
        # Ajustement éventuel si CropBox décale (ici c'est 0,0 mais on garde le correctif)
        x_pt -= crop.x0
        y_pt -= crop.y0
        out.append((x_pt * PT_TO_MM, y_pt * PT_TO_MM, int(m.group(1))))

    # On va travailler du HAUT vers le BAS (plus naturel visuellement)
    # En repère bas-gauche : y haut > y bas -> tri décroissant
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def overlay_pdf(page_size_pts: Tuple[float, float], items: list[Dict]) -> bytes:
    """
    Construit une page d’overlay (repère bas-gauche). Gère:
      - kind="mark": un petit marqueur centré
      - kind="rect": un rectangle (cadre)
      - kind="textbox": du texte multi-lignes auto-wrap dans un cadre
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size_pts)
    c.setFont("Helvetica", 12)  # fallback
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont("HandFont", str(FONT_HAND_PATH)))
        c.setFont("HandFont", 14)
    except Exception as e:
        print("Police non chargée, fallback Helvetica :", e)

    def to_xy(x_mm, y_mm):
        return (x_mm * mm, y_mm * mm)

    # Style de paragraphe par défaut (modifiable par item)
    base_style = ParagraphStyle(
        name="base",
        fontName="HandFont",      # ok pour FR de base ; si besoin, enregistre un TTF
        fontSize=10,
        leading=12,                # ~1.2 * fontSize
        alignment=TA_LEFT,         # TA_JUSTIFY si désiré
        spaceBefore=0,
        spaceAfter=0,
        textColor=Color(0.9, 0.3, 0.1),
    )

    for it in items:
        kind = it["kind"]
        if kind == "mark":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            c.setFillColorRGB(*PEN_COLOR)
            c.setFont("Helvetica-Bold", it.get("fontsize", 16))
            c.drawCentredString(x, y, it["text"])

        elif kind == "rect":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            c.setStrokeColor(magenta)
            c.setLineWidth(1)
            c.rect(x, y, it["w_mm"] * mm, it["h_mm"] * mm, fill=0)
        elif kind == "svg":
            x, y = to_xy(it["x_mm"], it["y_mm"])
            drawing = svg2rlg(str(it["path"]))       # Path vers ton SVG
            # Mise à l’échelle optionnelle (par mm)
            w = it.get("w_mm"); h = it.get("h_mm")
            if w and h:
                # scale prend des facteurs, pas des mm : calculer par bbox
                bw = drawing.minWidth(); bh = drawing.height
                sx = (w * mm) / max(bw, 1e-6)
                sy = (h * mm) / max(bh, 1e-6)
                drawing.scale(sx, sy)

            # 🎨 Recolorisation (ex. color=(0.1,0.6,0.2))
            if "color" in it:
                r, g, b = it["color"]
                recolor_drawing(drawing, colors.Color(r, g, b))

            renderPDF.draw(drawing, c, x, y)
        elif kind == "textbox":
            # Cadre + texte avec word-wrap
            x, y = to_xy(it["x_mm"], it["y_mm"])
            w = it["w_mm"] * mm
            h = it["h_mm"] * mm

            # Marges internes optionnelles
            pad_mm = it.get("pad_mm", 2.0)
            px = x + pad_mm * mm
            py = y + pad_mm * mm
            pw = max(0, w - 2 * pad_mm * mm)
            ph = max(0, h - 2 * pad_mm * mm)

            # Style spécifique (optionnel par item)
            st = ParagraphStyle(
                name="box",
                parent=base_style,
                fontSize=it.get("fontSize", base_style.fontSize),
                leading=it.get("leading", None) or int(1.2 * it.get("fontSize", base_style.fontSize)),
                alignment=it.get("align", TA_LEFT),
            )


            story = [Paragraph(it["text"], st)]
            # keepInFrame: 'shrink' pour tout faire tenir, sinon 'truncate' ou 'error'
            kif = KeepInFrame(pw, ph, story, mode=it.get("overflow", "shrink"))
            frame = Frame(px, py, pw, ph, showBoundary=0)
            frame.addFromList([kif], c)

        else:
            # Inconnu -> ignore
            pass

    c.showPage()
    c.save()
    return buf.getvalue()



def main():
    doc = fitz.open(str(PDF_INPUT))

    for pno, page in enumerate(doc):
        anchors = anchors_for_page(doc, pno)
        print(f"Page {pno+1}: {len(anchors)} anchors")

        if not anchors:
            continue

        items: list[Dict] = []

        # Marqueurs et rectangles "entre" deux ancres successives (haut -> bas)
        BOTTOM_MARGIN_MM = 10.0  # par ex.

        for j, (x_mm, y_mm, q) in enumerate(anchors):
            # ... tes items "textbox" habituels ...
            if j + 1 < len(anchors):
                _, y_next, _ = anchors[j + 1]
            else:
                # Dernière ancre de la page : utiliser le bas de la page comme "ancre suivante"
                y_next = BOTTOM_MARGIN_MM

            top_y = y_mm
            bottom_y = max(0.0, y_next + GAP_ABOVE_NEXT_ANCHOR_MM)
            if bottom_y < top_y:
                h = top_y - bottom_y
                w = max(0.0, max(LEFT_MARGIN_MM, x_mm - RIGHT_MARGIN_MM) - LEFT_MARGIN_MM)
                if w > 0.1 and h > 0.1:
                    glyph = random.choice(["check", "cross"])

                    items.append({
                        "kind": "svg",
                        "path": Path(glyph+ ".svg") ,
                        "x_mm": x_mm - 4.0,
                        "y_mm": top_y - 5,  # centré verticalement
                        "w_mm": 3.0,
                        "h_mm": 3.0,
                        "color": (0.9, 0.3, 0.1)
                    })

                    items.append({
                        "kind": "textbox",
                        "x_mm": LEFT_MARGIN_MM,
                        "y_mm": bottom_y,
                        "w_mm": w,
                        "h_mm": h - 3,
                        "pad_mm": 0.0,
                        "text": phrase_aleatoire(),
                        "fontSize": 10,
                        "align": TA_RIGHT,
                        "overflow": "shrink",
                    })

        # Superpose l’overlay (même page size)
        ov = fitz.open("pdf", overlay_pdf((page.rect.width, page.rect.height), items))
        page.show_pdf_page(page.rect, ov, 0)
        ov.close()

    doc.save(str(PDF_OUTPUT))
    doc.close()


if __name__ == "__main__":
    main()
