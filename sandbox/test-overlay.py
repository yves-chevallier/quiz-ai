from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta
import fitz  # PyMuPDF

# --- chemins ---
PDF_INPUT = Path("pdfs/2025-10-16-14-46-25.pdf")
PDF_OUTPUT = Path("mon_quiz_corrige.pdf")
FONT_HAND_PATH = Path("fonts/HomemadeApple-Regular.ttf")

penColor = (0.63921569, 0.0627451, 0.0627451)

# --- coordonnées en mm ---
marks = [
    {"x": 190, "y": 22, "text": "V"},
    {"x": 190, "y": 66, "text": "X"},
]
feedbacks = [
    {"x": 120, "y": 62, "w": 70, "h": 22, "text": "Les pommes sont cuites"},
]
frames = [
    {"x": 120, "y": 100, "w": 70, "h": 22},  # cadre magenta
]
circles = [
    {"x": 15, "y": 105, "diameter": 5},
]

# --- créer un overlay PDF transparent ---
overlay_path = Path("overlay.pdf")
c = canvas.Canvas(str(overlay_path), pagesize=A4)

# enregistrer la police manuscrite
c.setFont("Helvetica", 12)  # fallback
try:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(TTFont("HandFont", str(FONT_HAND_PATH)))
    c.setFont("HandFont", 14)
except Exception as e:
    print("Police non chargée, fallback Helvetica :", e)

# convertir coordonnées : A4 = 210×297 mm
width, height = A4


def mm_to_pts(x, y):
    """Convertit mm (origine en haut-gauche) → pts (origine en bas-gauche)."""
    return x * mm, height - y * mm


# --- texte dans mark anchors ---
for m in marks:
    x, y = mm_to_pts(m["x"], m["y"])
    c.setFillColorRGB(*penColor)
    c.drawCentredString(x + (m.get("w", 10) / 2) * mm, y, m["text"])

# --- feedbacks ---
for f in feedbacks:
    x, y = mm_to_pts(f["x"], f["y"])
    c.setFont("HandFont", 10)
    c.setFillColorRGB(*penColor)
    c.drawString(x, y - f["h"] * mm / 2, f["text"])

# --- cadre magenta ---

c.setStrokeColor(magenta)
for fr in frames:
    x, y = mm_to_pts(fr["x"], fr["y"])
    c.rect(x, y - fr["h"] * mm, fr["w"] * mm, fr["h"] * mm, fill=0)

# --- cercle ---
for circ in circles:
    x, y = mm_to_pts(circ["x"], circ["y"])
    r = (circ["diameter"] / 2) * mm
    c.circle(x, y, r)

circles_in_a4_corners = [
    {"x": 10, "y": 10},
    {"x": 200, "y": 287},
    {"x": 10, "y": 287},
    {"x": 200, "y": 10},
]
for circ in circles_in_a4_corners:
    x, y = mm_to_pts(circ["x"], circ["y"])
    r = 5 * mm
    c.setFillColorRGB(0.5, 0.8, 0.1)
    c.circle(x, y, r)


c.showPage()
c.save()

# --- fusionner overlay sur la page 2 du PDF d'origine ---
doc = fitz.open(PDF_INPUT)
overlay = fitz.open(overlay_path)

# page indices are 0-based
page = doc[1]  # page 2
page.show_pdf_page(page.rect, overlay, 0)

doc.save(PDF_OUTPUT)
doc.close()
overlay.close()

print("✅ PDF annoté enregistré :", PDF_OUTPUT)
