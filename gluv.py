#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ajoute automatiquement, pour chaque question détectée par une ancre "Q<n>-anchor"
dans un PDF (produit avec la classe exam + tags invisibles), deux éléments :
  1) Un "V" ou "X" rouge, choisi aléatoirement, placé à 1 cm à gauche de l'ancre (sauf la dernière ancre d'une page)
  2) Un rectangle magenta allant horizontalement de 5 mm du bord gauche jusqu'à l'abscisse de l'ancre,
     et verticalement de l'ordonnée de l'ancre jusqu'à 5 mm au-dessus de l'ancre suivante.

Les positions et tailles sont en millimètres, converties en points PDF pour le dessin.
Le script parcourt TOUTES les pages et ne dessine rien si une page ne contient pas d'ancres.

Pré-requis:
  pip install reportlab pymupdf

Note:
  Le script suppose que le PDF source contient des chaînes de texte invisibles
  "Q<numero>-anchor" dans le flux PDF (une par question), extractibles par PyMuPDF.
"""

from pathlib import Path
import io
import random
from typing import List, Dict, Tuple

import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import magenta

# -----------------------
# Configuration utilisateur
# -----------------------

PDF_INPUT = Path("exam2/exam.pdf")  # chemin du PDF d'entrée
PDF_OUTPUT = Path("fapouille.pdf")         # chemin du PDF de sortie

# Police manuscrite optionnelle pour les marques "V"/"X" (fallback si absente)
FONT_HAND_PATH = Path("fonts/HomemadeApple-Regular.ttf")

# Couleur rouge personnalisée pour "V"/"X" (RGB 0..1)
PEN_COLOR = (0.85, 0.10, 0.10)

# Décalages en millimètres
LEFT_MARGIN_MM = 5.0        # marge gauche minimale de la page pour la boîte
LEFT_OF_ANCHOR_TEXT_MM = 10 # "V"/"X" placé 10 mm (1 cm) à gauche de l'ancre
GAP_ABOVE_NEXT_ANCHOR_MM = 5  # la boîte s'arrête 5 mm AVANT l'ancre suivante

# Graine aléatoire optionnelle (décommentez pour rendre reproductible)
# random.seed(42)


# -----------------------
# Utilitaires
# -----------------------

PT_TO_MM = 0.352778  # 1 pt ≈ 0.352778 mm

def page_anchors_mm(page: fitz.Page) -> List[Tuple[float, float, int]]:
    """
    Extrait les ancres "Q<n>-anchor" de la page, renvoie une liste triée par Y:
    [(x_mm, y_mm, question_id), ...]
    """
    anchors = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not text:
            continue
        t = text.strip()
        # On cherche exactement Q<nombre>-anchor
        if t.startswith("Q") and t.endswith("-anchor"):
            try:
                qid = int(t.split("-")[0][1:])
            except Exception:
                continue
            # centre du bloc texte -> coordonnées en mm
            cx_pt = (x0 + x1) / 2.0
            cy_pt = (y0 + y1) / 2.0
            anchors.append((cx_pt * PT_TO_MM, cy_pt * PT_TO_MM, qid))
    # trier par ordonnée (haut -> bas ; dans PDF y croît vers le bas)
    anchors.sort(key=lambda el: el[1])
    print(f"Page {page.number + 1}: found anchors {anchors}")
    return anchors


def make_overlay_pdf_for_page(
    page_size_pts: Tuple[float, float],
    draw_items: List[Dict]
) -> bytes:
    """
    Construit un PDF (une page) en mémoire avec ReportLab, contenant:
     - marques "V"/"X" rouges,
     - rectangles magenta.
    draw_items: liste de dicts:
       {"kind": "mark", "x_mm": float, "y_mm": float, "text": "V"|"X", "fontsize": int}
       {"kind": "rect", "x_mm": float, "y_mm": float, "w_mm": float, "h_mm": float}
    Retourne le bytes du PDF.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size_pts)

    # Police: tenter la manuscrite, sinon fallback Helvetica
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        pdfmetrics.registerFont(TTFont("HandFont", str(FONT_HAND_PATH)))
        has_hand = True
    except Exception:
        has_hand = False

    width_pts, height_pts = page_size_pts

    def mm_to_pts_xy(x_mm: float, y_mm: float) -> Tuple[float, float]:
        """Convertit (mm relatif top-left) en points ReportLab (origine bas-gauche)."""
        return x_mm * mm, height_pts - y_mm * mm

    # Dessin
    for it in draw_items:
        if it["kind"] == "mark":
            x_mm, y_mm = it["x_mm"], it["y_mm"]
            text = it["text"]
            fontsize = it.get("fontsize", 14)

            x_pts, y_pts = mm_to_pts_xy(x_mm, y_mm)

            # Couleur rouge perso
            r, g, b = PEN_COLOR
            c.setFillColorRGB(r, g, b)

            if has_hand:
                c.setFont("HandFont", fontsize)
            else:
                # Helvetica Bold pour être bien lisible
                c.setFont("Helvetica-Bold", fontsize)

            # centré verticalement sur y_mm, et centré horizontalement sur x_mm
            # (x_mm ici est la position cible exacte)
            c.drawCentredString(x_pts, y_pts, text)

        elif it["kind"] == "rect":
            x_mm, y_mm = it["x_mm"], it["y_mm"]
            w_mm, h_mm = it["w_mm"], it["h_mm"]
            x_pts, y_pts = mm_to_pts_xy(x_mm, y_mm)
            # y_pts est le haut du rectangle -> reportlab rect attend le coin bas-gauche
            # On convertit donc la hauteur
            c.setStrokeColor(magenta)
            c.setLineWidth(1)  # ~0.35 mm
            c.rect(x_pts, y_pts - h_mm * mm, w_mm * mm, h_mm * mm, fill=0)

    c.showPage()
    c.save()
    return buf.getvalue()


# -----------------------
# Pipeline principal
# -----------------------

def main():
    doc = fitz.open(str(PDF_INPUT))

    for page_index in range(len(doc)):
        page = doc[page_index]
        anchors = page_anchors_mm(page)
        if len(anchors) < 2:
            # S'il n'y a pas au moins 2 ancres sur la page, on ne peut pas faire
            # la boîte verticale jusqu'à "5 mm avant la suivante"
            # (et on ne place rien à côté de la dernière ancre)
            continue

        # Taille réelle de la page en points (pour générer un overlay identique)
        page_rect = page.rect
        page_size_pts = (page_rect.width, page_rect.height)

        # Construire les items à dessiner pour CETTE page
        items: List[Dict] = []

        # Pour chaque ancre SAUF la dernière, on dessine:
        # 1) un "V" ou "X" rouge à 10 mm à gauche
        # 2) un rectangle magenta de 5 mm (gauche page) -> x_anchor,
        #    verticalement de y_anchor -> y_next_anchor - 5 mm
        for i in range(len(anchors) - 1):
            x_mm, y_mm, _qid = anchors[i]
            x_next_mm, y_next_mm, _ = anchors[i + 1]

            # (1) Mark V/X
            mark_text = random.choice(["V", "X"])
            mark_x_mm = max(LEFT_MARGIN_MM, x_mm - LEFT_OF_ANCHOR_TEXT_MM)
            items.append({
                "kind": "mark",
                "x_mm": mark_x_mm,
                "y_mm": y_mm,
                "text": mark_text,
                "fontsize": 16
            })

            # (2) Rectangle magenta
            left_x = LEFT_MARGIN_MM
            right_x = max(LEFT_MARGIN_MM, x_mm)  # borne droite = abscisse de l'ancre
            top_y = y_mm
            bottom_y = max(top_y, y_next_mm - GAP_ABOVE_NEXT_ANCHOR_MM)

            w_mm = max(0.0, right_x - left_x)
            h_mm = max(0.0, bottom_y - top_y)

            if w_mm > 0.1 and h_mm > 0.1:
                items.append({
                    "kind": "rect",
                    "x_mm": left_x,
                    "y_mm": top_y,
                    "w_mm": w_mm,
                    "h_mm": h_mm
                })

        # Générer overlay en mémoire et l'appliquer
        overlay_bytes = make_overlay_pdf_for_page(page_size_pts, items)
        overlay_doc = fitz.open("pdf", overlay_bytes)

        # show_pdf_page: colle la page 0 de l'overlay sur la page cible
        page.show_pdf_page(page.rect, overlay_doc, 0)

        overlay_doc.close()

    # Sauvegarde du PDF final
    doc.save(str(PDF_OUTPUT))
    doc.close()
    print(f"✅ PDF annoté enregistré : {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
