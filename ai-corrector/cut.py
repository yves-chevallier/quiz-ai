"""Extract questions regions from an annotated PDF based on anchors.
Used to cut out question areas for AI correction.
Used for post-correction annotation in 'annotate.py'.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

# =========================
# === Constantes & I/O ====
# =========================
PDF_INPUT = Path("exam2/exam.pdf")
OVERLAP_MM = 3.0  # chevauchement (en mm) ajouté en haut/bas de chaque région
INCLUDE_BOTTOM_SEGMENT = (
    True  # Inclure le segment entre la dernière ancre et le bas de page
)
# Si vous préférez ignorer la zone au-dessus de la 1ʳᵉ ancre, laissez False (comme le script d'origine)

PT_TO_MM = 25.4 / 72.0
R_TAG = re.compile(r"^Q(\d+)-anchor$")


# =========================
# ====== Fonctions ========
# =========================
def anchors_for_page(doc: fitz.Document, pno: int) -> List[Tuple[float, float, int]]:
    """
    Renvoie la liste des ancres sur la page 'pno' sous forme de tuples:
      (x_mm_from_bottom, y_mm_from_bottom, qnum)
    en se basant sur les destinations nommées créées par hyperref.
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


def regions_from_anchors(
    page_h_mm: float, anchors_mm: List[Tuple[float, float, int]], overlap_mm: float
) -> List[Dict[str, float]]:
    """
    Construit des régions [y_start_mm, y_end_mm] entre ancres successives avec chevauchement.
    - y augmente du bas vers le haut (repère PDF).
    - Chaque région chevauche ses voisines de 'overlap_mm' au-dessus et au-dessous, bornée à [0, page_h_mm].
    - Optionnellement, on inclut le segment bas de page -> dernière ancre.
    """
    regions: List[Dict[str, float]] = []
    if not anchors_mm:
        return regions

    # Paires (courante -> suivante), du haut vers le bas
    for j in range(len(anchors_mm) - 1):
        y_top = anchors_mm[j][1]  # ancre du haut (y plus grand)
        y_bottom = anchors_mm[j + 1][1]  # ancre suivante (plus bas)

        y_start = max(0.0, y_bottom - overlap_mm)
        y_end = min(page_h_mm, y_top + overlap_mm)

        if y_end - y_start > 0.1:
            regions.append({"y_start_mm": y_start, "y_end_mm": y_end})

    if INCLUDE_BOTTOM_SEGMENT:
        # Segment entre la dernière ancre et le bas de page (y=0)
        y_last = anchors_mm[-1][1]
        y_start = max(0.0, 0.0)  # bas de page
        y_end = min(page_h_mm, y_last + overlap_mm)
        if y_end - y_start > 0.1:
            regions.append({"y_start_mm": y_start, "y_end_mm": y_end})

    return regions


# =========================
# ========= Main ==========
# =========================
def main() -> None:
    if not PDF_INPUT.exists():
        raise FileNotFoundError(f"Fichier introuvable: {PDF_INPUT}")

    doc = fitz.open(str(PDF_INPUT))
    try:
        pages_out: List[Dict] = []
        for pno, page in enumerate(doc):
            page_h_mm = page.rect.height * PT_TO_MM
            anchors = anchors_for_page(doc, pno)
            regions = regions_from_anchors(page_h_mm, anchors, OVERLAP_MM)

            pages_out.append(
                {
                    "page_index": pno,
                    "page_height_mm": page_h_mm,
                    "anchors_mm": [
                        {"x_mm": x, "y_mm": y, "qnum": q} for (x, y, q) in anchors
                    ],
                    "regions_mm": regions,
                }
            )

        # Sortie JSON unique sur stdout
        print(json.dumps({"pages": pages_out}, ensure_ascii=False, indent=2))
    finally:
        doc.close()


if __name__ == "__main__":
    main()
