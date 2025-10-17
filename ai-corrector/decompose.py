"""
Using the information from cut.py, cut out specified regions from PDF pages rendered as images
to feed ai-corrector for further processing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import typer
from PIL import Image  # pillow
import fitz  # PyMuPDF


app = typer.Typer(add_completion=False)


# -------------------------
# Utilitaires découpe
# -------------------------
def mm_region_to_pixel_box(
    img_w: int,
    img_h: int,
    page_height_mm: float,
    y_start_mm: float,
    y_end_mm: float,
) -> Tuple[int, int, int, int]:
    """
    Convertit une région [y_start_mm, y_end_mm] en boîte PIL (left, top, right, bottom).
    Repère JSON: y=0 en bas ; Repère image: y=0 en haut.
    """
    if page_height_mm <= 0:
        raise ValueError("page_height_mm doit être > 0")

    scale = img_h / page_height_mm  # px par mm (vertical)

    top_px = int(round(img_h - (y_end_mm * scale)))
    bottom_px = int(round(img_h - (y_start_mm * scale)))

    top_px = max(0, min(img_h, top_px))
    bottom_px = max(0, min(img_h, bottom_px))

    if bottom_px <= top_px:
        bottom_px = min(img_h, top_px + 1)

    left_px = 0
    right_px = img_w
    return (left_px, top_px, right_px, bottom_px)


def crop_page_image(
    image_path: Path,
    regions_mm: List[Dict[str, float]],
    page_height_mm: float,
    output_dir: Path,
    base_output_stem: str,
    quality: int,
    out_ext: str,
) -> List[Path]:
    """
    Ouvre l'image de page, découpe les sous-régions, écrit les fichiers.
    Sort une liste des fichiers écrits.
    """
    written: List[Path] = []
    if not image_path.exists():
        typer.secho(f"[WARN] Image absente: {image_path}", fg=typer.colors.YELLOW)
        return written

    with Image.open(image_path) as im:
        # Pour JPG, forcer RGB (éviter mode P/LA/CMYK)
        if out_ext.lower() in {"jpg", "jpeg"} and im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size

        for i, reg in enumerate(regions_mm, start=1):
            y_start = float(reg["y_start_mm"])
            y_end = float(reg["y_end_mm"])
            box = mm_region_to_pixel_box(w, h, page_height_mm, y_start, y_end)
            cropped = im.crop(box)

            out_name = f"{base_output_stem}_{i}.{out_ext}"
            out_path = output_dir / out_name

            save_kwargs = {}
            if out_ext.lower() in {"jpg", "jpeg"}:
                save_kwargs.update(dict(format="JPEG", quality=quality, optimize=True))
            elif out_ext.lower() == "png":
                save_kwargs.update(dict(format="PNG", optimize=True))
            elif out_ext.lower() == "webp":
                save_kwargs.update(dict(format="WEBP", quality=quality, method=6))

            cropped.save(out_path, **save_kwargs)
            written.append(out_path)

    return written


# -------------------------
# Rendu PDF -> images
# -------------------------
def render_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    img_format: str,
) -> List[Tuple[int, Path]]:
    """
    Rend chaque page du PDF en image et l'écrit.
    Retourne [(page_index_zero_based, image_path), ...]
    """
    img_format = img_format.lower()
    valid = {"jpg", "jpeg", "png", "webp"}
    if img_format not in valid:
        raise typer.BadParameter(f"--format doit être dans {sorted(valid)}")

    out: List[Tuple[int, Path]] = []
    stem = pdf_path.stem

    with fitz.open(pdf_path) as doc:
        # Matrix pour DPI : 72dpi * scale = dpi
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        for pno in range(len(doc)):
            page = doc[pno]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            # Nom de base type: "{stem}_page{1}.{ext}"
            out_path = out_dir / f"{stem}_page{pno+1}.{img_format}"
            # Sauvegarde directe depuis le pixmap en PNG si non-JPG ?
            # On convertit via Pillow pour homogénéiser qualité/format.
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            save_kwargs = {}
            if img_format in {"jpg", "jpeg"}:
                save_kwargs.update(dict(format="JPEG", quality=95, optimize=True))
            elif img_format == "png":
                save_kwargs.update(dict(format="PNG", optimize=True))
            elif img_format == "webp":
                save_kwargs.update(dict(format="WEBP", quality=95, method=6))
            img.save(out_path, **save_kwargs)

            out.append((pno, out_path))

    return out


# -------------------------
# CLI
# -------------------------
@app.command()
def cut(
    pdf: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Chemin du PDF à traiter."
    ),
    json_path: Optional[Path] = typer.Option(
        None,
        "--json",
        "-j",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Fichier JSON d'entrée (sinon lecture sur stdin).",
    ),
    dpi: int = typer.Option(
        150, "--dpi", "-d", min=30, max=1200, help="DPI pour le rendu des pages PDF."
    ),
    img_format: str = typer.Option(
        "jpg", "--format", "-f", help="Format image de sortie des pages (jpg/png/webp)."
    ),
    output_dir: Path = typer.Option(
        Path("."), "--output-dir", "-o", help="Répertoire de sortie."
    ),
    quality: int = typer.Option(
        95, "--quality", min=1, max=100, help="Qualité JPEG/WEBP pour les découpes."
    ),
    skip_empty: bool = typer.Option(
        False, "--skip-empty", help="Ignorer les pages sans regions_mm."
    ),
):
    """
    Convertit le PDF en images (une par page), puis découpe chaque image
    selon 'regions_mm' du JSON. Écrit:
      - {pdf_stem}_page{N}.{ext}
      - {pdf_stem}_page{N}_{i}.{ext} pour chaque région.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) PDF -> images
    typer.secho(
        f"[INFO] Rendu PDF -> images à {dpi} DPI en {img_format}…", fg=typer.colors.CYAN
    )
    page_images = render_pdf_to_images(pdf, output_dir, dpi, img_format)
    typer.secho(f"[OK] {len(page_images)} page(s) rendue(s).", fg=typer.colors.GREEN)

    # 2) Charger JSON
    if json_path:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        data = json.load(sys.stdin)

    pages: List[Dict[str, Any]] = data.get("pages", [])
    pages_by_index = {p.get("page_index", 0): p for p in pages}

    total_written = 0

    # 3) Découpes
    for pno, img_path in page_images:
        entry = pages_by_index.get(pno)
        if not entry:
            if not skip_empty:
                typer.secho(
                    f"[WARN] Aucune entrée JSON pour page_index={pno}",
                    fg=typer.colors.YELLOW,
                )
            continue

        regions_mm = entry.get("regions_mm", []) or []
        if skip_empty and not regions_mm:
            continue

        page_height_mm = float(entry.get("page_height_mm", 0.0))
        if page_height_mm <= 0:
            typer.secho(
                f"[WARN] page_height_mm invalide pour page_index={pno}",
                fg=typer.colors.YELLOW,
            )
            continue

        base_output_stem = img_path.stem  # ex: "{stem}_page1"
        written = crop_page_image(
            image_path=img_path,
            regions_mm=regions_mm,
            page_height_mm=page_height_mm,
            output_dir=output_dir,
            base_output_stem=base_output_stem,
            quality=quality,
            out_ext=img_path.suffix.lstrip("."),
        )
        total_written += len(written)

    typer.secho(
        f"[DONE] Découpes terminées. Fichiers écrits: {total_written}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
