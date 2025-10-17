# -*- coding: utf-8 -*-
"""
(old pipeline, to be replaced with the new one)
Pipeline d'analyse & notation de copies scannées (quiz) avec cache des étapes.
Étapes:
  1) Charger la solution YAML (exam.yml)
  2) Charger le PDF source (ex: pdfs/2025-10-16-14-46-25.pdf)
  3) Extraire les pages en PNG (en mémoire)
  4) identify_content: envoyer TOUTES les images d'un coup au modèle (cache: out/analysis.json)
  5) grading: utiliser l'analyse + la solution pour calculer la note (cache: out/grades.json)
  6) Résumé CSV (out/summary.csv)

Dépendances:
  pip install typer[all] pyyaml pdf2image Pillow openai pymupdf

Notes:
  - Nécessite poppler installé pour pdf2image (Linux: apt-get install poppler-utils, macOS: brew install poppler).
  - OPENAI_API_KEY doit être présent dans l'environnement.
"""

from __future__ import annotations

import base64
import io
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import yaml
from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI

app = typer.Typer(
    help="Pipeline d'analyse et de grading de copies PDF",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()

# ================== CONFIG PAR DÉFAUT ==================
OUT_DIR = Path("out")
DEFAULT_SOLUTION_YAML = Path("exam.yml")
DEFAULT_PDF = Path("pdfs/2025-10-16-14-46-25.pdf")
DPI_IMAGES = 220
MODEL_VISION = "gpt-5"  # Modèle vision (ajuste si besoin)
# =======================================================


# ---------- OUTILS FICHIERS ----------
def ensure_out_dir(path: Path = OUT_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pil_image_to_jpeg_data_url(
    img: Image.Image, max_side: int = 1600, quality: int = 65
) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------- OUTILS IMAGES ----------
def pil_image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def pdf_to_pil_images_in_memory(
    pdf_path: Path, dpi: int = DPI_IMAGES
) -> List[Image.Image]:
    return convert_from_path(str(pdf_path), dpi=dpi)


# ---------- SOLUTIONS ----------
def load_solutions_yaml(path: Path) -> Dict[str, Any]:
    """
    Format attendu (souple, mais recommandé):
    title: "Quiz ..."
    questions:
      - id: 1
        type: single|multi|fillin|open
        points: 1
        answers: ["A"]           # pour single/multi; ou string/regex pour fillin; pour open: guidelines
      - id: 2
        points: 2
        answers: ["B","D"]
    total_points: 10            # facultatif (sinon somme des points)
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML solution invalide: racine non-objet.")
    q = data.get("questions", [])
    if not isinstance(q, list) or not q:
        raise ValueError("YAML solution invalide: 'questions' manquant ou vide.")
    # calcul total_points si absent
    if "total_points" not in data:
        total = 0.0
        for item in q:
            total += float(item.get("points", 1.0))
        data["total_points"] = total
    return data


def solution_points_map(solution: Dict[str, Any]) -> Dict[int, float]:
    mapping = {}
    for item in solution.get("questions", []):
        qid = int(item.get("id"))
        pts = float(item.get("points", 1.0))
        mapping[qid] = pts
    return mapping


# ---------- OPENAI HELPERS ----------
import httpx


def openai_client() -> OpenAI:
    # Client HTTPx personnalisable :
    http_client = httpx.Client(
        timeout=500.0,  # allonge franchement le timeout
        # http2=False,            # désactive HTTP/2 (certains proxies le cassent sur POST)
        verify=True,  # laisse True (si besoin, tu peux pointer vers un bundle certifi)
    )
    return OpenAI()
    #     timeout=500.0,
    #     max_retries=5,
    #     #http_client=http_client,
    # )


# ---------- IDENTIFY CONTENT ----------
def identify_content(
    images_data_urls: List[str], client: OpenAI, model: str
) -> Dict[str, Any]:
    """
    Envoie toutes les pages à la fois. Retourne un JSON structuré strict (voir prompt).
    """
    imgs = [
        {"type": "image_url", "image_url": {"url": url}} for url in images_data_urls
    ]
    messages = [
        {
            "role": "system",
            "content": open("identify.md", "r", encoding="utf-8").read(),
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Analyze this quiz (all pages):"}]
            + imgs,
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


# ---------- GRADING ----------
def run_grading(
    analysis: Dict[str, Any], solution: Dict[str, Any], client: OpenAI, model: str
) -> Dict[str, Any]:
    """
    Produit un JSON strict de grading. Le modèle calcule correctness + granted_points (0..1 par question).
    On gardera ensuite la conversion en points réels côté Python pour le CSV.
    """
    SYSTEM_PROMPT = """
You are an impartial and precise grader.

Input #1 is the student's quiz analysis JSON (multi-page). Input #2 is the official solution JSON.
Your job: produce a SINGLE strict JSON object with per-question evaluation.

Output schema:
{
  "name": "student name",
  "date": "YYYY-MM-DD or empty",
  "title": "exam title from solution if present or empty",
  "total_questions": int,
  "questions": [
    {
      "id": int,
      "answered": bool,
      "correct": bool,
      "granted_ratio": float,   // 0..1 fraction of points for this question
      "remark": "string",
      "confidence": float       // 0..1 confidence in evaluation
    }
  ]
}

Rules:
- Maintain a neutral, concise remark.
- For multi-part questions (multi-select), distribute partial credit proportionally via granted_ratio in [0,1].
- If the student left it blank, answered=false, correct=false, granted_ratio=0.0.
- Output ONLY valid JSON.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Student analysis JSON:"},
                {"type": "text", "text": json.dumps(analysis, ensure_ascii=False)},
                {"type": "text", "text": "Official solution JSON:"},
                {"type": "text", "text": json.dumps(solution, ensure_ascii=False)},
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


# ---------- RÉSUMÉ CSV ----------
def compute_points_from_grades(
    grades: Dict[str, Any], solution: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Convertit granted_ratio (0..1) -> points réels selon la solution.
    """
    pts_map = solution_points_map(solution)
    total_points = float(solution.get("total_points", sum(pts_map.values())))
    got = 0.0
    for q in grades.get("questions", []):
        qid = int(q.get("id"))
        ratio = float(q.get("granted_ratio", 0.0))
        got += ratio * float(pts_map.get(qid, 1.0))
    return got, total_points


def write_summary_csv(
    grades_path: Path, solution: Dict[str, Any], out_csv: Path
) -> None:
    grades = load_json(grades_path)
    got, tot = compute_points_from_grades(grades, solution)
    note = round(((got / tot) * 5.0 + 1.0), 1) if tot else None

    # minimal CSV (une ligne pour ce PDF)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "points_obtenus", "points_total", "note"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": grades.get("_source_pdf", ""),
                "points_obtenus": got,
                "points_total": tot,
                "note": note,
            }
        )


# ---------- PIPELINE ----------
@app.command("run")
def run_pipeline(
    solutions_yaml: Path = typer.Option(
        DEFAULT_SOLUTION_YAML,
        "--solutions",
        "-s",
        help="Chemin du fichier solutions YAML",
    ),
    pdf_path: Path = typer.Option(
        DEFAULT_PDF, "--pdf", "-p", help="Chemin du PDF à analyser"
    ),
    out_dir: Path = typer.Option(
        OUT_DIR, "--out", help="Dossier de sortie pour les JSON/CSV"
    ),
    model: str = typer.Option(MODEL_VISION, "--model", help="Modèle vision"),
    dpi: int = typer.Option(
        DPI_IMAGES, "--dpi", help="DPI pour l'extraction des pages"
    ),
    force: bool = typer.Option(
        False, "--force", help="Recalculer même si les caches existent"
    ),
):
    """
    Exécute la pipeline complète avec cache des étapes.
    """
    console.print(
        Panel.fit("[bold cyan]Pipeline Quiz → Analyse → Grading → Résumé[/bold cyan]")
    )

    # 0) Préparatifs
    ensure_out_dir(out_dir)
    analysis_json_path = out_dir / "analysis.json"
    grades_json_path = out_dir / "grades.json"
    summary_csv_path = out_dir / "summary.csv"

    # 1) Charger la solution YAML
    console.rule("[bold]Étape 1[/bold] • Chargement de la solution YAML")
    if not solutions_yaml.exists():
        typer.secho(
            f"Fichier solution introuvable: {solutions_yaml}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    solution = load_solutions_yaml(solutions_yaml)
    title = solution.get("title", "") or "—"
    table = Table(title="Solution chargée")
    table.add_column("Titre", style="magenta")
    table.add_column("Questions", style="green")
    table.add_column("Total points", style="yellow")
    table.add_row(
        str(title),
        str(len(solution.get("questions", []))),
        str(solution.get("total_points")),
    )
    console.print(table)

    # 2) Charger le PDF et extraire les pages en mémoire
    console.rule("[bold]Étape 2[/bold] • Extraction des pages du PDF (en mémoire)")
    if not pdf_path.exists():
        typer.secho(f"PDF introuvable: {pdf_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    images = pdf_to_pil_images_in_memory(pdf_path, dpi=dpi)
    typer.secho(f"Pages extraites: {len(images)} (DPI={dpi})", fg=typer.colors.GREEN)

    # --- sauvegarde les images extraites pour vérification ---
    images_out_dir = out_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images, start=1):
        out_path = images_out_dir / f"{pdf_path.stem}_page{i}.jpg"
        img.save(out_path, "JPEG", quality=90)
    print(f"Images sauvegardées dans {images_out_dir} ({len(images)} pages)")

    # Data URLs pour le modèle (pas d'écriture PNG)
    # images_data_urls = [pil_image_to_data_url(img) for img in images]
    images_data_urls = [
        pil_image_to_jpeg_data_url(img, max_side=1800, quality=70) for img in images
    ]

    approx_mb = (
        sum(len(u.split(",", 1)[1]) for u in images_data_urls) * 3 / 4 / (1024 * 1024)
    )
    print(f"Payload images ~ {approx_mb:.1f} MB")

    client = openai_client()

    # 3) identify_content (cache: out/analysis.json)
    console.rule(
        "[bold]Étape 3[/bold] • Analyse des images par le modèle (identify_content)"
    )
    if analysis_json_path.exists() and not force:
        typer.secho(
            f"Analyse déjà présente → {analysis_json_path}", fg=typer.colors.YELLOW
        )
        analysis = load_json(analysis_json_path)
    else:
        typer.secho(
            "Appel modèle (toutes les pages en une fois)…", fg=typer.colors.CYAN
        )
        analysis = identify_content(images_data_urls, client, model=model)
        # Ajout meta légère
        analysis["_source_pdf"] = str(pdf_path)
        save_json(analysis_json_path, analysis)
        typer.secho(
            f"Analyse sauvegardée → {analysis_json_path}", fg=typer.colors.GREEN
        )

    # 4) grading (cache: out/grades.json)
    console.rule("[bold]Étape 4[/bold] • Grading à partir de l'analyse + solution")
    if grades_json_path.exists() and not force:
        typer.secho(
            f"Grading déjà présent → {grades_json_path}", fg=typer.colors.YELLOW
        )
        grades = load_json(grades_json_path)
    else:
        typer.secho("Appel modèle (grading)…", fg=typer.colors.CYAN)
        grades = run_grading(analysis, solution, client, model=model)
        # Ajout meta + normalisation titre
        grades["_source_pdf"] = str(pdf_path)
        if not grades.get("title"):
            grades["title"] = solution.get("title", "")
        save_json(grades_json_path, grades)
        typer.secho(f"Grading sauvegardé → {grades_json_path}", fg=typer.colors.GREEN)

    # 5) Résumé CSV
    console.rule("[bold]Étape 5[/bold] • Génération du résumé CSV")
    write_summary_csv(grades_json_path, solution, summary_csv_path)
    typer.secho(f"Résumé écrit → {summary_csv_path}", fg=typer.colors.GREEN)

    console.print(Panel.fit("[bold green]Terminé ✔[/bold green]"))


# ---------- ENTRY ----------
if __name__ == "__main__":
    app()
