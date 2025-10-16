# script_pipeline.py
import json
import base64
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# =============== CONFIG ===============
DPI_IMAGES = 220
PDF_DIR = Path("pdfs")
OUT_DIR = Path("out")
IMG_DIR = Path("tmp_images")
SOLUTIONS_JSON = OUT_DIR / "solutions.json"
ARTIFACTS_JSON = OUT_DIR / "artifacts.json"
MODEL_VISION = "gpt-5"  # modèle vision
FONT_HAND_PATH = Path("fonts/HomemadeApple-Regular.ttf")  # police manuscrite
# =====================================

client = OpenAI()

def _has_hand_font() -> bool:
    return FONT_HAND_PATH.exists()

def _place_note_label(page, note_anchor, label):
    """
    note_anchor: dict {x, y, w, h}
    Écrit 'label' dans ce rectangle, centrée, en privilégiant la fonte manuscrite Homemade Apple.
    """
    page_rect = page.mediabox
    X = float(note_anchor.get("x", 0))
    Y = float(note_anchor.get("y", 0))
    W = float(note_anchor.get("w", 0))
    H = float(note_anchor.get("h", 0))
    X = max(0, min(X, float(page_rect.width) - 1))
    Y = max(0, min(Y, float(page_rect.height) - 1))
    W = max(1, min(W, float(page_rect.width) - X))
    H = max(1, min(H, float(page_rect.height) - Y))
    rect = fitz.Rect(X, Y, X + W, Y + H)

    font_kwargs = {}
    if _has_hand_font():
        font_kwargs["fontfile"] = str(FONT_HAND_PATH)

    # essaie de caser le texte, en réduisant la taille si besoin
    for fs in (22, 20, 18, 16, 14, 12, 11, 10):
        remaining = page.insert_textbox(
            rect,
            label,
            fontsize=fs,
            color=(1, 0, 0),
            align=1,   # centré
            **font_kwargs
        )
        if remaining == 0:
            return True
    # fallback coin haut-gauche du rectangle
    page.insert_text((X, Y), label, fontsize=10, color=(1,0,0), **font_kwargs)
    return False

def _g_points_got(g):
    # item de grade – supporte: points | points_obtenus | got
    return g.get("points", g.get("points_obtenus", g.get("got", 0.0)))

def _g_points_total(g):
    # item de grade – supporte: points_possible | points_total | total
    return g.get("points_possible", g.get("points_total", g.get("total", 0.0)))

def _g_comment(g):
    # item de grade – supporte: comment | remarque
    return g.get("comment", g.get("remarque", ""))

def _grades_totals(grades_dict):
    """
    top-level – supporte:
      - local: {got, total}
      - via modèle (ancienne version): {points_obtenus, total_points}
      - via modèle (nouvelle consigne): {total_got, total_points}
    """
    if "got" in grades_dict and "total" in grades_dict:
        return grades_dict["got"], grades_dict["total"]
    if "points_obtenus" in grades_dict and "total_points" in grades_dict:
        return grades_dict["points_obtenus"], grades_dict["total_points"]
    if "total_got" in grades_dict and "total_points" in grades_dict:
        return grades_dict["total_got"], grades_dict["total_points"]
    return 0.0, 0.0

def _read_json_or_passthrough(x):
    # accepte dict/list OU chemin -> retourne l'objet JSON
    if isinstance(x, (dict, list)):
        return x
    return json.loads(Path(x).read_text(encoding="utf-8"))

def _norm_str(x):
    # Convertit tout en str sûr, strip+lower
    if x is None:
        return ""
    return str(x).strip().lower()


def _norm_list_str(v):
    # Accepte liste / scalaire / None
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [_norm_str(x) for x in v if _norm_str(x) != ""]
    return [_norm_str(v)]  # si le modèle renvoie un scalaire plutôt qu'une liste


# ---------- Utils ----------
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def img_to_data_url(path_png: Path) -> str:
    b = path_png.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def save_json(path: Path, data: Any):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- 1) Extraction corrigé LaTeX ----------
def extract_solutions_from_tex(path_tex: Path, out_json: Path, force: bool = False):
    if out_json.exists() and not force:
        print(f"[1] solutions.json existe déjà → {out_json}")
        return load_json(out_json)

    tex = path_tex.read_text(encoding="utf-8")
    prompt = """
Tu es un assistant chargé de transformer la source d'un quiz LaTeX utilisant le package exam en un JSON structuré.
Exige le format suivant EXACTEMENT:
{
  "quiz_title": string,
  "questions": [
    {
      "number": int,
      "text": string,
      "type": "mcq" | "checkbox" | "fillin" | "open",
      "choices": [string],             // si mcq/checkbox
      "answers": [string],             // réponses correctes: CorrectChoice, fillin/fillna, ou solution
      "points": number                 // si absent dans le LaTeX: 1 par question
    }
  ],
  "total_points": number
}
Si un barème explicite n'est pas présent, déduis les points: 1 par question. Renvoie UNIQUEMENT le JSON.
    """

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Tu es un parseur de quiz LaTeX et tu renvoies un JSON strict.",
            },
            {"role": "user", "content": prompt + "\n\n" + tex},
        ],
        # temperature=0
    )
    data = json.loads(resp.choices[0].message.content)

    # filet de sécurité sur total_points
    if "total_points" not in data:
        total = 0.0
        for q in data.get("questions", []):
            pts = q.get("points")
            if pts is None:
                # déduire: 1 si rien
                pts = 1.0
            total += float(pts)
        data["total_points"] = total

    save_json(out_json, data)
    print(f"[1] solutions.json écrit → {out_json}")
    return data


# ---------- 2) PDF -> images + artifacts ----------
def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = DPI_IMAGES) -> List[str]:
    images = convert_from_path(str(pdf_path), dpi=dpi)
    base = pdf_path.stem
    out_paths = []
    for i, img in enumerate(images, start=1):
        out_path = out_dir / f"{base}_p{i}.png"
        img.save(out_path, "PNG")
        out_paths.append(str(out_path))
    return out_paths


def build_artifacts(pdfs: List[Path], force: bool = False) -> Dict[str, List[str]]:
    if ARTIFACTS_JSON.exists() and not force:
        print(f"[2] artifacts.json existe déjà → {ARTIFACTS_JSON}")
        return load_json(ARTIFACTS_JSON)

    artifacts = {}
    for pdf in pdfs:
        print(f"[2] PDF→PNG: {pdf.name}")
        paths = pdf_to_images(pdf, IMG_DIR, dpi=DPI_IMAGES)
        artifacts[str(pdf)] = paths

    save_json(ARTIFACTS_JSON, artifacts)
    print(f"[2] artifacts.json écrit → {ARTIFACTS_JSON}")
    return artifacts


# ---------- 3) Vision sur images (par PDF) ----------
def analyze_images_for_pdf(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Retourne une liste de pages JSON (une entrée par page).
    """
    pages_data = []
    base_prompt = """
        Tu es un correcteur humain impartial et précis.
        Pour cette page d'un quiz manuscrit, renvoie STRICTEMENT un JSON décrivant :
        - page_index (1-based),
        - éventuellement student_name/quiz_title si visibles sur cette page
        - note_anchor (x,y) sur la première page, dans une zone de 2x2 cm vide, plutôt en haut et à droite
        - pour chaque question:
            - numéro de question number
            - kind (single/multi/fillin/open)
            - selected_options[]
            - handwriting (note manuscrite de l'étudiant)
            - remarque de l'analyse (rature, barré recoché...)
            - indice de confidence float entre 0..1
            - mark_anchor (x,y,w,h), espace de 1x1 cm blanc dans marge de gauche préférablement sinon à droite
            - feedback_box (x,y,w,h), espace maximum blanc disponible dans la région de la question sans confusion pour autre question
            - Pour les question à choix multiple, détecte les options cochées en faisant attention aux ratures, corrections, consignes les dans
            - Tient compte toujours des annotations manuscrites (ex: barré mais recoché), texte explicatif, dessins, etc.
            - Explique les dessins s'il y a lieu (ex: schéma, graphique), dans handwriting.
        N'invente rien: si incertain/absent, laisse vide.
    """

    for i, path in enumerate(image_paths, start=1):
        data_url = img_to_data_url(Path(path))
        resp = client.chat.completions.create(
            model=MODEL_VISION,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Tu analyses visuellement une copie d'examen et tu renvoies un JSON strict.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": base_prompt + f" Indique page_index={i}.",
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            # temperature=0
        )
        page_json = json.loads(resp.choices[0].message.content)
        page_json.setdefault("page_index", i)
        page_json.setdefault("questions", [])
        pages_data.append(page_json)
    return pages_data


def run_analysis_all_pdfs(
    artifacts: Dict[str, List[str]], force: bool = False
) -> Dict[str, str]:
    """
    Pour chaque PDF => génère out/{basename}.student_data.json
    Retourne un mapping pdf -> chemin json
    """
    mapping = {}
    for pdf, imgs in artifacts.items():
        out_json = OUT_DIR / (Path(pdf).stem + ".student_data.json")
        if out_json.exists() and not force:
            print(f"[3] student_data existe déjà → {out_json}")
            mapping[pdf] = str(out_json)
            continue

        print(f"[3] Analyse vision: {Path(pdf).name}")
        pages = analyze_images_for_pdf(imgs)
        save_json(out_json, pages)
        mapping[pdf] = str(out_json)
        print(f"[3] Écrit → {out_json}")
    return mapping


def grade_student_via_model(solutions_in, student_data_in, model="gpt-4o"):
    """
    Accepte soit des chemins, soit des objets Python (dict/list).
    Retour JSON avec:
      grades: [{number, total, got, status, comment}]
      total_points, total_got, score
    """
    sol_obj = _read_json_or_passthrough(solutions_in)
    stu_obj = _read_json_or_passthrough(student_data_in)

    prompt = """
    Tu es un correcteur expérimenté chargé d'évaluer un étudiant.
    On te fournit deux JSON :
      1. solutions.json : les questions, types, réponses correctes et points.
      2. student_data.json : les réponses de l'étudiant (cases cochées, manuscrit, etc.).

    Règles :
    - Bonne réponse → points complets.
    - Partielle → points proportionnels.
    - Fausse/absente → 0.
    - Ajoute une remarque courte par question.

    Réponds UNIQUEMENT avec :
    {
      "grades": [
        {"number": <int>, "total": <float>, "got": <float>, "status": "correct"|"partial"|"wrong"|"absent", "comment": "<str>"}
      ],
      "total_points": <float>,
      "total_got": <float>,
      "score": <float>,  // (total_got / total_points * 5 + 1), arrondi 0,1
      "comment": "<str>",
      "need_human_review": <true|false>
    }
    """

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Tu es un correcteur humain impartial et précis."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "solutions.json = ```json\n" + json.dumps(sol_obj, ensure_ascii=False, indent=2) + "\n```"},
                {"type": "text", "text": "student_data.json = ```json\n" + json.dumps(stu_obj, ensure_ascii=False, indent=2) + "\n```"}
            ]}
        ],
        temperature=0
    )
    data = json.loads(resp.choices[0].message.content)

    # Garde-fous (si le modèle n'a pas calculé)
    if "grades" not in data:
        data["grades"] = []
    if "total_points" not in data:
        data["total_points"] = round(sum(float(it.get("total", it.get("points_total", it.get("points_possible", 0)))) for it in data["grades"]), 3)
    if "total_got" not in data:
        data["total_got"] = round(sum(float(it.get("got", it.get("points_obtenus", it.get("points", 0)))) for it in data["grades"]), 3)
    if "score" not in data:
        tot = data["total_points"] or 0.0
        got = data["total_got"] or 0.0
        data["score"] = round(((got / tot) * 5 + 1), 1) if tot else 1.0

    # Compat top-level avec ton pipeline existant (note/got/total)
    data.setdefault("note", data["score"])
    data.setdefault("got", data["total_got"])
    data.setdefault("total", data["total_points"])

    return data



def run_grading_all(
    solutions: Dict[str, Any], student_json_map: Dict[str, str], force: bool = False
) -> Dict[str, str]:
    """
    Pour chaque PDF => génère out/{basename}.grades.json
    Retourne mapping pdf -> grades_json
    """
    out_map = {}
    for pdf, stu_json in student_json_map.items():
        out_json = OUT_DIR / (Path(pdf).stem + ".grades.json")
        if out_json.exists() and not force:
            print(f"[4] grades existe déjà → {out_json}")
            out_map[pdf] = str(out_json)
            continue

        pages_data = load_json(Path(stu_json))

        grades = grade_student_via_model(solutions, pages_data)
        save_json(out_json, grades)
        out_map[pdf] = str(out_json)
        print(f"[4] Écrit → {out_json}")
    return out_map


# ---------- 5) Annotation PDF ----------
def annotate_pdf(pdf_path: Path, pages_data: List[Dict[str, Any]], grades: Dict[str, Any], out_path: Path):
    doc = fitz.open(str(pdf_path))
    grade_by_num = { str(g["number"]): g for g in grades["grades"] }

    # Prépare l’usage de la police manuscrite (si présente)
    font_kwargs_text = {}
    if _has_hand_font():
        font_kwargs_text["fontfile"] = str(FONT_HAND_PATH)

    for idx, page in enumerate(doc, start=1):
        page_data = next((p for p in pages_data if p.get("page_index")==idx), None)
        y = 50
        if page_data:
            # ---- NOTE FINALE ----
            if idx == 1:
                note_anchor = page_data.get("note_anchor")
                pts_got_all, pts_tot_all = _grades_totals(grades)
                note_val = grades.get("note")
                if note_val is None and pts_tot_all:
                    note_val = round(((pts_got_all / pts_tot_all) * 5 + 1), 1)
                label = f"Note finale : {note_val}/6" if note_val is not None else "Note finale : (indisponible)"

                if isinstance(note_anchor, dict) and all(k in note_anchor for k in ("x","y","w","h")):
                    _place_note_label(page, note_anchor, label)
                else:
                    page.insert_text((420, 100), label, fontsize=18, color=(1,0,0), **font_kwargs_text)

            # ---- ANNOTATIONS PAR QUESTION ----
            for q in page_data.get("questions", []):
                num = str(q.get("number"))
                g = grade_by_num.get(num)
                if not g:
                    continue

                pts_got = _g_points_got(g)       # au lieu de g['points'] ou g['points_obtenus'] ou g['got']
                pts_tot = _g_points_total(g)     # idem
                comment = _g_comment(g)          # idem

                # Pour la note affichée :
                pts_got_all, pts_tot_all = _grades_totals(grades)
                note_val = grades.get("note", grades.get("score"))
                if note_val is None and pts_tot_all:
                    note_val = round(((pts_got_all / pts_tot_all) * 5 + 1), 1)

                # Attention : Homemade Apple n’a pas toujours tous les glyphes (✓/✗/≈).
                # On garde les symboles; si glyph manquant, PyMuPDF fera un fallback. Autrement, on peut remplacer par "OK" / "X" / "~".
                sym = "✓" if pts_tot and pts_got == pts_tot else ("≈" if pts_got and pts_got > 0 else "✗")
                text = f"Q{num} {sym}  {pts_got}/{pts_tot}  – {comment}"

                mk = q.get("mark_anchor") or {}
                if isinstance(mk, dict) and "x" in mk and "y" in mk:
                    page.insert_text((float(mk["x"]), float(mk["y"])), text, fontsize=12, color=(1,0,0), **font_kwargs_text)
                else:
                    page.insert_text((50, y), text, fontsize=12, color=(1,0,0), **font_kwargs_text)
                    y += 18

                fb = q.get("feedback_box")
                if isinstance(fb, dict) and all(k in fb for k in ("x","y","w","h")):
                    rect = fitz.Rect(float(fb["x"]), float(fb["y"]),
                                     float(fb["x"])+float(fb["w"]), float(fb["y"])+float(fb["h"]))
                    page.draw_rect(rect, color=(1,0,0), width=1)

    doc.save(str(out_path))

def annotate_all_pdfs(
    student_json_map: Dict[str, str], grades_map: Dict[str, str], force: bool = False
):
    for pdf in student_json_map.keys():
        out_pdf = OUT_DIR / (Path(pdf).stem + ".annotated.pdf")
        if out_pdf.exists() and not force:
            print(f"[5] annotated existe déjà → {out_pdf}")
            continue
        pages_data = load_json(Path(student_json_map[pdf]))
        grades = load_json(Path(grades_map[pdf]))
        annotate_pdf(Path(pdf), pages_data, grades, out_pdf)
        print(f"[5] Écrit → {out_pdf}")


# ---------- 6) Résumé ----------
def build_summary_csv(grades_map: Dict[str, str], out_csv: Path):
    rows = []
    for pdf, gpath in grades_map.items():
        g = load_json(Path(gpath))
        pts_got_all, pts_tot_all = _grades_totals(g)
        note = g.get("note")
        if note is None and pts_tot_all:
            note = round(((pts_got_all / pts_tot_all) * 5 + 1), 1)
        rows.append(
            {
                "file": pdf,
                "points_obtenus": pts_got_all,
                "points_total": pts_tot_all,
                "note": note,
            }
        )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "points_obtenus", "points_total", "note"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[6] Résumé → {out_csv}")


# ---------- Orchestration ----------
def main(force: bool = False):
    ensure_dirs()

    # 1) solutions.json
    solutions = extract_solutions_from_tex(
        Path("exam.tex"), SOLUTIONS_JSON, force=force
    )

    # 2) artefacts (png pour tous les PDFs du dossier)
    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if not pdfs:
        print("Aucun PDF trouvé dans 'pdfs/'.")
        return
    artifacts = build_artifacts(pdfs, force=force)

    # 3) student_data pour tous les PDFs
    student_map = run_analysis_all_pdfs(artifacts, force=force)

    # 4) grading pour tous
    grades_map = run_grading_all(solutions, student_map, force=force)

    # 5) annotation pour tous
    annotate_all_pdfs(student_map, grades_map, force=force)

    # 6) résumé CSV
    build_summary_csv(grades_map, OUT_DIR / "summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regénère même si des sorties existent déjà",
    )
    args = parser.parse_args()
    main(force=args.force)
