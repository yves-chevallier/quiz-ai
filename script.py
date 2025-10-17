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
import time

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


def identify_content(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Identifie strictement le contenu et l'annotation des pages d'un quiz manuscrit.
    """

    SYSTEM_PROMPT = """
    You are an impartial and precise human anyliser.

    Your role is to analyze a scanned handwritten quiz page with printed questions
    and handwritten content. You should extract structured information with no omissions.
    You are not allowed to make assumptions beyond what is visually present on the page.

    ## Task
    You must visually interpret each provided image (A4 page, 210×297 mm)
    and return a **strict JSON object** describing what you see for each question.

    Do not return text explanations — only valid JSON.

    ---

    ## JSON Structure

    {
    "name": "string of the student, usually top right after 'Nom:' if present or else empty",
    "date": "string, centered under title, format YYYY-MM-DD if present or else empty",
    "page": "page number 1-based usually found at the bottom with 'Page X of Y' or else empty",
    "questions": [ // For each question found on the page.
        {
        "id": int,                       // Question number (1-based)
        "page": int,                     // Page index (1-based)
        "kind": "single|multi|fillin|open",
        "question_text": "string",      // Full text of the question
        "choices": [string],            // All options text (for mcq/multi)
        "annotations": {
          "text": "string",          // Any notes exactly handwritten by the student concerning the question
          "drawings": "string",        // Description of any drawings/sketches made by the student
          "analysis": "string",       // Any corrections, strikethroughs you observe and describe what options are selected with reasoning and justification
        }
    ]
    }

    ---

    ## Rules & Constraints

    - Each image is one full A4 page (210×297 mm).
    - Page index can be confirmed by counting "Page X of Y" if present.
    - If data is unclear or absent, leave fields empty (e.g. "").
    - Include **all questions**, even unanswered ones.
    - Accurately transcribe all handwritten content (text, formulas, drawings).
    - Maintain a **formal and neutral tone** in remarks.
    - The output must be a single valid JSON object — **no extra text**.

    ---

    ## Output Policy

    - Never return Markdown or explanations.
    - Never add comments or narrative text.
    - The model must produce **only valid JSON** following the structure above.
    """

    images = [{"type": "image_url", "image_url": {"url": img_to_data_url(Path(path))}} for path in image_paths]
    start = time.perf_counter()

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Analyze this quiz"}] + images,
            },
        ]
        # temperature=0
    )
    end = time.perf_counter()
    elapsed = end - start  # secondes (float)

    print(f"    - Prompt tokens: {resp.usage.prompt_tokens}\n"
          f"    - Completion tokens: {resp.usage.completion_tokens}\n"
          f"    - Total tokens: {resp.usage.total_tokens}"
          f"    - Time elapsed: {elapsed:.2f} seconds"
    )
    return json.loads(resp.choices[0].message.content)

# ---------- 3) Vision sur images (par PDF) ----------
def grading(analysis: str, solutions: str) -> Dict[str, Any]:
    """
    Retourne une liste de pages JSON (une entrée par page).
    """

    SYSTEM_PROMPT = """
    You are an impartial and precise human corrector.

    Your role is to analyze a summary of a scanned handwritten quiz page
    with printed questions and determine exactly what the student answered if the answers are correct or not.

    ## Task
    From the provided JSON analysis and the JSON answer key, build a **strict JSON object**

    Do not return text explanations — only valid JSON.

    ---

    ## Input analysis JSON Structure

    {
    "name": "string of the student, usually top right after 'Nom:' if present or else empty",
    "date": "string, centered under title, format YYYY-MM-DD if present or else empty",
    "page": "page number 1-based usually found at the bottom with 'Page X of Y' or else empty",
    "questions": [ // For each question found on the page.
        {
        "id": int,                       // Question number (1-based)
        "page": int,                     // Page index (1-based)
        "kind": "single|multi|fillin|open",
        "question_text": "string",      // Full text of the question
        "choices": [string],            // All options text (for mcq/multi)
        "annotations": {
          "text": "string",          // Any notes exactly handwritten by the student concerning the question
          "drawings": "string",        // Description of any drawings/sketches made by the student
          "analysis": "string",       // Any corrections, strikethroughs you observe and describe what options are selected with reasoning and justification
        }
    ]
    }

    ## Output JSON Structure

    {
    "name": "student name",
    "date": "exam date in YYYY-MM-DD format",
    "title": "exam title",
    "total_questions": int,
    "correct_answers": int,
    "grade": float,  // Computed from granted points over total points * 5 + 1, rounded to 0.1
    "questions": [
        {
        "id": int,                       // Question number (1-based)
        "answered": bool,               // Whether the student provided an answer
        "correct": bool,                // Whether the answer is correct
        "granted_points": float,        // Points awarded for this question 0..1, depending on how many subparts are correct
        "remark": "string",              // Relevant summary of analysis notes (strikethroughs, corrections, text not readable, etc.)
        "confidence": float,             // Between 0 and 1 indicating your confidence in the correctness of the evaluation based on reasoning
        }
    ]
    }

    ---

    ## Rules & Constraints

    - Maintain a **formal and neutral tone** in remarks.
    - The output must be a single valid JSON object — **no extra text**.

    ---

    ## Output Policy

    - Never return Markdown or explanations.
    - Never add comments or narrative text.
    - The model must produce **only valid JSON** following the structure above.
    """

    images = [{"type": "image_url", "image_url": {"url": img_to_data_url(Path(path))}} for path in image_paths]
    start = time.perf_counter()

    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Analyze this quiz"}] + images,
            },
        ]
        # temperature=0
    )
    end = time.perf_counter()
    elapsed = end - start  # secondes (float)

    print(f"    - Prompt tokens: {resp.usage.prompt_tokens}\n"
          f"    - Completion tokens: {resp.usage.completion_tokens}\n"
          f"    - Total tokens: {resp.usage.total_tokens}"
          f"    - Time elapsed: {elapsed:.2f} seconds"
    )
    return json.loads(resp.choices[0].message.content)


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
