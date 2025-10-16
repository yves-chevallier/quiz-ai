# script.py
import os, json, base64
from openai import OpenAI
from pdf2image import convert_from_path
import fitz  # PyMuPDF

client = OpenAI()  # Assure-toi d’avoir OPENAI_API_KEY dans l’environnement

# ---------- Utils ----------
def img_to_data_url(path_png: str) -> str:
    with open(path_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# ---------- 1) Extraction corrigé LaTeX ----------
def extract_solutions_from_tex(path_tex, model="gpt-4o"):
    tex = open(path_tex, "r", encoding="utf-8").read()
    prompt = """
    Tu es un assistant chargé de transformer la source d'un quiz LaTeX utilisant le package exam en un JSON structuré.
    Exige le format suivant:
    {
      "quiz_title": string,
      "questions": [
        {
          "number": int,
          "text": string,
          "type": "mcq" | "checkbox" | "fillin" | "open",
          "choices": [string],             // si mcq/checkbox
          "answers": [string],             // réponses correctes: CorrectChoice, fillin/fillna, ou solution
          "points": number                 // si absent dans le LaTeX: 1 par CorrectChoice ou par trou, sinon 1
        }
      ],
      "total_points": number
    }
    Si un barème explicite n’est pas dans le LaTeX, déduis les points: 1 par item (CorrectChoice ou trou).
    Donne UNIQUEMENT le JSON.
    """

    res = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Tu es un parseur de quiz LaTeX."},
            {"role": "user", "content": prompt + "\n\n" + tex}
        ]
    )

    data = json.loads(res.choices[0].message.content)

    # filet de sécurité (si total_points absent)
    if "total_points" not in data:
        total = 0.0
        for q in data.get("questions", []):
            pts = q.get("points")
            if pts is None:
                # si rien: 1
                pts = 1.0
            total += float(pts)
        data["total_points"] = total

    with open("solutions.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# ---------- 2) PDF -> images ----------
def pdf_to_images(pdf_path, out_dir="tmp_images", dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for i, img in enumerate(images, start=1):
        out_path = os.path.join(out_dir, f"{base}_p{i}.png")
        img.save(out_path, "PNG")
        paths.append(out_path)
    return paths

# ---------- 3) Vision sur images ----------


def img_to_data_url(path_png: str) -> str:
    with open(path_png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def analyze_student_images(image_paths, model="gpt-4o"):
    """
    Retourne un JSON par page:
    {
      "page_index": int (1-based),
      "student_name": string? ,
      "quiz_title": string? ,
      "note_anchor": {"x":int,"y":int}?,
      "questions":[
        {
          "number": int,
          "kind": "single"|"multi"|"fillin"|"open",
          "selected_options": [string],
          "handwriting": string,
          "confidence": number,
          "mark_anchor": {"x":int,"y":int}?,
          "feedback_box": {"x":int,"y":int,"w":int,"h":int}?
        }
      ]
    }
    """
    pages_data = []
    base_prompt = (
        "Tu es un correcteur. Pour cette page d’un quiz manuscrit, renvoie STRICTEMENT un JSON décrivant : "
        "page_index (1-based), éventuellement student_name/quiz_title si visibles, note_anchor sur la première page, "
        "et pour chaque question: number, kind (single/multi/fillin/open), selected_options[], handwriting, confidence (0..1), "
        "mark_anchor (x,y), feedback_box (x,y,w,h). Ne devine pas: si incertain, laisse vide."
    )

    for i, path in enumerate(image_paths, start=1):
        data_url = img_to_data_url(path)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Tu analyses visuellement une copie d'examen et tu renvoies un JSON strict."},
                {"role": "user", "content": [
                    {"type": "text", "text": base_prompt + f" Indique page_index={i}."},
                    # ⬇️ Corrigé : bloc image
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            temperature=0  # plus déterministe pour du JSON
        )
        page_json = json.loads(resp.choices[0].message.content)
        page_json.setdefault("page_index", i)
        page_json.setdefault("questions", [])
        pages_data.append(page_json)

    with open("student_data.json", "w", encoding="utf-8") as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=2)
    return pages_data


# ---------- 4) Notation ----------
def grade_student(solutions, pages_data):
    """
    solutions: dict (solutions.json)
    pages_data: liste de pages (chacune avec questions[])
    """
    # aplatir les questions prédites
    pred_qs = []
    for p in pages_data:
        for q in p.get("questions", []):
            pred_qs.append(q)

    # index par numéro
    pred_by_num = {}
    for q in pred_qs:
        n = str(q.get("number"))
        if not n:
            continue
        # si plusieurs occurrences, on garde celle avec meilleure confiance
        best = pred_by_num.get(n)
        if (best is None) or (q.get("confidence", 0) > best.get("confidence", 0)):
            pred_by_num[n] = q

    grades = []
    total = 0.0
    got = 0.0

    for qk in solutions.get("questions", []):
        sid = str(qk.get("number"))
        expected = [x.lower().strip() for x in qk.get("answers", [])]
        points = float(qk.get("points", 1.0))
        total += points

        stu = pred_by_num.get(sid)
        if not stu:
            grades.append({"number": sid, "points": 0.0, "comment": "Pas de réponse détectée"})
            continue

        kind = stu.get("kind", "single")
        comment = ""
        pts = 0.0

        if qk.get("type") in ("mcq", "checkbox", "mcq/checkbox") or kind in ("single","multi"):
            selected = [x.lower().strip() for x in stu.get("selected_options", [])]
            good = len(set(expected) & set(selected))
            # pénalité simple si sur-sélection
            bad = max(0, len(selected) - good)
            base = good / max(1, len(expected))
            penalty = 0.0 if bad == 0 else min(0.5, 0.1 * bad)
            pts = max(0.0, (base - penalty) * points)
            comment = f"{good}/{len(expected)} bons; {bad} en trop"
        elif qk.get("type") == "fillin":
            hw = (stu.get("handwriting") or "").lower()
            if not hw:
                pts = 0.0
                comment = "réponse manuscrite illisible/absente"
            else:
                match = sum(1 for e in expected if e in hw)
                pts = (match / max(1, len(expected))) * points
                comment = f"{match}/{len(expected)} éléments reconnus"
        else:  # open
            hw = (stu.get("handwriting") or "").lower()
            if not hw:
                pts = 0.0
                comment = "vide"
            else:
                # heuristique minimale (peut être raffinée)
                sol_txt = " ".join(expected).lower()
                if not sol_txt:
                    sol_txt = (qk.get("solution") or "").lower()
                hits = sum(1 for w in set(sol_txt.split()) if w and w in hw)
                denom = max(6, len(set(sol_txt.split())))
                ratio = min(1.0, hits / denom)
                pts = ratio * points
                comment = f"pertinence≈{int(ratio*100)}%"

        got += pts
        grades.append({
            "number": sid,
            "points": round(pts, 3),
            "points_possible": points,
            "comment": comment
        })

    note = round(((got / total) * 5 + 1), 1) if total > 0 else 1.0
    return {"grades": grades, "total": total, "got": round(got, 3), "note": note}

# ---------- 5) Annotation PDF ----------
def annotate_pdf(pdf_path, pages_data, grades, out_path="annotated.pdf"):
    """
    Ecrit en rouge (✓/✗/partiel) + points + note en page 1 si une ancre est fournie.
    Si les positions (mark_anchor/feedback_box/note_anchor) sont absentes, on pose les commentaires en marge.
    """
    doc = fitz.open(pdf_path)
    # Indexer grades par numéro de question pour lookup rapide
    grade_by_num = { str(g["number"]): g for g in grades["grades"] }

    for idx, page in enumerate(doc, start=1):
        page_data = next((p for p in pages_data if p.get("page_index")==idx), None)
        y = 50  # fallback pour marge
        if page_data:
            # Afficher la note finale si page 1 et ancre dispo
            if idx == 1:
                note_anchor = page_data.get("note_anchor")
                label = f"Note finale : {grades['note']}/6"
                if isinstance(note_anchor, dict) and "x" in note_anchor and "y" in note_anchor:
                    page.insert_text((note_anchor["x"], note_anchor["y"]), label, fontsize=16, color=(1,0,0))
                else:
                    page.insert_text((420, 100), label, fontsize=16, color=(1,0,0))

            for q in page_data.get("questions", []):
                num = str(q.get("number"))
                g = grade_by_num.get(num)
                if not g:
                    continue
                pts = f"{g['points']}/{g['points_possible']}"
                # symbole
                sym = "✓" if g["points"] == g["points_possible"] else ("≈" if g["points"]>0 else "✗")
                text = f"Q{num} {sym}  {pts}  – {g['comment']}"
                # où écrire ? ancre ou marge
                mk = q.get("mark_anchor") or {}
                if isinstance(mk, dict) and "x" in mk and "y" in mk:
                    page.insert_text((mk["x"], mk["y"]), text, fontsize=11, color=(1,0,0))
                else:
                    page.insert_text((50, y), text, fontsize=11, color=(1,0,0))
                    y += 16

                # zone feedback
                fb = q.get("feedback_box")
                if isinstance(fb, dict) and all(k in fb for k in ("x","y","w","h")):
                    rect = fitz.Rect(fb["x"], fb["y"], fb["x"]+fb["w"], fb["y"]+fb["h"])
                    page.draw_rect(rect, color=(1,0,0), width=1)

    doc.save(out_path)

# ---------- Main ----------
if __name__ == "__main__":
    # 1) Corrigé
    solutions = extract_solutions_from_tex("exam.tex")

    # 2) Un PDF d’exemple (ou boucle sur un dossier)
    pdf_paths = ["pdfs/2025-10-16-14-46-25.pdf"]

    for pdf in pdf_paths:
        # 2) PDF -> images
        imgs = pdf_to_images(pdf, out_dir="tmp_images", dpi=220)

        # 3) Vision -> JSON par page
        pages = analyze_student_images(imgs)

        # 4) Notation
        grades = grade_student(solutions, pages)

        # 5) Annotation
        out_pdf = pdf.replace(".pdf", "_annotated.pdf")
        annotate_pdf(pdf, pages, grades, out_path=out_pdf)

        print(f"{pdf}: {grades['got']}/{grades['total']}  => note {grades['note']}/6")
