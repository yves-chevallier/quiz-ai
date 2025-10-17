import fitz

PT_TO_MM = 0.352778

doc = fitz.open("exam2/exam.pdf")
for pageno, page in enumerate(doc, start=1):
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not text:
            continue
        t = text.strip()
        if t.startswith("Q") and t.endswith("-anchor"):
            qid = t.split("-")[0][1:]
            cx, cy = ((x0 + x1) / 2) * PT_TO_MM, ((y0 + y1) / 2) * PT_TO_MM
            print(f"p{pageno}  Q{qid}-anchor  @ ({cx:.1f} mm, {cy:.1f} mm)")
        elif t.startswith("Q") and t.endswith("-endchoices"):
            qid = t.split("-")[0][1:]
            cx, cy = ((x0 + x1) / 2) * PT_TO_MM, ((y0 + y1) / 2) * PT_TO_MM
            print(f"p{pageno}  Q{qid}-endchoices  @ ({cx:.1f} mm, {cy:.1f} mm)")
        elif t.startswith("Q") and t.endswith("-end"):
            qid = t.split("-")[0][1:]
            cx, cy = ((x0 + x1) / 2) * PT_TO_MM, ((y0 + y1) / 2) * PT_TO_MM
            print(f"p{pageno}  Q{qid}-end  @ ({cx:.1f} mm, {cy:.1f} mm)")
