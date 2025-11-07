#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="qz-02"
OUT_DIR="$BASE_DIR/out"
SRC_EXAM="$BASE_DIR/src/exam.pdf"
QUIZ_YAML="$BASE_DIR/src/exam.yml"
ROSTER="$BASE_DIR/students.xlsx"
BINDER="$BASE_DIR/qz-02.pdf"

# Step 0 – Clean slate
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Step 1 – Anchors
uv run quiz-ai anchors "$SRC_EXAM" -o "$OUT_DIR/anchors.json"

# Step 2 – Split binder into per-student PDFs
uv run quiz-ai split "$BINDER" \
  -t "$SRC_EXAM" \
  -o "$OUT_DIR/scans" \
  -p qz-02

# Step 3 – Analyse, grade, annotate each student
mkdir -p "$OUT_DIR/per-student"

for scan_pdf in "$OUT_DIR"/scans/qz-02-*.pdf; do
  student_stem=$(basename "$scan_pdf" .pdf)
  student_dir="$OUT_DIR/per-student/$student_stem"
  analysis_dir="$student_dir/analysis"

  echo "Processing $student_stem…"
  mkdir -p "$analysis_dir"

  uv run quiz-ai analysis "$scan_pdf" \
    -a "$OUT_DIR/anchors.json" \
    --quiz "$QUIZ_YAML" \
    --roster "$ROSTER" \
    -o "$analysis_dir" \
    --temperature 0

  uv run quiz-ai grading \
    "$analysis_dir/analysis.json" \
    "$QUIZ_YAML" \
    -o "$student_dir" \
    --model gpt-4o

  uv run quiz-ai annotate \
    "$scan_pdf" \
    -g "$student_dir/grading.json" \
    -a "$OUT_DIR/anchors.json" \
    "$student_dir/${student_stem}-annotated.pdf"
done

# Step 4 – Global summary (Excel, annotated copies, binder overlay)
uv run quiz-ai summary \
  "$OUT_DIR/per-student" \
  -q "$QUIZ_YAML" \
  --roster "$ROSTER" \
  --anchors "$OUT_DIR/anchors.json" \
  --binder "$BINDER" \
  --scans "$OUT_DIR/scans" \
  --template "$SRC_EXAM" \
  -o "$OUT_DIR/summary"

echo "Pipeline completed successfully."
