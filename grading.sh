#!/usr/bin/env bash
set -euo pipefail

for analysis_json in qz-02/out/per-student/*/analysis/analysis.json; do
  student_dir=$(dirname "$(dirname "$analysis_json")")
  student_stem=$(basename "$student_dir")
  scan_pdf="qz-02/out/scans/${student_stem}.pdf"

  echo "Regrading $student_stem…"
  uv run quiz-ai grading \
    "$analysis_json" \
    qz-02/src/exam.yml \
    -o "$student_dir" \
    --model gpt-4o

  echo "Annotating $student_stem…"
  uv run quiz-ai annotate \
    "$scan_pdf" \
    -g "$student_dir/grading.json" \
    -a qz-02/out/anchors.json \
    "$student_dir/${student_stem}-annotated.pdf"
done
