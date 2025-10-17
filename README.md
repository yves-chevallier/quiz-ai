# Experiment on Quiz automated grading system with LLMs

This repository contains code and resources for experimenting with automated grading of quizzes using Large Language Models (LLMs). The goal is to evaluate the effectiveness of LLMs in assessing student responses and providing accurate grades.

1. A quiz is written in YAML format, specifying questions, correct answers, and grading criteria.
2. Converter script transforms the YAML to LaTeX.
3. PDF is printed, distributed to students, and their responses are collected.
4. Anchors generated from LaTeX sources are extracted from the source PDF.
5. responses are scanned on a PDF.
6. The pdf is split into individual student response files.
7. Script extracts each page into individual images.
8. Each image is split according to the anchors to isolate each question.
9. Each question is sent to an LLM (GPT-5) for visual analysis only.
10. Meta data are merged into a single JSON file per student.
11. A grading prompt including the meta data analysis and the exam source file in YAML (converted to json) is sent to the LLM for grading.
12. Grades and feedback are collected and compiled into a final report.
13. (optional) Annotated PDF with feedback per question is generated.

## Usage

```python
# Convert quiz from YAML to LaTeX
quizai latex quiz.yaml -o quiz.tex

# Compile LaTeX to PDF
latexmk -lualatex quiz.tex

# Plumbing tool to manually extract anchors from the quiz PDF
quizai anchors quiz.pdf -o anchors.json

# LLM Processing of student responses using the anchors or
# using source PDF directly. If PDF may contains pages in order for N students.
quizai analysis responses.pdf -a anchors.json -o responses/
quizai analysis responses.pdf -i quiz.pdf -o responses/

# Grading student responses (folder containing multiple student analysis JSONs or a single JSON file)
quizai grade-one analysis.yaml -q quiz.yaml -o grade.json

# Generate a report from the grading JSON file
quizai report grade1.json grade2.json ... -o report.md

# Annotated PDF generation (optional)
quizai annotate responses.pdf grade.json -o annotated_responses.pdf

# Batch grading (analysis + grading + report)
quizai grade responses.pdf -q quiz.yaml -a anchors.json --report report.md --annotate=true
```
