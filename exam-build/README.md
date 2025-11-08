# Generate exam

L'objectif est de fournir une syntaxe robuste et canonique pour la saisie d'exercices d'examens ou de quiz entremélant des intitulés en Markdown, des questions à choix multiples, des questions ouvertes, des sections de code, etc. Le format YAML est utilisé pour sa lisibilité et sa simplicité.

La correction qui doit être automatisée est également spécifiée dans ce format, permettant une évaluation rapide et précise des réponses des étudiants.

- Soit avec des attentes précises qui peuvent être validées à la main.
- Soit avec une réflexion d'un LLM pour des questions ouvertes plus complexes.

Un LLM s'occupe de la reconnaissance des écritures manuscrites.

Le template LaTeX ajoute automatiquement des ancres pour identifier les région d'intérêt à fournir au modèle d'IA. Seules celles-ci sont extraites et envoyées au modèle pour analyse et correction.

Les examens sont prévus pour être générés en PDF, imprimés et distribués aux étudiants.

Sur la première page on ajoute un code 2D unique permettant d'identifier la seed utilisée pour générer l'examen, dans le cas ou les questions sont randomisées.

Chaque question, partie, sous-partie hérite de la seed actuelle permettant de
générer les variantes de l'examen de manière déterministe.

## Template

La conversion utilise TeXSmith pour la génération du PDF. La template est fournie dans `template/`.

## Auto-correction

Le module exam-ai s'occupe de l'auto-correction des examens scannés. Il utilise des LLMs pour analyser les réponses des étudiants et les comparer aux solutions attendues, il annote les PDFs avec les corrections et les scores et génère des rapports de performance, et des emails de feedback aux étudiants.

## TODO

- Écrire le module Python de génération Markdown de l'examen à partir du YAML.
- Etend TeXSmith pour supporter les éléments spécifiques à un examen:
  - section/subsection/subsubsection -> question/part/subpart
  - ajout des ancres pour l'IA
  - ajout du QR code avec la seed
  - Support des blocs de solution avec une admonition: ```!!!solution```
  - Support des éléments exams (`\fillwithdottedlines`, `\fillin`)
  - Implémente la syntaxe via une extension Markdown pour remplir avec:
    - des lignes pointillées `--- 3 ---`
    - des grilles `--- grid 5 ---`
- Écrire le module TexSmith-Exam fournissant les fonctionnalités spécifiques
  aux examens.

Utilisation:

```bash
# Build the PDF and extract the meta information
# - Anchors of interest for the LLM
# - Name anchor (student name)
# - QR code with the seed
exam generate exam.yml -n1 -o exam.pdf --meta meta.json
# Print exam, distribute to students
# Collect and scan completed exams
exam correct exam.yml -o out/ scan.pdf
# Generate:
# - annotated PDF with corrections
# - per-student reports
# - per-student feedback emails
```
