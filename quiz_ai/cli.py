"""
Typer command-line interface exposing the Quiz AI pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, List, Optional

import typer

from .analysis import PROMPT_PATH, load_or_extract_anchors, run_analysis
from .anchors import extract_anchors, save_anchors
from .grading import Solution, compute_points_from_grades, load_solution, run_grading
from .latex import write_latex_from_yaml
from .llm import DEFAULT_VISION_MODEL, build_openai_client
from .report import GradeSummary, build_markdown_report, summarise_grade, write_summary_csv
from .utils import ensure_directory, read_json, write_json
from .annotate import annotate_pdf
from .anchors import load_anchors


app = typer.Typer(
    help="Outils CLI pour la conversion, l'analyse et la notation de quiz scannés.",
    no_args_is_help=True,
)


@app.command()
def latex(
    quiz_yaml: Annotated[
        Path,
        typer.Argument(help="Fichier YAML décrivant le quiz.", exists=True, readable=True),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Chemin du fichier LaTeX généré."),
    ] = None,
) -> None:
    """
    Convertir un quiz YAML en fichier LaTeX.
    """
    tex_path = output or quiz_yaml.with_suffix(".tex")
    write_latex_from_yaml(quiz_yaml, tex_path)
    typer.echo(f"LaTeX généré → {tex_path}")


@app.command()
def anchors(
    pdf: Annotated[
        Path,
        typer.Argument(help="PDF source contenant les ancres hyperref.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Fichier JSON de sortie.", dir_okay=False),
    ] = Path("anchors.json"),
    overlap: Annotated[
        float,
        typer.Option(
            "--overlap",
            min=0.0,
            help="Chevauchement vertical en mm entre deux ancres consécutives.",
        ),
    ] = 3.0,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Expression régulière identifiant les ancres."),
    ] = r"^Q(\d+)-anchor$",
    include_bottom: Annotated[
        bool,
        typer.Option("--include-bottom/--no-include-bottom", help="Inclure la zone sous la dernière ancre."),
    ] = True,
) -> None:
    """
    Extraire les ancres d'un PDF et les sérialiser en JSON.
    """
    anchors_model = extract_anchors(
        pdf,
        overlap=overlap,
        anchor_pattern=pattern,
        include_bottom_segment=include_bottom,
    )
    save_anchors(anchors_model, output)
    typer.echo(f"Ancres sauvegardées → {output}")


@app.command()
def analysis(
    responses_pdf: Annotated[
        Path,
        typer.Argument(help="PDF des copies scannées à analyser.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Dossier de sortie pour l'analyse."),
    ] = Path("analysis"),
    anchors_path: Annotated[
        Optional[Path],
        typer.Option("-a", "--anchors", help="Fichier JSON d'ancres pré-calculées.", exists=True, readable=True),
    ] = None,
    source_pdf: Annotated[
        Optional[Path],
        typer.Option("-i", "--input", help="PDF source pour extraire les ancres si besoin.", exists=True, readable=True),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Modèle vision OpenAI à utiliser."),
    ] = DEFAULT_VISION_MODEL,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=60, max=600, help="DPI utilisés pour la rasterisation PDF."),
    ] = 220,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option("--prompt", help="Prompt Markdown personnalisé pour l'analyse.", exists=True, readable=True),
    ] = None,
) -> None:
    """
    Lancer l'analyse des copies PDF question par question.
    """
    out_dir = ensure_directory(output)
    try:
        anchors_model = load_or_extract_anchors(
            anchors_path=anchors_path,
            source_pdf=source_pdf,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    client = build_openai_client()
    items = run_analysis(
        responses_pdf=responses_pdf,
        anchors=anchors_model,
        output_dir=out_dir,
        client=client,
        model=model,
        dpi=dpi,
        prompt_path=prompt_path or PROMPT_PATH,
    )
    typer.echo(f"{len(items)} zones analysées. Résultat → {out_dir / 'analysis.json'}")


@app.command("grade-one")
def grade_one(
    analysis_json: Annotated[
        Path,
        typer.Argument(help="Fichier JSON produit par la commande d'analyse.", exists=True, readable=True),
    ],
    quiz_yaml: Annotated[
        Path,
        typer.Option("-q", "--quiz", help="Fichier YAML des solutions officielles.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Fichier JSON de notation.", dir_okay=False),
    ] = Path("grade.json"),
    model: Annotated[
        str,
        typer.Option("--model", help="Modèle texte OpenAI pour la notation."),
    ] = DEFAULT_VISION_MODEL,
) -> None:
    """
    Convertir un fichier d'analyse en notation à l'aide du LLM.
    """
    analysis_data = read_json(analysis_json)
    solution = load_solution(quiz_yaml)
    client = build_openai_client()
    grades = run_grading(
        analysis=analysis_data,
        solution=solution,
        client=client,
        model=model,
    )

    obtained, total = compute_points_from_grades(grades, solution)
    grades["_source_analysis"] = str(analysis_json)
    grades["_source_quiz"] = str(quiz_yaml)
    grades["_points_obtained"] = obtained
    grades["_points_total"] = total

    write_json(output, grades)
    typer.echo(f"Notation enregistrée → {output} ({obtained:.2f}/{total:.2f} pts)")


@app.command()
def report(
    grade_files: Annotated[
        List[Path],
        typer.Argument(help="Un ou plusieurs fichiers JSON de notation.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Chemin du rapport Markdown."),
    ] = Path("report.md"),
    format: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Format du rapport (md ou csv).",
            show_choices=True,
        ),
    ] = "md",
    quiz_yaml: Annotated[
        Optional[Path],
        typer.Option("--quiz", help="Fichier YAML des solutions pour recalculer les points.", exists=True, readable=True),
    ] = None,
) -> None:
    """
    Générer un rapport à partir de plusieurs fichiers de notation.
    """
    summaries = []
    solution: Optional[Solution] = load_solution(quiz_yaml) if quiz_yaml else None

    if format == "csv":
        if solution is None:
            raise typer.BadParameter("Le format CSV requiert --quiz pour connaître la pondération.")
        summaries = write_summary_csv(grade_files, solution, output)
        typer.echo(f"CSV de synthèse généré → {output}")
        return

    if solution:
        summaries = [summarise_grade(path, solution) for path in grade_files]
    else:
        summaries = []
        for path in grade_files:
            data = read_json(path)
            note_value = data.get("note")
            if isinstance(note_value, (int, float)):
                note = float(note_value)
            else:
                note = None
            summaries.append(
                GradeSummary(
                    path=path,
                    data=data,
                    points_obtained=float(data.get("_points_obtained", 0.0)),
                    points_total=float(data.get("_points_total", 0.0)),
                    note=note,
                )
            )

    build_markdown_report(summaries, output)
    typer.echo(f"Rapport Markdown généré → {output}")


@app.command()
def annotate(
    pdf_input: Annotated[
        Path,
        typer.Argument(help="PDF original des copies scannées.", exists=True, readable=True),
    ],
    grade_json: Annotated[
        Path,
        typer.Argument(help="Fichier JSON de notation contenant remarques et corrections.", exists=True, readable=True),
    ],
    anchors_json: Annotated[
        Path,
        typer.Option("-a", "--anchors", help="Fichier JSON d'ancres.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="PDF annoté à générer.", dir_okay=False),
    ] = Path("annotated.pdf"),
) -> None:
    """
    Produire un PDF annoté à partir des corrections du LLM.
    """
    anchors_model = load_anchors(anchors_json)
    grades = read_json(grade_json)
    feedback = [
        {
            "id": int(q.get("id")),
            "correct": bool(q.get("correct", False)),
            "comment": q.get("remark", ""),
        }
        for q in grades.get("questions", [])
    ]
    annotate_pdf(
        pdf_input=pdf_input,
        pdf_output=output,
        anchors=anchors_model,
        feedback=feedback,
    )
    typer.echo(f"PDF annoté généré → {output}")


@app.command()
def grade(
    responses_pdf: Annotated[
        Path,
        typer.Argument(help="PDF des copies scannées.", exists=True, readable=True),
    ],
    quiz_yaml: Annotated[
        Path,
        typer.Option("-q", "--quiz", help="Fichier YAML de la solution.", exists=True, readable=True),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Dossier de sortie global."),
    ] = Path("grade"),
    anchors_path: Annotated[
        Optional[Path],
        typer.Option("-a", "--anchors", help="Fichier d'ancres existant.", exists=True, readable=True),
    ] = None,
    source_pdf: Annotated[
        Optional[Path],
        typer.Option("-i", "--input", help="PDF source pour extraire les ancres.", exists=True, readable=True),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Modèle OpenAI (vision + texte)."),
    ] = DEFAULT_VISION_MODEL,
    report_path: Annotated[
        Optional[Path],
        typer.Option("--report", help="Générer un rapport Markdown à ce chemin."),
    ] = None,
    annotate_pdf_flag: Annotated[
        bool,
        typer.Option("--annotate/--no-annotate", help="Produire également un PDF annoté."),
    ] = False,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option("--prompt", help="Prompt Markdown personnalisé pour l'analyse.", exists=True, readable=True),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=60, max=600, help="DPI utilisés pour la rasterisation PDF."),
    ] = 220,
) -> None:
    """
    Chaîne complète analyse + notation (+ rapport/annotation optionnels).
    """
    base_dir = ensure_directory(output)
    analysis_dir = ensure_directory(base_dir / "analysis")
    try:
        anchors_model = load_or_extract_anchors(
            anchors_path=anchors_path,
            source_pdf=source_pdf,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    client = build_openai_client()
    run_analysis(
        responses_pdf=responses_pdf,
        anchors=anchors_model,
        output_dir=analysis_dir,
        client=client,
        model=model,
        prompt_path=prompt_path or PROMPT_PATH,
        dpi=dpi,
    )
    analysis_json = analysis_dir / "analysis.json"
    analysis_data = read_json(analysis_json)

    solution = load_solution(quiz_yaml)
    grades = run_grading(
        analysis=analysis_data,
        solution=solution,
        client=client,
        model=model,
    )
    obtained, total = compute_points_from_grades(grades, solution)
    grades["_source_analysis"] = str(analysis_json)
    grades["_source_quiz"] = str(quiz_yaml)
    grades["_source_pdf"] = str(responses_pdf)
    grades["_points_obtained"] = obtained
    grades["_points_total"] = total

    grades_path = base_dir / "grade.json"
    write_json(grades_path, grades)
    typer.echo(f"Notation terminée → {grades_path} ({obtained:.2f}/{total:.2f} pts)")

    summaries = None
    if report_path:
        summaries = [summarise_grade(grades_path, solution)]
        build_markdown_report(summaries, report_path)
        typer.echo(f"Rapport généré → {report_path}")

    if annotate_pdf_flag:
        annotated_path = base_dir / "annotated.pdf"
        feedback = [
            {
                "id": int(q.get("id")),
                "correct": bool(q.get("correct", False)),
                "comment": q.get("remark", ""),
            }
            for q in grades.get("questions", [])
        ]
        annotate_pdf(
            pdf_input=responses_pdf,
            pdf_output=annotated_path,
            anchors=anchors_model,
            feedback=feedback,
        )
        typer.echo(f"PDF annoté généré → {annotated_path}")
