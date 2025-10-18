"""
Typer command-line interface exposing the Quiz AI pipeline.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Annotated, Any, Deque, Dict, List, Optional

import typer

from .analysis import (
    PROMPT_PATH,
    analysis_output_schema,
    load_or_extract_anchors,
    run_analysis,
)
from .anchors import extract_anchors, save_anchors
from .grading import Solution, compute_points_from_grades, load_solution, run_grading
from .latex import write_latex_from_yaml
from .llm import DEFAULT_VISION_MODEL, build_openai_client
from .report import GradeSummary, build_markdown_report, summarise_grade, write_summary_csv
from .utils import ensure_directory, read_json, write_json
from .annotate import annotate_pdf
from .anchors import load_anchors

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

try:
    from typer.rich_utils import console as typer_console
except ImportError:  # pragma: no cover - fallback for older Typer versions
    typer_console = Console()


app = typer.Typer(
    help="Outils CLI pour la conversion, l'analyse et la notation de quiz scannés.",
    no_args_is_help=True,
)


def _format_duration(seconds: float) -> str:
    """Turn a duration in seconds into a short human-readable string."""
    if seconds < 1.0:
        return f"{seconds:.2f}s"
    total_seconds = int(seconds)
    millis = seconds - total_seconds
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    if millis >= 0.05:
        return f"{seconds:.2f}s"
    return f"{secs}s"


class RichAnalysisProgress:
    """Rich-powered progress rendering for the analysis pipeline."""

    def __init__(
        self,
        *,
        console: Optional[Console] = None,
        history_size: int = 6,
    ) -> None:
        self.console = console or Console()
        self.state: Dict[str, Any] = {
            "total_expected": 0,
            "pages_with_regions": 0,
            "total_pages": None,
            "questions_processed": 0,
            "questions_total": 0,
            "current_page": None,
            "current_page_regions": 0,
            "current_page_questions": 0,
            "current_question_id": None,
            "current_question_kind": "",
            "current_question_summary": "",
            "current_question_position": 0,
            "current_question_total": 0,
            "current_question_tokens": {"input": 0, "output": 0, "total": 0},
            "overall_tokens": {"input": 0, "output": 0, "total": 0},
        }
        self._messages: Deque[Text] = deque(maxlen=history_size)
        self._live: Optional[Live] = None
        self._pages_seen: set[int] = set()

    def __enter__(self) -> "RichAnalysisProgress":
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=6,
        )
        self._live.__enter__()
        self._log("Analyse initialisée. CTRL+C pour interrompre.", style="bold cyan")
        self._refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            self._log("Interruption utilisateur détectée.", style="bold yellow")
            self._refresh()
        if self._live:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        self._handle_event(event, payload or {})
        self._refresh()

    # Event handling -----------------------------------------------------------------
    def _handle_event(self, event: str, payload: Dict[str, Any]) -> None:
        if event == "run:init":
            self.state["total_expected"] = int(payload.get("total_regions_expected") or 0)
            self.state["pages_with_regions"] = int(payload.get("pages_with_regions") or 0)
            self._log(
                f"{self.state['total_expected']} zone(s) attendues sur "
                f"{self.state['pages_with_regions']} page(s) ancrées.",
                style="cyan",
            )
            return

        if event == "run:pages_ready":
            total_pages = payload.get("total_pages")
            if isinstance(total_pages, int):
                self.state["total_pages"] = total_pages
            self._log(f"PDF rasterisé → {total_pages} page(s).", style="cyan")
            return

        if event == "page:start":
            page_number = payload.get("page_number")
            self.state["current_page"] = page_number
            self.state["current_page_regions"] = 0
            self.state["current_page_questions"] = 0
            self._log(f"Page {page_number} : préparation.", style="magenta")
            return

        if event == "page:rendered":
            image_path = payload.get("image_path")
            filename = Path(image_path).name if image_path else "image"
            self._log(f"Page rendue → {filename}", style="magenta")
            return

        if event == "page:regions":
            count = int(payload.get("count", 0))
            self.state["current_page_regions"] = count
            self._log(f"Régions détectées : {count}", style="magenta")
            return

        if event == "page:crops-ready":
            count = int(payload.get("count", 0))
            self.state["current_page_regions"] = count
            self._log(f"Rognage terminé : {count} zone(s).", style="magenta")
            return

        if event == "page:skip":
            reason = payload.get("reason") or "motif inconnu"
            mapping = {
                "no-anchors": "aucune ancre correspondante",
                "no-regions": "aucune région définie",
                "invalid-page-height": "hauteur de page invalide",
            }
            detail = mapping.get(reason, reason)
            self._log(f"Page ignorée ({detail}).", style="yellow")
            return

        if event == "page:process":
            count = int(payload.get("question_count", 0))
            self.state["current_page_questions"] = count
            self._log(f"Traitement des {count} zone(s) détectées.", style="magenta")
            return

        if event == "question:start":
            question_id = payload.get("question_id")
            position = int(payload.get("position", 0))
            total = int(payload.get("total", 0))
            kind = payload.get("question_kind") or ""
            self.state["current_question_id"] = question_id
            self.state["current_question_kind"] = kind
            self.state["current_question_position"] = position
            self.state["current_question_total"] = total
            if total:
                self.state["questions_total"] = max(self.state["questions_total"], total)
            self._log(f"Question {question_id} [{position}/{total}] en cours.", style="green")
            return

        if event == "question:request":
            self._log("  ↳ Requête envoyée au modèle…", style="green")
            return

        if event == "question:result":
            question_id = payload.get("question_id")
            self.state["current_question_id"] = question_id
            kind = payload.get("question_kind") or ""
            summary = payload.get("summary") or ""
            position = int(payload.get("position", 0))
            total = int(payload.get("total", 0))
            usage = payload.get("usage") or {}
            in_tokens = int(usage.get("input_tokens") or 0)
            out_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (in_tokens + out_tokens))

            self.state["current_question_kind"] = kind
            self.state["current_question_summary"] = summary
            self.state["current_question_position"] = position
            self.state["current_question_total"] = total
            self.state["current_question_tokens"] = {
                "input": in_tokens,
                "output": out_tokens,
                "total": total_tokens,
            }
            self.state["questions_processed"] = position
            self.state["questions_total"] = max(self.state["questions_total"], total)

            self.state["overall_tokens"]["input"] += in_tokens
            self.state["overall_tokens"]["output"] += out_tokens
            self.state["overall_tokens"]["total"] += total_tokens

            page_number = payload.get("page_number")
            if isinstance(page_number, int):
                self._pages_seen.add(page_number)

            self._log(
                f"  ↳ Réponse reçue : {kind or 'question'} | {summary}",
                style="green",
            )
            self._log(
                f"    Jetons {total_tokens} (in {in_tokens}, out {out_tokens})",
                style="dim",
            )
            return

        if event == "run:complete":
            self._log("Analyse terminée.", style="bold green")
            return

        # Unknown events fallback
        self._log(f"{event} → {payload}", style="dim")

    # Rendering ---------------------------------------------------------------------
    def _render(self):
        size = self.console.size
        term_width = size.width if size else 120

        total_expected = self.state["total_expected"] or self.state["questions_total"] or 0
        processed = self.state["questions_processed"]
        expected_display = (
            f"{processed}/{total_expected}" if total_expected else f"{processed}/-"
        )

        total_pages = self.state["total_pages"] or 0
        pages_seen = len(self._pages_seen)
        pages_display = (
            f"{pages_seen}/{total_pages}" if total_pages else f"{pages_seen}/-"
        )

        summary_table = Table.grid(padding=(0, 1))
        summary_table.add_column(style="cyan", justify="right")
        summary_table.add_column()
        summary_table.add_column(justify="left")

        summary_table.add_row(
            "Questions",
            f"{expected_display}",
            self._progress_widget(processed, total_expected),
        )

        summary_table.add_row(
            "Pages",
            f"{pages_display} (avec régions {self.state['pages_with_regions']})",
            self._progress_widget(pages_seen, total_pages),
        )

        overall_tokens = self.state["overall_tokens"]
        summary_table.add_row(
            "Jetons",
            f"{overall_tokens['total']} (in {overall_tokens['input']}, out {overall_tokens['output']})",
            self._placeholder(),
        )

        summary_panel = Panel(
            summary_table,
            title="Progression",
            border_style="cyan",
            box=box.ROUNDED,
        )

        show_details = term_width >= 72
        show_log = term_width >= 110

        details_panel = None
        if show_details:
            details_table = Table.grid(padding=(0, 1))
            details_table.add_column(style="magenta", justify="right")
            details_table.add_column()

            if self.state["current_page"] is None:
                page_value = "-"
            else:
                page_value = (
                    f"Page {self.state['current_page']} • "
                    f"{self.state['current_page_regions']} zone(s) détectées"
                )
                if self.state["current_page_questions"]:
                    page_value += f" • {self.state['current_page_questions']} en traitement"
            details_table.add_row("Page actuelle", page_value)

            question_id = self.state["current_question_id"]
            if question_id is None:
                question_value = "-"
            else:
                total = self.state["current_question_total"] or self.state["questions_total"] or "-"
                kind = self.state["current_question_kind"] or "type inconnu"
                question_value = f"Q{question_id} ({kind}) • {self.state['current_question_position']}/{total}"
            details_table.add_row("Question en cours", question_value)

            summary_text = self.state["current_question_summary"] or "En attente de réponse…"
            details_table.add_row("Résumé", summary_text)

            tokens = self.state["current_question_tokens"]
            details_table.add_row(
                "Jetons (dernier)",
                f"{tokens['total']} (in {tokens['input']}, out {tokens['output']})",
            )

            details_panel = Panel(
                details_table,
                title="Contexte",
                border_style="magenta",
                box=box.ROUNDED,
            )

        if self._messages:
            messages_renderable = Group(*self._messages)
        else:
            messages_renderable = Text("En attente d'événements…", style="dim")

        log_panel = Panel(
            messages_renderable,
            title="Journal",
            border_style="green",
            box=box.ROUNDED,
        )

        if not show_details and not show_log:
            return summary_panel

        if not show_log:
            if details_panel is None:
                return summary_panel
            return Group(summary_panel, details_panel)

        left_content = Group(summary_panel, details_panel) if details_panel else summary_panel

        layout = Table.grid(expand=True)
        layout.add_column(ratio=2)
        layout.add_column(ratio=1)
        layout.add_row(left_content, log_panel)
        return layout

    # Helpers -----------------------------------------------------------------------
    def __init__(
        self,
        *,
        console: Optional[Console] = None,
        history_size: int = 6,
    ) -> None:
        self.console = console or Console()
        self.state: Dict[str, Any] = {
            "total_expected": 0,
            "pages_with_regions": 0,
            "total_pages": None,
            "questions_processed": 0,
            "questions_total": 0,
            "current_page": None,
            "current_page_regions": 0,
            "current_page_questions": 0,
            "current_question_id": None,
            "current_question_kind": "",
            "current_question_summary": "",
            "current_question_position": 0,
            "current_question_total": 0,
            "current_question_tokens": {"input": 0, "output": 0, "total": 0},
            "overall_tokens": {"input": 0, "output": 0, "total": 0},
        }
        self._messages: Deque[Text] = deque(maxlen=history_size)
        self._live: Optional[Live] = None
        self._pages_seen: set[int] = set()
        self._last_size: Optional[tuple[int, int]] = None

    @staticmethod
    def _placeholder() -> Text:
        return Text("—", style="dim")

    @staticmethod
    def _progress_widget_static(completed: Any, total: Any, *, width: int = 24):
        try:
            total_val = int(total)
            completed_val = int(completed)
        except (TypeError, ValueError):
            return RichAnalysisProgress._placeholder()

        if total_val <= 0:
            return RichAnalysisProgress._placeholder()

        total_val = max(total_val, 1)
        completed_val = max(0, min(completed_val, total_val))
        bar = ProgressBar(total=total_val, completed=completed_val, width=width)
        percent = (completed_val / total_val) * 100
        grid = Table.grid(padding=(0, 1))
        grid.add_column()
        grid.add_column(style="cyan", justify="right", width=6, no_wrap=True)
        grid.add_row(bar, Text(f"{percent:5.1f}%", style="cyan"))
        return grid

    def _progress_width(self) -> int:
        size = self.console.size
        width = size.width if size else 120
        if width >= 150:
            return 32
        if width >= 120:
            return 26
        if width >= 100:
            return 22
        if width >= 80:
            return 18
        return 12

    def _progress_widget(self, completed: Any, total: Any):
        width = self._progress_width()
        return RichAnalysisProgress._progress_widget_static(completed, total, width=width)

    def _log(self, message: str, *, style: str = "") -> None:
        text = Text(f"• {message}", style=style)
        if self._messages:
            last = self._messages[-1]
            if last and last.plain == text.plain:
                return
        self._messages.append(text)

    def _refresh(self) -> None:
        if self._live:
            size = self.console.size
            if size:
                current = (size.width, size.height)
                if self._last_size is None:
                    self._last_size = current
                elif self._last_size != current:
                    self._last_size = current
                    self._log(f"Dimension terminal ajustée → {current[0]}×{current[1]}", style="dim")
            self._live.update(self._render())


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
        Optional[Path],
        typer.Option("-o", "--output", help="Fichier JSON de sortie.", dir_okay=False),
    ] = None,
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
    if output:
        save_anchors(anchors_model, output)
        typer.echo(f"Ancres sauvegardées → {output}")
    else:
        typer.echo(anchors_model.json_pretty())


@app.command()
def analysis(
    responses_pdf: Annotated[
        Optional[Path],
        typer.Argument(
            help="PDF des copies scannées à analyser.",
            metavar="RESPONSES_PDF",
        ),
    ] = None,
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
    json_schema: Annotated[
        bool,
        typer.Option(
            "--json-schema",
            help="Afficher le schéma JSON de sortie et quitter.",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """
    Lancer l'analyse des copies PDF question par question.
    """
    if json_schema:
        typer.echo(json.dumps(analysis_output_schema(), ensure_ascii=False, indent=2))
        raise typer.Exit()

    if responses_pdf is None:
        raise typer.BadParameter("Aucun PDF fourni. Spécifiez RESPONSES_PDF ou utilisez --json-schema.")
    if not responses_pdf.exists() or not responses_pdf.is_file():
        raise typer.BadParameter(f"PDF introuvable ou illisible: {responses_pdf}")

    out_dir = ensure_directory(output)
    try:
        anchors_model = load_or_extract_anchors(
            anchors_path=anchors_path,
            source_pdf=source_pdf,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    client = build_openai_client()
    analysis_path = out_dir / "analysis.json"
    console = typer_console
    progress_reporter = RichAnalysisProgress(console=console)

    try:
        with progress_reporter:
            result = run_analysis(
                responses_pdf=responses_pdf,
                anchors=anchors_model,
                output_dir=out_dir,
                client=client,
                model=model,
                dpi=dpi,
                prompt_path=prompt_path or PROMPT_PATH,
                progress=progress_reporter,
            )
    except KeyboardInterrupt as exc:
        console.print(
            "\n[bold yellow]Analyse interrompue par l'utilisateur. Résultats partiels sauvegardés.[/]"
        )
        console.print(f"[bold]Fichier JSON[/bold] : {analysis_path}")
        raise typer.Exit(code=1) from exc

    question_range = "N/A"
    if result.question_ids:
        first = result.question_ids[0]
        last = result.question_ids[-1]
        question_range = f"{first}..{last}" if first != last else str(first)

    missing_display = "aucune"
    if result.missing_question_ids:
        missing_display = ", ".join(str(q) for q in result.missing_question_ids)

    ambiguous_display = "aucune"
    if result.ambiguous_question_ids:
        ambiguous_display = ", ".join(str(q) for q in result.ambiguous_question_ids)

    usage = result.usage
    duration_text = _format_duration(result.elapsed_seconds)

    summary_table = Table(
        title="Synthèse",
        box=box.SIMPLE_HEAVY,
        show_edge=True,
        expand=True,
    )
    summary_table.add_column("Indicateur", style="cyan", no_wrap=True)
    summary_table.add_column("Valeur", justify="right")
    summary_table.add_column("Progression", justify="left")

    pages_progress_total = result.total_pages or 0
    summary_table.add_row(
        "Pages analysées",
        f"{result.pages_processed}/{result.total_pages} (dont {result.pages_with_regions} avec régions)",
        RichAnalysisProgress._progress_widget_static(
            result.pages_processed, pages_progress_total, width=22
        ),
    )
    questions_total = (
        result.total_regions_expected
        or len(result.expected_question_ids)
        or result.questions_processed
    )
    summary_table.add_row(
        "Questions traitées",
        f"{result.questions_processed} (plage {question_range})",
        RichAnalysisProgress._progress_widget_static(
            result.questions_processed,
            questions_total,
            width=22,
        ),
    )
    summary_table.add_row("Questions manquantes", missing_display, RichAnalysisProgress._placeholder())
    summary_table.add_row("Ambiguïtés", ambiguous_display, RichAnalysisProgress._placeholder())
    summary_table.add_row(
        "Jetons consommés",
        f"{usage.total_tokens} (entrée {usage.input_tokens} / sortie {usage.output_tokens})",
        RichAnalysisProgress._placeholder(),
    )
    summary_table.add_row("Durée totale", duration_text, RichAnalysisProgress._placeholder())

    console.print()
    console.print(summary_table)
    console.print(f"[bold green]Fichier JSON[/bold green] : {analysis_path}")


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
