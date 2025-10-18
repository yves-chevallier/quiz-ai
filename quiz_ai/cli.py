"""
Typer command-line interface exposing the Quiz AI pipeline.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Annotated, Any, Deque, Dict, List, Optional, Tuple

import typer
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from .analysis import (
    PROMPT_PATH,
    analysis_output_schema,
    load_or_extract_anchors,
    run_analysis,
)
from .anchors import extract_anchors, load_anchors, save_anchors
from .annotate import annotate_pdf
from .feedback import (
    FeedbackInputs,
    build_feedback_inputs,
    generate_feedback_email,
    load_grading_file,
    resolve_student_name,
    write_feedback_file,
)
from .grading import (
    Solution,
    compute_points_from_grades,
    load_solution,
    run_grading,
    solution_points_map,
)
from .latex import write_latex_from_yaml
from .llm import DEFAULT_VISION_MODEL, build_openai_client
from .report import (
    GradeSummary,
    build_markdown_report,
    summarise_grade,
    write_summary_csv,
)
from .utils import ensure_directory, read_json, write_json

try:
    from typer.rich_utils import console as typer_console
except ImportError:  # pragma: no cover - fallback for older Typer versions
    typer_console = Console()


app = typer.Typer(
    help="CLI tools to convert, analyse, and grade scanned quizzes.",
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
        self._last_size: Optional[tuple[int, int]] = None

    def __enter__(self) -> "RichAnalysisProgress":
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=6,
        )
        self._live.__enter__()
        self._log("Analysis initialised. Press CTRL+C to interrupt.", style="bold cyan")
        self._refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is KeyboardInterrupt:
            self._log("User interruption detected.", style="bold yellow")
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
                f"{self.state['total_expected']} expected region(s) across "
                f"{self.state['pages_with_regions']} anchored page(s).",
                style="cyan",
            )
            return

        if event == "run:pages_ready":
            total_pages = payload.get("total_pages")
            if isinstance(total_pages, int):
                self.state["total_pages"] = total_pages
            self._log(f"PDF rasterised → {total_pages} page(s).", style="cyan")
            return

        if event == "page:start":
            page_number = payload.get("page_number")
            self.state["current_page"] = page_number
            self.state["current_page_regions"] = 0
            self.state["current_page_questions"] = 0
            self._log(f"Page {page_number}: preparing.", style="magenta")
            return

        if event == "page:rendered":
            image_path = payload.get("image_path")
            filename = Path(image_path).name if image_path else "image"
            self._log(f"Page rendered → {filename}", style="magenta")
            return

        if event == "page:regions":
            count = int(payload.get("count", 0))
            self.state["current_page_regions"] = count
            self._log(f"Detected regions: {count}", style="magenta")
            return

        if event == "page:crops-ready":
            count = int(payload.get("count", 0))
            self.state["current_page_regions"] = count
            self._log(f"Cropping finished: {count} region(s).", style="magenta")
            return

        if event == "page:skip":
            reason = payload.get("reason") or "motif inconnu"
            mapping = {
                "no-anchors": "no matching anchors",
                "no-regions": "no regions defined",
                "invalid-page-height": "invalid page height",
            }
            detail = mapping.get(reason, reason)
            self._log(f"Page skipped ({detail}).", style="yellow")
            return

        if event == "page:process":
            count = int(payload.get("question_count", 0))
            self.state["current_page_questions"] = count
            self._log(f"Processing {count} detected region(s).", style="magenta")
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
            self._log("  ↳ Request sent to the model…", style="green")
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
                f"  ↳ Response received: {kind or 'question'} | {summary}",
                style="green",
            )
            self._log(
                f"    Tokens {total_tokens} (in {in_tokens}, out {out_tokens})",
                style="dim",
            )
            return

        if event == "run:complete":
            self._log("Analysis completed.", style="bold green")
            return

        # Unknown events fallback
        self._log(f"{event} → {payload}", style="dim")

    # Rendering ---------------------------------------------------------------------
    def _render(self):
        size = self.console.size
        term_width = size.width if size else 120

        total_expected = self.state["total_expected"] or self.state["questions_total"] or 0
        processed = self.state["questions_processed"]
        expected_display = f"{processed}/{total_expected}" if total_expected else f"{processed}/-"

        total_pages = self.state["total_pages"] or 0
        pages_seen = len(self._pages_seen)
        pages_display = f"{pages_seen}/{total_pages}" if total_pages else f"{pages_seen}/-"

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
            f"{pages_display} (with regions {self.state['pages_with_regions']})",
            self._progress_widget(pages_seen, total_pages),
        )

        overall_tokens = self.state["overall_tokens"]
        summary_table.add_row(
            "Tokens",
            f"{overall_tokens['total']} (in {overall_tokens['input']}, out {overall_tokens['output']})",
            self._placeholder(),
        )

        summary_panel = Panel(
            summary_table,
            title="Progress",
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
                    f"{self.state['current_page_regions']} detected region(s)"
                )
                if self.state["current_page_questions"]:
                    page_value += f" • {self.state['current_page_questions']} in progress"
            details_table.add_row("Current page", page_value)

            question_id = self.state["current_question_id"]
            if question_id is None:
                question_value = "-"
            else:
                total = self.state["current_question_total"] or self.state["questions_total"] or "-"
                kind = self.state["current_question_kind"] or "unknown type"
                question_value = (
                    f"Q{question_id} ({kind}) • {self.state['current_question_position']}/{total}"
                )
            details_table.add_row("Active question", question_value)

            summary_text = self.state["current_question_summary"] or "Waiting for response…"
            details_table.add_row("Summary", summary_text)

            tokens = self.state["current_question_tokens"]
            details_table.add_row(
                "Tokens (latest)",
                f"{tokens['total']} (in {tokens['input']}, out {tokens['output']})",
            )

            details_panel = Panel(
                details_table,
                title="Context",
                border_style="magenta",
                box=box.ROUNDED,
            )

        if self._messages:
            messages_renderable = Group(*self._messages)
        else:
            messages_renderable = Text("Waiting for events…", style="dim")

        log_panel = Panel(
            messages_renderable,
            title="Log",
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
                    self._log(
                        f"Terminal size changed → {current[0]}×{current[1]}",
                        style="dim",
                    )
            self._live.update(self._render())


@app.command()
def latex(
    quiz_yaml: Annotated[
        Path,
        typer.Argument(help="Quiz YAML description file.", exists=True, readable=True),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Path to the generated LaTeX file."),
    ] = None,
) -> None:
    """
    Convert a quiz YAML file into a LaTeX document.
    """
    tex_path = output or quiz_yaml.with_suffix(".tex")
    write_latex_from_yaml(quiz_yaml, tex_path)
    typer.echo(f"LaTeX generated → {tex_path}")


@app.command()
def anchors(
    pdf: Annotated[
        Path,
        typer.Argument(help="Source PDF containing hyperref anchors.", exists=True, readable=True),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Output JSON file path.", dir_okay=False),
    ] = None,
    overlap: Annotated[
        float,
        typer.Option(
            "--overlap",
            min=0.0,
            help="Vertical overlap in millimetres between consecutive anchors.",
        ),
    ] = 3.0,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Regular expression selecting anchors."),
    ] = r"^Q(\d+)-anchor$",
    include_bottom: Annotated[
        bool,
        typer.Option(
            "--include-bottom/--no-include-bottom",
            help="Include the segment below the last anchor.",
        ),
    ] = True,
) -> None:
    """
    Extract anchors from a PDF and serialise them to JSON.
    """
    anchors_model = extract_anchors(
        pdf,
        overlap=overlap,
        anchor_pattern=pattern,
        include_bottom_segment=include_bottom,
    )
    if output:
        save_anchors(anchors_model, output)
        typer.echo(f"Anchors saved → {output}")
    else:
        typer.echo(anchors_model.json_pretty())


@app.command()
def analysis(
    responses_pdf: Annotated[
        Optional[Path],
        typer.Argument(
            help="Scanned responses PDF to analyse.",
            metavar="RESPONSES_PDF",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output directory for the analysis artefacts."),
    ] = Path("analysis"),
    anchors_path: Annotated[
        Optional[Path],
        typer.Option(
            "-a",
            "--anchors",
            help="Pre-calculated anchors JSON file.",
            exists=True,
            readable=True,
        ),
    ] = None,
    source_pdf: Annotated[
        Optional[Path],
        typer.Option(
            "-i",
            "--input",
            help="Source PDF used to extract anchors if required.",
            exists=True,
            readable=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI vision model to call."),
    ] = DEFAULT_VISION_MODEL,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=60, max=600, help="Rendering DPI for PDF rasterisation."),
    ] = 220,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option(
            "--prompt",
            help="Custom analysis prompt Markdown file.",
            exists=True,
            readable=True,
        ),
    ] = None,
    title_prompt_path: Annotated[
        Optional[Path],
        typer.Option(
            "--title-prompt",
            help="Prompt Markdown used to extract cover page metadata.",
            exists=True,
            readable=True,
        ),
    ] = None,
    json_schema: Annotated[
        bool,
        typer.Option(
            "--json-schema",
            help="Print the JSON output schema and exit.",
            is_flag=True,
        ),
    ] = False,
) -> None:
    """
    Run the visual analysis for a PDF of scanned responses.
    """
    if json_schema:
        typer.echo(json.dumps(analysis_output_schema(), ensure_ascii=False, indent=2))
        raise typer.Exit()

    if responses_pdf is None:
        raise typer.BadParameter("No PDF provided. Pass RESPONSES_PDF or use --json-schema.")
    if not responses_pdf.exists() or not responses_pdf.is_file():
        raise typer.BadParameter(f"PDF not found or unreadable: {responses_pdf}")

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
                title_prompt_path=title_prompt_path,
                progress=progress_reporter,
            )
    except KeyboardInterrupt as exc:
        console.print("\n[bold yellow]Analysis interrupted by the user. Partial results saved.[/]")
        console.print(f"[bold]JSON file[/bold] : {analysis_path}")
        raise typer.Exit(code=1) from exc

    question_range = "N/A"
    if result.question_ids:
        first = result.question_ids[0]
        last = result.question_ids[-1]
        question_range = f"{first}..{last}" if first != last else str(first)

    missing_display = "none"
    if result.missing_question_ids:
        missing_display = ", ".join(str(q) for q in result.missing_question_ids)

    ambiguous_display = "none"
    if result.ambiguous_question_ids:
        ambiguous_display = ", ".join(str(q) for q in result.ambiguous_question_ids)

    usage = result.usage
    duration_text = _format_duration(result.elapsed_seconds)

    summary_table = Table(
        title="Summary",
        box=box.SIMPLE_HEAVY,
        show_edge=True,
        expand=True,
    )
    summary_table.add_column("Indicator", style="cyan", no_wrap=True)
    summary_table.add_column("Value", justify="right")
    summary_table.add_column("Progress", justify="left")

    pages_progress_total = max(result.pages_with_regions, result.pages_processed, 1)
    pages_with_regions = result.pages_with_regions or result.pages_processed or result.total_pages or 1
    summary_table.add_row(
        "Pages processed",
        (
            f"{result.pages_processed}/{pages_with_regions} page(s) with regions"
            f" (full PDF: {result.total_pages})"
        ),
        RichAnalysisProgress._progress_widget_static(result.pages_processed, pages_progress_total, width=22),
    )
    questions_total = result.total_regions_expected or len(result.expected_question_ids) or result.questions_processed
    summary_table.add_row(
        "Questions processed",
        f"{result.questions_processed} (range {question_range})",
        RichAnalysisProgress._progress_widget_static(
            result.questions_processed,
            questions_total,
            width=22,
        ),
    )
    summary_table.add_row("Missing questions", missing_display, RichAnalysisProgress._placeholder())
    summary_table.add_row("Ambiguities", ambiguous_display, RichAnalysisProgress._placeholder())
    summary_table.add_row(
        "Tokens consumed",
        f"{usage.total_tokens} (input {usage.input_tokens} / output {usage.output_tokens})",
        RichAnalysisProgress._placeholder(),
    )
    summary_table.add_row("Total duration", duration_text, RichAnalysisProgress._placeholder())

    console.print()
    console.print(summary_table)
    console.print(f"[bold green]JSON file[/bold green] : {analysis_path}")


@app.command()
def grading(
    analysis_json: Annotated[
        Path,
        typer.Argument(
            help="JSON file produced by the visual analysis.",
            exists=True,
            readable=True,
        ),
    ],
    quiz_yaml: Annotated[
        Path,
        typer.Argument(
            help="Quiz YAML source (official solutions).",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Output directory for grading artefacts.",
            file_okay=False,
        ),
    ] = Path("out"),
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model to use for grading."),
    ] = DEFAULT_VISION_MODEL,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option(
            "--prompt",
            help="Custom grading prompt Markdown file.",
            exists=True,
            readable=True,
        ),
    ] = None,
    user_label: Annotated[
        Optional[str],
        typer.Option("--user", help="User label forwarded to the API."),
    ] = None,
) -> None:
    """
    Run automatic grading from an analysis JSON and the quiz YAML.
    """
    out_dir = ensure_directory(output)
    grade_path = out_dir / "grading.json"

    analysis_data = read_json(analysis_json)
    analysis_metadata = analysis_data.get("metadata") if isinstance(analysis_data, dict) else {}
    solution = load_solution(quiz_yaml)
    client = build_openai_client()
    solution_questions = solution.questions
    labels_map = {qid: entry.get("label", "") for qid, entry in solution_questions.items()}

    grades = run_grading(
        analysis=analysis_data,
        solution=solution,
        client=client,
        model=model,
        user_label=user_label,
        prompt_path=prompt_path,
    )

    points_mapping = solution_points_map(solution)
    total_questions = len(points_mapping) or len(solution_questions)
    questions = grades.get("questions")
    if isinstance(questions, list):
        normalised_questions = []
        for entry in questions:
            if not isinstance(entry, dict):
                continue
            try:
                qid = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            max_points = float(points_mapping.get(qid, 0.0))
            ratio_value = entry.get("awarded_ratio")
            if ratio_value is None:
                ratio_value = entry.get("granted_ratio", 0.0)
            try:
                ratio = float(ratio_value)
            except (TypeError, ValueError):
                ratio = 0.0
            ratio = max(0.0, min(1.0, ratio))
            entry["id"] = qid
            entry["max_points"] = max_points
            entry["awarded_ratio"] = ratio
            entry["awarded_points"] = round(max_points * ratio, 4)
            label = entry.get("label")
            if not isinstance(label, str) or not label.strip():
                entry["label"] = labels_map.get(qid) or f"Question {qid}"
            status = entry.get("status")
            if not isinstance(status, str):
                if ratio == 0.0:
                    entry["status"] = "incorrect"
                elif ratio == 1.0:
                    entry["status"] = "correct"
                else:
                    entry["status"] = "partial"
            flags = entry.get("flags")
            if not isinstance(flags, list):
                flags = []
            cleaned_flags: List[str] = []
            for flag in flags:
                if isinstance(flag, str):
                    text = flag.strip()
                    if text:
                        cleaned_flags.append(text)
                elif isinstance(flag, (int, float)):
                    cleaned_flags.append(str(flag))
            entry["flags"] = cleaned_flags
            normalised_questions.append(entry)
        grades["questions"] = sorted(normalised_questions, key=lambda item: item["id"])
        total_questions = len(normalised_questions)
    else:
        grades["questions"] = []

    obtained, total = compute_points_from_grades(grades, solution)
    percentage = (obtained / total * 100.0) if total else 0.0

    score_block = grades.get("score")
    if not isinstance(score_block, dict):
        score_block = {}
    score_block["points_obtained"] = round(obtained, 4)
    score_block["points_total"] = round(total, 4)
    score_block["percentage"] = round(percentage, 2)
    score_block["total_questions"] = total_questions
    grades["score"] = score_block

    # Ensure high-level metadata is present
    student_block = grades.get("student")
    if not isinstance(student_block, dict):
        student_block = {}
    metadata_name = ""
    metadata_date = ""
    if isinstance(analysis_metadata, dict):
        metadata_name = str(analysis_metadata.get("student_name") or "").strip()
        metadata_date = str(analysis_metadata.get("grading_date") or "").strip()
    student_block.setdefault("identifier", "")
    student_block.setdefault("name", metadata_name or "")
    student_block.setdefault("date", metadata_date or "")
    if metadata_name and not student_block.get("name"):
        student_block["name"] = metadata_name
    if metadata_date and not student_block.get("date"):
        student_block["date"] = metadata_date
    grades["student"] = student_block

    quiz_block = grades.get("quiz")
    if not isinstance(quiz_block, dict):
        quiz_block = {}
    raw_meta = solution.raw.get("meta") if isinstance(solution.raw, dict) else {}
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    title = quiz_block.get("title") or raw_meta.get("title") or ""
    source_reference = (
        quiz_block.get("source_reference") or raw_meta.get("code") or (solution.path.stem if solution.path else "")
    )
    quiz_block["title"] = title
    quiz_block["source_reference"] = source_reference
    quiz_block["total_questions"] = total_questions
    grades["quiz"] = quiz_block

    if not isinstance(grades.get("final_report"), str) or not grades["final_report"].strip():
        grades["final_report"] = (
            "No narrative report was provided by the model. Numeric results remain available."
        )

    grades["_source_analysis"] = str(analysis_json)
    grades["_source_quiz"] = str(quiz_yaml)
    grades["_points_obtained"] = round(obtained, 4)
    grades["_points_total"] = round(total, 4)
    if isinstance(analysis_metadata, dict):
        grades["_analysis_metadata"] = analysis_metadata

    write_json(grade_path, grades)
    typer.echo(f"Grading saved → {grade_path} ({obtained:.2f}/{total:.2f} pts, {percentage:.1f} %)")


@app.command("grade-one")
def grade_one(
    analysis_json: Annotated[
        Path,
        typer.Argument(
            help="Analysis JSON produced by the analysis command.",
            exists=True,
            readable=True,
        ),
    ],
    quiz_yaml: Annotated[
        Path,
        typer.Option(
            "-q",
            "--quiz",
            help="YAML file with the official solutions.",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output grading JSON file.", dir_okay=False),
    ] = Path("grade.json"),
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI text model to use for grading."),
    ] = DEFAULT_VISION_MODEL,
) -> None:
    """
    Convert a single analysis file into grading results using the LLM.
    """
    analysis_data = read_json(analysis_json)
    analysis_metadata = analysis_data.get("metadata") if isinstance(analysis_data, dict) else {}
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
    if isinstance(analysis_metadata, dict):
        grades["_analysis_metadata"] = analysis_metadata

    student_block = grades.get("student")
    if not isinstance(student_block, dict):
        student_block = {}
    metadata_name = ""
    metadata_date = ""
    if isinstance(analysis_metadata, dict):
        metadata_name = str(analysis_metadata.get("student_name") or "").strip()
        metadata_date = str(analysis_metadata.get("grading_date") or "").strip()
    student_block.setdefault("identifier", "")
    student_block.setdefault("name", metadata_name or "")
    student_block.setdefault("date", metadata_date or "")
    if metadata_name and not student_block.get("name"):
        student_block["name"] = metadata_name
    if metadata_date and not student_block.get("date"):
        student_block["date"] = metadata_date
    grades["student"] = student_block

    write_json(output, grades)
    typer.echo(f"Grading saved → {output} ({obtained:.2f}/{total:.2f} pts)")


@app.command()
def report(
    grade_files: Annotated[
        List[Path],
        typer.Argument(
            help="One or more grading JSON files.",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Path to the generated report (Markdown by default)."),
    ] = Path("report.md"),
    format: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Report format (md or csv).",
            show_choices=True,
        ),
    ] = "md",
    quiz_yaml: Annotated[
        Optional[Path],
        typer.Option(
            "--quiz",
            help="Solution YAML file to recompute scores.",
            exists=True,
            readable=True,
        ),
    ] = None,
) -> None:
    """
    Generate a report from one or more grading files.
    """
    summaries = []
    solution: Optional[Solution] = load_solution(quiz_yaml) if quiz_yaml else None

    if format == "csv":
        if solution is None:
            raise typer.BadParameter("CSV output requires --quiz to know the question weighting.")
        summaries = write_summary_csv(grade_files, solution, output)
        typer.echo(f"Summary CSV generated → {output}")
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
    typer.echo(f"Markdown report generated → {output}")


@app.command()
def feedback(
    grading_json: Annotated[
        Path,
        typer.Argument(
            help="Grading JSON file produced by the grade command.",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--output",
            help="Markdown file where the feedback email will be written.",
            dir_okay=False,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI model to use for feedback generation."),
    ] = DEFAULT_VISION_MODEL,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option(
            "--prompt",
            help="Custom prompt used to craft the email.",
            exists=True,
            readable=True,
        ),
    ] = None,
    user_label: Annotated[
        Optional[str],
        typer.Option("--user", help="User label forwarded to the API."),
    ] = None,
) -> None:
    """
    Generate a motivational feedback email from a grading file.
    """
    grades = load_grading_file(grading_json)
    client = build_openai_client()

    resolved_name = resolve_student_name(grading_json, grades)

    student_block = grades.get("student")
    if resolved_name:
        if not isinstance(student_block, dict):
            student_block = {}
        student_block["name"] = resolved_name
        grades["student"] = student_block
    elif not isinstance(student_block, dict):
        grades["student"] = {}

    payload = build_feedback_inputs(grades)
    if resolved_name and not payload.student_name:
        payload = FeedbackInputs(
            student_name=resolved_name,
            score_points=payload.score_points,
            score_total=payload.score_total,
            score_percentage=payload.score_percentage,
            quiz_title=payload.quiz_title,
            final_report=payload.final_report,
            positive_topics=payload.positive_topics,
            improvement_topics=payload.improvement_topics,
        )

    email_text = generate_feedback_email(
        payload,
        client=client,
        model=model,
        prompt_path=prompt_path,
        user_label=user_label,
    )

    feedback_path = output or grading_json.with_name("feedback.md")
    write_feedback_file(feedback_path, email_text)

    if resolved_name:
        typer.echo(f"Feedback generated for {resolved_name} → {feedback_path}")
    else:
        typer.echo(f"Feedback generated (student name unresolved) → {feedback_path}")


def _build_feedback_payload(
    grades: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[Tuple[float, float]]]:
    """Transform grading data into annotate-ready feedback payloads."""
    feedback: List[Dict[str, Any]] = []
    for q in grades.get("questions", []):
        q_id = q.get("id")
        if q_id is None:
            continue
        status = str(q.get("status") or "").lower()
        ratio_value: Optional[float] = None
        raw_ratio = q.get("awarded_ratio")
        try:
            if raw_ratio is not None:
                ratio_value = float(raw_ratio)
        except (TypeError, ValueError):
            ratio_value = None
        if ratio_value is None:
            try:
                ap = q.get("awarded_points")
                mp = q.get("max_points")
                if ap is not None and mp not in (None, 0):
                    ratio_value = float(ap) / float(mp)  # type: ignore[arg-type]
            except (TypeError, ValueError, ZeroDivisionError):
                ratio_value = None

        if not status:
            if ratio_value is not None and ratio_value >= 0.999:
                status = "correct"
            elif ratio_value is not None and ratio_value <= 0.001:
                status = "incorrect"
            elif ratio_value is not None:
                status = "partial"
            else:
                status = "unknown"

        feedback.append(
            {
                "id": q_id,
                "status": status,
                "awarded_points": q.get("awarded_points"),
                "max_points": q.get("max_points"),
                "awarded_ratio": q.get("awarded_ratio"),
                "remarks": q.get("remarks"),
                "justification": q.get("justification"),
                "comment": q.get("comment"),
                "flags": q.get("flags"),
            }
        )

    score_block = grades.get("score") or {}
    try:
        obtained = float(score_block.get("points_obtained"))
        total = float(score_block.get("points_total"))
    except (TypeError, ValueError):
        overall_points = None
    else:
        overall_points = (obtained, total) if total else None

    return feedback, overall_points


@app.command()
def annotate(
    pdf_input: Annotated[
        Path,
        typer.Argument(help="Original PDF of the scanned responses.", exists=True, readable=True),
    ],
    grades_json: Annotated[
        Path,
        typer.Option(
            "-g",
            "--grades",
            help="Grading JSON file containing remarks and corrections.",
            exists=True,
            readable=True,
        ),
    ],
    anchors_json: Annotated[
        Path,
        typer.Option("-a", "--anchors", help="Anchors JSON file.", exists=True, readable=True),
    ],
    pdf_output: Annotated[
        Path,
        typer.Argument(
            help="Annotated PDF to generate.",
            dir_okay=False,
        ),
    ] = Path("annotated.pdf"),
) -> None:
    """
    Produce an annotated PDF from the LLM corrections.
    """
    anchors_model = load_anchors(anchors_json)
    grades = read_json(grades_json)
    feedback, overall_points = _build_feedback_payload(grades)

    annotate_pdf(
        pdf_input=pdf_input,
        pdf_output=pdf_output,
        anchors=anchors_model,
        feedback=feedback,
        overall_points=overall_points,
    )
    typer.echo(f"Annotated PDF generated → {pdf_output}")


@app.command()
def grade(
    responses_pdf: Annotated[
        Path,
        typer.Argument(help="PDF of the scanned responses.", exists=True, readable=True),
    ],
    quiz_yaml: Annotated[
        Path,
        typer.Option(
            "-q",
            "--quiz",
            help="YAML solution file.",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Global output directory."),
    ] = Path("grade"),
    anchors_path: Annotated[
        Optional[Path],
        typer.Option(
            "-a",
            "--anchors",
            help="Existing anchors JSON file.",
            exists=True,
            readable=True,
        ),
    ] = None,
    source_pdf: Annotated[
        Optional[Path],
        typer.Option(
            "-i",
            "--input",
            help="Source PDF used to extract anchors.",
            exists=True,
            readable=True,
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="OpenAI multimodal model (vision + text)."),
    ] = DEFAULT_VISION_MODEL,
    report_path: Annotated[
        Optional[Path],
        typer.Option("--report", help="Write a Markdown report at this path."),
    ] = None,
    annotate_pdf_flag: Annotated[
        bool,
        typer.Option("--annotate/--no-annotate", help="Also produce an annotated PDF."),
    ] = False,
    prompt_path: Annotated[
        Optional[Path],
        typer.Option(
            "--prompt",
            help="Custom analysis prompt Markdown file.",
            exists=True,
            readable=True,
        ),
    ] = None,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=60, max=600, help="Rendering DPI for PDF rasterisation."),
    ] = 220,
) -> None:
    """
    Full pipeline: analysis + grading (with optional report/annotation).
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
    typer.echo(f"Grading completed → {grades_path} ({obtained:.2f}/{total:.2f} pts)")

    summaries = None
    if report_path:
        summaries = [summarise_grade(grades_path, solution)]
        build_markdown_report(summaries, report_path)
        typer.echo(f"Report generated → {report_path}")

    if annotate_pdf_flag:
        annotated_path = base_dir / "annotated.pdf"
        feedback, overall_points = _build_feedback_payload(grades)
        annotate_pdf(
            pdf_input=responses_pdf,
            pdf_output=annotated_path,
            anchors=anchors_model,
            feedback=feedback,
            overall_points=overall_points,
        )
        typer.echo(f"Annotated PDF generated → {annotated_path}")
