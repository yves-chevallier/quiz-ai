
import os
import typer
import shellingham
from pathlib import Path
from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from typing_extensions import Annotated

app = typer.Typer(
    help="Pipeline d'analyse et de grading de copies PDF",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()

def provide_default():
    if os.name == 'posix':
        return os.environ['SHELL']
    elif os.name == 'nt':
        return os.environ['COMSPEC']
    raise NotImplementedError(f'OS {os.name!r} support not available')

try:
    shell = shellingham.detect_shell()
except shellingham.ShellDetectionFailure:
    shell = provide_default()

import typer
from pathlib import Path
from typing import Annotated, Literal

app = typer.Typer()

@app.command()
def report(
    # Positional argument: input YAML file
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the configuration file (.yml)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],

    # Option: output report path
    output: Annotated[
        Path,
        typer.Option(
            "-o", "--output",
            help="Path to the output report file",
            writable=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = Path("report.md"),

    # Option: output format (limited choices)
    format: Annotated[
        Literal["md", "csv", "xlsx"],
        typer.Option(
            "-f", "--format",
            help="Report format (md, csv, xlsx)",
            show_choices=True,
        ),
    ] = "md",
):
    """
    Generate a report from a YAML configuration file.
    """
    typer.echo(f"Input file: {input_file}")
    typer.echo(f"Output file: {output}")
    typer.echo(f"Format: {format}")





if __name__ == "__main__":
    app()