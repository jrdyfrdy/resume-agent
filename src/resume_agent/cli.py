"""Command line entry point.

M0 ships exactly one command: `build`, the deterministic pipeline end to end.

    resume-agent build --profile profile.example --out out/

No `analyze`, no `search`, no job description argument -- those need the
retrieval and scoring machinery from M1-M3 and there is nothing useful to stub.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer

from resume_agent.kb.loader import ProfileLoadError, load_profile
from resume_agent.latex.compile import INSTALL_MESSAGE, compile_tex, find_compiler
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.latex.inspect import inspect_output

RESUME_TEMPLATE = "jake_resume.tex.j2"


class ExitCode(IntEnum):
    """Distinct codes so a script calling this can tell the failures apart."""

    OK = 0
    COMPILE_FAILED = 1
    PROFILE_INVALID = 2
    NO_COMPILER = 3


app = typer.Typer(
    add_completion=False,
    help="Truthful, one-page, ATS-clean resumes from a structured career knowledge base.",
)


@app.callback()
def main() -> None:
    """Root callback.

    Present only so that `build` stays a named subcommand. Typer collapses a
    single-command app into a bare `resume-agent [OPTIONS]` invocation, which
    would make `resume-agent build` an error today and silently change the CLI's
    shape when `search` (M1) and `analyze` (M3) arrive.
    """


@app.command()
def build(
    profile: Annotated[
        Path,
        typer.Option("--profile", help="Profile directory to render.", show_default=True),
    ] = Path("profile.example"),
    out: Annotated[
        Path,
        typer.Option("--out", help="Directory for resume.tex, resume.pdf and compile.log."),
    ] = Path("out"),
) -> None:
    """Render a profile to a one-page PDF. No LLM involved."""
    try:
        loaded = load_profile(profile)
    except ProfileLoadError as exc:
        typer.secho(f"Could not load profile:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.PROFILE_INVALID) from exc

    typer.echo(
        f"Loaded {profile}: {len(loaded.experience)} roles, "
        f"{len(loaded.projects)} projects, {len(loaded.all_bullets())} bullets"
    )

    # Checked before rendering so a missing compiler is reported in a second
    # rather than after the template work, and with instructions rather than a
    # traceback (spec 6.4).
    if find_compiler() is None:
        typer.secho(INSTALL_MESSAGE, fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_COMPILER)

    tex_source = render_template(RESUME_TEMPLATE, build_resume_context(loaded))
    result = compile_tex(tex_source, out, job_name="resume")

    # The log is written whether or not the compile succeeded -- on failure it is
    # the only thing that explains why (CLAUDE.md: never swallow a LaTeX failure).
    log_path = out / "compile.log"
    log_path.write_text(result.log, encoding="utf-8")

    report = inspect_output(result)

    if not result.ok:
        typer.secho(f"Compile failed using {result.compiler}.", fg=typer.colors.RED, err=True)
        if report.first_error:
            typer.secho(report.first_error, fg=typer.colors.RED, err=True)
        typer.secho(f"Full log: {log_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(ExitCode.COMPILE_FAILED)

    typer.echo(f"Compiled with {result.compiler} -> {result.pdf_path}")
    typer.echo(f"Pages: {report.page_count}")
    typer.echo(f"Overfull hboxes: {len(report.overfull_boxes)}")
    for box in report.overfull_boxes:
        typer.secho(f"  {box.raw}", fg=typer.colors.YELLOW)

    # A warning, not an error. M0 has no selection node to fix it with -- that
    # is exactly the job of M5's layout loop. Saying so is more useful than
    # failing on something this milestone cannot act on.
    if report.page_count != 1:
        typer.secho(
            f"Warning: {report.page_count} pages. M0 renders the whole profile with no "
            f"selection step; trimming to one page is M3/M5's job.",
            fg=typer.colors.YELLOW,
        )


if __name__ == "__main__":
    app()
