"""Rendering a FitReport for a human. Spec 8: "the CLI output needs to be genuinely readable".

Separate from `cli.py` because the ordering choices here are decisions, not
plumbing, and they are what make the command worth running daily:

* The recommendation comes first. It is the question you opened the tool to ask.
* Gaps come before coverage. What you are missing is actionable; what you have
  is reassurance, and reassurance can wait.
* Must-have gaps are separated from nice-to-have gaps, because one of those
  kinds decides whether to apply and the other decides what to study.
* Inferred requirements are marked, so you never think you failed to meet
  something the posting never actually asked for.
"""

from __future__ import annotations

import typer

from resume_agent.analyze import AnalysisResult
from resume_agent.models.fit import FitReport
from resume_agent.models.job import Requirement

RECOMMENDATION_STYLE = {
    "strong_apply": (typer.colors.GREEN, "STRONG APPLY"),
    "apply": (typer.colors.GREEN, "APPLY"),
    "stretch": (typer.colors.YELLOW, "STRETCH"),
    "skip": (typer.colors.RED, "SKIP"),
}


def _bar(fraction: float, width: int = 24) -> str:
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)


def _requirement_line(requirement: Requirement) -> str:
    kind = "must-have" if requirement.is_must_have else "nice-to-have"
    inferred = "  (inferred)" if requirement.is_inferred else ""
    return f"[{requirement.weight}] {kind:<12} {requirement.text}{inferred}"


def render_fit_report(result: AnalysisResult) -> None:
    """Print the whole analysis."""
    job = result.job
    fit: FitReport = result.fit

    colour, label = RECOMMENDATION_STYLE[fit.recommendation]

    typer.echo()
    typer.secho(f"{job.title}", bold=True)
    typer.echo(f"{job.company} - {job.seniority} - {job.domain}")
    typer.echo()

    must_haves = job.must_haves()
    covered_texts = {m.requirement_text for m in fit.covered}
    covered_must_haves = sum(1 for r in must_haves if r.text in covered_texts)

    typer.secho(f"  {label}", fg=colour, bold=True)
    typer.echo(f"  fit {fit.overall_fit:.0%}  [{_bar(fit.overall_fit)}]")
    if must_haves:
        typer.echo(f"  must-haves covered: {covered_must_haves}/{len(must_haves)}")
    typer.echo()

    # --- gaps first: this is the actionable half ----------------------------
    must_gaps = fit.must_have_gaps()
    nice_gaps = [r for r in fit.gaps if not r.is_must_have]

    if must_gaps:
        typer.secho(f"MISSING - must-haves ({len(must_gaps)})", fg=typer.colors.RED, bold=True)
        for requirement in must_gaps:
            typer.secho(f"  {_requirement_line(requirement)}", fg=typer.colors.RED)
        typer.echo()

    if nice_gaps:
        typer.secho(f"Missing - nice-to-haves ({len(nice_gaps)})", fg=typer.colors.YELLOW)
        for requirement in nice_gaps:
            typer.echo(f"  {_requirement_line(requirement)}")
        typer.echo()

    if fit.partial:
        typer.secho(f"Partial evidence ({len(fit.partial)})", fg=typer.colors.YELLOW)
        for match in fit.partial:
            typer.echo(f"  {match.relevance:.2f}  {match.requirement_text}")
            typer.secho(f"        {match.rationale}", fg=typer.colors.BRIGHT_BLACK)
        typer.echo()

    if fit.covered:
        typer.secho(f"Covered ({len(fit.covered)})", fg=typer.colors.GREEN)
        for match in fit.covered:
            typer.echo(f"  {match.relevance:.2f}  {match.requirement_text}")
            typer.secho(
                f"        {match.bullet_id}: {match.rationale}", fg=typer.colors.BRIGHT_BLACK
            )
        typer.echo()

    if job.red_flags:
        typer.secho("Red flags in the posting", fg=typer.colors.MAGENTA)
        for flag in job.red_flags:
            typer.echo(f"  - {flag}")
        typer.echo()

    _render_selection(result)


def _render_selection(result: AnalysisResult) -> None:
    selection = result.selection
    typer.secho("Selected for the resume", bold=True)
    typer.echo(
        f"  {len(selection.selected_bullet_ids)} bullets, "
        f"{selection.total_estimated_lines}/{selection.line_budget} lines "
        f"(from {result.candidates_considered} candidates)"
    )
    for bullet_id in selection.selected_bullet_ids:
        typer.echo(f"    {bullet_id}")

    # Printed because "why is my best bullet missing?" is the first question
    # anyone asks of a selector, and answering it should not require a debugger.
    if selection.rejected:
        typer.echo()
        typer.secho("Not selected", fg=typer.colors.BRIGHT_BLACK)
        for bullet_id, reason in sorted(selection.rejected.items()):
            typer.secho(f"    {bullet_id}: {reason}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo()
