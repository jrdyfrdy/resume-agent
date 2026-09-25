"""Command line entry point.

    resume-agent build  --profile profile.example --out out/   (M0)
    resume-agent index  --profile profile.example              (M1)
    resume-agent search "kubernetes" --explain                 (M1)
    resume-agent parse-jd --jd evals/datasets/jds/mid.txt       (M2)
    resume-agent analyze  --jd evals/datasets/jds/mid.txt       (M3)
    resume-agent run      --jd evals/datasets/jds/mid.txt       (M5)
    resume-agent run      --jd ... --interactive                (M7)
    resume-agent resume-run --jd ...                            (M7)
    resume-agent applications                                   (M7)
    resume-agent serve                                          (M9)
    resume-agent users list | approve <email> | reject <email>  (M10)
"""

from __future__ import annotations

import os
from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer
from langgraph.types import Command

from resume_agent.analyze import analyze as run_analysis
from resume_agent.graph.build import build_graph, initial_state
from resume_agent.graph.checkpoint import open_checkpointer, thread_config, thread_id_for
from resume_agent.graph.nodes.parse_jd import JobDescriptionParseError, parse_job_description
from resume_agent.graph.state import RunOptions
from resume_agent.kb.index import ProfileIndex, index_path_for
from resume_agent.kb.loader import ProfileLoadError, load_profile
from resume_agent.kb.retriever import HybridRetriever, SearchMode
from resume_agent.latex.compile import INSTALL_MESSAGE, compile_tex, find_compiler
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.latex.inspect import inspect_output
from resume_agent.llm import (
    MissingCredentialsError,
    credentials_message,
    has_credentials,
    resolve_provider,
)
from resume_agent.report import render_fit_report
from resume_agent.tracker.db import bullets_by_outcome, list_applications, set_outcome

RESUME_TEMPLATE = "jake_resume.tex.j2"


class ExitCode(IntEnum):
    """Distinct codes so a script calling this can tell the failures apart."""

    OK = 0
    COMPILE_FAILED = 1
    PROFILE_INVALID = 2
    NO_COMPILER = 3
    NO_CREDENTIALS = 4
    JD_PARSE_FAILED = 5
    # A flag that could never be right, as opposed to work that failed.
    USAGE = 6
    NO_SUCH_USER = 7


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



def _load_or_exit(profile: Path):
    """Load a profile, turning a load failure into a clean CLI exit."""
    try:
        return load_profile(profile)
    except ProfileLoadError as exc:
        typer.secho(f"Could not load profile:\n{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.PROFILE_INVALID) from exc


@app.command()
def index(
    profile: Annotated[
        Path, typer.Option("--profile", help="Profile directory to index.")
    ] = Path("profile.example"),
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Re-embed even if the index looks current.")
    ] = False,
) -> None:
    """Build the SQLite + vector index for a profile.

    Rarely needed by hand -- `search` rebuilds automatically when the YAML has
    changed. Useful for warming the index, or after editing the embed recipe.
    """
    loaded = _load_or_exit(profile)
    db_path = index_path_for(profile)

    built, was_rebuilt = ProfileIndex.open(loaded, profile, rebuild=rebuild)
    verb = "Rebuilt" if was_rebuilt else "Already current"
    typer.echo(f"{verb}: {db_path} ({len(built.bullets())} bullets)")
    built.close()


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    profile: Annotated[
        Path, typer.Option("--profile", help="Profile directory to search.")
    ] = Path("profile.example"),
    k: Annotated[int, typer.Option("--k", help="How many results to show.")] = 8,
    mode: Annotated[
        SearchMode,
        typer.Option("--mode", help="Fused, or one retriever alone (for comparison)."),
    ] = "hybrid",
    explain: Annotated[
        bool,
        typer.Option("--explain", help="Show each retriever's rank next to the fused score."),
    ] = False,
) -> None:
    """Search the knowledge base by hand.

    The point of this command is inspection: spec 9 notes that judges drift and
    your eye is the calibration set, so there needs to be a way to look at what
    retrieval is actually doing before anything is built on top of it.
    """
    loaded = _load_or_exit(profile)

    built, was_rebuilt = ProfileIndex.open(loaded, profile)
    if was_rebuilt:
        typer.secho(f"(rebuilt index for {profile})", fg=typer.colors.BRIGHT_BLACK)

    retriever = HybridRetriever(built, loaded)
    hits = retriever.search(query, k=k, mode=mode)

    expanded = retriever_expansion_note(retriever, query)
    if expanded:
        typer.secho(expanded, fg=typer.colors.BRIGHT_BLACK)

    if not hits:
        typer.secho(f"No results for {query!r} in {mode} mode.", fg=typer.colors.YELLOW)
        built.close()
        return

    typer.echo(f"\n{len(hits)} result(s) for {query!r} [{mode}]\n")
    for position, hit in enumerate(hits, start=1):
        if explain:
            bm25 = f"bm25 #{hit.bm25_rank}" if hit.bm25_rank else "bm25   -"
            dense = f"dense #{hit.dense_rank}" if hit.dense_rank else "dense   -"
            detail = f"  [{bm25}, {dense}]"
        else:
            detail = ""
        typer.secho(
            f"{position:2d}. {hit.score:.5f}  {hit.bullet_id}{detail}",
            fg=typer.colors.CYAN,
        )
        typer.echo(f"    {hit.entry_label}: {hit.bullet.canonical}")
        typer.echo("")

    built.close()


def retriever_expansion_note(retriever: HybridRetriever, query: str) -> str:
    """A one-line note about what the alias table added to the query, if anything."""
    from resume_agent.kb.retriever import expand_query

    terms = expand_query(query, retriever.profile)
    added = terms[1:]
    return f"(expanded with: {', '.join(added)})" if added else ""


@app.command("parse-jd")
def parse_jd(
    jd: Annotated[Path, typer.Option("--jd", help="File containing the job posting.")],
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Re-parse even if a cached result exists.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the raw JobSpec JSON instead of a summary.")
    ] = False,
) -> None:
    """Parse a job posting into a structured JobSpec.

    Results are cached on a hash of the posting, the model and the prompt, so
    re-running this while debugging downstream nodes costs nothing.
    """
    if not jd.is_file():
        typer.secho(f"No such file: {jd}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED)

    raw = jd.read_text(encoding="utf-8")

    try:
        spec, cache_hit = parse_job_description(raw, use_cache=not no_cache)
    except MissingCredentialsError:
        typer.secho(credentials_message(), fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_CREDENTIALS) from None
    except JobDescriptionParseError as exc:
        typer.secho(f"Could not parse {jd}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED) from exc

    if as_json:
        typer.echo(spec.model_dump_json(indent=2))
        return

    source = "cache" if cache_hit else "model"
    typer.secho(f"{spec.title} at {spec.company}  [{source}]", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  seniority : {spec.seniority}")
    typer.echo(f"  domain    : {spec.domain}")
    typer.echo(f"  tone      : {spec.tone}")
    typer.echo(f"  hash      : {spec.source_hash[:16]}")

    stated = spec.stated_requirements()
    inferred = spec.inferred_requirements()

    typer.echo(
        f"\nRequirements ({len(spec.must_haves())} must-have of {len(spec.requirements)}):"
    )
    for requirement in sorted(stated, key=lambda r: -r.weight):
        marker = "!" if requirement.is_must_have else " "
        typer.echo(
            f"  {marker} [{requirement.weight}] "
            f"{requirement.category:10s} {requirement.text}"
        )

    # Printed separately because the distinction is the point: these are things
    # the posting never actually asked for, and a candidate should read them
    # differently from the stated list.
    if inferred:
        typer.secho("\nInferred priorities (not stated in the posting):", fg=typer.colors.YELLOW)
        for requirement in sorted(inferred, key=lambda r: -r.weight):
            typer.echo(f"    [{requirement.weight}] {requirement.category:10s} {requirement.text}")

    if spec.ats_keywords:
        typer.echo(f"\nATS keywords: {', '.join(spec.ats_keywords)}")

    if spec.red_flags:
        typer.secho("\nRed flags:", fg=typer.colors.RED)
        for flag in spec.red_flags:
            typer.echo(f"  - {flag}")


@app.command("check-credentials")
def check_credentials() -> None:
    """Report which provider is active and whether its key is visible."""
    if has_credentials():
        provider = resolve_provider()
        # The key itself is never printed or logged anywhere in this project --
        # only the name of the variable it was read from.
        typer.secho(f"{provider.key_env_var} is set.", fg=typer.colors.GREEN)
        typer.echo(f"  provider    {provider.name}")
        typer.echo(f"  generation  {provider.generation_model}")
        typer.echo(f"  judge       {provider.judge_model}")
        if provider.base_url:
            typer.echo(f"  endpoint    {provider.base_url}")
        return
    typer.secho(credentials_message(), fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(ExitCode.NO_CREDENTIALS)


@app.command()
def analyze(
    jd: Annotated[Path, typer.Option("--jd", help="File containing the job posting.")],
    profile: Annotated[
        Path, typer.Option("--profile", help="Profile directory to match against.")
    ] = Path("profile.example"),
    strict: Annotated[
        bool, typer.Option("--strict", help="Exclude bullets marked confidence: claim.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Re-parse the posting even if cached.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the FitReport as JSON instead of a summary.")
    ] = False,
) -> None:
    """Score a profile against a job posting and report the fit.

    Spec 8 calls this the first genuinely useful deliverable: it answers
    "should I apply, and what am I missing?" before any resume is generated.
    """
    if not jd.is_file():
        typer.secho(f"No such file: {jd}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED)

    loaded = _load_or_exit(profile)

    try:
        result = run_analysis(jd, loaded, profile, strict=strict, use_cache=not no_cache)
    except MissingCredentialsError:
        typer.secho(credentials_message(), fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_CREDENTIALS) from None
    except JobDescriptionParseError as exc:
        typer.secho(f"Could not parse {jd}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED) from exc

    if as_json:
        typer.echo(result.fit.model_dump_json(indent=2))
        return

    render_fit_report(result)



@app.command()
def run(
    jd: Annotated[Path, typer.Option("--jd", help="File containing the job posting.")],
    profile: Annotated[
        Path, typer.Option("--profile", help="Profile directory to tailor from.")
    ] = Path("profile.example"),
    out: Annotated[Path, typer.Option("--out", help="Where to write the run directory.")] = Path(
        "out"
    ),
    strict: Annotated[
        bool, typer.Option("--strict", help="Exclude bullets marked confidence: claim.")
    ] = False,
    no_judge: Annotated[
        bool,
        typer.Option("--no-judge", help="Skip the LLM grounding judge; free layers stay on."),
    ] = False,
    no_cover_letter: Annotated[
        bool, typer.Option("--no-cover-letter", help="Skip the cover letter subgraph.")
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", help="Pause for review before finalizing (spec 5)."),
    ] = False,
    max_pages: Annotated[
        int,
        typer.Option(
            "--max-pages",
            min=1,
            max=2,
            help="Page target. One is the default and what most readers expect.",
        ),
    ] = 1,
    layout: Annotated[
        str,
        typer.Option(
            "--layout",
            help="Section order: auto (from your profile), student, or experienced.",
        ),
    ] = "auto",
    summary: Annotated[
        bool,
        typer.Option("--summary", help="Write an opening summary, gated like every other line."),
    ] = False,
) -> None:
    """Run the whole graph: parse, retrieve, score, select, tailor, verify, compile.

    This is the agent. Both feedback loops are live -- a bullet that fails
    grounding is retried twice and then dropped, and a resume that spills onto a
    second page shrinks its budget and reselects.
    """
    if not jd.is_file():
        typer.secho(f"No such file: {jd}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED)

    _load_or_exit(profile)

    if find_compiler() is None:
        typer.secho(INSTALL_MESSAGE, fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_COMPILER)

    raw_jd = jd.read_text(encoding="utf-8")
    if layout not in ("auto", "student", "experienced"):
        typer.secho(
            f"--layout must be auto, student or experienced (got {layout!r})",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(ExitCode.USAGE)

    options = RunOptions(
        out_dir=str(out),
        strict=strict,
        use_judge=not no_judge,
        write_cover_letter=not no_cover_letter,
        interactive=interactive,
        max_pages=max_pages,
        layout=layout,
        summary=summary,
    )

    # A checkpointer is only needed when the run can pause, but attaching it
    # always would mean every batch run writing checkpoint rows nobody reads.
    saver, connection = (open_checkpointer() if interactive else (None, None))
    config = {"recursion_limit": 100}
    if interactive:
        config |= thread_config(thread_id_for(raw_jd, profile))

    try:
        graph = build_graph(checkpointer=saver)
        final = graph.invoke(initial_state(raw_jd, profile, options), config)
    except MissingCredentialsError:
        typer.secho(credentials_message(), fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_CREDENTIALS) from None
    finally:
        if connection is not None:
            connection.close()

    # A paused run returns the interrupt payload instead of a finished state.
    if "__interrupt__" in final:
        _render_review(final["__interrupt__"][0].value)
        typer.secho(
            f"\nPaused. Resume with:\n"
            f"  resume-agent resume-run --jd {jd}\n"
            f'  resume-agent resume-run --jd {jd} --revise "your notes"',
            fg=typer.colors.CYAN,
        )
        return

    _report_run(final)


def _report_run(final: dict, resumed: bool = False) -> None:
    """Summarise a finished run.

    Shared by `run` and `resume-run` so a resumed run reports identically to one
    that never paused -- the outcome is the same artifact either way.
    """
    run_dir = final.get("out_dir")
    typer.echo()
    verb = "Run resumed and complete" if resumed else "Run complete"
    typer.secho(f"{verb}: {run_dir}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  pages           : {final.get('page_count')}")
    typer.echo(f"  bullets on page : {len(final.get('tailored', []))}")
    typer.echo(f"  line budget     : {final.get('line_budget')}")
    typer.echo(f"  layout attempts : {final.get('layout_attempts', 0)}")

    letter = final.get("cover_letter")
    if letter is not None:
        typer.echo(f"  cover letter    : {letter.word_count} words")
    elif final.get("options") and final["options"].write_cover_letter:
        typer.secho(
            "  cover letter    : ABANDONED (failed verification; see run.json)",
            fg=typer.colors.RED,
        )

    # Dropped bullets are the one outcome worth interrupting for: the resume is
    # weaker than it could be, and the only alternative was shipping a claim the
    # verifier could not stand behind.
    dropped = final.get("dropped_bullets", [])
    if dropped:
        typer.secho(
            f"\n  {len(dropped)} bullet(s) dropped for failing grounding:", fg=typer.colors.YELLOW
        )
        for bullet_id in dropped:
            typer.echo(f"    {bullet_id}")

    if (final.get("page_count") or 0) != 1:
        typer.secho(
            f"\nWarning: finished at {final.get('page_count')} pages after "
            f"{final.get('layout_attempts')} layout attempts.",
            fg=typer.colors.RED,
        )




def _render_review(payload: dict) -> None:
    """Show the human what they are approving. Spec 5's interrupt payload."""
    typer.echo()
    typer.secho("REVIEW REQUIRED", fg=typer.colors.YELLOW, bold=True)
    typer.echo(f"  {payload.get('title')} at {payload.get('company')}")
    typer.echo(
        f"  recommendation : {payload.get('recommendation')} "
        f"(fit {(payload.get('overall_fit') or 0):.0%})"
    )
    typer.echo(f"  pages          : {payload.get('page_count')}")
    if payload.get("cover_letter_words"):
        typer.echo(f"  cover letter   : {payload['cover_letter_words']} words")

    # Gaps first, for the same reason `analyze` puts them first: they are what
    # decides whether to send this at all.
    if payload.get("must_have_gaps"):
        typer.secho("\n  Missing must-haves:", fg=typer.colors.RED)
        for gap in payload["must_have_gaps"]:
            typer.echo(f"    - {gap}")

    if payload.get("dropped_bullets"):
        typer.secho("\n  Dropped for failing grounding:", fg=typer.colors.YELLOW)
        for bullet_id in payload["dropped_bullets"]:
            typer.echo(f"    - {bullet_id}")

    typer.echo("\n  Bullets on the resume:")
    for text in payload.get("selected_bullets", []):
        typer.echo(f"    - {text}")

    if payload.get("cover_letter"):
        typer.echo("\n  Cover letter:")
        for line in payload["cover_letter"].splitlines():
            typer.echo(f"    {line}")

    typer.echo(f"\n  PDF: {payload.get('pdf_path')}")


@app.command()
def resume_run(
    jd: Annotated[Path, typer.Option("--jd", help="The posting this run was started from.")],
    profile: Annotated[
        Path, typer.Option("--profile", help="Profile directory the run used.")
    ] = Path("profile.example"),
    revise: Annotated[
        str, typer.Option("--revise", help="Ask for changes instead of approving.")
    ] = "",
) -> None:
    """Resume a paused run. Spec 8's M7: survives a process restart.

    Nothing is carried over from the earlier process except the arguments: the
    thread id is recomputed from the posting and the profile, and the state
    comes back out of the checkpoint database.
    """
    if not jd.is_file():
        typer.secho(f"No such file: {jd}", fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.JD_PARSE_FAILED)

    raw_jd = jd.read_text(encoding="utf-8")
    thread = thread_id_for(raw_jd, profile)
    saver, connection = open_checkpointer()

    try:
        graph = build_graph(checkpointer=saver)
        config = thread_config(thread)

        snapshot = graph.get_state(config)
        if not snapshot.next:
            typer.secho(
                f"No paused run for {jd} + {profile}. Start one with "
                f"`resume-agent run --jd {jd} --interactive`.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(ExitCode.OK)

        decision = (
            {"action": "revise", "notes": revise} if revise else {"action": "approve"}
        )
        final = graph.invoke(Command(resume=decision), config)
        _report_run(final, resumed=True)
    finally:
        connection.close()


@app.command()
def applications(
    limit: Annotated[int, typer.Option("--limit", help="How many rows to show.")] = 20,
    outcome: Annotated[
        str, typer.Option("--outcome", help="Only show applications with this outcome.")
    ] = "",
    set_outcome_for: Annotated[
        int, typer.Option("--set-outcome-for", help="Application id to update.")
    ] = 0,
    to: Annotated[
        str, typer.Option("--to", help="The outcome to record, e.g. callback.")
    ] = "",
    by_bullet: Annotated[
        bool,
        typer.Option("--by-bullet", help="Which bullets appear in which outcomes."),
    ] = False,
) -> None:
    """The application tracker.

    `--by-bullet` answers spec 11's question -- "which bullets appear in
    applications that got callbacks?" -- which is only as good as the outcomes
    you have recorded.
    """
    if set_outcome_for:
        if not to:
            typer.secho("--set-outcome-for needs --to.", fg=typer.colors.RED, err=True)
            raise typer.Exit(ExitCode.JD_PARSE_FAILED)
        if set_outcome(set_outcome_for, to):
            typer.secho(f"Application #{set_outcome_for}: {to}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"No application #{set_outcome_for}.", fg=typer.colors.RED, err=True)
        return

    if by_bullet:
        counts = bullets_by_outcome()
        if not counts:
            typer.secho(
                "No outcomes recorded yet. Use --set-outcome-for <id> --to callback.",
                fg=typer.colors.YELLOW,
            )
            return
        typer.echo()
        for bullet_id, outcomes in sorted(counts.items()):
            rendered = ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
            typer.echo(f"  {bullet_id:34s} {rendered}")
        typer.echo()
        return

    rows = list_applications(limit=limit, outcome=outcome or None)
    if not rows:
        typer.secho("No applications recorded yet.", fg=typer.colors.YELLOW)
        return

    typer.echo()
    typer.secho(f"{'id':>4}  {'date':<11} {'company':<24} {'fit':>5}  outcome", bold=True)
    for row in rows:
        fit = f"{row.overall_fit:.0%}" if row.overall_fit is not None else "-"
        typer.echo(
            f"{row.id:>4}  {row.applied_on:<11} {row.company[:24]:<24} {fit:>5}  "
            f"{row.outcome or '-'}"
        )
    typer.echo()


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option("--reload", help="Reload on code changes.")] = False,
) -> None:
    """Serve the web UI. Spec 8's M9.

    Binds to localhost by default. In local mode the API has no authentication
    and the run registry is an in-process dict, so this is a single-user tool on
    your own machine -- binding it to 0.0.0.0 would expose an unauthenticated
    endpoint that spends money. Multi-user mode (`RESUME_AGENT_MULTIUSER=1`) is
    the one that is safe to put on a public address.
    """
    import uvicorn

    typer.secho(f"resume-agent UI: http://{host}:{port}", fg=typer.colors.CYAN, bold=True)
    if not has_credentials():
        typer.secho(
            "  (no credentials -- the page will load and the Run button will be "
            "disabled; run `resume-agent check-credentials` for what to set)",
            fg=typer.colors.YELLOW,
        )

    uvicorn.run(
        "resume_agent.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ===========================================================================
# Multi-user accounts (M10)
# ===========================================================================

users_app = typer.Typer(
    no_args_is_help=True,
    help="Who is waiting for access, and letting them in. Reads DATABASE_URL.",
)
app.add_typer(users_app, name="users")


def _accounts_db():
    """The accounts database the deployed app uses, or a clear exit.

    Only `DATABASE_URL` is needed -- not the Google or session settings -- so
    this still works when the web app cannot start. That is the point of it.
    """
    from resume_agent.accounts.auth import DATABASE_URL_ENV_VAR  # noqa: PLC0415
    from resume_agent.accounts.db import AccountsDB  # noqa: PLC0415

    url = os.environ.get(DATABASE_URL_ENV_VAR, "").strip()
    if not url:
        typer.secho(
            f"Set {DATABASE_URL_ENV_VAR} to the database the web app uses "
            "(the Neon connection string).",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(ExitCode.USAGE)
    try:
        db = AccountsDB(url)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.USAGE) from None
    db.create_schema()
    return db


@users_app.command("list")
def users_list() -> None:
    """Everyone who has signed in, the people waiting first."""
    db = _accounts_db()
    try:
        people = db.users()
    finally:
        db.close()
    if not people:
        typer.secho("Nobody has signed in yet.", fg=typer.colors.YELLOW)
        return

    colours = {"pending": typer.colors.YELLOW, "approved": typer.colors.GREEN,
               "rejected": typer.colors.RED}
    typer.echo()
    typer.secho(f"  {'status':<9} {'email':<34} {'signed up':<11} name", bold=True)
    for user in people:
        status = typer.style(f"{user.status:<9}", fg=colours.get(user.status))
        owner = "  (you)" if user.is_admin else ""
        typer.echo(
            f"  {status} {user.email[:34]:<34} {user.created_at[:10]:<11} {user.name}{owner}"
        )
    waiting = sum(user.status == "pending" for user in people)
    typer.echo()
    if waiting:
        typer.echo(f"  {waiting} waiting. resume-agent users approve <email>")
        typer.echo()


def _decide(email: str, status: str) -> None:
    db = _accounts_db()
    try:
        user = db.user_by_email(email)
        if user is None:
            typer.secho(
                f"Nobody has signed in as {email}. They need to sign in once "
                "before they can be approved.",
                fg=typer.colors.RED, err=True,
            )
            raise typer.Exit(ExitCode.NO_SUCH_USER)
        db.decide(user.id, status, by="cli")
    finally:
        db.close()
    typer.secho(f"{user.email}: {status}", fg=typer.colors.GREEN)


@users_app.command("approve")
def users_approve(
    email: Annotated[str, typer.Argument(help="The address they signed in with.")],
) -> None:
    """Let someone in. Takes effect on their next click; no need to sign in again."""
    _decide(email, "approved")


@users_app.command("reject")
def users_reject(
    email: Annotated[str, typer.Argument(help="The address they signed in with.")],
) -> None:
    """Turn someone away, or take access back from someone already approved."""
    _decide(email, "rejected")


if __name__ == "__main__":
    app()
