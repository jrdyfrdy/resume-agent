"""Command line entry point.

    resume-agent build  --profile profile.example --out out/   (M0)
    resume-agent index  --profile profile.example              (M1)
    resume-agent search "kubernetes" --explain                 (M1)
    resume-agent parse-jd --jd evals/datasets/jds/mid.txt       (M2)
    resume-agent analyze  --jd evals/datasets/jds/mid.txt       (M3)
    resume-agent run      --jd evals/datasets/jds/mid.txt       (M5)
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Annotated

import typer

from resume_agent.analyze import analyze as run_analysis
from resume_agent.graph.build import build_graph, initial_state
from resume_agent.graph.nodes.parse_jd import JobDescriptionParseError, parse_job_description
from resume_agent.graph.state import RunOptions
from resume_agent.kb.index import ProfileIndex, index_path_for
from resume_agent.kb.loader import ProfileLoadError, load_profile
from resume_agent.kb.retriever import HybridRetriever, SearchMode
from resume_agent.latex.compile import INSTALL_MESSAGE, compile_tex, find_compiler
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.latex.inspect import inspect_output
from resume_agent.llm import CREDENTIALS_MESSAGE, MissingCredentialsError, has_credentials
from resume_agent.report import render_fit_report

RESUME_TEMPLATE = "jake_resume.tex.j2"


class ExitCode(IntEnum):
    """Distinct codes so a script calling this can tell the failures apart."""

    OK = 0
    COMPILE_FAILED = 1
    PROFILE_INVALID = 2
    NO_COMPILER = 3
    NO_CREDENTIALS = 4
    JD_PARSE_FAILED = 5


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
        typer.secho(CREDENTIALS_MESSAGE, fg=typer.colors.RED, err=True)
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
    """Report whether an Anthropic API key is visible to resume-agent."""
    if has_credentials():
        typer.secho("ANTHROPIC_API_KEY is set.", fg=typer.colors.GREEN)
        return
    typer.secho(CREDENTIALS_MESSAGE, fg=typer.colors.YELLOW, err=True)
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
        typer.secho(CREDENTIALS_MESSAGE, fg=typer.colors.RED, err=True)
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

    options = RunOptions(out_dir=str(out), strict=strict, use_judge=not no_judge)
    graph = build_graph()

    try:
        final = graph.invoke(
            initial_state(jd.read_text(encoding="utf-8"), profile, options),
            {"recursion_limit": 100},
        )
    except MissingCredentialsError:
        typer.secho(CREDENTIALS_MESSAGE, fg=typer.colors.RED, err=True)
        raise typer.Exit(ExitCode.NO_CREDENTIALS) from None

    run_dir = final.get("out_dir")
    typer.echo()
    typer.secho(f"Run complete: {run_dir}", fg=typer.colors.CYAN, bold=True)
    typer.echo(f"  pages           : {final.get('page_count')}")
    typer.echo(f"  bullets on page : {len(final.get('tailored', []))}")
    typer.echo(f"  line budget     : {final.get('line_budget')}")
    typer.echo(f"  layout attempts : {final.get('layout_attempts', 0)}")

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

if __name__ == "__main__":
    app()
