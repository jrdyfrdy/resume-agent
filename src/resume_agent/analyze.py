"""The `analyze` pipeline, end to end. Spec 5 and spec 8's M3.

    "Ship the FitReport as a standalone CLI command. `resume-agent analyze
     --jd job.txt` telling you 'you cover 7/9 must-haves, you're missing
     Kubernetes and Kafka, recommendation: apply' is genuinely useful on day
     one, before any of the LaTeX machinery exists."

Kept out of `cli.py` so the pipeline is importable and testable without going
through Typer, and so M5 can call the same sequence from graph nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from resume_agent.graph.nodes.parse_jd import parse_job_description
from resume_agent.graph.nodes.retrieve import retrieve_evidence
from resume_agent.graph.nodes.score import build_fit_report, score_fit
from resume_agent.graph.nodes.select import select_content
from resume_agent.kb.index import ProfileIndex
from resume_agent.kb.retriever import HybridRetriever
from resume_agent.latex.context import credential_rows, group_skills_by_category
from resume_agent.latex.layout import (
    SUMMARY_LINES,
    item_lines,
    line_budget,
    skill_row_lines,
)
from resume_agent.models.fit import EvidenceMatch, FitReport, SelectionResult
from resume_agent.models.job import JobSpec
from resume_agent.models.profile import Profile
from resume_agent.sections import Stage, career_stage, this_month


@dataclass
class AnalysisResult:
    job: JobSpec
    fit: FitReport
    selection: SelectionResult
    matches: list[EvidenceMatch]
    candidates_considered: int
    jd_cache_hit: bool


def budget_for_profile(
    profile: Profile,
    *,
    pages: int = 1,
    stage: Stage | None = None,
    today: str | None = None,
    reserve_summary: bool = False,
) -> int:
    """The line budget this profile's shape leaves for bullets.

    Sections counted are the ones the template will actually emit, so an empty
    projects list does not pay for a Projects heading.

    The student layout costs more furniture than the experienced one -- GPA,
    honours, coursework and the awards under them are all printed lines -- so
    the stage is part of the shape, not a presentation detail applied later.
    """
    today = today or this_month()
    stage = stage or career_stage(profile, today)
    compact = stage == "experienced"

    skill_rows = group_skills_by_category(profile.skills)
    sections = sum(
        [
            bool(profile.education),
            bool(profile.experience),
            bool(profile.projects),
            bool(skill_rows),
        ]
    )

    # Measured from the text that will actually print, for the reason given on
    # `item_lines`: a two-word honour and a wrapping coursework list are both
    # one `\resumeItem` and are not the same number of lines.
    education_extras: list[str] = []
    credentials: list[str] = []
    if not compact:
        for entry in profile.education:
            if entry.gpa:
                education_extras.append(f"GPA: {entry.gpa}")
            education_extras.extend(entry.honors)
            if entry.coursework:
                education_extras.append(
                    "Relevant coursework: " + ", ".join(entry.coursework)
                )
        credentials = [row["text"] for row in credential_rows(profile)]

    return line_budget(
        experience_entries=len(profile.experience),
        project_entries=len(profile.projects),
        education_entries=len(profile.education),
        skill_lines=skill_row_lines(skill_rows),
        sections=sections,
        leadership_entries=len(profile.leadership),
        publication_entries=len(profile.publications),
        education_extra_lines=item_lines(education_extras),
        credential_lines=item_lines(credentials),
        summary_lines=SUMMARY_LINES if reserve_summary else 0,
        pages=pages,
    )


def analyze(
    jd_path: Path,
    profile: Profile,
    profile_dir: Path,
    *,
    strict: bool = False,
    use_cache: bool = True,
) -> AnalysisResult:
    """Parse -> retrieve -> score -> report -> select."""
    raw_jd = jd_path.read_text(encoding="utf-8")
    job, jd_cache_hit = parse_job_description(raw_jd, use_cache=use_cache)

    index, _ = ProfileIndex.open(profile, profile_dir)
    try:
        retriever = HybridRetriever(index, profile)
        candidates = retrieve_evidence(job, retriever)
        matches = score_fit(job, candidates.bullet_ids, profile)
    finally:
        index.close()

    fit = build_fit_report(job, matches)
    selection = select_content(
        job, matches, profile, budget_for_profile(profile), strict=strict
    )

    return AnalysisResult(
        job=job,
        fit=fit,
        selection=selection,
        matches=matches,
        candidates_considered=len(candidates),
        jd_cache_hit=jd_cache_hit,
    )
