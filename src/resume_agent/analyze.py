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
from resume_agent.latex.context import group_skills_by_category
from resume_agent.latex.layout import line_budget, skill_row_lines
from resume_agent.models.fit import EvidenceMatch, FitReport, SelectionResult
from resume_agent.models.job import JobSpec
from resume_agent.models.profile import Profile


@dataclass
class AnalysisResult:
    job: JobSpec
    fit: FitReport
    selection: SelectionResult
    matches: list[EvidenceMatch]
    candidates_considered: int
    jd_cache_hit: bool


def budget_for_profile(profile: Profile) -> int:
    """The line budget this profile's shape leaves for bullets.

    Sections counted are the ones the template will actually emit, so an empty
    projects list does not pay for a Projects heading.
    """
    skill_rows = group_skills_by_category(profile.skills)
    sections = sum(
        [
            bool(profile.education),
            bool(profile.experience),
            bool(profile.projects),
            bool(skill_rows),
        ]
    )
    return line_budget(
        experience_entries=len(profile.experience),
        project_entries=len(profile.projects),
        education_entries=len(profile.education),
        skill_lines=skill_row_lines(skill_rows),
        sections=sections,
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
