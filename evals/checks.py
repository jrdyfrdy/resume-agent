"""The deterministic half of the eval. Spec 9.

    "Deterministic checks (run on every JD in the eval set, no LLM, no cost):
       compiles without error
       exactly 1 page
       zero overfull hboxes
       zero fabricated numbers (verifier layer 1)
       must-have keyword coverage >= 70% where evidence exists
       no bullet exceeds the character cap"

Six checks, all free, and **not one line of new judgement**. Every one is
`inspect_output`, `unsupported_numbers`, `FitReport` or `estimate_lines` applied
to a finished run -- the same code the agent used to produce it.

That reuse is the point. An eval that reimplements its own idea of "fabricated"
is measuring the reimplementation, and would drift from the gate it is supposed
to be auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from resume_agent.graph.nodes.tailor import target_characters
from resume_agent.grounding.numbers import unsupported_numbers
from resume_agent.kb.loader import load_profile
from resume_agent.latex.inspect import find_overfull_boxes, first_latex_error
from resume_agent.models.fit import COVERED_THRESHOLD

# Spec 9: "must-have keyword coverage >= 70% where evidence exists".
MIN_MUST_HAVE_COVERAGE = 0.70


@dataclass(frozen=True)
class DeterministicResult:
    """Spec 9's six checks, for one resume."""

    jd_name: str
    compiled: bool
    page_count: int | None
    overfull_boxes: int
    first_error: str | None
    fabricated_numbers: dict[str, list[str]] = field(default_factory=dict)
    must_have_coverage: float = 0.0
    must_haves_total: int = 0
    over_length_bullets: list[str] = field(default_factory=list)
    dropped_bullets: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        """Every check that did not pass, named. Empty means clean."""
        problems = []
        if not self.compiled:
            first_line = (self.first_error or "").splitlines()
            problems.append(
                f"did not compile: {first_line[0]}" if first_line else "did not compile"
            )
        if self.page_count != 1:
            problems.append(f"{self.page_count} pages")
        if self.overfull_boxes:
            problems.append(f"{self.overfull_boxes} overfull hboxes")
        if self.fabricated_numbers:
            problems.append(f"fabricated numbers in {sorted(self.fabricated_numbers)}")
        # Coverage is only meaningful when the posting actually stated
        # must-haves -- spec 9 says "where evidence exists", and a posting with
        # none is not a resume that failed.
        if self.must_haves_total and self.must_have_coverage < MIN_MUST_HAVE_COVERAGE:
            problems.append(f"must-have coverage {self.must_have_coverage:.0%}")
        if self.over_length_bullets:
            problems.append(f"{len(self.over_length_bullets)} bullets over the character cap")
        return problems

    @property
    def passed(self) -> bool:
        return not self.failures


def run_deterministic_checks(jd_name: str, state: dict) -> DeterministicResult:
    """Apply spec 9's six checks to a finished run's state."""
    log = state.get("compile_log") or ""
    profile = load_profile(Path(state["profile_path"]))
    tailored = state.get("tailored", [])

    # --- fabricated numbers: verifier layer 1, re-run on the shipped text ----
    #
    # Re-run rather than trusted from the run, because the eval's job is to
    # audit the gate, not to take its word for it. If the verifier ever stops
    # being called, this is what notices.
    fabricated: dict[str, list[str]] = {}
    over_length: list[str] = []
    for bullet in tailored:
        try:
            source = profile.bullet_by_id(bullet.source_id)
        except KeyError:
            fabricated[bullet.source_id] = ["source bullet does not exist"]
            continue

        unsupported = unsupported_numbers(bullet.text, source.metrics, source.canonical)
        if unsupported:
            fabricated[bullet.source_id] = sorted(unsupported)

        if len(bullet.text) > target_characters(source):
            over_length.append(bullet.source_id)

    # --- must-have coverage --------------------------------------------------
    fit = state.get("fit_report")
    job = state.get("job_spec")

    total, coverage = 0, 0.0
    if fit is not None and job is not None:
        # Counted from the posting's own must-haves against the report's covered
        # set, rather than read off a field. Recomputing means a change to how
        # the report buckets requirements shows up here as a measured difference
        # instead of being invisible.
        covered_texts = {
            match.requirement_text
            for match in fit.covered
            if match.relevance >= COVERED_THRESHOLD
        }
        must_haves = job.must_haves()
        total = len(must_haves)
        covered = sum(1 for requirement in must_haves if requirement.text in covered_texts)
        coverage = (covered / total) if total else 0.0

    return DeterministicResult(
        jd_name=jd_name,
        compiled=state.get("pdf_path") is not None and first_latex_error(log) is None,
        page_count=state.get("page_count"),
        overfull_boxes=len(find_overfull_boxes(log)),
        first_error=first_latex_error(log),
        fabricated_numbers=fabricated,
        must_have_coverage=round(coverage, 4),
        must_haves_total=total,
        over_length_bullets=over_length,
        dropped_bullets=list(state.get("dropped_bullets", [])),
    )
