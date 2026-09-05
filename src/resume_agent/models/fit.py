"""How well a candidate matches a posting, and what to put on the page. Spec 4.

The division of labour in this file is the project's central discipline
(CLAUDE.md rule 2). An `EvidenceMatch` carries a judgement -- how relevant is
this achievement to that requirement, and why -- and a judgement is what a
language model is for. Everything built *on top* of those judgements here
(`overall_fit`, the covered/partial/gap buckets, the recommendation) is
arithmetic, and arithmetic is Python's.

Asking a model "so overall, how good a fit is this?" would produce a number that
drifts between runs, cannot be unit-tested, and quietly encodes whatever mood
the prompt put it in. Computing it means the thresholds are visible, tunable,
and defensible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from resume_agent.models.job import Requirement

Recommendation = Literal["strong_apply", "apply", "stretch", "skip"]

# --- thresholds -------------------------------------------------------------
#
# Deliberately module constants rather than magic numbers buried in a function:
# spec 8 says to use `analyze` on real postings for a week and let that tell you
# whether the scoring is calibrated. These are the dials that week will move.

COVERED_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.40

# Recommendation cut points, applied to overall_fit. `skip` is deliberately
# reachable: a tool that never tells you not to apply is not giving you
# information, it is flattering you.
STRONG_APPLY_FIT = 0.75
APPLY_FIT = 0.55
STRETCH_FIT = 0.35

# A posting can look good on aggregate while missing something non-negotiable.
# If fewer than this fraction of must-haves are covered, the recommendation is
# capped at "stretch" no matter how high overall_fit is.
MUST_HAVE_COVERAGE_FLOOR = 0.60


class EvidenceMatch(BaseModel):
    """One achievement judged against one requirement. Spec 4."""

    model_config = ConfigDict(extra="forbid")

    bullet_id: str
    requirement_text: str
    relevance: float = Field(ge=0.0, le=1.0)
    rationale: str

    @property
    def is_covered(self) -> bool:
        return self.relevance >= COVERED_THRESHOLD

    @property
    def is_partial(self) -> bool:
        return PARTIAL_THRESHOLD <= self.relevance < COVERED_THRESHOLD


class FitReport(BaseModel):
    """The honest answer to "should I apply, and what am I missing?". Spec 4."""

    model_config = ConfigDict(extra="forbid")

    overall_fit: float = Field(ge=0.0, le=1.0)
    covered: list[EvidenceMatch] = Field(default_factory=list)
    partial: list[EvidenceMatch] = Field(default_factory=list)
    # Requirements with no supporting evidence at all. Spec 4 annotates this
    # field "honest: no evidence exists", and that honesty is the whole value of
    # the report -- a fit report that cannot say "you don't have this" is a
    # flattery machine.
    gaps: list[Requirement] = Field(default_factory=list)
    recommendation: Recommendation

    def must_have_gaps(self) -> list[Requirement]:
        return [r for r in self.gaps if r.is_must_have]

    def covered_bullet_ids(self) -> set[str]:
        return {match.bullet_id for match in self.covered}


class SelectionResult(BaseModel):
    """What `select_content` chose, and what it had to leave out.

    The rejected list is not decoration. When a resume comes back missing an
    achievement you expected, the only useful question is "why was it dropped?",
    and a selector that cannot answer is one you end up debugging by bisecting
    the profile.
    """

    model_config = ConfigDict(extra="forbid")

    selected_bullet_ids: list[str] = Field(default_factory=list)
    total_estimated_lines: int = 0
    line_budget: int = 0
    # bullet_id -> the constraint that excluded it, in plain words.
    rejected: dict[str, str] = Field(default_factory=dict)

    @property
    def fits(self) -> bool:
        return self.total_estimated_lines <= self.line_budget


def compute_overall_fit(
    requirements: list[Requirement],
    best_relevance: dict[str, float],
) -> float:
    """Weighted coverage of the posting's requirements.

    Each requirement contributes its `weight`, scaled by the best relevance any
    single achievement reached against it. Weighting by the posting's own sense
    of importance is the point: missing a weight-5 must-have should hurt far
    more than missing a weight-1 nice-to-have, and an unweighted mean would
    treat them identically.

    Returns 0.0 for a posting with no requirements rather than dividing by zero.
    """
    total_weight = sum(r.weight for r in requirements)
    if total_weight == 0:
        return 0.0

    earned = sum(r.weight * best_relevance.get(r.text, 0.0) for r in requirements)
    return round(earned / total_weight, 4)


def recommend(
    overall_fit: float,
    requirements: list[Requirement],
    best_relevance: dict[str, float],
) -> Recommendation:
    """Turn a fit score into advice, with a hard floor on must-have coverage."""
    must_haves = [r for r in requirements if r.is_must_have]
    if must_haves:
        covered = sum(1 for r in must_haves if best_relevance.get(r.text, 0.0) >= COVERED_THRESHOLD)
        must_have_coverage = covered / len(must_haves)
    else:
        must_have_coverage = 1.0

    if overall_fit >= STRONG_APPLY_FIT:
        base: Recommendation = "strong_apply"
    elif overall_fit >= APPLY_FIT:
        base = "apply"
    elif overall_fit >= STRETCH_FIT:
        base = "stretch"
    else:
        base = "skip"

    # A strong aggregate can hide a fatal gap -- 90% fit is no comfort if the
    # one thing you cannot do is the thing they cannot compromise on.
    if must_have_coverage < MUST_HAVE_COVERAGE_FLOOR and base in ("strong_apply", "apply"):
        return "stretch"

    return base
