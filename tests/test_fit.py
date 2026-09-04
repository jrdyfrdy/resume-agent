"""FitReport assembly, retrieval, and the scoring node.

Everything here is either pure arithmetic (the report) or runs against a fake
model (the scoring node), so the whole file works without an API key.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.graph.nodes.retrieve import retrieve_evidence
from resume_agent.graph.nodes.score import ScoredMatch, ScoringResult, build_fit_report, score_fit
from resume_agent.models.fit import (
    COVERED_THRESHOLD,
    MUST_HAVE_COVERAGE_FLOOR,
    PARTIAL_THRESHOLD,
    EvidenceMatch,
    compute_overall_fit,
    recommend,
)
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.profile import Profile


def make_job(*requirements: Requirement) -> JobSpec:
    return JobSpec.from_fields(
        JobSpecFields(
            company="C",
            title="T",
            seniority="mid",
            domain="d",
            requirements=list(requirements),
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )


def req(text: str, weight: int = 3, must: bool = False) -> Requirement:
    return Requirement(text=text, category="practice", weight=weight, is_must_have=must)


# --- overall_fit is arithmetic ---------------------------------------------


def test_overall_fit_is_weight_weighted() -> None:
    """Missing a weight-5 requirement must hurt more than missing a weight-1."""
    requirements = [req("heavy", weight=5), req("light", weight=1)]

    missing_heavy = compute_overall_fit(requirements, {"light": 1.0})
    missing_light = compute_overall_fit(requirements, {"heavy": 1.0})

    assert missing_light > missing_heavy
    # `compute_overall_fit` rounds to 4dp so the number reads cleanly in the
    # report; the tolerance here matches that, not float precision.
    assert missing_heavy == pytest.approx(1 / 6, abs=1e-4)
    assert missing_light == pytest.approx(5 / 6, abs=1e-4)


def test_overall_fit_of_no_requirements_is_zero_not_a_crash() -> None:
    assert compute_overall_fit([], {}) == 0.0


def test_perfect_coverage_is_one() -> None:
    requirements = [req("a", weight=5), req("b", weight=2)]
    assert compute_overall_fit(requirements, {"a": 1.0, "b": 1.0}) == 1.0


# --- the recommendation, including its safety floor ------------------------


def test_high_fit_recommends_applying() -> None:
    requirements = [req("a", weight=5, must=True)]
    assert recommend(0.9, requirements, {"a": 0.9}) == "strong_apply"


def test_low_fit_recommends_skipping() -> None:
    """A tool that never says "skip" is flattering you, not informing you."""
    requirements = [req("a", weight=5, must=True)]
    assert recommend(0.1, requirements, {"a": 0.1}) == "skip"


def test_missing_must_haves_caps_the_recommendation() -> None:
    """A strong aggregate can hide a fatal gap.

    Four covered nice-to-haves and one uncovered must-have can produce a
    respectable overall_fit, but the must-have is the thing they will not
    compromise on -- so the advice is capped at "stretch".
    """
    requirements = [
        req("must_a", weight=5, must=True),
        req("must_b", weight=5, must=True),
        req("nice", weight=1),
    ]
    best = {"must_a": 0.95, "must_b": 0.0, "nice": 1.0}  # 1 of 2 must-haves = 50%

    assert MUST_HAVE_COVERAGE_FLOOR > 0.5
    assert recommend(0.85, requirements, best) == "stretch"


def test_no_must_haves_means_no_cap() -> None:
    requirements = [req("nice", weight=3)]
    assert recommend(0.9, requirements, {"nice": 0.9}) == "strong_apply"


# --- FitReport bucketing ----------------------------------------------------


def match(bullet_id: str, requirement: str, relevance: float) -> EvidenceMatch:
    return EvidenceMatch(
        bullet_id=bullet_id, requirement_text=requirement, relevance=relevance, rationale="r"
    )


def test_buckets_follow_the_thresholds() -> None:
    job = make_job(req("covered"), req("partial"), req("weak"))
    report = build_fit_report(
        job,
        [
            match("b1", "covered", COVERED_THRESHOLD),
            match("b2", "partial", PARTIAL_THRESHOLD),
            match("b3", "weak", PARTIAL_THRESHOLD - 0.01),
        ],
    )
    assert [m.requirement_text for m in report.covered] == ["covered"]
    assert [m.requirement_text for m in report.partial] == ["partial"]
    assert [r.text for r in report.gaps] == ["weak"]


def test_a_requirement_with_no_evidence_becomes_a_gap() -> None:
    """The most important property in the file.

    Gaps are derived from the requirement side. If they came only from scored
    matches, a requirement that retrieved nothing would vanish from the report
    entirely -- and silently omitting what the candidate lacks is exactly the
    dishonesty this tool exists to avoid.
    """
    job = make_job(req("kubernetes", weight=5, must=True), req("python"))
    report = build_fit_report(job, [match("b1", "python", 0.9)])

    assert [r.text for r in report.gaps] == ["kubernetes"]
    assert [r.text for r in report.must_have_gaps()] == ["kubernetes"]


def test_best_match_per_requirement_wins() -> None:
    job = make_job(req("python"))
    report = build_fit_report(job, [match("b1", "python", 0.4), match("b2", "python", 0.95)])
    assert len(report.covered) == 1
    assert report.covered[0].bullet_id == "b2"


def test_gaps_are_sorted_must_haves_first_then_by_weight() -> None:
    job = make_job(
        req("nice_heavy", weight=5),
        req("must_light", weight=1, must=True),
        req("must_heavy", weight=5, must=True),
    )
    report = build_fit_report(job, [])
    assert [r.text for r in report.gaps] == ["must_heavy", "must_light", "nice_heavy"]


def test_empty_job_produces_a_skip() -> None:
    report = build_fit_report(make_job(), [])
    assert report.overall_fit == 0.0
    assert report.recommendation == "skip"


# --- retrieval --------------------------------------------------------------


def test_retrieve_evidence_deduplicates_across_requirements(
    example_profile: Profile, retriever
) -> None:
    job = make_job(req("redis caching"), req("cache invalidation"), req("kubernetes"))
    candidates = retrieve_evidence(job, retriever, k=5)

    assert set(candidates.by_requirement) == {r.text for r in job.requirements}
    assert len(candidates.bullet_ids) == len(set(candidates.bullet_ids))
    # Overlapping queries must not inflate the deduplicated union.
    assert len(candidates.bullet_ids) < sum(len(v) for v in candidates.by_requirement.values())


def test_retrieve_evidence_with_no_requirements(example_profile: Profile, retriever) -> None:
    candidates = retrieve_evidence(make_job(), retriever)
    assert candidates.bullet_ids == []


# --- the scoring node, against a fake model ---------------------------------


class FakeScoringModel(BaseChatModel):
    result: ScoringResult = ScoringResult(matches=[])
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-scoring"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> ScoringResult:
                fake.call_count += 1
                return fake.result

        return _Runnable()


def test_scoring_is_a_single_batched_call(example_profile: Profile) -> None:
    """Spec 5 is explicit that batching is what keeps scores comparable."""
    job = make_job(req("a"), req("b"), req("c"))
    bullet_ids = [b.id for b in example_profile.all_bullets()][:4]

    fake = FakeScoringModel(
        result=ScoringResult(
            matches=[
                ScoredMatch(
                    bullet_id=bullet_ids[0], requirement_text="a", relevance=0.9, rationale="r"
                )
            ]
        )
    )
    matches = score_fit(job, bullet_ids, example_profile, llm=fake)

    assert fake.call_count == 1, "scored requirements in separate calls"
    assert len(matches) == 1


def test_invented_bullet_ids_are_dropped(example_profile: Profile) -> None:
    """An invented bullet id is the retrieval equivalent of a fabricated claim.

    It must never reach selection, where it would become evidence for something
    the candidate never did.
    """
    job = make_job(req("a"))
    bullet_ids = [b.id for b in example_profile.all_bullets()][:2]

    fake = FakeScoringModel(
        result=ScoringResult(
            matches=[
                ScoredMatch(
                    bullet_id=bullet_ids[0], requirement_text="a", relevance=0.9, rationale="r"
                ),
                ScoredMatch(
                    bullet_id="exp_nonexistent.b9",
                    requirement_text="a",
                    relevance=1.0,
                    rationale="r",
                ),
            ]
        )
    )
    matches = score_fit(job, bullet_ids, example_profile, llm=fake)

    assert [m.bullet_id for m in matches] == [bullet_ids[0]]


def test_invented_requirements_are_dropped(example_profile: Profile) -> None:
    job = make_job(req("a"))
    bullet_ids = [b.id for b in example_profile.all_bullets()][:2]

    fake = FakeScoringModel(
        result=ScoringResult(
            matches=[
                ScoredMatch(
                    bullet_id=bullet_ids[0],
                    requirement_text="a requirement the posting never had",
                    relevance=1.0,
                    rationale="r",
                )
            ]
        )
    )
    assert score_fit(job, bullet_ids, example_profile, llm=fake) == []


def test_no_candidates_means_no_call(example_profile: Profile) -> None:
    fake = FakeScoringModel()
    assert score_fit(make_job(req("a")), [], example_profile, llm=fake) == []
    assert fake.call_count == 0, "spent a call with nothing to score"
