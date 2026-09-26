"""The eval harness. M8's definition of done.

Spec 8: "15 JDs, deterministic + judge scores, `make eval` produces a table; a
deliberately worsened prompt shows a measurable score drop."

The load-bearing test is `test_the_gate_fires_on_a_worsened_score`. Spec 9 wants
prompt changes gated on measured quality; a harness whose gate never fires is
worse than no harness, because it looks like a safety net.

Nothing here spends money -- the judge runs against a fake, and the deterministic
checks never needed a model at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from evals.checks import (
    MIN_MUST_HAVE_COVERAGE,
    DeterministicResult,
    run_deterministic_checks,
)
from evals.judges.resume_judge import DIMENSIONS, JudgeScores, judge_resume
from evals.run_eval import (
    REGRESSION_THRESHOLD,
    VARIANTS,
    CaseResult,
    EvalReport,
    check_regression,
    render_table,
)
from resume_agent.llm import PROMPT_OVERRIDE_ENV_VAR, load_prompt, prompt_path
from resume_agent.models.fit import EvidenceMatch, FitReport
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.resume import TailoredBullet

REPO_ROOT = Path(__file__).resolve().parent.parent
JD_DIR = REPO_ROOT / "evals" / "datasets" / "jds"

SOURCE_ID = "exp_halvorsen_bright.b1"
TRUTHFUL = (
    "Cut p95 checkout latency from 820ms to 50ms with a Redis read-through cache "
    "and by removing N+1 queries across 12 endpoints"
)


def make_job(**overrides: Any) -> JobSpec:
    fields = {
        "company": "Acme Corp",
        "title": "Backend Engineer",
        "seniority": "mid",
        "domain": "payments",
        "requirements": [
            Requirement(text="caching", category="practice", weight=5, is_must_have=True),
            Requirement(text="kubernetes", category="tool", weight=4, is_must_have=True),
        ],
        "responsibilities": [],
        "ats_keywords": ["Redis"],
        "culture_signals": [],
        "tone": "formal",
        "red_flags": [],
    }
    fields.update(overrides)
    return JobSpec.from_fields(JobSpecFields(**fields), "raw")


def make_state(**overrides: Any) -> dict:
    job = make_job()
    state = {
        "profile_path": str(REPO_ROOT / "profile.example"),
        "job_spec": job,
        "tailored": [TailoredBullet(source_id=SOURCE_ID, text=TRUTHFUL, estimated_lines=2)],
        # Both must-haves covered by default, so `make_state()` is a genuinely
        # clean run. Tests that want a coverage failure ask for one explicitly.
        "fit_report": FitReport(
            overall_fit=0.9,
            covered=[
                EvidenceMatch(
                    bullet_id=SOURCE_ID, requirement_text=text, relevance=0.9, rationale="r"
                )
                for text in ("caching", "kubernetes")
            ],
            recommendation="strong_apply",
        ),
        "pdf_path": "out/resume.pdf",
        "page_count": 1,
        "compile_log": "Output written on resume.pdf (1 page).",
        "dropped_bullets": [],
    }
    state.update(overrides)
    return state


# ===========================================================================
# The dataset
# ===========================================================================


def test_the_dataset_has_at_least_fifteen_jds() -> None:
    """M8 DoD: "15 JDs"."""
    assert len(list(JD_DIR.glob("*.txt"))) >= 15


def test_the_dataset_spans_seniorities() -> None:
    """An eval set that is all mid-level backend measures one thing repeatedly."""
    names = " ".join(p.stem for p in JD_DIR.glob("*.txt"))
    for level in ["intern", "junior", "mid", "senior", "staff", "lead"]:
        assert level in names, f"no {level} posting in the eval set"


def test_the_dataset_includes_postings_the_profile_should_fail() -> None:
    """An eval set where everything scores 0.8 cannot detect a regression.

    These four are outside the example profile's experience on purpose -- no ML,
    no security, no people management, no embedded work.
    """
    stems = {p.stem for p in JD_DIR.glob("*.txt")}
    assert {"senior_ml", "senior_security", "lead_engineering_manager", "mid_embedded"} <= stems


# ===========================================================================
# Deterministic checks -- spec 9's six, all free
# ===========================================================================


def test_a_clean_run_passes_every_check() -> None:
    result = run_deterministic_checks("case", make_state())
    assert result.passed, result.failures


def test_a_two_page_resume_fails() -> None:
    result = run_deterministic_checks("case", make_state(page_count=2))
    assert not result.passed
    assert any("2 pages" in f for f in result.failures)


def test_overfull_boxes_fail() -> None:
    log = "Overfull \\hbox (12.0pt too wide) in paragraph at lines 1--2"
    result = run_deterministic_checks("case", make_state(compile_log=log))
    assert result.overfull_boxes == 1
    assert not result.passed


def test_a_compile_error_fails() -> None:
    state = make_state(compile_log="! Undefined control sequence.\nl.7 \\nope", pdf_path=None)
    result = run_deterministic_checks("case", state)
    assert result.compiled is False
    assert not result.passed


def test_a_fabricated_number_is_caught_by_the_eval_itself() -> None:
    """Re-run, not trusted from the run.

    The eval's job is to audit the fabrication gate. If the verifier ever stops
    being called, this is what notices -- so it recomputes rather than reading a
    field the agent filled in.
    """
    state = make_state(
        tailored=[
            TailoredBullet(
                source_id=SOURCE_ID,
                text="Cut latency by 94% and saved 7919 hours",
                estimated_lines=1,
            )
        ]
    )
    result = run_deterministic_checks("case", state)

    assert SOURCE_ID in result.fabricated_numbers
    assert "7919" in result.fabricated_numbers[SOURCE_ID]
    assert not result.passed


def test_must_have_coverage_is_measured() -> None:
    """One of two must-haves covered = 50%, under spec 9's 70% floor."""
    job = make_job()
    partial = make_state(
        fit_report=FitReport(
            overall_fit=0.5,
            covered=[
                EvidenceMatch(
                    bullet_id=SOURCE_ID, requirement_text="caching", relevance=0.9, rationale="r"
                )
            ],
            gaps=[job.requirements[1]],
            recommendation="apply",
        )
    )
    result = run_deterministic_checks("case", partial)
    assert result.must_haves_total == 2
    assert result.must_have_coverage == 0.5
    assert MIN_MUST_HAVE_COVERAGE == 0.70
    assert any("coverage" in f for f in result.failures)


def test_coverage_is_not_counted_when_there_are_no_must_haves() -> None:
    """Spec 9 says "where evidence exists"; a posting with no must-haves is not
    a resume that failed."""
    job = make_job(requirements=[])
    state = make_state(
        job_spec=job,
        fit_report=FitReport(overall_fit=1.0, recommendation="strong_apply"),
    )
    result = run_deterministic_checks("case", state)
    assert result.must_haves_total == 0
    assert result.passed


def test_an_over_length_bullet_is_caught() -> None:
    state = make_state(
        tailored=[TailoredBullet(source_id=SOURCE_ID, text="x" * 900, estimated_lines=9)]
    )
    result = run_deterministic_checks("case", state)
    assert result.over_length_bullets == [SOURCE_ID]
    assert not result.passed


def test_a_bullet_with_no_real_source_is_caught() -> None:
    state = make_state(
        tailored=[TailoredBullet(source_id="nope.b9", text="Did a thing", estimated_lines=1)]
    )
    result = run_deterministic_checks("case", state)
    assert "nope.b9" in result.fabricated_numbers
    assert not result.passed


# ===========================================================================
# The judge
# ===========================================================================


class FakeJudge(BaseChatModel):
    scores: JudgeScores = JudgeScores(
        relevance=4,
        specificity=4,
        ats_alignment=4,
        tone_match=4,
        absence_of_filler=4,
        reasoning="fine",
    )
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-resume-judge"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> JudgeScores:
                fake.call_count += 1
                return fake.scores

        return _Runnable()


def test_the_rubric_has_spec_9s_five_dimensions() -> None:
    assert DIMENSIONS == [
        "relevance", "specificity", "ats_alignment", "tone_match", "absence_of_filler",
    ]  # fmt: skip


def test_mean_is_computed_not_returned() -> None:
    """CLAUDE.md rule 2. A model asked for parts and an average will sometimes
    return an average that is not the average."""
    assert "mean" not in JudgeScores.model_fields
    scores = JudgeScores(
        relevance=5,
        specificity=4,
        ats_alignment=3,
        tone_match=2,
        absence_of_filler=1,
        reasoning="r",
    )
    assert scores.mean == 3.0


def test_one_judge_call_per_resume() -> None:
    """Spec 9: "one judge call per resume"."""
    judge = FakeJudge()
    bullets = [
        TailoredBullet(source_id=SOURCE_ID, text=TRUTHFUL, estimated_lines=2),
        TailoredBullet(source_id="x.b2", text="Another bullet", estimated_lines=1),
    ]
    judge_resume(make_job(), bullets, llm=judge)
    assert judge.call_count == 1


def test_an_empty_resume_is_not_sent_to_the_judge() -> None:
    """Already a deterministic failure; judging it would charge for the news."""
    judge = FakeJudge()
    assert judge_resume(make_job(), [], llm=judge) is None
    assert judge.call_count == 0


def test_scores_are_bounded() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeScores(
            relevance=6,
            specificity=4,
            ats_alignment=4,
            tone_match=4,
            absence_of_filler=4,
            reasoning="r",
        )


# ===========================================================================
# M8 DoD: the table, and the gate that must fire
# ===========================================================================


def _report(variant: str, means: list[float]) -> EvalReport:
    """A report whose judge scores are exactly `means`."""
    report = EvalReport(variant=variant, ran_at="2026-09-05T00:00:00")
    for index, mean in enumerate(means):
        value = int(round(mean))
        report.cases.append(
            CaseResult(
                jd_name=f"jd_{index}",
                deterministic=run_deterministic_checks(f"jd_{index}", make_state()),
                judge=JudgeScores(
                    relevance=value,
                    specificity=value,
                    ats_alignment=value,
                    tone_match=value,
                    absence_of_filler=value,
                    reasoning="r",
                ),
            )
        )
    return report


def test_the_table_renders() -> None:
    """M8 DoD: "`make eval` produces a table"."""
    table = render_table(_report("baseline", [4.0, 3.0]))
    for expected in ["jd_0", "jd_1", "mean", "deterministic pass rate", "spot-check"]:
        assert expected in table


def test_the_table_names_three_resumes_to_spot_check() -> None:
    """Spec 9: "Human spot-check: 3 resumes per eval run ... your eye is the
    calibration set." Lowest-scoring first, so the three are worth looking at."""
    table = render_table(_report("baseline", [5.0, 1.0, 2.0, 4.0, 3.0]))
    section = table.split("spot-check")[1]
    assert "jd_1" in section and "jd_2" in section


def test_the_gate_passes_when_scores_hold() -> None:
    baseline = _report("baseline", [4.0, 4.0]).to_dict()
    ok, message = check_regression(_report("candidate", [4.0, 4.0]), baseline)
    assert ok
    assert "held" in message


def test_the_gate_fires_on_a_worsened_score() -> None:
    """The most important test in this file.

    Spec 9: "a PR that drops mean judge score by >0.3 doesn't merge." A harness
    whose gate never fires is worse than no harness, because it looks like a
    safety net.
    """
    baseline = _report("baseline", [4.0, 4.0]).to_dict()
    worse = _report("worsened", [3.0, 3.0])  # a full point down

    ok, message = check_regression(worse, baseline)

    assert ok is False
    assert "REGRESSION" in message
    assert "4.000" in message and "3.000" in message


def test_the_gate_tolerates_a_drop_within_the_threshold() -> None:
    """Judges are noisy; a gate that fires on noise gets switched off."""
    baseline = {"mean_judge_score": 4.0}
    small_drop = _report("candidate", [3.8, 3.8])
    ok, _ = check_regression(small_drop, baseline)
    assert ok
    assert REGRESSION_THRESHOLD == 0.3


def test_a_report_round_trips_through_json() -> None:
    """It is written to disk and read back by `--check-against`."""
    payload = json.loads(json.dumps(_report("baseline", [4.0]).to_dict()))
    assert payload["mean_judge_score"] == 4.0
    assert payload["cases"][0]["judge"]["mean"] == 4.0


# ===========================================================================
# The worsened variant
# ===========================================================================


def test_the_worsened_variant_exists_and_is_registered() -> None:
    assert "worsened" in VARIANTS
    for path in VARIANTS["worsened"].values():
        assert path.is_file(), f"variant prompt missing: {path}"


def test_the_variant_actually_changes_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override has to be visible to `load_prompt`.

    Caching on the prompt *name* rather than the resolved path would make the
    variant invisible for the rest of the process -- and a variant that silently
    scores identically to the baseline is the one result an eval must never
    produce by accident.
    """
    baseline_text = load_prompt("tailor_bullets")

    variant = VARIANTS["worsened"]["tailor_bullets"]
    monkeypatch.setenv(PROMPT_OVERRIDE_ENV_VAR, f"tailor_bullets={variant}")

    assert prompt_path("tailor_bullets") == variant
    assert load_prompt("tailor_bullets") != baseline_text
    assert "results-driven" in load_prompt("tailor_bullets")


def test_the_worsened_prompt_inverts_the_real_rules() -> None:
    """It must be bad in ways the *judge* can see, not ways a regex catches.

    It deliberately never invites inventing a number or a technology -- those
    are caught for free by the deterministic layer, and the point of this
    variant is to show the judge detecting a quality drop no check could.
    """
    worsened = VARIANTS["worsened"]["tailor_bullets"].read_text(encoding="utf-8")
    body = worsened.split("<!--")[0]

    assert "results-driven" in body and "passionate" in body
    assert "high level" in body.lower() or "broad" in body.lower()
    # And it must not tell the writer to make things up.
    assert "invent" not in body.lower()
    assert "make up" not in body.lower()


def test_normal_runs_are_unaffected_by_the_variant_machinery() -> None:
    """No override set means the real prompt, byte for byte."""
    os.environ.pop(PROMPT_OVERRIDE_ENV_VAR, None)
    assert prompt_path("tailor_bullets").name == "tailor_bullets.md"
    assert "You may rephrase. You may not add facts." in load_prompt("tailor_bullets")


# ===========================================================================
# M11 J3: scoring by Jev, and the comparison that decides the default
# ===========================================================================


def test_choosing_jev_needs_its_own_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.run_eval import _choose_scorer  # noqa: PLC0415

    monkeypatch.delenv("RESUME_AGENT_JEV_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="RESUME_AGENT_JEV_API_KEY"):
        _choose_scorer("jev")


def test_choosing_a_scorer_sets_and_clears_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    from evals.run_eval import _choose_scorer  # noqa: PLC0415

    monkeypatch.setenv("RESUME_AGENT_JEV_API_KEY", "sk-or-test")
    monkeypatch.delenv("RESUME_AGENT_JEV_SCORING", raising=False)

    _choose_scorer("jev")
    assert os.environ["RESUME_AGENT_JEV_SCORING"] == "1"

    _choose_scorer("llm")
    assert "RESUME_AGENT_JEV_SCORING" not in os.environ


def test_the_comparison_shows_what_scoring_changed() -> None:
    """Scoring decides what gets selected, so the comparison has to show the
    picks and the coverage, not only the judge score."""
    from evals.run_eval import compare  # noqa: PLC0415

    before = {"decisions": "llm", "mean_judge_score": 3.8, "cases": [{
        "jd_name": "mid", "covered": 5, "selected": ["a", "b", "c", "d"],
        "judge": {"mean": 3.8}, "scoring": {"by": "model", "cached": True},
    }]}
    after = {"decisions": "jev", "mean_judge_score": 3.75, "cases": [{
        "jd_name": "mid", "covered": 6, "selected": ["a", "b", "c", "e"],
        "judge": {"mean": 3.75}, "scoring": {"by": "jev", "cached": False, "seconds": 1.4},
    }]}

    text = compare(before, after)

    assert "llm -> jev" in text
    assert "5->6" in text, "requirements covered, before and after"
    assert "60%" in text, "3 of the 5 distinct picks are shared"
    assert "3.80->3.75" in text
    assert "cached->1.4" in text
    assert "3.800 -> 3.750" in text
    assert "within 0.1" in text, "the rule for switching the default is printed with it"


def test_a_report_records_who_scored(tmp_path: Path) -> None:
    report = EvalReport(variant="baseline", ran_at="now", decisions="jev")
    report.cases.append(CaseResult(
        jd_name="mid",
        deterministic=DeterministicResult(
            jd_name="mid", compiled=True, page_count=1, overfull_boxes=0, first_error=None
        ),
        scoring={"by": "jev", "seconds": 1.2}, covered=4, selected=["a"],
    ))

    saved = json.loads(json.dumps(report.to_dict()))

    assert saved["decisions"] == "jev"
    assert saved["cases"][0]["scoring"]["by"] == "jev"
    assert saved["cases"][0]["covered"] == 4
