"""The fabrication gate.

Spec 5: "Write a test that feeds the verifier a deliberately fabricated bullet
and asserts it gets caught. **That test is the most important one in the repo.**"

M4's definition of done: "`test_verifier.py` catches an injected fabrication;
verified bullets never contain out-of-vocabulary numbers."

The three fabrications the kickoff names each target a different layer:

    invented metric        -> layer 1, numbers      (free)
    out-of-vocab tech      -> layer 1, vocabulary   (free)
    semantic inflation     -> layer 2, judge        (one cheap call)

The inverse matters just as much and is tested alongside: **truthful rephrasings
must pass.** A verifier that rejects honest rewrites makes the generator worse,
and CLAUDE.md is explicit that the fix for an inconvenient check is better
generation, never a looser check.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.graph.nodes.verify import (
    MAX_GROUNDING_ATTEMPTS,
    verify_grounding,
    verify_numbers,
    verify_vocabulary,
    verify_with_judge,
)
from resume_agent.grounding.numbers import unsupported_numbers
from resume_agent.grounding.vocabulary import unsupported_technologies
from resume_agent.models.profile import Bullet, Profile
from resume_agent.models.resume import JudgeVerdict, TailoredBullet

# The real bullet from profile.example that most of these hang off.
SOURCE = Bullet(
    id="exp_halvorsen_bright.b1",
    canonical=(
        "Cut p95 checkout latency from 820ms to ~50ms with a Redis read-through "
        "cache and by removing N+1 queries across 12 endpoints"
    ),
    skills=["redis", "caching", "sql-optimization", "performance", "fastapi"],
    metrics={"latency_before_ms": 820, "latency_after_ms": 50, "endpoints_touched": 12},
)

COLLABORATION_SOURCE = Bullet(
    id="exp_x.b1",
    canonical="Collaborated with two engineers to migrate the billing service",
    metrics={"engineers": 2},
)


def tailored(text: str, source: Bullet = SOURCE) -> TailoredBullet:
    return TailoredBullet(source_id=source.id, text=text, estimated_lines=1)


class FakeJudge(BaseChatModel):
    """A judge whose verdict is fixed by the test."""

    verdict: JudgeVerdict = JudgeVerdict(verdict="supported", reason="fine")
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-judge"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> JudgeVerdict:
                fake.call_count += 1
                return fake.verdict

        return _Runnable()


# ===========================================================================
# THE THREE INJECTED FABRICATIONS
# ===========================================================================


def test_catches_an_invented_metric(example_profile: Profile) -> None:
    """FABRICATION 1: a number that exists nowhere in the source data.

    The source records 820ms, 50ms and 12 endpoints. "94% reduction" is
    arithmetic the candidate never recorded -- and even though it happens to be
    roughly true of 820 -> 50, computing it is inventing a claim. Spec 3.1:
    "any digit in a generated bullet must trace to a value in `metrics`".
    """
    fabricated = tailored("Cut p95 checkout latency by 94% with a Redis read-through cache")

    result = verify_numbers(fabricated, SOURCE)

    assert result.passed is False
    assert result.layer == "numbers"
    assert "94" in result.critique
    # The critique must be actionable -- it is fed straight back to the writer.
    assert "latency_before_ms" in result.critique


def test_catches_an_out_of_vocabulary_technology(example_profile: Profile) -> None:
    """FABRICATION 2: a technology the candidate has no recorded experience with.

    Spec 3.2: "if the generated bullet says 'Kafka' and Kafka isn't in your
    vocabulary, that's a fabrication." Here the trap is subtler -- Kafka IS in
    profile.example's skills.yaml, so the test uses Cassandra, which is not.
    """
    vocabulary = example_profile.skill_vocabulary()
    assert "cassandra" not in vocabulary, "the fixture must not know this technology"

    fabricated = tailored(
        "Cut p95 checkout latency from 820ms to 50ms with a Cassandra-backed cache"
    )

    result = verify_vocabulary(fabricated, SOURCE, vocabulary)

    assert result.passed is False
    assert result.layer == "vocabulary"
    assert "cassandra" in result.critique.lower()


def test_catches_semantic_inflation() -> None:
    """FABRICATION 3: numerically and technologically clean, but a bigger claim.

    Spec 5's own example. "Led a team of engineers" from "collaborated with two
    engineers" contains no invented number and no unknown technology -- layer 1
    passes it. Only the judge can see that collaboration has been upgraded to
    leadership.
    """
    inflated = tailored(
        "Led a team of engineers to migrate the billing service", COLLABORATION_SOURCE
    )

    # Layer 1 is blind to this, which is exactly why layer 2 exists.
    assert verify_numbers(inflated, COLLABORATION_SOURCE).passed is True
    assert verify_vocabulary(inflated, COLLABORATION_SOURCE, set()).passed is True

    judge = FakeJudge(
        verdict=JudgeVerdict(
            verdict="unsupported",
            reason="'Led a team' overstates 'collaborated with two engineers'.",
        )
    )
    result = verify_with_judge(inflated, COLLABORATION_SOURCE, llm=judge)

    assert result.passed is False
    assert result.layer == "judge"
    assert "collaborated" in result.critique


# ===========================================================================
# The inverse: truthful rewrites must survive
# ===========================================================================


@pytest.mark.parametrize(
    "text",
    [
        # Identical to the source.
        "Cut p95 checkout latency from 820ms to ~50ms with a Redis read-through cache "
        "and by removing N+1 queries across 12 endpoints",
        # Reworded, same facts.
        "Reduced p95 checkout latency from 820ms to 50ms by adding a Redis read-through "
        "cache and eliminating N+1 queries across 12 endpoints",
        # Unit re-expression -- spec 5's own example of an allowed form.
        "Cut p95 checkout latency from 0.82s to 50ms using a Redis read-through cache",
        # Shorter. Dropping detail is not a fabrication.
        "Cut p95 checkout latency from 820ms to 50ms with a Redis read-through cache",
    ],
)
def test_truthful_rewrites_pass_both_free_layers(text: str, example_profile: Profile) -> None:
    """A verifier that rejects honest rewrites is a broken verifier."""
    candidate = tailored(text)
    vocabulary = example_profile.skill_vocabulary()

    numbers = verify_numbers(candidate, SOURCE)
    assert numbers.passed, f"rejected a truthful rewrite: {numbers.critique}"

    vocab = verify_vocabulary(candidate, SOURCE, vocabulary)
    assert vocab.passed, f"rejected a truthful rewrite: {vocab.critique}"


def test_sentence_openers_are_not_read_as_technologies(example_profile: Profile) -> None:
    """The failure mode the naive "capitalised token" check would have.

    Every bullet starts with a capitalised verb. If those were treated as
    technology mentions, the vocabulary layer would reject all of them.
    """
    for opener in ["Cut", "Built", "Shipped", "Migrated", "Reduced", "Designed"]:
        candidate = tailored(f"{opener} p95 checkout latency from 820ms to 50ms with Redis")
        result = verify_vocabulary(candidate, SOURCE, example_profile.skill_vocabulary())
        assert result.passed, f"{opener!r} was misread as a technology"


def test_numbers_already_in_the_source_need_no_metric() -> None:
    """ "N+1 queries" contains a literal 1 that belongs to no metric key.

    Without allowing numbers present in the canonical text, this check would
    reject the source sentence against itself.
    """
    assert unsupported_numbers(SOURCE.canonical, SOURCE.metrics, SOURCE.canonical) == set()


def test_technology_already_in_the_source_needs_no_vocabulary_entry() -> None:
    """A technology the source names is grounded by definition."""
    candidate = tailored("Cut latency with a Redis read-through cache")
    assert unsupported_technologies(candidate.text, SOURCE.canonical, set()) == set()


# ===========================================================================
# Layer ordering and the cheap-first guarantee
# ===========================================================================


def test_layer_one_short_circuits_before_the_judge(example_profile: Profile) -> None:
    """A fabricated metric must never reach a paid call.

    Spec 5: "Two layers, cheap first. [...] LLM judge (runs only on bullets that
    pass layer 1)."
    """
    judge = FakeJudge()
    fabricated = tailored("Cut p95 checkout latency by 94% with a Redis cache")

    result = verify_grounding(fabricated, SOURCE, example_profile.skill_vocabulary(), llm=judge)

    assert result.passed is False
    assert result.layer == "numbers"
    assert judge.call_count == 0, "spent money judging a bullet a regex already rejected"


def test_numbers_are_checked_before_vocabulary(example_profile: Profile) -> None:
    """When a rewrite breaks both, the cheaper and more actionable critique wins."""
    both = tailored("Cut latency by 94% using Cassandra")
    result = verify_grounding(both, SOURCE, example_profile.skill_vocabulary(), llm=FakeJudge())
    assert result.layer == "numbers"


def test_judge_can_be_disabled_for_free_verification(example_profile: Profile) -> None:
    """`judge=False` makes the whole gate free, which the test suite relies on."""
    judge = FakeJudge()
    candidate = tailored("Cut p95 checkout latency from 820ms to 50ms with Redis")

    result = verify_grounding(
        candidate, SOURCE, example_profile.skill_vocabulary(), judge=False, llm=judge
    )

    assert result.passed is True
    assert judge.call_count == 0


def test_clean_bullet_reaches_and_passes_the_judge(example_profile: Profile) -> None:
    judge = FakeJudge()
    candidate = tailored("Cut p95 checkout latency from 820ms to 50ms with Redis")

    result = verify_grounding(candidate, SOURCE, example_profile.skill_vocabulary(), llm=judge)

    assert result.passed is True
    assert judge.call_count == 1


# ===========================================================================
# The DoD's second clause, and the retry cap
# ===========================================================================


def test_verified_bullets_never_contain_out_of_vocabulary_numbers(
    example_profile: Profile,
) -> None:
    """M4 DoD, stated directly: sweep every bullet in the example profile.

    For each one, a rewrite carrying an invented figure must be rejected, and
    the untouched source must be accepted.
    """
    vocabulary = example_profile.skill_vocabulary()

    for source in example_profile.all_bullets():
        clean = TailoredBullet(source_id=source.id, text=source.canonical, estimated_lines=1)
        assert verify_grounding(clean, source, vocabulary, judge=False).passed, (
            f"{source.id} failed verification against its own canonical text"
        )

        # 7919 is prime and appears in no metric anywhere in profile.example.
        fabricated = TailoredBullet(
            source_id=source.id,
            text=f"{source.canonical} across 7919 services",
            estimated_lines=1,
        )
        result = verify_grounding(fabricated, source, vocabulary, judge=False)
        assert result.passed is False, f"{source.id} accepted an invented number"
        assert result.layer == "numbers"


def test_retry_cap_is_two() -> None:
    """CLAUDE.md rule 7: "Grounding: 2 retries then drop the bullet"."""
    assert MAX_GROUNDING_ATTEMPTS == 2


def test_dropping_is_logged_loudly(caplog: pytest.LogCaptureFixture) -> None:
    """Spec 5: "drop the bullet and log it loudly".

    A dropped bullet silently weakens the resume. The log is the only trace,
    so it is an error, and it carries the source text and every critique.
    """
    from resume_agent.graph.nodes.verify import drop_unverified

    with caplog.at_level(logging.ERROR):
        drop_unverified(SOURCE, ["invented 94%", "still invented 94%"])

    assert caplog.records, "dropping a bullet produced no log record at all"
    record = caplog.records[-1]
    assert record.levelno >= logging.ERROR
    assert SOURCE.id in record.getMessage()
    assert "invented 94%" in record.getMessage()
