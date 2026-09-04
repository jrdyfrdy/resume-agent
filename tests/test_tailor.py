"""Tailoring, and the grounding cycle's hard cap.

The cap is the part that matters. Spec 5: "After 2 failures, **drop the bullet
and log it loudly**. Never ship an unverified claim because the retry budget ran
out." A loop that quietly shipped its last attempt would defeat the verifier
entirely, so `test_unfixable_bullet_is_dropped_not_shipped` is the test standing
between this project and the failure mode it exists to prevent.

Everything runs against fake models; nothing here costs money.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.graph.nodes.tailor import (
    TailoringResult,
    tailor_bullets,
    tailor_until_grounded,
    target_characters,
)
from resume_agent.latex.metrics import CHARS_PER_LINE
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.profile import Bullet, Profile
from resume_agent.models.resume import TailoredBulletFields

SOURCE = Bullet(
    id="exp_halvorsen_bright.b1",
    canonical=(
        "Cut p95 checkout latency from 820ms to ~50ms with a Redis read-through "
        "cache and by removing N+1 queries across 12 endpoints"
    ),
    skills=["redis", "caching"],
    metrics={"latency_before_ms": 820, "latency_after_ms": 50, "endpoints_touched": 12},
)

JOB = JobSpec.from_fields(
    JobSpecFields(
        company="Acme",
        title="Backend Engineer",
        seniority="mid",
        domain="payments",
        requirements=[
            Requirement(text="caching", category="practice", weight=5, is_must_have=True)
        ],
        responsibilities=[],
        ats_keywords=["Redis", "caching"],
        culture_signals=[],
        tone="formal",
        red_flags=[],
    ),
    "raw",
)


class ScriptedWriter(BaseChatModel):
    """A tailoring model that returns a fixed sequence of texts, one per attempt."""

    texts: list[str] = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-writer"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> TailoringResult:
                index = min(fake.call_count, len(fake.texts) - 1)
                fake.call_count += 1
                return TailoringResult(
                    bullets=[TailoredBulletFields(source_id=SOURCE.id, text=fake.texts[index])]
                )

        return _Runnable()


@pytest.fixture
def profile_with_source(example_profile: Profile) -> Profile:
    """The example profile, guaranteed to contain SOURCE's real counterpart."""
    return example_profile


# --- the happy path ---------------------------------------------------------


def test_tailors_and_verifies_in_one_pass(profile_with_source: Profile) -> None:
    writer = ScriptedWriter(
        texts=["Cut p95 checkout latency from 820ms to 50ms with a Redis read-through cache"]
    )
    verified, dropped = tailor_until_grounded(
        JOB, [SOURCE], profile_with_source, llm=writer, use_judge=False
    )

    assert dropped == []
    assert len(verified) == 1
    assert writer.call_count == 1, "retried a bullet that already passed"


def test_estimated_lines_is_computed_not_returned(profile_with_source: Profile) -> None:
    """CLAUDE.md rule 2 -- the model never counts lines."""
    text = "Cut p95 checkout latency from 820ms to 50ms with a Redis read-through cache"
    writer = ScriptedWriter(texts=[text])

    verified, _ = tailor_until_grounded(
        JOB, [SOURCE], profile_with_source, llm=writer, use_judge=False
    )

    assert verified[0].estimated_lines == 1
    assert "estimated_lines" not in TailoredBulletFields.model_fields


# --- the retry cycle --------------------------------------------------------


def test_a_rejected_bullet_is_retried_and_can_recover(profile_with_source: Profile) -> None:
    """The critique goes back to the writer, and a fixed rewrite is accepted."""
    writer = ScriptedWriter(
        texts=[
            "Cut p95 checkout latency by 94% with a Redis cache",  # invented metric
            "Cut p95 checkout latency from 820ms to 50ms with a Redis cache",  # fixed
        ]
    )
    verified, dropped = tailor_until_grounded(
        JOB, [SOURCE], profile_with_source, llm=writer, use_judge=False
    )

    assert dropped == []
    assert len(verified) == 1
    assert "94%" not in verified[0].text
    assert writer.call_count == 2


def test_unfixable_bullet_is_dropped_not_shipped(
    profile_with_source: Profile, caplog: pytest.LogCaptureFixture
) -> None:
    """The most important behaviour in this module.

    A writer that keeps fabricating must end with the bullet **absent**, not
    with its last attempt quietly on the page. Spec 5: "Never ship an unverified
    claim because the retry budget ran out."
    """
    writer = ScriptedWriter(texts=["Cut p95 checkout latency by 94% with a Redis cache"])

    with caplog.at_level(logging.ERROR):
        verified, dropped = tailor_until_grounded(
            JOB, [SOURCE], profile_with_source, llm=writer, use_judge=False
        )

    assert verified == [], "shipped an unverified bullet after exhausting retries"
    assert dropped == [SOURCE.id]
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_retries_are_capped_at_two(profile_with_source: Profile) -> None:
    """Two retries means three total attempts, and then it stops."""
    writer = ScriptedWriter(texts=["Cut latency by 94% with Redis"])

    tailor_until_grounded(JOB, [SOURCE], profile_with_source, llm=writer, use_judge=False)

    assert writer.call_count == 3, f"made {writer.call_count} attempts, expected 3"


def test_a_dropped_bullet_does_not_stop_the_others(profile_with_source: Profile) -> None:
    """One unfixable bullet must not take a good one down with it."""
    good = Bullet(id="exp_a.b1", canonical="Built a cache", metrics={})
    bad = Bullet(id="exp_b.b1", canonical="Built a queue", metrics={})

    class MixedWriter(ScriptedWriter):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            fake = self

            class _Runnable:
                def invoke(self, _messages: Any, **_kw: Any) -> TailoringResult:
                    fake.call_count += 1
                    return TailoringResult(
                        bullets=[
                            TailoredBulletFields(source_id="exp_a.b1", text="Built a cache"),
                            TailoredBulletFields(
                                source_id="exp_b.b1", text="Built a queue serving 9973 users"
                            ),
                        ]
                    )

            return _Runnable()

    verified, dropped = tailor_until_grounded(
        JOB, [good, bad], profile_with_source, llm=MixedWriter(), use_judge=False
    )

    assert [v.source_id for v in verified] == ["exp_a.b1"]
    assert dropped == ["exp_b.b1"]


def test_a_missing_rewrite_is_retried_then_dropped(profile_with_source: Profile) -> None:
    """A model that returns nothing for a bullet must not silently lose it."""

    class SilentWriter(ScriptedWriter):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            fake = self

            class _Runnable:
                def invoke(self, _messages: Any, **_kw: Any) -> TailoringResult:
                    fake.call_count += 1
                    return TailoringResult(bullets=[])

            return _Runnable()

    verified, dropped = tailor_until_grounded(
        JOB, [SOURCE], profile_with_source, llm=SilentWriter(), use_judge=False
    )

    assert verified == []
    assert dropped == [SOURCE.id]


# --- inputs to the prompt ---------------------------------------------------


def test_invented_source_ids_are_dropped(profile_with_source: Profile) -> None:
    """A rewrite attached to a bullet that does not exist is a claim about nothing."""

    class LiarWriter(ScriptedWriter):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            class _Runnable:
                def invoke(self, _messages: Any, **_kw: Any) -> TailoringResult:
                    return TailoringResult(
                        bullets=[TailoredBulletFields(source_id="exp_fake.b9", text="Did a thing")]
                    )

            return _Runnable()

    assert tailor_bullets(JOB, [SOURCE], profile_with_source, llm=LiarWriter()) == []


def test_target_characters_tracks_the_measured_line_width() -> None:
    """The budget the writer is given comes from the same constant selection spent."""
    one_liner = Bullet(id="x.b1", canonical="short")
    assert target_characters(one_liner) == CHARS_PER_LINE

    two_lines = Bullet(id="x.b2", canonical="y" * (CHARS_PER_LINE + 10))
    assert target_characters(two_lines) == CHARS_PER_LINE * 2 - 6


def test_empty_input_makes_no_call(profile_with_source: Profile) -> None:
    writer = ScriptedWriter(texts=["unused"])
    assert tailor_bullets(JOB, [], profile_with_source, llm=writer) == []
    assert writer.call_count == 0
