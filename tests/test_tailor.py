"""The tailoring node itself.

The grounding *cycle* -- retry, cap, and dropping rather than shipping -- lives
in the graph now (spec 2 draws it as a cycle, and M5 made it one), so those
tests are in `test_graph.py` where they exercise the real routing. What is left
here is the single-pass behaviour of `tailor_bullets`.

Everything runs against fake models; nothing here costs money.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.graph.nodes.tailor import (
    TailoringResult,
    tailor_bullets,
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
