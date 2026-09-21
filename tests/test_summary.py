"""The professional summary, and the gate that keeps it honest.

The summary is the only generated line on the resume that is not a rewrite of
one `canonical` sentence, so it is the only one where the per-bullet gate has no
single source to check against. These tests are about the substitute: the joined
text of the selected achievements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from resume_agent.graph.nodes.summary import (
    MAX_SUMMARY_ATTEMPTS,
    check_summary,
    summary_source,
    write_summary,
)
from resume_agent.kb.loader import load_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
AN_ENTRY_BULLET = "exp_halvorsen_bright.b1"


@pytest.fixture
def profile():
    return load_profile(REPO_ROOT / "profile.example")


class ScriptedWriter(BaseChatModel):
    """Returns each scripted reply in turn, and records what it was asked."""

    replies: list[str] = []
    calls: int = 0
    prompts: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-writer"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    def invoke(self, messages: Any, **_kw: Any) -> AIMessage:
        self.prompts.append("\n".join(str(m.content) for m in messages))
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return AIMessage(content=reply)


# ===========================================================================
# The source
# ===========================================================================


def test_the_source_is_the_selected_achievements_only(profile) -> None:
    """Not the whole profile. A summary that promises something the page does
    not then evidence is worse than no summary."""
    source = summary_source(profile, [AN_ENTRY_BULLET], {})

    assert profile.bullet_by_id(AN_ENTRY_BULLET).canonical in source
    assert len(source.splitlines()) == 1


def test_tailored_text_wins_over_canonical(profile) -> None:
    """The gate should check against the words that will actually print under
    the summary, not the ones they were rewritten from."""
    source = summary_source(
        profile, [AN_ENTRY_BULLET], {AN_ENTRY_BULLET: "A rewritten sentence."}
    )

    assert source == "A rewritten sentence."


# ===========================================================================
# The gate
# ===========================================================================


def test_a_number_no_achievement_supports_is_caught() -> None:
    source = "Cut p95 checkout latency from 820ms to 50ms across 12 endpoints."

    # "94%" normalises to "94" -- `extract_numbers` strips the unit so that
    # "50ms" in a bullet supports "50 ms" in a rewrite.
    assert check_summary(
        "Backend engineer who cut latency 94% across the platform.", source
    ) == {"94"}


def test_a_number_the_achievements_do_state_is_allowed() -> None:
    source = "Cut p95 checkout latency from 820ms to 50ms across 12 endpoints."

    assert not check_summary(
        "Backend engineer who took a checkout path from 820ms to 50ms.", source
    )


def test_a_technology_in_skills_but_not_on_the_page_is_caught() -> None:
    """Kubernetes IS in profile.example's skills, and is still caught -- the
    claim here is about the page, and no selected achievement shows it. Passing
    `skill_vocabulary()` here would have let the gate contradict rule 2 of
    `write_summary.md`, which this test was written to check."""
    source = "Held 100% uptime on the payments ingest using replay-safe consumers."

    # Lowercased: `extract_tech_tokens` normalises case, since a bullet writing
    # "postgres" must support a rewrite writing "PostgreSQL".
    assert "kubernetes" in check_summary(
        "Backend engineer experienced with Kubernetes.", source
    )


# ===========================================================================
# The retry, and what happens at the cap
# ===========================================================================


def test_an_honest_summary_is_returned_on_the_first_try(profile) -> None:
    model = ScriptedWriter(replies=["Backend engineer who works on payment reliability."])

    text, complaints = write_summary(
        profile, "Backend Engineer", [AN_ENTRY_BULLET], {}, llm=model
    )

    assert text == "Backend engineer who works on payment reliability."
    assert complaints == []
    assert model.calls == 1


def test_the_retry_is_told_what_it_invented(profile) -> None:
    model = ScriptedWriter(replies=[
        "Backend engineer with 9 years of experience.",
        "Backend engineer who works on payment reliability.",
    ])

    text, complaints = write_summary(
        profile, "Backend Engineer", [AN_ENTRY_BULLET], {}, llm=model
    )

    assert text == "Backend engineer who works on payment reliability."
    assert complaints == []
    assert model.calls == 2
    # The information the first attempt was missing, handed back to it.
    assert "9" in model.prompts[1]
    assert "rejected_last_attempt" in model.prompts[1]


def test_at_the_cap_the_summary_is_dropped_rather_than_printed(profile) -> None:
    """A resume with no summary is complete. A resume whose opening line claims
    something it does not evidence is worse than both."""
    model = ScriptedWriter(replies=["Backend engineer with 9 years of experience."])

    text, complaints = write_summary(
        profile, "Backend Engineer", [AN_ENTRY_BULLET], {}, llm=model
    )

    assert text == ""
    assert len(complaints) == MAX_SUMMARY_ATTEMPTS
    assert model.calls == MAX_SUMMARY_ATTEMPTS


def test_nothing_selected_means_nothing_to_summarise(profile) -> None:
    model = ScriptedWriter(replies=["Should never be asked for."])

    text, complaints = write_summary(profile, "Backend Engineer", [], {}, llm=model)

    assert text == ""
    assert complaints == ["no selected achievements to summarise"]
    assert model.calls == 0
