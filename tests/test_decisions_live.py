"""One real call to Jev, when a key is set. Skipped otherwise.

    RESUME_AGENT_JEV_API_KEY=<an OpenRouter key>  uv run pytest tests/test_decisions_live.py

One request carrying all three question types -- Jev answers every question in
a request together -- costs a fraction of a cent. It is the check that the
settings (the OpenRouter base URL and the pinned model name) are ones the
service actually accepts, which no mock can tell us.
"""

from __future__ import annotations

import pytest

from resume_agent.decisions import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    ask,
    jev_enabled,
)

requires_jev = pytest.mark.skipif(
    not jev_enabled(), reason="RESUME_AGENT_JEV_API_KEY not set -- skipping live Jev call"
)

POSTING = (
    "Backend Engineer, Acme Payments. You will build and run Python services on "
    "Kubernetes. Requirements: 3+ years of Python; experience operating services "
    "in production; PostgreSQL. Nice to have: Go."
)


@requires_jev
def test_jev_answers_every_kind_of_question() -> None:
    decision = ask(POSTING, {
        "is_posting": Noul(instructions="Is this text a job posting?"),
        "seniority": Choice(
            instructions="What level is this role?",
            criteria={
                "entry": "New graduates and interns",
                "mid": "A few years of experience",
                "senior": "Many years, leading others",
            },
        ),
        "python": Score(
            instructions="How central is Python to this role?",
            criteria=["Not mentioned", "Mentioned in passing", "A core requirement"],
        ),
    })

    posting, level, python = (decision.answers[k] for k in ("is_posting", "seniority", "python"))
    assert isinstance(posting, NoulAnswer) and posting.noul > 0.8
    assert isinstance(level, ChoiceAnswer) and level.choice in {"entry", "mid", "senior"}
    assert isinstance(python, ScoreAnswer) and python.score > 1.0
