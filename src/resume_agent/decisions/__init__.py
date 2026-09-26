"""Jev: fast, typed judgements for the pipeline's decisions (M11). Optional.

Jev (TypeSafe AI) answers questions about a piece of text with numbers -- the
probability a statement is true, which option fits, where something sits on a
scale. It writes nothing. This project uses it only where a step *judges*, and
keeps every step that writes, counts or compares dates exactly as it was
(CLAUDE.md rule 2, and Jev's own documented weak spots).

**Off unless `RESUME_AGENT_JEV_API_KEY` is set**, and every use falls back to
the pre-Jev path on `DecisionsUnavailable`. Its own variable, not
`OPENROUTER_API_KEY`: that one also takes part in choosing the *writing*
provider, and switching on a judge must never change who writes.

Settings are read on every call, like `resolve_provider`, so a test or an eval
can change them without restarting anything.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from urllib.parse import urlparse

from resume_agent.decisions.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DecisionsUnavailable,
    JevClient,
    JevSettings,
    Questions,
    State,
)
from resume_agent.decisions.questions import (
    Choice,
    ChoiceAnswer,
    Decision,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    load_question,
)

API_KEY_ENV_VAR = "RESUME_AGENT_JEV_API_KEY"
BASE_URL_ENV_VAR = "RESUME_AGENT_JEV_BASE_URL"
MODEL_ENV_VAR = "RESUME_AGENT_JEV_MODEL"

__all__ = [
    "API_KEY_ENV_VAR",
    "BASE_URL_ENV_VAR",
    "MODEL_ENV_VAR",
    "Choice",
    "ChoiceAnswer",
    "Decision",
    "DecisionsUnavailable",
    "Noul",
    "NoulAnswer",
    "Score",
    "ScoreAnswer",
    "ask",
    "ask_many",
    "jev_enabled",
    "jev_settings",
    "load_question",
    "recipient",
    "uses",
]


def jev_settings() -> JevSettings | None:
    """The configured settings, or None when Jev is switched off."""
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        return None
    return JevSettings(
        api_key=key,
        base_url=os.environ.get(BASE_URL_ENV_VAR, "").strip() or DEFAULT_BASE_URL,
        model=os.environ.get(MODEL_ENV_VAR, "").strip() or DEFAULT_MODEL,
    )


def jev_enabled() -> bool:
    return jev_settings() is not None


@lru_cache(maxsize=4)
def _client_for(settings: JevSettings) -> JevClient:
    # One client per configuration, so connections are reused across calls.
    return JevClient(settings)


def _client() -> JevClient:
    settings = jev_settings()
    if settings is None:
        raise DecisionsUnavailable("Jev is switched off")
    return _client_for(settings)


def ask(state: State, questions: Questions) -> Decision:
    """Every question about one state. Raises `DecisionsUnavailable` to mean
    "take the path you took before Jev"."""
    return _client().ask(state, questions)


def ask_many(requests: Sequence[tuple[State, Questions]]) -> list[Decision]:
    """Many small requests at once, in order. All of them, or `DecisionsUnavailable`."""
    return _client().ask_many(requests)


def recipient() -> str:
    """Who receives what Jev is sent, as the privacy page should name them."""
    settings = jev_settings()
    host = urlparse(settings.base_url).hostname if settings else ""
    if host and host.endswith("openrouter.ai"):
        return "TypeSafe, through OpenRouter"
    if host and host.endswith("typesafe.ai"):
        return "TypeSafe"
    return f"TypeSafe, through {host}" if host else "TypeSafe"


def uses() -> list[str]:
    """What Jev is sent, in the privacy page's words, for each use that is on.

    Grows as each use is added, so the privacy page never claims more, or
    less, than the code does.
    """
    return [
        # api/posting_check.py (M11 J2)
        "each job ad you paste, to check it is one before a resume is made",
    ]
