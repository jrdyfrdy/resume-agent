"""Jev over HTTP: one endpoint, behind this project's own interface.

    POST {base}/v1/systemone      Authorization: Bearer <key>
         {"model", "state", "questions": {name: question}}
      -> {"model", "answers": {name: answer}, "usage": {...}}

Through OpenRouter the base is `https://openrouter.ai/api`. That is its
Decisions API, not the OpenAI-compatible chat endpoint, which cannot carry typed
questions. Directly with TypeSafe the base is `https://api.typesafe.ai`.

**Why not TypeSafe's SDK.** It was ten days old when this was written, and this
is one POST. An interface of our own keeps churn in their SDK out of the nodes,
and a fake of our own keeps the tests independent of theirs.

**Failure is never a crash.** Every error ends in `DecisionsUnavailable`, and
every caller treats that as "do what you did before Jev". An outage makes a run
slower, never broken -- and the key never appears in a log or an error message.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from resume_agent.decisions.questions import Choice, Decision, Noul, Score

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api"
# Pinned, not `jev-latest`: results measured by the eval harness belong to one
# model version, for the same reason prompts are versioned.
DEFAULT_MODEL = "jev-1.13"

TIMEOUT_S = 10.0
# One retry, never more (CLAUDE.md rule 7). 429 is the rate limit and 529
# "overloaded", which TypeSafe's reference says to back off from.
MAX_ATTEMPTS = 2
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
BACKOFF_S = 0.5
# OpenRouter does not publish a rate limit for this API, so many small requests
# are sent at most this many at a time.
MAX_IN_FLIGHT = 16

State = str | Mapping[str, Any] | Sequence[Any]
Questions = Mapping[str, Noul | Choice | Score]


class DecisionsUnavailable(RuntimeError):
    """Jev could not answer: switched off, unreachable, refused, or unreadable.

    Never an error for the person using the app. Callers take the path they
    took before Jev existed.
    """


@dataclass(frozen=True)
class JevSettings:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/systemone"


class JevClient:
    """Asks Jev, and turns every way that can fail into `DecisionsUnavailable`."""

    def __init__(
        self,
        settings: JevSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._sleep = sleep
        self._http = httpx.Client(
            timeout=TIMEOUT_S,
            transport=transport,
            headers={"Authorization": f"Bearer {settings.api_key}"},
        )

    def ask(self, state: State, questions: Questions) -> Decision:
        """Every question about one state, answered together."""
        body = {
            "model": self.settings.model,
            "state": state,
            "questions": {
                name: question.model_dump(exclude_none=True)
                for name, question in questions.items()
            },
        }
        problem = "no attempt was made"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._http.post(self.settings.endpoint, json=body)
            except httpx.TimeoutException:
                problem = f"no answer within {TIMEOUT_S:g}s"
            except httpx.TransportError as exc:
                problem = f"could not reach it ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    return _read(response, questions)
                problem = f"HTTP {response.status_code}: {_detail(response)}"
                if response.status_code not in RETRY_STATUSES:
                    break  # a bad key or a bad request will not improve on retry
            if attempt < MAX_ATTEMPTS:
                self._sleep(BACKOFF_S * (1 + random.random()))

        logger.warning("Jev unavailable, falling back: %s", problem)
        raise DecisionsUnavailable(problem)

    def ask_many(self, requests: Sequence[tuple[State, Questions]]) -> list[Decision]:
        """Many small requests at once, in order. All of them, or none.

        All or none because a caller mixing Jev's answers with a fallback's
        would be comparing two different scales. The first failure cancels
        whatever has not started and is raised.
        """
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=min(MAX_IN_FLIGHT, len(requests))) as pool:
            futures = [pool.submit(self.ask, state, questions) for state, questions in requests]
            try:
                return [future.result() for future in futures]
            except DecisionsUnavailable:
                for future in futures:
                    future.cancel()
                raise


def _read(response: httpx.Response, questions: Questions) -> Decision:
    """The answers, checked against what was asked."""
    try:
        decision = Decision.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise DecisionsUnavailable(f"an answer that could not be read: {exc}") from exc

    for name, question in questions.items():
        answer = decision.answers.get(name)
        if answer is None:
            raise DecisionsUnavailable(f"no answer to {name!r}")
        if answer.type != question.type:
            raise DecisionsUnavailable(f"{name!r} was answered as a {answer.type}")
    return decision


def _detail(response: httpx.Response) -> str:
    """What the error said, briefly. The request is not echoed: it can hold career text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200] or response.reason_phrase
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("detail") or error)[:200]
        return str(error)[:200]
    return str(payload)[:200]
