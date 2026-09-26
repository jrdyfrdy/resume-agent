"""The Jev client (M11 J1): request shape, answers, and every way it can fail.

Driven through `httpx.MockTransport`, so the real client code runs -- URL,
headers, body, retries, parsing -- with no network and no key. Live calls are
in `test_decisions_live.py`, and skip without `RESUME_AGENT_JEV_API_KEY`.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import resume_agent.decisions as decisions
from resume_agent.decisions import (
    Choice,
    ChoiceAnswer,
    DecisionsUnavailable,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    load_question,
)
from resume_agent.decisions.client import JevClient, JevSettings

KEY = "sk-or-test-not-a-real-key"
SETTINGS = JevSettings(api_key=KEY)

QUESTIONS = {
    "is_posting": Noul(
        instructions="Is this a job posting?",
        criteria={"true": "It describes a role and what it needs.", "false": "Anything else."},
    ),
    "intent": Choice(
        instructions="What is the person doing?",
        criteria={"advice": "Asking for advice.", "dictation": "Describing their work."},
    ),
    "evidence": Score(
        instructions="How well does the achievement evidence the requirement?",
        criteria=["No evidence", "Partial evidence", "Direct evidence"],
    ),
}

ANSWERS = {
    "model": "jev-1.13",
    "answers": {
        "is_posting": {"type": "noul", "noul": 0.97},
        "intent": {
            "type": "choice", "choice": "dictation", "confidence": 0.9,
            "probabilities": {"advice": 0.05, "dictation": 0.95},
            "a_field_added_next_month": True,
        },
        "evidence": {
            "type": "score", "score": 1.7, "confidence": 0.62,
            "probabilities": {"0": 0.05, "1": 0.2, "2": 0.75},
            "legend": {"0": "No evidence", "1": "Partial evidence", "2": "Direct evidence"},
        },
    },
    "usage": {"input_tokens": 212, "output_tokens": 0},
}


def client_answering(*responses, seen: list | None = None) -> JevClient:
    """A client whose server replies with `responses` in turn. Each is a
    `httpx.Response`, or an exception to raise instead."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return JevClient(SETTINGS, transport=httpx.MockTransport(handler), sleep=lambda _s: None)


def ok(payload: dict = ANSWERS) -> httpx.Response:
    return httpx.Response(200, json=payload)


# ===========================================================================
# The request
# ===========================================================================


def test_the_request_goes_to_openrouters_decisions_api() -> None:
    seen: list[httpx.Request] = []
    client_answering(ok(), seen=seen).ask("Backend Engineer at Acme ...", QUESTIONS)

    request = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.method == "POST"
    assert request.headers["authorization"] == f"Bearer {KEY}"

    body = json.loads(request.content)
    assert body["model"] == "jev-1.13", "pinned, not jev-latest"
    assert body["state"] == "Backend Engineer at Acme ..."
    assert body["questions"]["is_posting"] == {
        "type": "noul",
        "instructions": "Is this a job posting?",
        "criteria": {"true": "It describes a role and what it needs.", "false": "Anything else."},
    }
    assert body["questions"]["intent"]["criteria"] == {
        "advice": "Asking for advice.", "dictation": "Describing their work."
    }
    assert body["questions"]["evidence"]["criteria"] == [
        "No evidence", "Partial evidence", "Direct evidence"
    ]


def test_a_state_can_be_structured() -> None:
    """TypeSafe's re-ranking recipe sends each pair as named fields."""
    seen: list[httpx.Request] = []
    state = {"requirement": "Kubernetes", "achievement": "Ran 40 services on EKS"}
    client_answering(ok(), seen=seen).ask(state, QUESTIONS)

    assert json.loads(seen[0].content)["state"] == state


def test_a_noul_without_criteria_sends_none() -> None:
    seen: list[httpx.Request] = []
    payload = {"answers": {"q": {"type": "noul", "noul": 0.5}}}
    client_answering(ok(payload), seen=seen).ask("x", {"q": Noul(instructions="True?")})

    assert "criteria" not in json.loads(seen[0].content)["questions"]["q"]


# ===========================================================================
# The answers
# ===========================================================================


def test_every_kind_of_answer_is_read() -> None:
    decision = client_answering(ok()).ask("x", QUESTIONS)

    noul, choice, score = (decision.answers[k] for k in ("is_posting", "intent", "evidence"))
    assert isinstance(noul, NoulAnswer) and noul.noul == 0.97
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "dictation"
    assert choice.probabilities["dictation"] == 0.95
    assert isinstance(score, ScoreAnswer) and score.score == 1.7
    assert score.probabilities[2] == 0.75, "levels keyed by number"
    assert decision.usage.input_tokens == 212


def test_an_answer_missing_or_of_the_wrong_kind_is_unavailable() -> None:
    missing = {"answers": {"is_posting": {"type": "noul", "noul": 0.9}}}
    with pytest.raises(DecisionsUnavailable, match="no answer to 'intent'"):
        client_answering(ok(missing)).ask("x", QUESTIONS)

    wrong = {"answers": {"q": {"type": "noul", "noul": 0.9}}}
    with pytest.raises(DecisionsUnavailable, match="answered as a noul"):
        client_answering(ok(wrong)).ask("x", {"q": QUESTIONS["intent"]})


def test_an_unreadable_answer_is_unavailable() -> None:
    with pytest.raises(DecisionsUnavailable, match="could not be read"):
        client_answering(httpx.Response(200, text="<html>gateway</html>")).ask("x", QUESTIONS)


# ===========================================================================
# Failing without failing the run
# ===========================================================================


@pytest.mark.parametrize("status", [429, 503, 529])
def test_a_busy_service_is_asked_once_more(status: int) -> None:
    seen: list[httpx.Request] = []
    decision = client_answering(httpx.Response(status), ok(), seen=seen).ask("x", QUESTIONS)

    assert len(seen) == 2
    assert decision.answers["is_posting"].noul == 0.97


def test_it_gives_up_after_one_retry_and_says_why() -> None:
    seen: list[httpx.Request] = []
    busy = httpx.Response(529, json={"error": {"message": "overloaded"}})

    with pytest.raises(DecisionsUnavailable, match="HTTP 529: overloaded") as caught:
        client_answering(busy, busy, seen=seen).ask("x", QUESTIONS)

    assert len(seen) == 2, "one retry, never more (CLAUDE.md rule 7)"
    assert KEY not in str(caught.value)


def test_a_bad_key_is_not_retried() -> None:
    seen: list[httpx.Request] = []
    refused = httpx.Response(401, json={"error": {"message": "invalid api key"}})

    with pytest.raises(DecisionsUnavailable, match="HTTP 401") as caught:
        client_answering(refused, seen=seen).ask("x", QUESTIONS)

    assert len(seen) == 1
    assert KEY not in str(caught.value)


def test_a_timeout_or_a_dropped_connection_is_unavailable() -> None:
    timeout = httpx.ReadTimeout("slow")
    with pytest.raises(DecisionsUnavailable, match="no answer within 10s"):
        client_answering(timeout, timeout).ask("x", QUESTIONS)

    dropped = httpx.ConnectError("refused")
    assert client_answering(dropped, ok()).ask("x", QUESTIONS).answers, "retried, then fine"


def test_the_key_is_not_in_the_settings_repr() -> None:
    assert KEY not in repr(SETTINGS)


# ===========================================================================
# Many at once
# ===========================================================================


def test_many_requests_come_back_in_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        return ok({"answers": {"q": {"type": "noul", "noul": float(state) / 100}}})

    client = JevClient(SETTINGS, transport=httpx.MockTransport(handler), sleep=lambda _s: None)
    question = {"q": Noul(instructions="?")}

    results = client.ask_many([(str(n), question) for n in range(40)])

    assert [round(d.answers["q"].noul * 100) for d in results] == list(range(40))


def test_many_is_all_or_nothing() -> None:
    """A caller mixing Jev's scores with a fallback's would compare two scales."""
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["state"] == "7":
            return httpx.Response(401)
        return ok({"answers": {"q": {"type": "noul", "noul": 0.5}}})

    client = JevClient(SETTINGS, transport=httpx.MockTransport(handler), sleep=lambda _s: None)

    with pytest.raises(DecisionsUnavailable):
        client.ask_many([(str(n), {"q": Noul(instructions="?")}) for n in range(20)])


# ===========================================================================
# Settings
# ===========================================================================


def test_off_unless_its_own_key_is_set(monkeypatch) -> None:
    monkeypatch.delenv(decisions.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-other")  # the writing provider's

    assert decisions.jev_enabled() is False
    with pytest.raises(DecisionsUnavailable, match="switched off"):
        decisions.ask("x", QUESTIONS)


def test_settings_default_to_openrouter_and_a_pinned_model(monkeypatch) -> None:
    monkeypatch.setenv(decisions.API_KEY_ENV_VAR, KEY)
    monkeypatch.delenv(decisions.BASE_URL_ENV_VAR, raising=False)
    monkeypatch.delenv(decisions.MODEL_ENV_VAR, raising=False)

    settings = decisions.jev_settings()

    assert settings.endpoint == "https://openrouter.ai/api/v1/systemone"
    assert settings.model == "jev-1.13"
    assert decisions.recipient() == "TypeSafe, through OpenRouter"


def test_a_direct_typesafe_account_works_too(monkeypatch) -> None:
    monkeypatch.setenv(decisions.API_KEY_ENV_VAR, KEY)
    monkeypatch.setenv(decisions.BASE_URL_ENV_VAR, "https://api.typesafe.ai/")
    monkeypatch.setenv(decisions.MODEL_ENV_VAR, "jev-latest")

    settings = decisions.jev_settings()

    assert settings.endpoint == "https://api.typesafe.ai/v1/systemone"
    assert settings.model == "jev-latest"
    assert decisions.recipient() == "TypeSafe"


# ===========================================================================
# Question wording lives in prompts/decide_*.md (CLAUDE.md rule 5)
# ===========================================================================


def _question_file(tmp_path: Path, monkeypatch, text: str) -> None:
    path = tmp_path / "decide_example.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("RESUME_AGENT_PROMPT_OVERRIDES", f"decide_example={path}")


def test_a_question_file_loads(tmp_path, monkeypatch) -> None:
    _question_file(tmp_path, monkeypatch, (
        "---\n"
        "type: noul\n"
        "criteria:\n"
        "  true: It describes a job someone could apply for.\n"   # a bare YAML boolean key
        "  false: It is something else.\n"
        "---\n"
        "Is this text a job posting?\n"
    ))

    question = load_question("example")

    assert isinstance(question, Noul)
    assert question.instructions == "Is this text a job posting?"
    assert question.criteria == {
        "true": "It describes a job someone could apply for.",
        "false": "It is something else.",
    }


@pytest.mark.parametrize("text, problem", [
    ("Is this true?\n", "front matter"),
    ("---\ntype: maybe\n---\nIs this true?\n", "type must be"),
    ("---\ntype: score\ncriteria: [only one]\n---\nHow good?\n", "2 to 10 levels"),
])
def test_a_broken_question_file_fails_at_load_time(tmp_path, monkeypatch, text, problem) -> None:
    _question_file(tmp_path, monkeypatch, text)

    with pytest.raises(ValueError, match=problem):
        load_question("example")


# ===========================================================================
# The privacy page names Jev only for what it is actually sent
# ===========================================================================


def test_the_privacy_page_says_nothing_about_jev_until_a_use_is_on(monkeypatch) -> None:
    from resume_agent.accounts.auth import _jev_notice

    monkeypatch.setenv(decisions.API_KEY_ENV_VAR, KEY)
    assert _jev_notice() == "", "switched on, but nothing uses it yet"

    monkeypatch.setattr(decisions, "uses", lambda: ["the job ad, to check it is one"])
    notice = _jev_notice()
    assert "TypeSafe, through OpenRouter" in notice
    assert "the job ad, to check it is one" in notice

    monkeypatch.delenv(decisions.API_KEY_ENV_VAR)
    assert _jev_notice() == "", "switched off"
