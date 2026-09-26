"""Each place Jev is used (M11), against a stand-in Jev server.

`FakeJev` answers through `httpx.MockTransport` behind the real `JevClient`, so
every test runs the whole path -- question file, request, answer, threshold,
fallback -- with no network and no key. It records what it was asked, so a
test can also prove Jev was *not* asked.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import resume_agent.decisions as decisions
from resume_agent.api.app import create_app
from resume_agent.api.posting_check import REFUSAL, posting_refusal
from resume_agent.decisions.client import JevClient
from tests.test_api import FakeGraph

KEY = "sk-or-test-not-a-real-key"

Answerer = Callable[[Any, dict], dict]  # (state, questions) -> {name: answer}


class FakeJev:
    """A Jev server. `answer` decides each reply; `down` makes it fail instead."""

    def __init__(self, answer: Answerer | None = None) -> None:
        self.answer = answer or (lambda _state, questions: {})
        self.down = False
        self.asked: list[tuple[Any, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.asked.append((body["state"], body["questions"]))
        if self.down:
            return httpx.Response(529, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={
            "model": body["model"],
            "answers": self.answer(body["state"], body["questions"]),
            "usage": {"input_tokens": 100, "output_tokens": 0},
        })


@pytest.fixture
def jev(monkeypatch) -> FakeJev:
    """Jev switched on, answered by a `FakeJev`."""
    fake = FakeJev()
    monkeypatch.setenv(decisions.API_KEY_ENV_VAR, KEY)
    monkeypatch.setattr(
        decisions,
        "_client_for",
        lambda settings: JevClient(
            settings, transport=httpx.MockTransport(fake.handler), sleep=lambda _s: None
        ),
    )
    return fake


def noul(probability: float) -> Answerer:
    return lambda _state, questions: {
        name: {"type": "noul", "noul": probability} for name in questions
    }


# ===========================================================================
# J2: the posting check
# ===========================================================================


def test_a_paste_that_is_not_a_job_ad_is_turned_away(jev: FakeJev) -> None:
    jev.answer = noul(0.05)

    assert posting_refusal("Jane Doe\nSoftware engineer, 5 years of Python...") == REFUSAL
    state, questions = jev.asked[0]
    assert state.startswith("Jane Doe")
    assert questions["is_posting"]["type"] == "noul"
    assert "job posting" in questions["is_posting"]["instructions"]


@pytest.mark.parametrize("probability", [0.2, 0.55, 0.99])
def test_anything_not_clearly_junk_goes_ahead(jev: FakeJev, probability: float) -> None:
    """Refusing a real posting costs someone a resume; letting junk through
    costs one run, as it always did. So only the clear cases are refused."""
    jev.answer = noul(probability)

    assert posting_refusal("Backend Engineer at Acme...") is None


def test_jev_down_or_off_means_the_run_goes_ahead(jev: FakeJev, monkeypatch) -> None:
    jev.down = True
    assert posting_refusal("anything at all") is None

    monkeypatch.delenv(decisions.API_KEY_ENV_VAR)
    asked = len(jev.asked)
    assert posting_refusal("anything at all") is None
    assert len(jev.asked) == asked, "switched off: not even asked"


def test_a_huge_paste_is_trimmed_not_refused(jev: FakeJev) -> None:
    jev.answer = noul(0.9)

    posting_refusal("x" * 100_000)

    assert len(jev.asked[0][0]) == 24_000


def test_the_run_endpoint_refuses_before_starting_anything(jev: FakeJev) -> None:
    jev.answer = noul(0.03)
    client = TestClient(create_app(graph_factory=lambda **_kw: FakeGraph(), mode="local"))

    response = client.post("/api/runs", json={"jd": "my resume, pasted by mistake"})

    assert response.status_code == 422
    assert response.json()["detail"] == REFUSAL


def test_a_real_posting_starts_a_run(jev: FakeJev) -> None:
    jev.answer = noul(0.98)
    with TestClient(create_app(graph_factory=lambda **_kw: FakeGraph(), mode="local")) as client:
        response = client.post("/api/runs", json={"jd": "Backend Engineer at Acme..."})

    assert response.status_code == 202


def test_a_turned_away_paste_does_not_use_up_a_run_on_the_hosted_site(jev: FakeJev, tmp_path):
    """The point of checking first: the daily limit counts runs started."""
    from resume_agent.accounts.auth import AuthSettings, MultiUser  # noqa: PLC0415
    from resume_agent.accounts.db import AccountsDB  # noqa: PLC0415
    from resume_agent.accounts.workspace import Workspaces  # noqa: PLC0415
    from tests.test_accounts_isolation import FakeGoogle, ReadsTheProfile  # noqa: PLC0415

    db = AccountsDB(f"sqlite:///{tmp_path / 'a.db'}")
    db.create_schema()
    google = FakeGoogle()
    multi = MultiUser(
        db=db, workspaces=Workspaces(db, tmp_path / "ws"), google_identity=google,
        settings=AuthSettings(session_secret="s" * 48, admin_email="owner@example.com"),
    )
    graph = ReadsTheProfile()
    with TestClient(create_app(graph_factory=lambda **_kw: graph, mode="multiuser",
                               multiuser=multi)) as client:
        google.claims = {"sub": "o", "email": "owner@example.com",
                         "email_verified": True, "name": "Owner"}
        client.get("/auth/callback", follow_redirects=False)
        client.post("/api/profile/create", json={"mode": "empty", "display_name": "Owner"})
        me = client.get("/api/me").json()

        jev.answer = noul(0.01)
        response = client.post("/api/runs", json={"jd": "hello?", "profile": me["id"]})

    assert response.status_code == 422
    assert db.runs(me["id"]) == [], "nothing counted against today's runs"


def test_the_privacy_page_says_job_ads_are_checked(jev: FakeJev) -> None:
    assert any("job ad" in use for use in decisions.uses())
