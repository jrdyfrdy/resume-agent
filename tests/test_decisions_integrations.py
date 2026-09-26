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


# ===========================================================================
# J3: scoring
# ===========================================================================


def rated(rate: Callable[[dict, str], float]) -> Answerer:
    """Answer each rating question with `rate(state, requirement text)`."""
    def answer(state, questions):
        return {
            name: {
                "type": "score",
                "score": rate(state, question["instructions"].rsplit("Requirement: ", 1)[1]),
                "confidence": 0.9,
                "probabilities": {"0": 1.0},
            }
            for name, question in questions.items()
        }
    return answer


@pytest.fixture
def scoring(jev: FakeJev, monkeypatch) -> FakeJev:
    monkeypatch.setenv(decisions.SCORING_ENV_VAR, "1")
    return jev


def test_one_request_per_achievement_with_a_question_per_requirement(
    scoring: FakeJev, example_profile
) -> None:
    from resume_agent.graph.nodes.score import score_fit  # noqa: PLC0415
    from tests.test_fit import make_job, req  # noqa: PLC0415

    job = make_job(req("kubernetes"), req("python"), req("leadership"))
    ids = [b.id for b in example_profile.all_bullets()][:4]
    scoring.answer = rated(lambda _state, _req: 4.0)
    details: dict = {}

    score_fit(job, ids, example_profile, use_cache=False, details=details)

    assert len(scoring.asked) == 4, "one request per achievement"
    state, questions = scoring.asked[0]
    assert state["achievement"] == example_profile.bullet_by_id(ids[0]).canonical.strip()
    assert sorted(q["instructions"].rsplit("Requirement: ", 1)[1] for q in questions.values()) == [
        "kubernetes", "leadership", "python"
    ]
    assert all(len(q["criteria"]) == 5 for q in questions.values())
    seconds = details.pop("seconds")
    assert details == {"by": "jev", "model": "jev-1.13", "cached": False,
                       "requests": 4, "input_tokens": 400}
    assert isinstance(seconds, float), "how long scoring took, for the eval comparison"


def test_levels_land_in_the_bands_the_thresholds_expect(scoring: FakeJev, example_profile):
    """Level / 4: direct 1.0 and strong 0.75 are covered (>= 0.7), partial 0.5 is
    partial (>= 0.4), weak 0.25 is a gap, and unrelated is left out entirely."""
    from resume_agent.graph.nodes.score import build_fit_report, score_fit  # noqa: PLC0415
    from tests.test_fit import make_job, req  # noqa: PLC0415

    job = make_job(req("direct"), req("strong"), req("partial"), req("weak"), req("unrelated"))
    level = {"direct": 4, "strong": 3, "partial": 2, "weak": 1, "unrelated": 0.2}
    ids = [example_profile.all_bullets()[0].id]
    scoring.answer = rated(lambda _state, requirement: level[requirement])

    matches = score_fit(job, ids, example_profile, use_cache=False)
    relevance = {m.requirement_text: m.relevance for m in matches}
    report = build_fit_report(job, matches)

    assert relevance == {"direct": 1.0, "strong": 0.75, "partial": 0.5, "weak": 0.25}
    assert {m.requirement_text for m in report.covered} == {"direct", "strong"}
    assert {m.requirement_text for m in report.partial} == {"partial"}
    assert {r.text for r in report.gaps} == {"weak", "unrelated"}
    assert all(m.rationale == "" for m in matches), "Jev rates; it does not explain"


def test_scoring_stays_with_the_model_unless_switched_on(jev: FakeJev, example_profile):
    """Jev on for the posting check does not mean Jev scores: that changes what
    gets selected, so it has its own switch, off until the evals say so."""
    from resume_agent.graph.nodes.score import ScoringResult, score_fit  # noqa: PLC0415
    from tests.test_fit import FakeScoringModel, make_job, req  # noqa: PLC0415

    model = FakeScoringModel(result=ScoringResult(matches=[]))
    details: dict = {}

    score_fit(make_job(req("a")), [example_profile.all_bullets()[0].id], example_profile,
              llm=model, use_cache=False, details=details)

    assert jev.asked == []
    assert model.call_count == 1
    assert details["by"] == "model"


def test_jev_down_means_the_model_scores(scoring: FakeJev, example_profile, caplog) -> None:
    from resume_agent.graph.nodes.score import ScoringResult, score_fit  # noqa: PLC0415
    from tests.test_fit import FakeScoringModel, make_job, req  # noqa: PLC0415

    scoring.down = True
    model = FakeScoringModel(result=ScoringResult(matches=[]))
    details: dict = {}

    score_fit(make_job(req("a")), [example_profile.all_bullets()[0].id], example_profile,
              llm=model, use_cache=False, details=details)

    assert model.call_count == 1
    assert details["by"] == "model"
    assert "asking the model instead" in caplog.text


def test_jev_scores_are_cached_like_the_models(scoring: FakeJev, example_profile) -> None:
    """The layout loop re-runs everything after scoring; the scores never change."""
    from resume_agent.graph.nodes.score import score_fit  # noqa: PLC0415
    from tests.test_fit import make_job, req  # noqa: PLC0415

    job = make_job(req("a"))
    ids = [example_profile.all_bullets()[0].id]
    scoring.answer = rated(lambda _state, _req: 3.0)

    first = score_fit(job, ids, example_profile)
    details: dict = {}
    second = score_fit(job, ids, example_profile, details=details)

    assert first == second
    assert len(scoring.asked) == 1
    assert details["cached"] is True


def test_the_score_node_says_who_scored(scoring: FakeJev, example_profile) -> None:
    from pathlib import Path  # noqa: PLC0415

    from resume_agent.graph.build import node_score  # noqa: PLC0415
    from resume_agent.graph.state import RunOptions  # noqa: PLC0415
    from tests.test_fit import make_job, req  # noqa: PLC0415

    scoring.answer = rated(lambda _state, _req: 4.0)
    example = Path(__file__).resolve().parent.parent / "profile.example"
    state = {
        "job_spec": make_job(req("a")),
        "candidates": [example_profile.all_bullets()[0].id],
        "profile_path": str(example),
        "options": RunOptions(use_cache=False),
    }

    result = node_score(state)

    assert result["scoring"]["by"] == "jev"
    assert result["fit_report"].covered


def test_the_privacy_page_mentions_achievements_only_when_jev_scores(jev: FakeJev, monkeypatch):
    scoring_line = "how well each shows each requirement"
    assert not any(scoring_line in use for use in decisions.uses())

    monkeypatch.setenv(decisions.SCORING_ENV_VAR, "1")

    assert any(scoring_line in use for use in decisions.uses())


# ===========================================================================
# J4: chat routing
# ===========================================================================

# No "I", under 25 words, no question word: the rules read it as advice.
DICTATION_THE_RULES_MISS = "Built a Kubernetes operator at Acme that cut deploys to 5 minutes."


def intent(choice: str, confidence: float) -> Answerer:
    return lambda _state, questions: {
        name: {"type": "choice", "choice": choice, "confidence": confidence,
               "probabilities": {choice: confidence}}
        for name in questions
    }


def test_jev_routes_the_message_when_it_is_sure(jev: FakeJev) -> None:
    from resume_agent.chat.session import classify, route  # noqa: PLC0415

    jev.answer = intent("dictation", 0.92)

    assert classify(DICTATION_THE_RULES_MISS) == "advise", "the rules get this one wrong"
    assert route(DICTATION_THE_RULES_MISS) == "extract"
    state, questions = jev.asked[0]
    assert state == DICTATION_THE_RULES_MISS
    assert set(questions["intent"]["criteria"]) == {"advice", "dictation"}


def test_an_unsure_jev_leaves_it_to_the_rules(jev: FakeJev) -> None:
    from resume_agent.chat.session import route  # noqa: PLC0415

    jev.answer = intent("dictation", 0.55)

    assert route(DICTATION_THE_RULES_MISS) == "advise"


def test_a_removal_is_never_put_to_jev(jev: FakeJev) -> None:
    """The removal rule stays first: answering a removal with advice produced
    prose agreeing to do it, and nothing done."""
    from resume_agent.chat.session import route  # noqa: PLC0415

    jev.answer = intent("advice", 0.99)

    assert route("Can you remove my narratives?") == "extract"
    assert jev.asked == []


def test_jev_down_leaves_it_to_the_rules(jev: FakeJev) -> None:
    from resume_agent.chat.session import route  # noqa: PLC0415

    jev.down = True

    assert route("I rewrote the exporter so it streams.") == "extract"
    assert route("What is weak about my file?") == "advise"


def test_the_chat_endpoint_routes_with_jev(jev: FakeJev) -> None:
    from resume_agent.chat.extract import ExtractionFields  # noqa: PLC0415
    from tests.test_chat import ScriptedExtractor  # noqa: PLC0415

    jev.answer = intent("dictation", 0.9)
    app = create_app(
        graph_factory=lambda **_kw: FakeGraph(), mode="local",
        chat_model_factory=lambda: ScriptedExtractor(fields=ExtractionFields(reply="noted")),
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/chat", json={"profile": "profile.example", "message": DICTATION_THE_RULES_MISS}
        )

    assert created.status_code == 202
    assert created.json()["intent"] == "extract"


# ===========================================================================
# J4: the "not said" flag
# ===========================================================================

SAID = "I helped with the billing migration at Halvorsen, moving about 40 jobs to the new queue."


def proposal_with(example_profile, *canonicals: str):
    """A proposal holding one new job with these achievements."""
    from resume_agent.chat.extract import (  # noqa: PLC0415
        ExtractionFields,
        ProposedBullet,
        ProposedEntry,
        build_proposal,
    )

    fields = ExtractionFields(reply="ok", entries=[ProposedEntry(
        kind="experience", name="Halvorsen", title="Engineer", start="2023-01",
        bullets=[ProposedBullet(canonical=text) for text in canonicals],
    )])
    return build_proposal(fields, SAID, example_profile)


def stated_unless(doubted: str) -> Answerer:
    """Supported, except the achievement whose text contains `doubted`."""
    return lambda _state, questions: {
        name: {"type": "noul", "noul": 0.1 if doubted in question["instructions"] else 0.95}
        for name, question in questions.items()
    }


def test_an_achievement_that_says_more_than_you_did_is_flagged(jev: FakeJev, example_profile):
    """"Led" from "helped with": no number or technology changed, so no other
    check could see it."""
    jev.answer = stated_unless("Led")

    proposal = proposal_with(
        example_profile,
        "Helped migrate about 40 billing jobs to the new queue.",
        "Led the billing migration to the new queue.",
    )

    [entry] = proposal.items
    flags = [f for f in entry.flags if f.kind == "not_stated"]
    assert len(flags) == 1
    assert "Led the billing migration" in flags[0].detail
    assert proposal.flagged() == [entry], "a flagged item arrives unticked"

    assert len(jev.asked) == 1, "one request for the whole proposal"
    state, questions = jev.asked[0]
    assert state == {"what_the_person_wrote": SAID}
    assert len(questions) == 2


def test_a_faithful_proposal_is_not_flagged(jev: FakeJev, example_profile) -> None:
    jev.answer = stated_unless("nothing matches this")

    proposal = proposal_with(example_profile, "Helped migrate about 40 billing jobs.")

    assert not any(f.kind == "not_stated" for f in proposal.items[0].flags)


def test_jev_down_keeps_the_other_checks(jev: FakeJev, example_profile) -> None:
    jev.down = True

    proposal = proposal_with(example_profile, "Migrated 400 billing jobs to Kafka.")

    kinds = {f.kind for f in proposal.items[0].flags}
    assert "not_stated" not in kinds
    assert {"invented_number", "invented_technology"} <= kinds, "the free checks still ran"


def test_jev_off_is_not_asked(jev: FakeJev, example_profile, monkeypatch) -> None:
    monkeypatch.delenv(decisions.API_KEY_ENV_VAR)

    proposal_with(example_profile, "Helped migrate about 40 billing jobs.")

    assert jev.asked == []


def test_the_privacy_page_lists_the_chat_uses(jev: FakeJev) -> None:
    uses = " ".join(decisions.uses())

    assert "chat messages" in uses
    assert "achievements the chat proposes" in uses
