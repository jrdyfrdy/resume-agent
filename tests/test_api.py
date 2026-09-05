"""The API and its SSE stream. M9.

Spec 8: "FastAPI + SSE streaming of node events; minimal frontend."

Every test drives the real app with a **fake graph** injected through
`create_app(graph_factory=...)`. No API key, no compiler, no network -- which is
the point of having that seam: the endpoints, the event filtering and the SSE
framing are all testable without any of the machinery they front.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from resume_agent.api.app import GRAPH_NODES, create_app
from resume_agent.api.models import summarise_state
from resume_agent.models.fit import FitReport
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.resume import TailoredBullet


def make_job() -> JobSpec:
    return JobSpec.from_fields(
        JobSpecFields(
            company="Acme Corp",
            title="Backend Engineer",
            seniority="mid",
            domain="payments",
            requirements=[
                Requirement(text="kubernetes", category="tool", weight=5, is_must_have=True)
            ],
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )


class FakeGraph:
    """A graph whose `astream_events` yields a scripted event sequence."""

    def __init__(self, events: list[dict] | None = None, boom: str | None = None) -> None:
        self.events = events if events is not None else _default_events()
        self.boom = boom

    async def astream_events(self, _state, _config=None, **_kw) -> Any:
        if self.boom:
            raise RuntimeError(self.boom)
        for event in self.events:
            await asyncio.sleep(0)
            yield event


def _default_events() -> list[dict]:
    """A plausible run: three nodes, one model call, a final state."""
    final = {
        "job_spec": make_job(),
        "fit_report": FitReport(
            overall_fit=0.62, gaps=make_job().requirements, recommendation="stretch"
        ),
        "tailored": [
            TailoredBullet(source_id="a.b1", text="Cut p95 latency to 50ms", estimated_lines=1)
        ],
        "dropped_bullets": ["a.b9"],
        "page_count": 1,
        "line_budget": 32,
        "layout_attempts": 0,
        "pdf_path": None,
        "errors": [],
    }
    return [
        {"event": "on_chain_start", "name": "parse_jd", "data": {}},
        {"event": "on_chain_end", "name": "parse_jd", "data": {"output": {}}},
        {"event": "on_chat_model_start", "name": "ChatAnthropic", "data": {}},
        {"event": "on_chat_model_end", "name": "ChatAnthropic", "data": {}},
        # An internal runnable the page has no use for; must be filtered out.
        {"event": "on_chain_start", "name": "RunnableSequence", "data": {}},
        {"event": "on_chain_start", "name": "tailor", "data": {}},
        {"event": "on_chain_end", "name": "tailor", "data": {"output": {}}},
        {"event": "on_chain_start", "name": "finalize", "data": {}},
        {"event": "on_chain_end", "name": "finalize", "data": {"output": final}},
    ]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(graph_factory=lambda **_kw: FakeGraph()))


def read_events(client: TestClient, run_id: str) -> list[dict]:
    """Drain the SSE stream into a list of parsed events."""
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() != "{}":
                    events.append(json.loads(payload))
        return events


# ===========================================================================
# Read-only endpoints
# ===========================================================================


def test_the_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "resume-agent" in response.text
    assert "EventSource" in response.text, "the page does not actually stream"


def test_profile_summary(client: TestClient) -> None:
    body = client.get("/api/profile").json()
    assert body["name"] == "John Doe"
    assert body["bullets"] == 13
    # The page uses these to disable Run before you wait for a failure.
    assert "has_credentials" in body
    assert "has_compiler" in body


def test_an_unknown_profile_is_a_400_not_a_500(client: TestClient) -> None:
    response = client.get("/api/profile", params={"profile": "does_not_exist"})
    assert response.status_code == 400


def test_applications_endpoint(client: TestClient) -> None:
    assert client.get("/api/applications").status_code == 200


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.get("/api/runs/nope/pdf").status_code == 404


# ===========================================================================
# Starting a run
# ===========================================================================


def test_starting_a_run_returns_an_id(client: TestClient) -> None:
    response = client.post("/api/runs", json={"jd": "Backend engineer wanted."})
    assert response.status_code == 202
    assert len(response.json()["run_id"]) == 12


def test_an_empty_jd_is_rejected(client: TestClient) -> None:
    """`min_length=1` on the request model, so it never reaches the graph."""
    assert client.post("/api/runs", json={"jd": ""}).status_code == 422


# ===========================================================================
# The SSE stream -- M9's actual subject
# ===========================================================================


def test_node_events_stream(client: TestClient) -> None:
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    events = read_events(client, run_id)

    kinds = [(e["kind"], e["name"]) for e in events]
    assert ("node_start", "parse_jd") in kinds
    assert ("node_end", "parse_jd") in kinds
    assert ("node_start", "tailor") in kinds
    assert ("status", "done") in kinds


def test_internal_runnables_are_filtered_out(client: TestClient) -> None:
    """A UI showing every internal runnable is noise, not progress."""
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    names = {e["name"] for e in read_events(client, run_id)}

    assert "RunnableSequence" not in names
    assert names <= GRAPH_NODES | {"model", "done", "failed"}


def test_model_calls_are_surfaced(client: TestClient) -> None:
    """The reason for `astream_events` over updates-mode.

    Without this the page is a frozen spinner for the twenty seconds a tailoring
    call takes, and a frozen spinner is indistinguishable from a hang.
    """
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    events = read_events(client, run_id)

    model_events = [e for e in events if e["kind"].startswith("model")]
    assert model_events, "no model activity reached the page"
    assert model_events[0]["detail"] == "ChatAnthropic"


def test_events_carry_elapsed_time(client: TestClient) -> None:
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    events = read_events(client, run_id)
    assert all(e["elapsed_s"] is not None for e in events if e["kind"] != "error")


def test_reconnecting_replays_the_whole_log(client: TestClient) -> None:
    """The reason the stream reads a history list rather than draining a queue.

    A queue would have been emptied by the first connection, so a refresh
    mid-run would show an empty log.
    """
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]

    first = read_events(client, run_id)
    second = read_events(client, run_id)

    assert first == second
    assert len(second) > 3


# ===========================================================================
# Finished runs
# ===========================================================================


def test_the_summary_is_available_after_the_run(client: TestClient) -> None:
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    read_events(client, run_id)  # drain to completion

    body = client.get(f"/api/runs/{run_id}").json()
    assert body["status"] == "done"
    assert body["company"] == "Acme Corp"
    assert body["recommendation"] == "stretch"
    assert body["bullets"] == ["Cut p95 latency to 50ms"]
    # Surfaced, not buried: the resume is weaker than it could have been.
    assert body["dropped_bullets"] == ["a.b9"]
    assert body["must_have_gaps"] == ["kubernetes"]


def test_a_failing_run_reports_rather_than_hanging() -> None:
    """A page left spinning forever is the worst outcome; it must be told."""
    app = create_app(graph_factory=lambda **_kw: FakeGraph(boom="tectonic exploded"))
    client = TestClient(app)

    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    events = read_events(client, run_id)

    assert any(e["kind"] == "error" for e in events)
    body = client.get(f"/api/runs/{run_id}").json()
    assert body["status"] == "failed"
    assert "tectonic exploded" in body["errors"][0]


def test_no_pdf_is_a_404_not_a_broken_embed(client: TestClient) -> None:
    run_id = client.post("/api/runs", json={"jd": "Backend engineer."}).json()["run_id"]
    read_events(client, run_id)
    assert client.get(f"/api/runs/{run_id}/pdf").status_code == 404


# ===========================================================================
# The wire shape
# ===========================================================================


def test_summarise_state_is_json_safe() -> None:
    """The browser gets a flat view, not AgentState with Pydantic objects in it."""
    summary = summarise_state(
        "abc",
        "done",
        {
            "job_spec": make_job(),
            "fit_report": FitReport(overall_fit=0.5, recommendation="apply"),
            "tailored": [TailoredBullet(source_id="a.b1", text="Did a thing", estimated_lines=1)],
        },
    )
    payload = json.loads(summary.model_dump_json())
    assert payload["company"] == "Acme Corp"
    assert payload["bullets"] == ["Did a thing"]


def test_summarising_an_empty_state_does_not_crash() -> None:
    """A run that failed before parse_jd still has to render something."""
    summary = summarise_state("abc", "failed", {})
    assert summary.company is None
    assert summary.bullets == []


def test_graph_nodes_list_matches_the_real_graph() -> None:
    """The event filter names nodes explicitly; a renamed node would silently
    stop appearing in the UI, so the list is checked against the real graph."""
    from resume_agent.graph.build import build_graph

    real = {n for n in build_graph().get_graph().nodes if not n.startswith("__")}
    assert real <= GRAPH_NODES, f"nodes missing from the UI filter: {real - GRAPH_NODES}"


def test_the_page_has_no_build_step() -> None:
    """Spec 8 asks for a "minimal frontend"; a bundler in a Python repo is a
    second toolchain to install and explain."""
    page = (Path(__file__).resolve().parent.parent
            / "src" / "resume_agent" / "api" / "static" / "index.html")
    text = page.read_text(encoding="utf-8")
    assert "<script src=" not in text, "the page pulls in an external script"
    assert "import " not in text.split("<script>")[1][:200]
