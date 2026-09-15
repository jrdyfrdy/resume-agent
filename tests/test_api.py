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
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from resume_agent.api.app import GRAPH_NODES, create_app, discover_profiles
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


def test_profile_summary_names_the_provider_to_configure(monkeypatch, client: TestClient) -> None:
    """The page tells you which variable to set, so it has to be told which
    provider is active. Hardcoding ANTHROPIC_API_KEY in the page would send a
    DeepSeek user to set a variable that changes nothing."""
    monkeypatch.setenv("RESUME_AGENT_PROVIDER", "deepseek")
    body = client.get("/api/profile").json()

    assert body["provider"] == "deepseek"
    assert body["credentials_var"] == "DEEPSEEK_API_KEY"


def test_a_bad_provider_name_blocks_the_page_rather_than_500ing(
    monkeypatch, client: TestClient
) -> None:
    monkeypatch.setenv("RESUME_AGENT_PROVIDER", "nonsense")
    response = client.get("/api/profile")

    assert response.status_code == 200
    assert response.json()["has_credentials"] is False


def test_the_page_has_four_wired_up_tabs() -> None:
    """Tabs are only an improvement if they are reachable. Each one needs a
    button and the panel it names, or a keyboard user lands on a dead control."""
    text = page_source()

    for tab in ("tailor", "profile", "guide", "history"):
        assert f'id="tab-{tab}"' in text
        assert f'id="panel-{tab}"' in text
        assert f'aria-controls="panel-{tab}"' in text


def test_the_tabs_are_a_real_tablist() -> None:
    """`role="tablist"` without roving tabindex and arrow keys is a div that
    looks like tabs to a screen reader and behaves like nothing."""
    text = page_source()

    assert 'role="tablist"' in text
    assert 'role="tabpanel"' in text
    assert "ArrowRight" in text and "ArrowLeft" in text


def test_the_guide_says_where_the_knowledge_base_goes() -> None:
    """The question this tab exists to answer. `profile/` is gitignored and is
    not created by anything, so nothing else on disk tells you."""
    text = page_source()

    assert "cp -r profile.example profile" in text
    assert "gitignored" in text


def test_the_run_request_carries_the_chosen_profile() -> None:
    """The picker is decoration unless the value reaches the run."""
    assert "profile: activeProfile," in page_source()


def test_the_profile_name_is_only_written_through_its_guard() -> None:
    """Found in the browser, not in review.

    The cost note is rewritten wholesale when something blocks a run, which
    destroys the `<b id="note-profile">` inside it. Writing to that element
    directly then throws, and because it happens inside `loadProfiles` the whole
    page stops updating with nothing in the UI to say why. `showActiveProfile`
    is the guarded accessor; nothing may bypass it.
    """
    text = page_source()

    assert "function showActiveProfile()" in text
    assert '$("note-profile").textContent =' not in text.replace(
        'const el = $("note-profile");', ""
    ), "write to note-profile outside showActiveProfile()"


def test_every_class_the_page_uses_is_defined() -> None:
    """Caught a real one: `.sr-only` survived in the markup but its rule was
    lost in a rewrite, so a screen-reader-only label rendered as a heading in
    the middle of the editor. A class that styles nothing is either dead markup
    or a missing rule, and both are worth knowing about."""
    text = page_source()
    styles = text.split("<style>")[1].split("</style>")[0]

    used = set(re.findall(r'class="([^"]+)"', text))
    # Class attributes inside template literals interpolate, so a raw split
    # yields fragments like `bullet${bullet.confidence`. Only real CSS
    # identifiers are candidates.
    names = {
        name
        for group in used
        for name in group.split()
        if re.fullmatch(r"[a-zA-Z][\w-]*", name)
    }

    undefined = {name for name in names if f".{name}" not in styles}
    assert not undefined, f"classes used in markup but never styled: {sorted(undefined)}"


def test_the_editor_explains_what_a_save_does() -> None:
    """The user is editing non-regenerable data through a browser. The three
    guarantees that make that safe are worth stating where they act."""
    text = page_source()
    assert "validated before anything" in text
    assert "backup" in text


def test_the_page_hardcodes_no_vendor_key_name() -> None:
    """The blocked state is rendered from `credentials_var`, so a vendor
    variable spelled out in the page would be a bug that only shows up for
    whoever is not using that vendor."""
    assert "ANTHROPIC_API_KEY" not in page_source()


def test_an_unknown_profile_is_a_400_not_a_500(client: TestClient) -> None:
    response = client.get("/api/profile", params={"profile": "does_not_exist"})
    assert response.status_code == 400


def test_applications_endpoint(client: TestClient) -> None:
    assert client.get("/api/applications").status_code == 200


# ===========================================================================
# Choosing and browsing a knowledge base
# ===========================================================================


def test_profiles_are_discovered(client: TestClient) -> None:
    names = [p["name"] for p in client.get("/api/profiles").json()]
    assert "profile.example" in names


def test_a_profile_that_does_not_load_is_listed_with_its_error(tmp_path: Path) -> None:
    """Hiding it would be worse. A validation error is the likeliest thing to go
    wrong while someone is first writing their YAML, and "my profile vanished
    from the dropdown" is far harder to act on than an error next to its name."""
    broken = tmp_path / "profile"
    broken.mkdir()
    (broken / "identity.yaml").write_text("name: [this is not an identity", encoding="utf-8")

    options = discover_profiles(tmp_path)

    assert len(options) == 1
    assert options[0].name == "profile" and options[0].loads is False
    assert options[0].error


def test_real_profiles_sort_above_the_example(tmp_path: Path) -> None:
    for name in ("profile.example", "profile"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "identity.yaml").write_text("broken", encoding="utf-8")

    assert [o.name for o in discover_profiles(tmp_path)] == ["profile", "profile.example"]


def test_a_directory_without_identity_is_not_a_profile(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "resume.pdf").write_bytes(b"%PDF")

    assert discover_profiles(tmp_path) == []


def test_profile_detail_exposes_the_bullets(client: TestClient) -> None:
    """The Profile tab's whole purpose: until now the knowledge base was
    invisible from the UI, so you could not tell whether a bullet you had
    written was being read at all."""
    detail = client.get("/api/profile/detail").json()

    assert detail["person"] == "John Doe"
    assert len(detail["entries"]) == 5
    assert sum(len(e["bullets"]) for e in detail["entries"]) == 13

    bullets = [b for e in detail["entries"] for b in e["bullets"]]
    with_metrics = [b for b in bullets if b["metrics"]]
    assert with_metrics, "no metrics reached the page"
    # Stringified on the wire: the page renders them either way, and the
    # grounding gate compares them as text.
    assert all(isinstance(v, str) for b in with_metrics for v in b["metrics"].values())


def test_profile_detail_groups_skills_by_category(client: TestClient) -> None:
    categories = client.get("/api/profile/detail").json()["skills_by_category"]
    assert "language" in categories
    assert categories["language"] == sorted(categories["language"])


def test_profile_detail_on_an_unknown_profile_is_a_400(client: TestClient) -> None:
    assert client.get("/api/profile/detail", params={"profile": "nope"}).status_code == 400


# ===========================================================================
# Editing
# ===========================================================================


@pytest.mark.parametrize(
    "endpoint,params",
    [
        ("/api/profile", {"profile": "../.."}),
        ("/api/profile/detail", {"profile": "../.."}),
        ("/api/profile/files", {"profile": "../.."}),
        ("/api/profile/file", {"profile": "../..", "path": "identity.yaml"}),
    ],
)
def test_the_profile_parameter_cannot_escape_the_allow_list(
    client: TestClient, endpoint: str, params: dict
) -> None:
    """`profile` comes from the browser. Read-only it was merely sloppy; with a
    save endpoint in the same app it chooses where writes land, so every
    endpoint resolves it through `discover_profiles()`."""
    assert client.get(endpoint, params=params).status_code == 400


def test_a_run_cannot_name_a_profile_outside_the_allow_list(client: TestClient) -> None:
    """Refused on the request that named it, rather than inside a stream the
    page has not connected to yet."""
    response = client.post("/api/runs", json={"jd": "Backend engineer.", "profile": "../.."})
    assert response.status_code == 400


def test_the_file_list_is_offered(client: TestClient) -> None:
    body = client.get("/api/profile/files").json()
    assert "identity.yaml" in body["files"]
    assert "experience/halvorsen_bright.yaml" in body["files"]


def test_reading_a_file_returns_it_verbatim(client: TestClient) -> None:
    text = client.get("/api/profile/file", params={"path": "identity.yaml"}).json()["text"]
    on_disk = (Path("profile.example") / "identity.yaml").read_text(encoding="utf-8")
    assert text == on_disk


@pytest.mark.parametrize("path", ["../pyproject.toml", "/etc/passwd", "experience/x.yml"])
def test_reading_outside_the_profile_is_refused(client: TestClient, path: str) -> None:
    assert client.get("/api/profile/file", params={"path": path}).status_code == 400


def test_a_save_with_a_bad_path_is_a_400_not_a_result(client: TestClient) -> None:
    """A path the editor should never have sent is a bug, not bad YAML, so it
    gets a status code rather than an `ok=False` body the page would render as
    a validation message."""
    response = client.put(
        "/api/profile/file",
        json={"profile": "profile.example", "path": "../escape.yaml", "text": "x"},
    )
    assert response.status_code == 400


def test_invalid_content_is_a_result_not_an_error_status(client: TestClient) -> None:
    """Invalid YAML is an ordinary outcome of editing. 200 with `ok=False` keeps
    the editor's two branches honest: one renders a message, the other is a bug.

    Uses `profile.example` deliberately -- a refused save must not touch disk,
    and this asserts that on the very file the golden snapshot renders from.
    """
    before = (Path("profile.example") / "identity.yaml").read_bytes()

    response = client.put(
        "/api/profile/file",
        json={"profile": "profile.example", "path": "identity.yaml", "text": "name: [broken\n"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False and body["backup"] is None
    assert "not valid YAML" in body["error"]
    assert (Path("profile.example") / "identity.yaml").read_bytes() == before


def test_creating_a_profile_refuses_an_unsafe_name(client: TestClient) -> None:
    """The name becomes a directory. A slash or a dot-dot must never reach the
    filesystem layer, so it is constrained by the request model itself."""
    for name in ["../escape", "a/b", "", "x" * 65]:
        response = client.post("/api/profile/create", json={"name": name})
        assert response.status_code == 422, name


def test_creating_over_an_existing_profile_is_refused(client: TestClient) -> None:
    response = client.post("/api/profile/create", json={"name": "profile.example"})
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


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


def page_source() -> str:
    return (Path(__file__).resolve().parent.parent
            / "src" / "resume_agent" / "api" / "static" / "index.html").read_text(encoding="utf-8")


def test_the_page_has_no_build_step() -> None:
    """Spec 8 asks for a "minimal frontend"; a bundler in a Python repo is a
    second toolchain to install and explain."""
    text = page_source()
    assert "<script src=" not in text, "the page pulls in an external script"
    assert "import " not in text.split("<script>")[1][:200]


def test_the_page_fetches_nothing_off_this_machine() -> None:
    """`serve` is a localhost tool that has to come up with the network
    unplugged, so a webfont or a CDN stylesheet is a real failure mode and not
    just a preference. Only same-origin `/api/...` URLs are allowed."""
    for attribute in re.findall(r'(?:src|href)="([^"]*)"', page_source()):
        assert not attribute.startswith(("http://", "https://", "//")), attribute


def test_the_rail_matches_the_real_graph() -> None:
    """The pipeline rail names all sixteen nodes so it can light them up one by
    one. A renamed node would otherwise become a tick that never lights -- a
    silent, plausible-looking wrong answer rather than a visible break."""
    phases = re.search(r"const PHASES = \[(.*?)\n\];", page_source(), re.S)
    assert phases, "the page no longer declares PHASES"

    on_the_rail = set()
    for group in re.findall(r"nodes: \[(.*?)\]", phases.group(1), re.S):
        on_the_rail.update(re.findall(r'"([a-z_]+)"', group))

    assert on_the_rail == GRAPH_NODES, (
        f"missing from the rail: {GRAPH_NODES - on_the_rail}; "
        f"not real nodes: {on_the_rail - GRAPH_NODES}"
    )
