"""Two people on one server, and nothing crosses between them.

Alex and Blake are both approved. Each makes a profile with their own name in
it; every test then has one of them reach for the other's data -- by the
other's id, by path tricks, by a run or chat id -- and asserts they find
nothing, and that the other's data is unchanged.

These are the tests that matter most in multi-user mode. The quota and queue
tests are at the end.

One `TestClient` is shared and each person carries their own cookies onto it,
so every request runs on the same event loop -- as it does under uvicorn --
and a run started by one request is still alive for the next.
"""

from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from resume_agent.accounts.auth import AuthSettings, MultiUser
from resume_agent.accounts.db import AccountsDB
from resume_agent.accounts.workspace import Workspaces
from resume_agent.api.app import create_app
from resume_agent.api.limits import AccountQuota
from resume_agent.chat.extract import ExtractionFields
from resume_agent.kb.index import index_path_for
from resume_agent.kb.loader import load_profile
from tests.test_api import make_job
from tests.test_chat import ScriptedExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
OWNER = "owner@example.com"


class FakeGoogle:
    def __init__(self) -> None:
        self.claims: dict = {}

    async def __call__(self, _request) -> dict:
        return dict(self.claims)


class ReadsTheProfile:
    """Stands in for the graph: loads the profile the run was pointed at and
    'compiles' a PDF whose bytes say whose profile that was.

    `hold`, when set, keeps the run going until it is released -- which is how
    the queue test has two runs overlap.
    """

    def __init__(self) -> None:
        self.read: list[str] = []
        self.hold: threading.Event | None = None
        self.boom: str | None = None

    async def astream_events(self, state, _config=None, **_kw):
        import asyncio  # noqa: PLC0415

        yield {"event": "on_chain_start", "name": "retrieve", "data": {}}
        if self.boom:
            raise RuntimeError(self.boom)
        while self.hold is not None and not self.hold.is_set():
            await asyncio.sleep(0.01)
        name = load_profile(Path(state["profile_path"])).identity.name
        self.read.append(name)

        out = Path(state["options"].out_dir)
        out.mkdir(parents=True, exist_ok=True)
        pdf = out / "resume.pdf"
        pdf.write_bytes(b"%PDF-1.4 resume of " + name.encode())
        final = {"job_spec": make_job(), "pdf_path": str(pdf), "errors": []}
        yield {"event": "on_chain_end", "name": "finalize", "data": {"output": final}}


def copy_cookies(client: TestClient, jar):
    """A copy of `jar`, in the cookie class `client` actually uses -- Starlette's
    test client is built on its own `httpx2`, whose `Cookies` does not accept
    httpx's."""
    return type(client.cookies)(jar)


@dataclass
class Person:
    """One signed-in browser: its own cookies, on the shared client."""

    client: TestClient
    jar: object  # the client's own cookie class; see `copy_cookies`
    id: str
    profile: str = ""

    def _as_me(self) -> TestClient:
        self.client.cookies = copy_cookies(self.client, self.jar)
        return self.client

    def get(self, url, **kw):
        return self._as_me().get(url, **kw)

    def post(self, url, **kw):
        return self._as_me().post(url, **kw)

    def put(self, url, **kw):
        return self._as_me().put(url, **kw)

    def request(self, method, url, **kw):
        return self._as_me().request(method, url, **kw)

    def stream(self, method, url, **kw):
        return self._as_me().stream(method, url, **kw)


@dataclass
class World:
    client: TestClient
    google: FakeGoogle
    multi: MultiUser
    graph: ReadsTheProfile
    root: Path

    def sign_in(self, email: str, *, approve: bool = True) -> Person:
        self.client.cookies.clear()
        self.google.claims = {
            "sub": f"sub-{email}", "email": email, "email_verified": True, "name": email[:5],
        }
        response = self.client.get("/auth/callback", follow_redirects=False)
        assert response.status_code == 303, response.text
        user = self.multi.db.user_by_email(email)
        if approve and not user.approved:
            self.multi.db.decide(user.id, "approved", by="test")
        return Person(self.client, copy_cookies(self.client, self.client.cookies), user.id)

    def person_with_profile(self, email: str, display_name: str) -> Person:
        person = self.sign_in(email)
        created = person.post(
            "/api/profile/create", json={"mode": "empty", "display_name": display_name}
        )
        assert created.status_code == 201, created.text
        person.profile = created.json()["profile"]
        return person


@pytest.fixture
def make_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A multi-user server in a scratch directory; environment first, if any."""
    # The search index and the editor's backups are written relative to the
    # working directory; keep them in the scratch one.
    monkeypatch.chdir(tmp_path)
    shutil.copytree(REPO_ROOT / "profile.example", tmp_path / "profile.example")
    clients: list[TestClient] = []

    def build(**env: str) -> World:
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        db = AccountsDB(f"sqlite:///{tmp_path / 'accounts.db'}")
        db.create_schema()
        google = FakeGoogle()
        graph = ReadsTheProfile()
        multi = MultiUser(
            db=db,
            workspaces=Workspaces(db, tmp_path / "ws"),
            settings=AuthSettings(session_secret="s" * 48, admin_email=OWNER),
            google_identity=google,
        )
        app = create_app(
            graph_factory=lambda **_kw: graph,
            chat_model_factory=lambda: ScriptedExtractor(fields=ExtractionFields(reply="noted")),
            mode="multiuser",
            multiuser=multi,
        )
        client = TestClient(app)
        client.__enter__()  # one event loop for every request, as under uvicorn
        clients.append(client)
        return World(client, google, multi, graph, tmp_path)

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def world(make_world) -> World:
    return make_world()


@pytest.fixture
def alex_and_blake(world: World) -> tuple[Person, Person]:
    return (
        world.person_with_profile("alex@example.com", "Alex Lee"),
        world.person_with_profile("blake@example.com", "Blake Moss"),
    )


def wait_for_run(person: Person, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = person.get(f"/api/runs/{run_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} did not finish")


def start_run(person: Person) -> str:
    response = person.post("/api/runs", json={"jd": "a job", "profile": person.profile})
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


# ===========================================================================
# A profile of one's own
# ===========================================================================


def test_a_new_person_has_no_profile_until_they_make_one(world: World) -> None:
    alex = world.sign_in("alex@example.com")
    assert alex.get("/api/profiles").json() == []

    created = alex.post(
        "/api/profile/create", json={"mode": "empty", "display_name": "Alex Lee"}
    )

    assert created.status_code == 201
    # Named after the account, whatever name the request asked for: that is
    # what keeps every per-profile store on the server apart.
    assert created.json()["profile"] == alex.id
    assert [p["name"] for p in alex.get("/api/profiles").json()] == [alex.id]
    assert "identity.yaml" in world.multi.db.profile_files(alex.id)


def test_each_person_sees_only_their_own(alex_and_blake) -> None:
    alex, blake = alex_and_blake

    assert [p["name"] for p in alex.get("/api/profiles").json()] == [alex.id]
    assert alex.get("/api/profile", params={"profile": alex.profile}).json()["name"] == "Alex Lee"
    assert blake.get("/api/profile", params={"profile": blake.profile}).json()["name"] == (
        "Blake Moss"
    )


def test_naming_someone_elses_profile_finds_nothing(world: World, alex_and_blake) -> None:
    """Every route that takes a profile, pointed at the other person's id.

    A 400 for an unknown profile -- identical to a typo -- and never their
    data, and never a change to it.
    """
    alex, blake = alex_and_blake
    theirs = blake.profile
    before = world.multi.db.profile_files(blake.id)

    attempts = [
        ("GET", "/api/profile", {"params": {"profile": theirs}}),
        ("GET", "/api/profile/detail", {"params": {"profile": theirs}}),
        ("GET", "/api/profile/files", {"params": {"profile": theirs}}),
        ("GET", "/api/profile/file", {"params": {"profile": theirs, "path": "identity.yaml"}}),
        ("GET", "/api/profile/form", {"params": {"profile": theirs, "path": "identity.yaml"}}),
        ("PUT", "/api/profile/file",
         {"json": {"profile": theirs, "path": "identity.yaml", "text": "name: Pwned\n"}}),
        ("PUT", "/api/profile/form",
         {"json": {"profile": theirs, "path": "identity.yaml", "data": {"name": "Pwned"}}}),
        ("POST", "/api/profile/entry",
         {"json": {"profile": theirs, "role": "project", "name": "Pwned"}}),
        ("DELETE", "/api/profile/file", {"json": {"profile": theirs, "path": "identity.yaml"}}),
        ("POST", "/api/chat", {"json": {"profile": theirs, "message": "what is here?"}}),
        ("POST", "/api/chat/apply", {"json": {"profile": theirs, "turn_id": "x"}}),
        ("POST", "/api/runs", {"json": {"profile": theirs, "jd": "a job"}}),
    ]
    for method, path, kwargs in attempts:
        response = alex.request(method, path, **kwargs)
        assert response.status_code == 400, f"{method} {path} -> {response.status_code}"
        assert "Blake" not in response.text, f"{method} {path} leaked Blake's data"

    assert world.multi.db.profile_files(blake.id) == before
    assert world.multi.db.runs(blake.id) == []


@pytest.mark.parametrize(
    "name",
    ["profile.example", "..", "../..", "users", "{blake}", "../{blake}",
     "../../users/{blake}/{blake}"],
)
def test_path_tricks_find_nothing_either(alex_and_blake, name: str) -> None:
    alex, blake = alex_and_blake
    name = name.format(blake=blake.id)

    response = alex.get("/api/profile/files", params={"profile": name})

    assert response.status_code == 400
    assert "Blake" not in response.text


def test_only_the_shipped_example_can_be_copied(world: World) -> None:
    """The copy is read from the server's own directory. If the owner's real
    profile were ever deployed alongside the app, it must not be copyable."""
    shutil.copytree(world.root / "profile.example", world.root / "profile")
    alex = world.sign_in("alex@example.com")

    refused = alex.post("/api/profile/create", json={"mode": "example", "source": "profile"})
    assert refused.status_code == 400
    assert world.multi.db.profile_files(alex.id) == {}

    copied = alex.post(
        "/api/profile/create", json={"mode": "example", "source": "profile.example"}
    )
    assert copied.status_code == 201


# ===========================================================================
# Saving lands in the database
# ===========================================================================


def test_a_save_is_stored_and_versioned(world: World, alex_and_blake) -> None:
    alex, _ = alex_and_blake
    text = alex.get(
        "/api/profile/file", params={"profile": alex.profile, "path": "identity.yaml"}
    ).json()["text"]

    saved = alex.put(
        "/api/profile/file",
        json={"profile": alex.profile, "path": "identity.yaml",
              "text": text.replace("Alex Lee", "Alex Q. Lee")},
    )

    assert saved.json() == {"ok": True, "error": None, "backup": None}
    assert "Alex Q. Lee" in world.multi.db.profile_files(alex.id)["identity.yaml"]
    history = world.multi.db.profile_history(alex.id, "identity.yaml")
    assert len(history) == 2, "the creation, then the edit"


def test_a_save_that_does_not_validate_changes_nothing(world: World, alex_and_blake) -> None:
    alex, _ = alex_and_blake
    before = world.multi.db.profile_files(alex.id)

    saved = alex.put(
        "/api/profile/file",
        json={"profile": alex.profile, "path": "identity.yaml", "text": "name: [unclosed\n"},
    )

    assert saved.json()["ok"] is False
    assert world.multi.db.profile_files(alex.id) == before


def test_added_and_deleted_entries_land_in_the_database(world: World, alex_and_blake) -> None:
    alex, _ = alex_and_blake

    added = alex.post(
        "/api/profile/entry", json={"profile": alex.profile, "role": "project", "name": "Kiln"}
    )
    assert added.status_code == 201
    path = added.json()["path"]
    assert path in world.multi.db.profile_files(alex.id)

    removed = alex.request(
        "DELETE", "/api/profile/file", json={"profile": alex.profile, "path": path}
    )
    assert removed.status_code == 200
    assert path not in world.multi.db.profile_files(alex.id)


# ===========================================================================
# Runs
# ===========================================================================


def test_each_run_reads_its_owners_profile(world: World, alex_and_blake) -> None:
    alex, blake = alex_and_blake

    a, b = start_run(alex), start_run(blake)
    assert wait_for_run(alex, a)["status"] == "done"
    assert wait_for_run(blake, b)["status"] == "done"

    assert sorted(world.graph.read) == ["Alex Lee", "Blake Moss"]
    assert alex.get(f"/api/runs/{a}/pdf").content == b"%PDF-1.4 resume of Alex Lee"
    assert blake.get(f"/api/runs/{b}/pdf").content == b"%PDF-1.4 resume of Blake Moss"


def test_someone_elses_run_is_not_found(alex_and_blake) -> None:
    alex, blake = alex_and_blake
    run_id = start_run(blake)
    wait_for_run(blake, run_id)

    for path in (f"/api/runs/{run_id}", f"/api/runs/{run_id}/events", f"/api/runs/{run_id}/pdf"):
        response = alex.get(path)
        # 404, not 403: the id's existence is not Alex's business either.
        assert response.status_code == 404, f"{path} -> {response.status_code}"
        assert b"Blake" not in response.content


def test_history_is_per_person(alex_and_blake) -> None:
    alex, blake = alex_and_blake
    wait_for_run(alex, start_run(alex))
    wait_for_run(blake, start_run(blake))

    mine = alex.get("/api/applications").json()

    assert len(mine) == 1
    assert mine[0]["company"] == "Acme Corp"
    assert mine[0]["has_pdf"] is True
    assert alex.get(f"/api/runs/{mine[0]['run_id']}/pdf").status_code == 200


def test_a_pdf_outlives_the_server_restarting(make_world) -> None:
    """Free hosting sleeps after fifteen idle minutes and forgets everything in
    memory. The PDF is in the database, so Download still works."""
    world = make_world()
    alex = world.person_with_profile("alex@example.com", "Alex Lee")
    run_id = start_run(alex)
    wait_for_run(alex, run_id)

    restarted = make_world()  # same database, a brand-new process
    again = Person(restarted.client, alex.jar, alex.id)  # the same browser cookie

    response = again.get(f"/api/runs/{run_id}/pdf")
    assert response.status_code == 200
    assert response.content.endswith(b"Alex Lee")


def test_a_run_leaves_nothing_on_disk(world: World, alex_and_blake) -> None:
    alex, _ = alex_and_blake
    run_id = start_run(alex)
    wait_for_run(alex, run_id)

    # Cleared just after the run reports done, so give it a moment.
    run_dir = world.root / "ws" / "runs" / run_id
    deadline = time.monotonic() + 5
    while run_dir.exists():
        assert time.monotonic() < deadline, "the run's files were not cleared"
        time.sleep(0.02)
    assert list(world.root.glob("out/applications.db")) == [], "no shared tracker file"


def test_a_failed_run_is_recorded_as_failed(world: World, alex_and_blake) -> None:
    alex, _ = alex_and_blake
    world.graph.boom = "the model provider is down"

    run_id = start_run(alex)

    assert wait_for_run(alex, run_id)["status"] == "failed"
    assert world.multi.db.runs(alex.id)[0]["status"] == "failed"
    assert alex.get("/api/applications").json() == []


# ===========================================================================
# Chat
# ===========================================================================


def test_someone_elses_chat_turn_is_not_found(alex_and_blake) -> None:
    alex, blake = alex_and_blake
    turn = blake.post(
        "/api/chat", json={"profile": blake.profile, "message": "I shipped the Kiln rewrite."}
    )
    assert turn.status_code == 202
    turn_id = turn.json()["turn_id"]
    deadline = time.monotonic() + 5
    while blake.get(f"/api/chat/{turn_id}").json()["status"] not in ("done", "failed"):
        assert time.monotonic() < deadline
        time.sleep(0.02)

    assert alex.get(f"/api/chat/{turn_id}").status_code == 404
    assert alex.get(f"/api/chat/{turn_id}/events").status_code == 404
    # Alex's own profile, Blake's turn: applying it would copy Blake's words
    # into Alex's file.
    applied = alex.post(
        "/api/chat/apply", json={"profile": alex.profile, "turn_id": turn_id, "accept": []}
    )
    assert applied.status_code == 404

    assert blake.post(
        "/api/chat/apply", json={"profile": blake.profile, "turn_id": turn_id, "accept": []}
    ).status_code == 200


# ===========================================================================
# Deleting an account
# ===========================================================================


def test_deleting_an_account_clears_the_server_disk(world: World, alex_and_blake) -> None:
    alex, blake = alex_and_blake
    wait_for_run(alex, start_run(alex))
    folder = world.multi.workspaces.folder_for(world.multi.db.user(alex.id))
    index = index_path_for(folder)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes(b"an index")  # the fake graph does not build one; a real run does
    alex.put(  # a save leaves an editor backup behind
        "/api/profile/form",
        json={"profile": alex.profile, "path": "identity.yaml",
              "data": {"name": "Alex Lee", "email": "alex@example.com"}},
    )

    assert alex.request("DELETE", "/api/me", json={}).status_code == 200

    assert not folder.parent.exists()
    assert not index.exists()
    assert not list(world.root.rglob(f"*{alex.id}*")), "something of Alex's is still on disk"
    assert world.multi.db.runs(alex.id) == []
    # And Blake is untouched.
    assert blake.get("/api/profile", params={"profile": blake.profile}).status_code == 200


# ===========================================================================
# Cost and capacity
# ===========================================================================


def test_a_person_runs_out_of_runs_for_the_day(make_world) -> None:
    world = make_world(RESUME_AGENT_RUNS_PER_USER_PER_DAY="2")
    alex = world.person_with_profile("alex@example.com", "Alex Lee")
    blake = world.person_with_profile("blake@example.com", "Blake Moss")

    for _ in range(2):
        wait_for_run(alex, start_run(alex))
    assert alex.get("/api/profile", params={"profile": alex.profile}).json()[
        "runs_left_today"
    ] == 0

    refused = alex.post("/api/runs", json={"jd": "a job", "profile": alex.profile})
    assert refused.status_code == 429
    assert "2 resumes for today" in refused.json()["detail"]
    assert len(world.multi.db.runs(alex.id)) == 2, "a refused run is not recorded"

    # Alex's limit is Alex's.
    allowed = blake.post("/api/runs", json={"jd": "a job", "profile": blake.profile})
    assert allowed.status_code == 202


def test_the_site_runs_out_for_everyone(make_world) -> None:
    world = make_world(RESUME_AGENT_RUNS_PER_DAY="2")
    alex = world.person_with_profile("alex@example.com", "Alex Lee")
    blake = world.person_with_profile("blake@example.com", "Blake Moss")

    wait_for_run(alex, start_run(alex))
    wait_for_run(blake, start_run(blake))

    refused = blake.post("/api/runs", json={"jd": "a job", "profile": blake.profile})
    assert refused.status_code == 429
    assert "limit for today" in refused.json()["detail"]


def test_a_second_run_waits_its_turn(world: World, alex_and_blake) -> None:
    """One run at a time fits in 512 MB. The second queues; it does not fail."""
    alex, blake = alex_and_blake
    world.graph.hold = threading.Event()

    first = start_run(alex)
    second = start_run(blake)
    time.sleep(0.2)

    assert blake.get(f"/api/runs/{second}").json()["status"] == "queued"
    assert alex.get(f"/api/runs/{first}").json()["status"] == "running"

    world.graph.hold.set()
    assert wait_for_run(alex, first)["status"] == "done"
    assert wait_for_run(blake, second)["status"] == "done"

    with blake.stream("GET", f"/api/runs/{second}/events") as response:
        body = "".join(response.iter_text())
    assert "waiting for another resume to finish" in body


def test_done_means_the_pdf_is_already_downloadable(world: World, alex_and_blake) -> None:
    """Found as an intermittent failure: the run's summary -- which says "done"
    -- was visible before its PDF reached the database, so a quick Download
    click could 404. A slow save makes that window impossible to miss."""
    alex, _ = alex_and_blake
    finish = world.multi.db.finish_run

    def slow_finish(*args, **kwargs):
        time.sleep(0.3)
        finish(*args, **kwargs)

    world.multi.db.finish_run = slow_finish
    run_id = start_run(alex)

    assert wait_for_run(alex, run_id)["status"] == "done"
    assert alex.get(f"/api/runs/{run_id}/pdf").status_code == 200


def test_the_quota_arithmetic() -> None:
    quota = AccountQuota(per_user=5, per_day=40, concurrent=1)

    assert quota.refusal(4, 39) is None
    assert "5 resumes for today" in quota.refusal(5, 0)
    assert "limit for today" in quota.refusal(0, 40)
    assert quota.remaining(2, 38) == 2, "the site's limit is the nearer one"
    assert quota.remaining(9, 0) == 0


def test_a_nonsense_limit_falls_back_rather_than_lifting_it(monkeypatch) -> None:
    monkeypatch.setenv("RESUME_AGENT_RUNS_PER_USER_PER_DAY", "-3")
    monkeypatch.setenv("RESUME_AGENT_CONCURRENT_RUNS", "lots")

    quota = AccountQuota()

    assert quota.per_user == 5
    assert quota.concurrent == 1
