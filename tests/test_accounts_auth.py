"""Signing in, the approval gate, and the owner's controls.

Google is replaced by `FakeGoogle`, which hands the callback whatever identity a
test chooses -- the same seam as `graph_factory`. Nothing here touches the
network. The session is real: Starlette's signed cookie, carried by the test
client exactly as a browser would carry it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from resume_agent.accounts.auth import (
    PUBLIC_PATHS,
    AuthSettings,
    ConfigurationError,
    MultiUser,
)
from resume_agent.accounts.db import AccountsDB
from resume_agent.accounts.workspace import Workspaces
from resume_agent.api.app import create_app, resolve_mode

OWNER = "owner@example.com"


class FakeGoogle:
    """Whoever `claims` says, verified or not. Set it, then hit the callback.

    Or set `raises`, to be the Google that says no -- a cancelled consent
    screen, or a callback whose state no longer checks out.
    """

    def __init__(self) -> None:
        self.claims: dict = {}
        self.raises: Exception | None = None

    async def __call__(self, _request) -> dict:
        if self.raises is not None:
            raise self.raises
        return dict(self.claims)


class NoGraph:
    """Phase 3 starts no runs; this only has to exist."""

    async def astream_events(self, *_a, **_kw):  # pragma: no cover - unused
        if False:
            yield


@pytest.fixture
def google() -> FakeGoogle:
    return FakeGoogle()


@pytest.fixture
def multi(tmp_path: Path, google: FakeGoogle) -> MultiUser:
    db = AccountsDB(f"sqlite:///{tmp_path / 'accounts.db'}")
    db.create_schema()
    return MultiUser(
        db=db,
        workspaces=Workspaces(db, tmp_path / "ws"),
        settings=AuthSettings(session_secret="s" * 48, admin_email=OWNER),
        google_identity=google,
    )


@pytest.fixture
def app(multi: MultiUser):
    return create_app(graph_factory=lambda **_kw: NoGraph(), mode="multiuser", multiuser=multi)


def sign_in(app, google: FakeGoogle, email: str, *, verified: bool = True) -> TestClient:
    """A fresh browser, signed in as `email`."""
    client = TestClient(app)
    google.claims = {
        "sub": f"sub-{email}", "email": email, "email_verified": verified, "name": email[:5],
    }
    response = client.get("/auth/callback", follow_redirects=False)
    assert response.status_code in (303, 403), response.text
    return client


def approve(multi: MultiUser, email: str) -> None:
    user = multi.db.user_by_email(email)
    multi.db.decide(user.id, "approved", by="test")


# ===========================================================================
# The gate denies by default
# ===========================================================================


def _concrete(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


def test_every_route_that_is_not_public_needs_a_session(app) -> None:
    """The allow-list, held to account.

    Walks every route the app actually registers -- including ones added after
    this test was written -- and asserts a request with no session is refused
    before any handler runs. A new endpoint that forgot to be protected would
    fail here, not in production.
    """
    from fastapi.routing import APIRoute  # noqa: PLC0415

    client = TestClient(app)
    checked = 0
    for route in app.routes:
        # The gate is an app-level dependency, which FastAPI applies to API
        # routes only. A plain Starlette route or a `Mount` never runs it --
        # that is how the schema and docs pages were public until this test
        # found them -- so none may exist unless the path is deliberately public.
        if route.path not in PUBLIC_PATHS:
            assert isinstance(route, APIRoute), (
                f"{type(route).__name__} {route.path} bypasses the gate entirely"
            )
        for method in sorted(getattr(route, "methods", None) or ()):
            if method in ("HEAD", "OPTIONS") or route.path in PUBLIC_PATHS:
                continue
            response = client.request(method, _concrete(route.path), json={})
            assert response.status_code == 401, f"{method} {route.path} -> {response.status_code}"
            checked += 1
    assert checked > 20, "the walk found suspiciously few routes"


def test_the_public_paths_are_the_only_ones(app) -> None:
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/profile").status_code == 401
    # Not gated -- not served. FastAPI's own schema and docs routes cannot run
    # an app-level dependency, so in multi-user mode they do not exist.
    for path in ("/openapi.json", "/api/docs", "/redoc"):
        assert client.get(path).status_code == 404, path


# ===========================================================================
# Pending, approved, rejected
# ===========================================================================


def test_a_new_friend_is_signed_in_but_waiting(app, google, multi) -> None:
    client = sign_in(app, google, "alex@example.com")

    me = client.get("/api/me")
    assert me.status_code == 200
    assert me.json()["status"] == "pending"

    refused = client.get("/api/profile")
    assert refused.status_code == 403
    assert "waiting for approval" in refused.json()["detail"]


def test_approval_opens_the_app(app, google, multi) -> None:
    client = sign_in(app, google, "alex@example.com")
    approve(multi, "alex@example.com")

    # No need to sign in again: approval is read fresh on every request.
    assert client.get("/api/me").json()["status"] == "approved"
    assert client.get("/api/profiles").status_code == 200


def test_revoking_access_takes_effect_on_the_next_click(app, google, multi) -> None:
    """The session is not trusted to say whether you are approved."""
    client = sign_in(app, google, "alex@example.com")
    approve(multi, "alex@example.com")
    assert client.get("/api/profiles").status_code == 200

    user = multi.db.user_by_email("alex@example.com")
    multi.db.decide(user.id, "rejected", by="test")

    refused = client.get("/api/profiles")
    assert refused.status_code == 403
    assert "does not have access" in refused.json()["detail"]


def test_an_unverified_email_is_turned_away_entirely(app, google, multi) -> None:
    """Approval only means something if the email is real. An unverified one
    is not even queued."""
    client = sign_in(app, google, "impostor@example.com", verified=False)

    assert client.get("/api/me").status_code == 401
    assert multi.db.user_by_email("impostor@example.com") is None
    # And the page is told why, rather than shown a JSON error.
    back = client.get("/auth/callback", follow_redirects=False)
    assert back.headers["location"] == "/?signin=unverified"


@pytest.mark.parametrize("error", ["access_denied", "mismatching_state"])
def test_a_sign_in_that_does_not_finish_goes_back_to_the_start(
    app, google, multi, error
) -> None:
    """Cancel on Google's consent screen, or Back into a used callback. Found
    before the first deploy: the stand-in Google never failed, and this was a
    bare 500 page."""
    from authlib.integrations.starlette_client import OAuthError  # noqa: PLC0415

    google.raises = OAuthError(error=error)
    client = TestClient(app)

    response = client.get("/auth/callback", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/?signin=incomplete"
    assert client.get("/api/me").status_code == 401
    assert multi.db.users() == []


def test_the_page_can_explain_every_way_back_from_sign_in() -> None:
    """The server says why a sign-in did not work with `?signin=<reason>`; the
    page has one sentence per reason. A reason with no sentence would send
    someone back to the sign-in page with no idea why."""
    root = Path(__file__).resolve().parent.parent / "src" / "resume_agent"
    sent = set(re.findall(r'"/\?signin=([a-z]+)"', (root / "accounts" / "auth.py").read_text(
        encoding="utf-8")))
    page = (root / "api" / "static" / "index.html").read_text(encoding="utf-8")
    notes = re.search(r"const SIGNIN_NOTES = \{(.*?)\n\};", page, re.S)

    assert sent == {"incomplete", "unverified"}
    assert notes, "the page no longer declares SIGNIN_NOTES"
    assert set(re.findall(r"^\s*([a-z]+):", notes.group(1), re.M)) == sent


def test_a_deleted_account_ends_its_session(app, google, multi) -> None:
    client = sign_in(app, google, "alex@example.com")

    assert client.request("DELETE", "/api/me", json={}).status_code == 200

    assert client.get("/api/me").status_code == 401
    assert multi.db.user_by_email("alex@example.com") is None


# ===========================================================================
# The owner
# ===========================================================================


def test_the_owner_is_let_straight_in(app, google) -> None:
    client = sign_in(app, google, OWNER)
    me = client.get("/api/me").json()

    assert me["status"] == "approved"
    assert me["is_admin"] is True


def test_the_owner_can_see_and_approve_the_queue(app, google, multi) -> None:
    sign_in(app, google, "alex@example.com")
    owner = sign_in(app, google, OWNER)

    queue = owner.get("/api/admin/users").json()
    alex = next(u for u in queue if u["email"] == "alex@example.com")
    assert alex["status"] == "pending"
    assert queue[0]["status"] == "pending", "the people waiting come first"

    decided = owner.post(
        f"/api/admin/users/{alex['id']}/decision", json={"status": "approved"}
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"


def test_a_friend_cannot_reach_the_owner_controls(app, google, multi) -> None:
    friend = sign_in(app, google, "alex@example.com")
    approve(multi, "alex@example.com")

    assert friend.get("/api/admin/users").status_code == 403
    victim = multi.db.user_by_email("alex@example.com")
    assert friend.post(
        f"/api/admin/users/{victim.id}/decision", json={"status": "approved"}
    ).status_code == 403


def test_the_owner_cannot_lock_themselves_out(app, google, multi) -> None:
    owner = sign_in(app, google, OWNER)
    me = owner.get("/api/me").json()

    response = owner.post(f"/api/admin/users/{me['id']}/decision", json={"status": "rejected"})

    assert response.status_code == 400
    assert owner.get("/api/me").json()["status"] == "approved"


# ===========================================================================
# Request hygiene
# ===========================================================================


def test_a_mutating_request_must_be_json(app, google, multi) -> None:
    """A cross-site form cannot send JSON without a CORS preflight, which this
    app never answers -- so requiring JSON closes the form-based CSRF route."""
    owner = sign_in(app, google, OWNER)
    alex = sign_in(app, google, "alex@example.com")
    target = multi.db.user_by_email("alex@example.com")
    del alex

    response = owner.post(
        f"/api/admin/users/{target.id}/decision",
        content="status=approved",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 415
    assert multi.db.user(target.id).status == "pending"


def test_signing_out_needs_a_post(app, google) -> None:
    """A GET cannot sign you out, so a cross-site `<img>` cannot either."""
    client = sign_in(app, google, OWNER)

    assert client.get("/auth/logout").status_code == 405
    assert client.get("/api/me").status_code == 200

    assert client.post("/auth/logout").status_code == 200
    assert client.get("/api/me").status_code == 401


# ===========================================================================
# Being told someone is waiting
# ===========================================================================


def test_a_new_signup_pings_the_owner(tmp_path, google, monkeypatch) -> None:
    posted: list[bytes] = []

    class Sink:
        def close(self) -> None:
            pass

    def fake_urlopen(request, timeout):  # noqa: ARG001 - signature match
        posted.append(request.data)
        return Sink()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    db = AccountsDB(f"sqlite:///{tmp_path / 'a.db'}")
    db.create_schema()
    multi = MultiUser(
        db=db, workspaces=Workspaces(db, tmp_path / "ws"), google_identity=google,
        settings=AuthSettings(
            session_secret="s" * 48, admin_email=OWNER,
            signup_webhook="https://hooks.example.com/x",
        ),
    )
    app = create_app(graph_factory=lambda **_kw: NoGraph(), mode="multiuser", multiuser=multi)

    sign_in(app, google, "alex@example.com")
    sign_in(app, google, "alex@example.com")  # a return visit is not news
    sign_in(app, google, OWNER)               # nor is the owner

    import time  # noqa: PLC0415 - the post runs on a worker thread
    for _ in range(50):
        if posted:
            break
        time.sleep(0.02)
    assert len(posted) == 1
    assert b"alex@example.com" in posted[0]


# ===========================================================================
# Refusing to start when it could not run safely
# ===========================================================================


def test_missing_settings_stop_the_deploy_with_their_names(monkeypatch) -> None:
    for name in ("DATABASE_URL", "SESSION_SECRET", "GOOGLE_CLIENT_ID",
                 "GOOGLE_CLIENT_SECRET", "RESUME_AGENT_ADMIN_EMAIL",
                 "RESUME_AGENT_PUBLIC_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="SESSION_SECRET"):
        AuthSettings.from_env()


def test_a_short_session_secret_is_refused(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
    monkeypatch.setenv("SESSION_SECRET", "too-short")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("RESUME_AGENT_ADMIN_EMAIL", OWNER)
    monkeypatch.setenv("RESUME_AGENT_PUBLIC_URL", "https://example.com")

    with pytest.raises(ConfigurationError, match="at least 32"):
        AuthSettings.from_env()


def test_demo_and_multiuser_together_are_refused(monkeypatch) -> None:
    """Opposite permission models. A silent guess about which one wins is the
    wrong thing to get wrong on a public URL."""
    monkeypatch.setenv("RESUME_AGENT_DEMO", "1")
    monkeypatch.setenv("RESUME_AGENT_MULTIUSER", "1")

    with pytest.raises(ConfigurationError, match="not both"):
        resolve_mode()


def test_cookies_are_secure_only_over_https() -> None:
    """A `Secure` cookie is never sent over plain HTTP, so setting it locally
    would make signing in impossible."""
    assert AuthSettings(session_secret="s" * 48, public_url="https://x.onrender.com").secure_cookies
    assert not AuthSettings(session_secret="s" * 48, public_url="http://localhost:8000").secure_cookies
    assert not AuthSettings(session_secret="s" * 48).secure_cookies


# ===========================================================================
# The page speaks the gate's language
# ===========================================================================


def test_every_change_the_page_sends_is_json() -> None:
    """The gate refuses a mutating /api request that is not JSON (415) -- that is
    the CSRF defence. So a page call that forgot the header would work locally
    and fail only on the shared site, which is the worst place to find out.
    Account deletion is the likeliest to be written without one: it has no body
    worth sending."""
    page = (Path(__file__).resolve().parent.parent
            / "src" / "resume_agent" / "api" / "static" / "index.html").read_text(encoding="utf-8")

    calls = re.findall(r"fetch\(\s*([^,)]+),\s*\{(.*?)\}\s*\)", page, re.S)
    mutating = [
        (url, options) for url, options in calls
        if re.search(r'method:\s*"(POST|PUT|PATCH|DELETE)"', options) and "/api/" in url
    ]

    assert len(mutating) >= 8, f"found only {len(mutating)}; is the pattern still right?"
    for url, options in mutating:
        assert '"Content-Type": "application/json"' in options, f"{url.strip()} is not sent as JSON"
