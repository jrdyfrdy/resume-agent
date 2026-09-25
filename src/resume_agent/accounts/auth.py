"""Signing in with Google, and the gate every request passes through.

**Why Google rather than passwords.** The owner approves every account by hand,
and approval only means something if the email is real. With passwords, anyone
can sign up as `your.friend@gmail.com`; Google's `email_verified` claim is what
makes "Alex wants access" actually be Alex. It also means there is no password
store to leak and no reset flow to build.

**The gate is an allow-list, and it denies by default.** Every request runs
through `gate` as a global dependency. A small set of paths is public, a smaller
set needs only a signed-in session, and *everything else* needs an approved
account -- including routes that do not exist yet. Forgetting to protect a new
endpoint is not possible, because nothing has to remember to protect it.
`test_every_route_that_is_not_public_needs_a_session` enumerates the app's
routes to hold that line.

**A dependency, not middleware.** Starlette's `BaseHTTPMiddleware` wraps the
response, which has a history of buffering streaming responses -- and this app
streams every run over SSE for about a minute. A global dependency runs before
the handler and never touches the response.

**The user is re-read from the database on every request**, rather than
trusting the session to say whether they are approved. Revoking someone takes
effect on their next click, not when their cookie expires.

**CSRF.** The session cookie is `SameSite=Lax`, so a cross-site POST does not
carry it. Belt and braces: every mutating `/api` request must be
`Content-Type: application/json`, which a cross-site form cannot send without a
CORS preflight -- and this app answers no preflights.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from resume_agent.accounts.db import AccountsDB, User
from resume_agent.accounts.workspace import Workspaces

logger = logging.getLogger(__name__)

# -- configuration -----------------------------------------------------------

MULTIUSER_ENV_VAR = "RESUME_AGENT_MULTIUSER"
DATABASE_URL_ENV_VAR = "DATABASE_URL"
SESSION_SECRET_ENV_VAR = "SESSION_SECRET"
GOOGLE_CLIENT_ID_ENV_VAR = "GOOGLE_CLIENT_ID"
GOOGLE_CLIENT_SECRET_ENV_VAR = "GOOGLE_CLIENT_SECRET"
ADMIN_EMAIL_ENV_VAR = "RESUME_AGENT_ADMIN_EMAIL"
PUBLIC_URL_ENV_VAR = "RESUME_AGENT_PUBLIC_URL"
SIGNUP_WEBHOOK_ENV_VAR = "RESUME_AGENT_SIGNUP_WEBHOOK"
WORKSPACE_DIR_ENV_VAR = "RESUME_AGENT_WORKSPACE_DIR"

GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# A signing key shorter than this is guessable enough to forge a session with.
MIN_SECRET_LENGTH = 32

# Anyone may reach these. `/privacy` because Google links to it from its
# consent screen, and someone deciding whether to sign in should be able to
# read it first.
PUBLIC_PATHS = frozenset({"/", "/auth/login", "/auth/callback", "/favicon.ico", "/privacy"})

# These need a session but not approval: a pending user has to be able to ask
# who they are (so the page can say "you're on the list"), sign out, and take
# their own account back.
SIGNED_IN_PATHS = frozenset({"/api/me", "/auth/logout"})

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# What the Google exchange returns, reduced to what this module uses. A seam,
# like `graph_factory`: the tests supply their own and sign in as anyone.
GoogleIdentity = Callable[[Request], Awaitable[dict]]


class ConfigurationError(RuntimeError):
    """Multi-user mode was asked for but cannot run safely as configured."""


@dataclass(frozen=True)
class AuthSettings:
    session_secret: str
    google_client_id: str | None = None
    google_client_secret: str | None = None
    admin_email: str | None = None
    public_url: str | None = None
    signup_webhook: str | None = None

    @property
    def secure_cookies(self) -> bool:
        # Only when served over HTTPS. A `Secure` cookie is never sent over
        # plain HTTP, so setting it locally would make signing in impossible.
        return bool(self.public_url and self.public_url.startswith("https://"))

    @classmethod
    def from_env(cls) -> AuthSettings:
        """Read and validate everything multi-user mode needs, or refuse to start.

        Refusing is the point. A missing secret should stop the deploy with a
        sentence naming it, not produce a server that half works.
        """
        env = os.environ
        missing = [
            name for name in (
                DATABASE_URL_ENV_VAR, SESSION_SECRET_ENV_VAR,
                GOOGLE_CLIENT_ID_ENV_VAR, GOOGLE_CLIENT_SECRET_ENV_VAR,
                ADMIN_EMAIL_ENV_VAR, PUBLIC_URL_ENV_VAR,
            )
            if not env.get(name, "").strip()
        ]
        if missing:
            raise ConfigurationError(
                "multi-user mode needs these environment variables: " + ", ".join(missing)
            )
        secret = env[SESSION_SECRET_ENV_VAR].strip()
        if len(secret) < MIN_SECRET_LENGTH:
            raise ConfigurationError(
                f"{SESSION_SECRET_ENV_VAR} must be at least {MIN_SECRET_LENGTH} characters; "
                'generate one with: python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return cls(
            session_secret=secret,
            google_client_id=env[GOOGLE_CLIENT_ID_ENV_VAR].strip(),
            google_client_secret=env[GOOGLE_CLIENT_SECRET_ENV_VAR].strip(),
            admin_email=env[ADMIN_EMAIL_ENV_VAR].strip().lower(),
            public_url=env[PUBLIC_URL_ENV_VAR].strip().rstrip("/"),
            signup_webhook=env.get(SIGNUP_WEBHOOK_ENV_VAR, "").strip() or None,
        )


@dataclass
class MultiUser:
    """Everything multi-user mode needs, bundled so the tests can build one."""

    db: AccountsDB
    workspaces: Workspaces
    settings: AuthSettings
    google_identity: GoogleIdentity | None = None
    _oauth: object = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> MultiUser:
        settings = AuthSettings.from_env()
        db = AccountsDB(os.environ[DATABASE_URL_ENV_VAR].strip())
        db.create_schema()
        root = Path(os.environ.get(WORKSPACE_DIR_ENV_VAR) or _default_workspace_root())
        return cls(db=db, workspaces=Workspaces(db, root), settings=settings)


def _default_workspace_root() -> Path:
    import tempfile  # noqa: PLC0415 - only needed here

    return Path(tempfile.gettempdir()) / "resume-agent-workspaces"


# -- the gate ----------------------------------------------------------------


async def current_user(request: Request, multi: MultiUser) -> User | None:
    """The signed-in user, freshly read -- so a revocation applies at once."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = await asyncio.to_thread(multi.db.user, user_id)
    if user is None:
        # The account was deleted out from under this session.
        request.session.clear()
    return user


def make_gate(multi: MultiUser) -> Callable[[Request], Awaitable[None]]:
    """The dependency every route runs. Denies by default; see the module docstring."""

    async def gate(request: Request) -> None:
        path = request.url.path
        if path in PUBLIC_PATHS:
            return

        user = await current_user(request, multi)
        if user is None:
            raise HTTPException(status_code=401, detail="Sign in to continue.")
        request.state.user = user

        if request.method in MUTATING_METHODS and path.startswith("/api/"):
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                raise HTTPException(
                    status_code=415, detail="This endpoint only accepts JSON."
                )

        if path in SIGNED_IN_PATHS:
            return

        if not user.approved:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Your account is waiting for approval."
                    if user.status == "pending"
                    else "This account does not have access."
                ),
            )

        if path.startswith("/api/admin") and not user.is_admin:
            raise HTTPException(status_code=403, detail="Only the owner can do that.")

    return gate


def signed_in_user(request: Request) -> User:
    """The user the gate already loaded. Only valid on gated routes."""
    return request.state.user


# -- routes ------------------------------------------------------------------


def install(app: FastAPI, multi: MultiUser) -> None:
    """Sessions, the Google routes, `/api/me`, and the admin endpoints."""
    from authlib.integrations.starlette_client import OAuthError  # noqa: PLC0415
    from starlette.middleware.sessions import SessionMiddleware  # noqa: PLC0415

    app.add_middleware(
        SessionMiddleware,
        secret_key=multi.settings.session_secret,
        session_cookie="resume_agent_session",
        same_site="lax",
        https_only=multi.settings.secure_cookies,
        max_age=60 * 60 * 24 * 14,  # a fortnight; approval is re-checked each request anyway
    )
    identity = multi.google_identity or _google_identity(multi)
    privacy = privacy_page(multi)

    @app.get("/privacy", response_class=HTMLResponse)
    async def privacy_policy() -> str:
        return privacy

    @app.get("/auth/login")
    async def auth_login(request: Request):
        if multi.google_identity is not None:
            # The injected identity has no Google to redirect to.
            return RedirectResponse("/auth/callback")
        oauth = _oauth_client(multi)
        return await oauth.google.authorize_redirect(request, _callback_url(request, multi))

    @app.get("/auth/callback", name="auth_callback")
    async def auth_callback(request: Request):
        try:
            claims = await identity(request)
        except OAuthError as exc:
            # Cancelled on Google's screen, or a callback that can no longer be
            # checked: the Back button, a reload, a session that expired on the
            # way. Found before the first deploy -- the tests' stand-in Google
            # never fails, and this was a bare 500. It is the sign-in page
            # again, with a line saying it did not finish.
            logger.info("sign-in did not complete: %s", exc.error)
            return RedirectResponse("/?signin=incomplete", status_code=303)
        if not claims.get("email_verified"):
            # Approval depends on the email being real. An unverified one is
            # refused outright rather than admitted to the queue.
            return RedirectResponse("/?signin=unverified", status_code=303)
        existing = await asyncio.to_thread(multi.db.user_by_email, claims["email"])
        user = await asyncio.to_thread(
            multi.db.sign_in,
            google_sub=claims["sub"],
            email=claims["email"],
            name=claims.get("name") or "",
            admin_email=multi.settings.admin_email,
        )
        request.session.clear()  # a fresh session per sign-in; nothing carries over
        request.session["user_id"] = user.id
        if existing is None and user.status == "pending":
            _notify_signup(multi.settings.signup_webhook, user)
        return RedirectResponse("/", status_code=303)

    @app.post("/auth/logout")
    async def auth_logout(request: Request):
        # POST, not GET: a cross-site `<img src="/auth/logout">` cannot sign
        # anyone out.
        request.session.clear()
        return JSONResponse({"ok": True})

    @app.get("/api/me")
    async def me(request: Request):
        return _user_view(signed_in_user(request))

    @app.delete("/api/me")
    async def delete_me(request: Request):
        """Your career history is yours. This removes the account and, by
        cascade, every file, every earlier version and every run."""
        user = signed_in_user(request)
        await asyncio.to_thread(multi.db.delete_user, user.id)
        # The database is the record; this is the copies on the server's disk.
        await multi.workspaces.forget(user)
        request.session.clear()
        return JSONResponse({"ok": True})

    @app.get("/api/admin/users")
    async def admin_users():
        users = await asyncio.to_thread(multi.db.users)
        return [_user_view(user) for user in users]

    @app.post("/api/admin/users/{user_id}/decision")
    async def admin_decide(user_id: str, request: Request):
        body = await request.json()
        status = body.get("status")
        if status not in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="status must be approved or rejected")
        admin = signed_in_user(request)
        if user_id == admin.id:
            # The one decision that would lock the owner out of their own site.
            raise HTTPException(status_code=400, detail="You cannot change your own access.")
        user = await asyncio.to_thread(multi.db.decide, user_id, status, by=admin.id)
        if user is None:
            raise HTTPException(status_code=404, detail="No such account.")
        return _user_view(user)


def _user_view(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "status": user.status,
        "is_admin": user.is_admin,
        "created_at": user.created_at,
        "decided_at": user.decided_at,
    }


# -- Google ------------------------------------------------------------------


def _oauth_client(multi: MultiUser):
    if multi._oauth is None:  # noqa: SLF001 - lazily built, once
        from authlib.integrations.starlette_client import OAuth  # noqa: PLC0415

        oauth = OAuth()
        oauth.register(
            name="google",
            server_metadata_url=GOOGLE_DISCOVERY_URL,
            client_id=multi.settings.google_client_id,
            client_secret=multi.settings.google_client_secret,
            client_kwargs={"scope": "openid email profile"},
        )
        multi._oauth = oauth  # noqa: SLF001
    return multi._oauth  # noqa: SLF001


def _callback_url(request: Request, multi: MultiUser) -> str:
    """Built from the configured public URL, not from the request.

    Behind Render's proxy the request arrives as plain `http://`, so
    `request.url_for` would produce a callback Google has not registered and
    refuse to redirect to.
    """
    if multi.settings.public_url:
        return multi.settings.public_url + "/auth/callback"
    return str(request.url_for("auth_callback"))


def _google_identity(multi: MultiUser) -> GoogleIdentity:
    async def exchange(request: Request) -> dict:
        oauth = _oauth_client(multi)
        # authlib verifies the ID token's signature, issuer, audience and nonce
        # here, and hands back its claims as `userinfo`.
        token = await oauth.google.authorize_access_token(request)
        return dict(token.get("userinfo") or {})

    return exchange


# How the privacy page names each provider. The page says where people's career
# data is sent, so it names the company rather than the setting.
PROVIDER_NAMES = {
    "anthropic": "Anthropic (Claude)",
    "deepseek": "DeepSeek",
    "openrouter": "OpenRouter",
}


def privacy_page(multi: MultiUser) -> str:
    """`privacy.html` with the two facts that depend on configuration filled in.

    Built once at startup, like the app's own page. The provider and the webhook
    are environment variables, so they cannot change while the process runs.
    """
    from resume_agent.llm import ProviderConfigError, resolve_provider  # noqa: PLC0415

    fallback = "the one this site is set up with"
    try:
        provider = PROVIDER_NAMES.get(resolve_provider().name, fallback)
    except ProviderConfigError:
        provider = fallback
    notice = (
        "<li><b>To the owner, when you first sign in:</b> a message with your name and "
        "email address, through the chat service they chose, so they know someone is "
        "waiting.</li>"
        if multi.settings.signup_webhook
        else ""
    )
    page = Path(__file__).with_name("privacy.html").read_text(encoding="utf-8")
    return page.replace("__PROVIDER__", provider).replace("__SIGNUP_NOTICE__", notice)


def _notify_signup(webhook: str | None, user: User) -> None:
    """Tell the owner someone is waiting, if they asked to be told.

    Fire-and-forget on a thread with a short timeout. A webhook that is down
    must never stop someone signing up.
    """
    if not webhook:
        return
    payload = json.dumps({
        "content": f"{user.email} ({user.name or 'no name'}) is waiting for approval."
    }).encode()

    def post() -> None:
        try:
            request = urllib.request.Request(
                webhook, data=payload, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(request, timeout=5).close()  # noqa: S310 - owner-configured URL
        except Exception as exc:  # noqa: BLE001 - never block a sign-up on this
            logger.warning("signup webhook failed: %s", exc)

    asyncio.get_running_loop().run_in_executor(None, post)
