"""FastAPI app with SSE streaming of node events. Spec 8's M9.

    "FastAPI + SSE streaming of node events; minimal frontend."

**Why `astream_events` and not `astream(stream_mode="updates")`.** Updates-mode
is simpler and gives exactly one delta per node. But spec 10 maps
"Streaming (`astream_events`)" to this milestone, and the events carry more than
node boundaries: `on_chat_model_start` is what lets the page say "calling the
model" during the twenty seconds when tailoring is otherwise a frozen spinner.
The stream is filtered down to node boundaries plus model calls rather than
firehosing every internal runnable.

**Why runs execute in a background task.** Letting the SSE connection drive the
graph directly is fewer lines, and means closing the tab kills a run you are
paying for. A run is started once and appends to its own event log; the SSE
endpoint reads that log by index, so a refresh replays and reconnects instead
of restarting.

**The run registry is an in-process dict.** Locally this is a single-user
tool, and anything more would be inventing a deployment story. Multi-user mode
keeps the dict for what is in flight -- one process serves everyone -- but
records every run and its PDF in the accounts database, which is what outlives
a restart.

**Multi-user mode scopes by directory.** Every route that touches a profile gets
it from `open_profile`, which in that mode hands back the signed-in person's own
workspace folder. Runs and chat turns carry their owner and are a 404 to anyone
else. See `accounts/` for the rest.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

from resume_agent.accounts.auth import (
    MULTIUSER_ENV_VAR,
    ConfigurationError,
    MultiUser,
    make_gate,
    signed_in_user,
)
from resume_agent.accounts.auth import install as install_auth
from resume_agent.accounts.db import ago
from resume_agent.api.limits import AccountQuota, RunLimiter
from resume_agent.api.models import (
    ApplicationSummary,
    ApplyProposalRequest,
    ChatCreated,
    ChatEvent,
    ChatRequest,
    ChatTurnView,
    CreateEntryRequest,
    CreateProfileRequest,
    DeleteFileRequest,
    FormDocument,
    NodeEvent,
    ProfileDetail,
    ProfileFile,
    ProfileFileList,
    ProfileOption,
    ProfileSummary,
    RunCreated,
    RunRequest,
    RunSummary,
    SaveFileRequest,
    SaveFormRequest,
    SaveResult,
    build_profile_detail,
    summarise_state,
)
from resume_agent.chat.advise import advise
from resume_agent.chat.extract import apply as apply_proposal_items
from resume_agent.chat.extract import extract
from resume_agent.chat.session import Registry
from resume_agent.graph.build import build_graph, initial_state
from resume_agent.graph.state import RunOptions
from resume_agent.kb.forms import (
    create_entry,
    delete_document,
    read_document,
    skill_usage,
    write_document,
)
from resume_agent.kb.loader import ProfileLoadError, load_profile
from resume_agent.kb.writer import (
    ProfileWriteError,
    create_empty_profile,
    read_profile_file,
    relative_profile_files,
    resolve_editable_path,
    scaffold_profile,
    write_profile_file,
)
from resume_agent.latex.compile import find_compiler
from resume_agent.llm import (
    PROVIDER_ENV_VAR,
    ProviderConfigError,
    has_credentials,
    resolve_provider,
)
from resume_agent.tracker.db import list_applications

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The graph's own node names. Everything else in the event stream is an internal
# runnable the page has no use for.
GRAPH_NODES = {
    "parse_jd", "retrieve", "score", "select", "tailor", "verify", "summary",
    "render", "compile", "inspect", "fix_latex", "shrink_budget", "note_overfull",
    "cover_letter", "human_review", "revision_cap", "finalize",
}  # fmt: skip

# How often the SSE endpoint checks for new events. Node transitions take
# seconds, so a tenth of a second of latency is invisible.
POLL_INTERVAL_S = 0.1

# The shipped fixture. Kept as the server-side default so the endpoints behave
# the same on every machine; the page prefers `profile/` when it finds one,
# which is a decision about presentation rather than about the API.
DEFAULT_PROFILE = "profile.example"

# The file whose absence means "this directory is not a knowledge base". Chosen
# because it is required and is the first thing `load_profile` reads.
PROFILE_MARKER = "identity.yaml"

# Serve a public, read-only tour instead of the real tool.
#
# This app has no authentication and eight endpoints that write career data or
# spend an API key, which is exactly why `serve` binds to localhost. Demo mode
# is what makes a public URL defensible: every mutating route is removed, so
# there is nothing to authenticate *to*.
DEMO_ENV_VAR = "RESUME_AGENT_DEMO"


def demo_mode() -> bool:
    """Whether this process is serving the public tour."""
    return os.environ.get(DEMO_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


Mode = Literal["local", "demo", "multiuser"]


def resolve_mode() -> Mode:
    """Which of the three ways to run this app, read once from the environment.

    `local` is the default and is this tool as it has always been: one person,
    their own machine, no sign-in. `demo` is a public read-only tour. `multiuser`
    is accounts and approval. Asking for demo *and* multiuser is refused rather
    than resolved by precedence -- they have opposite permission models, and a
    silent guess about which one wins is exactly the wrong thing to get wrong on
    a public URL.
    """
    multi = os.environ.get(MULTIUSER_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
    if multi and demo_mode():
        raise ConfigurationError(
            f"set {DEMO_ENV_VAR} or {MULTIUSER_ENV_VAR}, not both: they are different sites"
        )
    return "multiuser" if multi else "demo" if demo_mode() else "local"


def resolve_profile_dir(name: str, root: Path | None = None) -> Path:
    """Turn a profile name from the browser into a directory, or refuse.

    `profile` arrives as a query parameter, and until now it went straight into
    `load_profile(Path(profile))`. Read-only that was merely sloppy. With a save
    endpoint in the same app it is directory traversal: `?profile=../..` would
    let the page choose where writes land.

    The allow-list is `discover_profiles()` -- the same directories the picker
    offers -- so the API can only ever touch something the UI could name.
    """
    known = {option.name for option in discover_profiles(root)}
    if name not in known:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown profile {name!r}. Available: {', '.join(sorted(known)) or 'none'}"
            ),
        )
    return (root or Path.cwd()) / name


def resolve_writable_profile_dir(name: str, root: Path | None = None) -> Path:
    """Like `resolve_profile_dir`, but refuses the shipped example.

    `profile.example/` is a fixture: the test suite loads it and the golden
    `.tex` snapshot renders from it, and unlike `profile/` it is **tracked by
    git**. Editing it through the UI therefore breaks a test and commits
    whatever you typed -- which is exactly what happened, because the only
    profile that exists on a fresh checkout is the example, so it was the only
    thing the editor offered.

    Read-only here rather than in `kb/writer.py`: the writer is a general tool
    the tests use against temporary copies, and this is a rule about what the
    *browser* may reach.
    """
    directory = resolve_profile_dir(name, root)
    if name.endswith(".example"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{name} is the shipped example and is read-only -- it is what the tests "
                f"load, and it is committed to git. Create your own knowledge base from "
                f"it instead; it will be gitignored."
            ),
        )
    return directory


def discover_profiles(root: Path | None = None) -> list[ProfileOption]:
    """Every knowledge base sitting next to the project.

    A directory that looks like a profile but fails to load is listed **with its
    error** rather than filtered out. "My profile is missing from the dropdown"
    is a much harder thing to debug than "my profile is listed and tells me what
    is wrong with it", and a validation error here is the single most likely
    thing to go wrong while someone is first writing their YAML.
    """
    base = root or Path.cwd()
    options: list[ProfileOption] = []

    for candidate in sorted(base.iterdir() if base.is_dir() else []):
        if not candidate.is_dir() or not (candidate / PROFILE_MARKER).is_file():
            continue
        try:
            loaded = load_profile(candidate)
        except ProfileLoadError as exc:
            options.append(
                ProfileOption(
                    name=candidate.name,
                    is_example=candidate.name.endswith(".example"),
                    loads=False,
                    error=str(exc),
                )
            )
            continue
        options.append(
            ProfileOption(
                name=candidate.name,
                is_example=candidate.name.endswith(".example"),
                loads=True,
                bullets=len(loaded.all_bullets()),
            )
        )

    # Real profiles first: someone who has written one wants it at the top, and
    # the example is a reference rather than a thing you run against.
    options.sort(key=lambda option: (option.is_example, option.name))
    return options


@dataclass
class Run:
    """One in-flight or finished run."""

    run_id: str
    status: str = "queued"
    # Multi-user mode: whose run this is. Every lookup checks it.
    user_id: str | None = None
    summary: RunSummary | None = None
    pdf_path: Path | None = None
    task: asyncio.Task | None = None
    # The append-only event log. The SSE endpoint reads it by index rather than
    # consuming a queue, which is what makes a browser refresh reconnect to a
    # run in progress and replay what it missed -- a queue would have been
    # drained by the connection that dropped.
    history: list[NodeEvent] = field(default_factory=list)


def create_app(
    graph_factory=build_graph,
    chat_model_factory=None,
    mode: Mode | None = None,
    multiuser: MultiUser | None = None,
) -> FastAPI:
    """Build the app.

    `graph_factory` and `chat_model_factory` are injectable so the tests can
    drive every endpoint with fakes -- no API key, no network, no compiler.
    `multiuser` is the same idea for accounts: the tests pass a SQLite store and
    a fake Google, and sign in as whoever they like.

    `mode` is read from the environment when not given; see `resolve_mode`.
    """
    mode = mode or resolve_mode()
    demo = mode == "demo"
    if mode == "multiuser" and multiuser is None:
        multiuser = MultiUser.from_env()
    limiter = RunLimiter() if demo else None

    # In multi-user mode every route runs the gate. It has to be a dependency of
    # the app itself, passed here, because FastAPI copies an app's dependencies
    # onto each route at the moment the route is registered.
    gate = make_gate(multiuser) if mode == "multiuser" else None

    # FastAPI registers its schema and docs pages as plain Starlette routes, not
    # API routes, so the app-level dependency above is never applied to them --
    # the gate would silently not run, and the whole API schema would be public.
    # In multi-user mode they are simply not served. Found by
    # `test_every_route_that_is_not_public_needs_a_session`, which walks every
    # registered route rather than trusting the dependency to reach them all.
    docs = {} if gate is None else {"openapi_url": None, "docs_url": None, "redoc_url": None}
    app = FastAPI(
        title="resume-agent",
        dependencies=[Depends(gate)] if gate else [],
        **({"docs_url": "/api/docs"} | docs),
    )
    # Everything below checks `multi`, never `mode`: it is the accounts store
    # when there is one, and None otherwise.
    multi = multiuser if mode == "multiuser" else None
    if multi is not None:
        install_auth(app, multi)
    quota = AccountQuota() if multi is not None else None
    # Held across the quota check and the row that uses it up, so two clicks at
    # once cannot both be granted the last run of the day.
    quota_lock = asyncio.Lock()
    run_slots = asyncio.Semaphore(quota.concurrent) if quota is not None else None
    runs: dict[str, Run] = {}
    # Conversations live as long as the process, like `runs`. A transcript is
    # not career data; what you accepted out of it is, and that is on disk.
    chats = Registry()

    # -- read-only ---------------------------------------------------------

    # Read once, at startup, rather than per request.
    #
    # Serving it from disk meant the page always reflected the working tree
    # while the routes stayed frozen at process start -- so a server left
    # running from before a change would hand the browser a new page that
    # called endpoints it did not have. That showed up as a bare "Not Found"
    # in the editor, with nothing to suggest the server was the stale part.
    # Page and API now ship as one vintage: an old process serves its own old
    # page, which works. `serve --reload` is the way to iterate.
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    # -- whose profile, whose run -------------------------------------------
    #
    # Every route that touches a profile gets its directory here, and in
    # multi-user mode this is the one change that scopes them all. The folder
    # is the signed-in user's workspace, rebuilt from the database for this one
    # operation and saved back if it changed, and the requested name is looked
    # up *inside it*. Another person's id is therefore not a forbidden profile
    # but an unknown one: the same 400 as a typo, which confirms nothing.

    @asynccontextmanager
    async def open_profile(http: Request, name: str, *, writable: bool = False):
        if multi is None:
            yield (resolve_writable_profile_dir if writable else resolve_profile_dir)(name)
            return
        async with multi.workspaces.profile(signed_in_user(http)) as folder:
            yield resolve_profile_dir(name, folder.parent)

    def owned_run(http: Request, run_id: str) -> Run:
        run = runs.get(run_id)
        # Someone else's run is a 404, never a 403: a 403 would confirm the id
        # exists, and that is already more than another user should learn.
        if run is None or (multi is not None and run.user_id != signed_in_user(http).id):
            raise HTTPException(status_code=404, detail="no such run")
        return run

    def owned_turn(http: Request, turn_id: str):
        turn = chats.get(turn_id)
        # A turn belongs to the profile it was about, which in multi-user mode
        # is named after its owner.
        if turn is None or (multi is not None and turn.profile != signed_in_user(http).id):
            raise HTTPException(status_code=404, detail="no such turn")
        return turn

    def backup_shown(backup: Path) -> str | None:
        # In multi-user mode every save is already a version in the database,
        # and this path is on a server disk the person cannot reach.
        return None if multi is not None else str(backup)

    async def runs_used(user_id: str) -> tuple[int, int]:
        """This person's runs and everyone's, in the last 24 hours."""
        since = ago(24)
        mine = await asyncio.to_thread(multi.db.runs_since, since, user_id=user_id)
        everyone = await asyncio.to_thread(multi.db.runs_since, since)
        return mine, everyone

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page

    @app.get("/api/profile", response_model=ProfileSummary)
    async def profile_summary(http: Request, profile: str = DEFAULT_PROFILE) -> ProfileSummary:
        async with open_profile(http, profile) as directory:
            try:
                loaded = load_profile(directory)
            except ProfileLoadError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        runs_left = None
        if limiter is not None:
            runs_left = limiter.remaining_today()
        elif quota is not None:
            runs_left = quota.remaining(*await runs_used(signed_in_user(http).id))

        try:
            provider = resolve_provider()
            provider_name, credentials_var = provider.name, provider.key_env_var
        except ProviderConfigError:
            # A bad RESUME_AGENT_PROVIDER should light up the page's blocked
            # state pointing at the thing that is actually wrong, not 500.
            provider_name, credentials_var = "unconfigured", PROVIDER_ENV_VAR

        return ProfileSummary(
            name=loaded.identity.name,
            experience=len(loaded.experience),
            projects=len(loaded.projects),
            bullets=len(loaded.all_bullets()),
            skills=len(loaded.skills),
            # Surfaced so the page can say what will fail before you click Run,
            # rather than after you have waited for it.
            has_credentials=has_credentials(),
            has_compiler=find_compiler() is not None,
            provider=provider_name,
            credentials_var=credentials_var,
            demo=demo,
            runs_left_today=runs_left,
        )

    @app.get("/api/profiles", response_model=list[ProfileOption])
    async def profiles(http: Request) -> list[ProfileOption]:
        """Every knowledge base on disk that a run could point at.

        In multi-user mode, the person's own -- or nothing, before they have
        made it.
        """
        if multi is None:
            return discover_profiles()
        async with multi.workspaces.profile(signed_in_user(http)) as folder:
            return discover_profiles(folder.parent)

    @app.get("/api/profile/detail", response_model=ProfileDetail)
    async def profile_detail(http: Request, profile: str = DEFAULT_PROFILE) -> ProfileDetail:
        async with open_profile(http, profile) as directory:
            try:
                loaded = load_profile(directory)
            except ProfileLoadError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return build_profile_detail(profile, loaded)

    # -- editing -----------------------------------------------------------
    #
    # Writes go through `kb/writer.py` and nothing else. It owns path
    # containment, whole-directory validation and the pre-write backup; this
    # layer only translates its errors into status codes.

    @app.get("/api/profile/files", response_model=ProfileFileList)
    async def profile_files(http: Request, profile: str = DEFAULT_PROFILE) -> ProfileFileList:
        async with open_profile(http, profile) as directory:
            try:
                return ProfileFileList(profile=profile, files=relative_profile_files(directory))
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/profile/file", response_model=ProfileFile)
    async def profile_file(
        http: Request, path: str, profile: str = DEFAULT_PROFILE
    ) -> ProfileFile:
        async with open_profile(http, profile) as directory:
            try:
                return ProfileFile(
                    profile=profile, path=path, text=read_profile_file(directory, path)
                )
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/profile/file", response_model=SaveResult)
    async def save_profile_file(request: SaveFileRequest, http: Request) -> SaveResult:
        async with open_profile(http, request.profile, writable=True) as directory:
            try:
                resolve_editable_path(directory, request.path)
            except ProfileWriteError as exc:
                # A path the editor should never have sent: a bug, not bad YAML.
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            try:
                backup = write_profile_file(directory, request.path, request.text)
            except ProfileWriteError as exc:
                # Invalid content is an ordinary part of editing YAML, so it is
                # a result rather than an error status. The file is unchanged.
                return SaveResult(ok=False, error=str(exc))

        logger.info("saved %s in %s (backup: %s)", request.path, request.profile, backup)
        return SaveResult(ok=True, backup=backup_shown(backup))

    # -- structured editing ------------------------------------------------
    #
    # The form endpoints are a thin layer over `forms.py`, which is itself a layer
    # over `writer.py`. Nothing here writes: it translates `ProfileWriteError`
    # into a status code and lets the writer keep its guarantees.

    @app.get("/api/profile/form", response_model=FormDocument)
    async def profile_form(
        http: Request, path: str, profile: str = DEFAULT_PROFILE
    ) -> FormDocument:
        async with open_profile(http, profile) as directory:
            try:
                document = read_document(directory, path)
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return FormDocument(**document, skill_usage=skill_usage(directory))

    @app.put("/api/profile/form", response_model=SaveResult)
    async def save_profile_form(request: SaveFormRequest, http: Request) -> SaveResult:
        async with open_profile(http, request.profile, writable=True) as directory:
            try:
                resolve_editable_path(directory, request.path)
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            try:
                backup = write_document(directory, request.path, request.data)
            except ProfileWriteError as exc:
                # Same contract as the text editor: content that does not
                # validate is a result, not an error status, and the file is
                # unchanged.
                return SaveResult(ok=False, error=str(exc))

        logger.info("saved %s in %s (backup: %s)", request.path, request.profile, backup)
        return SaveResult(ok=True, backup=backup_shown(backup))

    @app.post("/api/profile/entry", response_model=ProfileFile, status_code=201)
    async def add_entry(request: CreateEntryRequest, http: Request) -> ProfileFile:
        async with open_profile(http, request.profile, writable=True) as directory:
            try:
                relative = create_entry(directory, request.role, request.name)
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return ProfileFile(
                profile=request.profile,
                path=relative,
                text=read_profile_file(directory, relative),
            )

    @app.delete("/api/profile/file", response_model=SaveResult)
    async def remove_file(request: DeleteFileRequest, http: Request) -> SaveResult:
        async with open_profile(http, request.profile, writable=True) as directory:
            try:
                backup = delete_document(directory, request.path)
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SaveResult(ok=True, backup=backup_shown(backup))

    @app.post("/api/profile/create", response_model=ProfileFileList, status_code=201)
    async def create_profile(request: CreateProfileRequest, http: Request) -> ProfileFileList:
        """Start a new profile directory, empty or copied from the example.

        The answer to "where does my knowledge base go", made a button. It
        defaults to empty: copying the example produces a directory that loads
        straight away, but everything in it then has to be deleted before the
        profile describes its actual owner.

        In multi-user mode `request.name` is not used. Each person has exactly
        one profile, its folder is named after their id -- which is what keeps
        their search index and history apart from everyone else's -- and the
        response says what it is called.
        """
        source = None
        if request.mode == "example":
            # The copy is read from the server's own directory, which is not the
            # person's. Only the shipped example may be copied out of it.
            if multi is not None and not request.source.endswith(".example"):
                raise HTTPException(status_code=400, detail="only the example can be copied")
            source = resolve_profile_dir(request.source)

        target_of = (
            nullcontext(Path.cwd() / request.name)
            if multi is None
            else multi.workspaces.profile(signed_in_user(http))
        )
        async with target_of as target:
            try:
                if source is None:
                    create_empty_profile(target, name=request.display_name)
                else:
                    scaffold_profile(source, target)
                return ProfileFileList(profile=target.name, files=relative_profile_files(target))
            except ProfileWriteError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/applications", response_model=list[ApplicationSummary])
    async def applications(http: Request, limit: int = 20) -> list[ApplicationSummary]:
        if multi is not None:
            # The shared tracker file would list everyone's applications to
            # everyone; this person's runs come from the accounts database.
            rows = await asyncio.to_thread(multi.db.runs, signed_in_user(http).id, limit)
            return [
                ApplicationSummary(
                    id=None,
                    applied_on=row["created_at"][:10],
                    company=row["company"] or "",
                    title=row["title"] or "",
                    overall_fit=row["overall_fit"],
                    outcome=None,
                    run_id=row["id"],
                    has_pdf=row["has_pdf"],
                )
                for row in rows
                if row["status"] == "done"
            ]
        return [
            ApplicationSummary(
                id=row.id,
                applied_on=row.applied_on,
                company=row.company,
                title=row.title,
                overall_fit=row.overall_fit,
                outcome=row.outcome,
            )
            for row in list_applications(limit=limit)
        ]

    # -- chat ---------------------------------------------------------------
    #
    # Same shape as a run: start it detached, stream its append-only event log
    # by index, then fetch the settled turn. A dictation turn ends holding a
    # proposal, which is a suggestion in memory -- `/api/chat/apply` is the only
    # thing that writes, and it goes through the form layer like everything else.

    @app.post("/api/chat", response_model=ChatCreated, status_code=202)
    async def start_chat(request: ChatRequest, http: Request) -> ChatCreated:
        # Loaded here, inside the request, and handed to the turn: in multi-user
        # mode the folder is rebuilt by the next request, so a turn that read it
        # later could catch it half-written.
        async with open_profile(http, request.profile) as directory:
            try:
                profile = load_profile(directory)
            except ProfileLoadError as exc:
                # A broken profile is exactly when you most want to ask about
                # it, so this says which file rather than failing silently.
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        turn = chats.start(request.profile, request.message)
        asyncio.create_task(_chat_turn(turn, profile, chats, chat_model_factory))
        return ChatCreated(turn_id=turn.turn_id, intent=turn.intent)

    @app.get("/api/chat/{turn_id}/events")
    async def stream_chat(turn_id: str, http: Request) -> StreamingResponse:
        turn = owned_turn(http, turn_id)
        return StreamingResponse(
            _chat_sse(turn),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/chat/{turn_id}", response_model=ChatTurnView)
    async def get_chat_turn(turn_id: str, http: Request) -> ChatTurnView:
        return _turn_view(owned_turn(http, turn_id))

    @app.post("/api/chat/apply", response_model=ChatTurnView)
    async def apply_proposal(request: ApplyProposalRequest, http: Request) -> ChatTurnView:
        async with open_profile(http, request.profile, writable=True) as directory:
            turn = chats.get(request.turn_id)
            if turn is None or turn.proposal is None:
                raise HTTPException(status_code=404, detail="no proposal for that turn")
            owned_turn(http, request.turn_id)

            try:
                turn.applied = apply_proposal_items(turn.proposal, directory, request.accept)
            except (ProfileWriteError, ValueError) as exc:
                # Includes the whole-directory validation refusing the write,
                # which is a normal outcome worth reading rather than a crash.
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info("chat applied %d item(s) to %s", len(turn.applied), request.profile)
        return _turn_view(turn)

    # -- running -----------------------------------------------------------

    @app.post("/api/runs", response_model=RunCreated, status_code=202)
    async def start_run(request: RunRequest, http: Request) -> RunCreated:
        if multi is not None:
            return await start_run_for(signed_in_user(http), request)

        # Checked here rather than inside the run: an unknown profile should be
        # a 400 on the request that named it, not an error buried in a stream
        # the page is still waiting to connect to.
        resolve_profile_dir(request.profile)

        # Only on the public demo. Running this on your own machine, against
        # your own key, is nobody's business but yours.
        if limiter is not None:
            refusal = limiter.check(_client_ip(http))
            if refusal:
                raise HTTPException(status_code=429, detail=refusal)

        run = Run(run_id=uuid.uuid4().hex[:12])
        runs[run.run_id] = run
        run.task = asyncio.create_task(_execute(run, request, graph_factory))
        return RunCreated(run_id=run.run_id)

    async def start_run_for(user, request: RunRequest) -> RunCreated:
        """A multi-user run: counted, snapshotted, and recorded before it starts."""
        run = Run(run_id=uuid.uuid4().hex[:12], user_id=user.id)
        async with quota_lock:
            # The run reads a private copy taken now, so saving the profile
            # while it runs cannot change what it is reading.
            snapshot = await multi.workspaces.snapshot_for_run(user, run.run_id)
            try:
                resolve_profile_dir(request.profile, snapshot.parent)
                refusal = quota.refusal(*await runs_used(user.id))
                if refusal:
                    raise HTTPException(status_code=429, detail=refusal)
            except HTTPException:
                multi.workspaces.discard_run(run.run_id)
                raise
            # Recorded as it starts, inside the lock: a run in flight counts
            # against today's quota, so ten quick clicks cannot start ten.
            await asyncio.to_thread(multi.db.start_run, run.run_id, user.id)

        runs[run.run_id] = run
        run.task = asyncio.create_task(
            _execute_for_user(run, request, graph_factory, multi, run_slots, snapshot)
        )
        return RunCreated(run_id=run.run_id)

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    async def get_run(run_id: str, http: Request) -> RunSummary:
        run = owned_run(http, run_id)
        if run.summary is not None:
            return run.summary
        return RunSummary(run_id=run_id, status=run.status)

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str, http: Request) -> StreamingResponse:
        run = owned_run(http, run_id)
        return StreamingResponse(
            _sse(run),
            media_type="text/event-stream",
            # Without these a proxy or the browser will happily buffer the whole
            # stream and deliver it at the end, which looks exactly like the
            # feature not working.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/pdf")
    async def get_pdf(run_id: str, http: Request) -> Response:
        if multi is not None:
            # From the database, with ownership in the query: it outlives the
            # server restarting, and another person's run id finds nothing.
            pdf = await asyncio.to_thread(multi.db.run_pdf, signed_in_user(http).id, run_id)
            if pdf is None:
                raise HTTPException(status_code=404, detail="no PDF for this run")
            return Response(
                pdf,
                media_type="application/pdf",
                headers={"Content-Disposition": 'attachment; filename="resume.pdf"'},
            )
        run = runs.get(run_id)
        if run is None or run.pdf_path is None or not run.pdf_path.is_file():
            raise HTTPException(status_code=404, detail="no PDF for this run")
        return FileResponse(run.pdf_path, media_type="application/pdf", filename="resume.pdf")

    if demo:
        _strip_mutating_routes(app)
    return app


# The only mutating endpoint a public visitor may reach. Everything else that
# writes career data or deletes a file is removed outright in demo mode.
#
# Deliberately an allow-list, not a block-list. A block-list has one failure
# mode that matters: the endpoint added next year by someone who has never read
# this file. With an allow-list, a new route is excluded by default and somebody
# has to make a decision to expose it.
DEMO_ALLOWED_MUTATIONS = {("POST", "/api/runs")}


def _client_ip(request: Request) -> str:
    """The visitor's address, as best we can tell behind a proxy.

    `X-Forwarded-For` is set by the platform's load balancer and is trivially
    spoofable by a client that sends its own. That is tolerable because the
    per-address limit is a courtesy, not a control -- the global daily cap in
    `limits.py` is what actually protects the account, and no header can move
    that one.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _strip_mutating_routes(app: FastAPI) -> None:
    """Remove every mutating route except the one the demo exists to show.

    The routes are *removed*, not made to return 403: there is nothing to probe,
    nothing to get past, and no handler reachable by a path some guard failed to
    anticipate. `test_demo_exposes_nothing_but_the_run_endpoint` fails loudly if
    that ever stops being true.
    """
    mutating = {"POST", "PUT", "PATCH", "DELETE"}
    kept = []
    for route in app.router.routes:
        methods = getattr(route, "methods", None) or set()
        touched = methods & mutating
        if not touched:
            kept.append(route)
            continue
        path = getattr(route, "path", "")
        if all((method, path) in DEMO_ALLOWED_MUTATIONS for method in touched):
            kept.append(route)
    app.router.routes = kept


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


async def _execute(
    run: Run,
    request: RunRequest,
    graph_factory,
    *,
    profile_path: Path | None = None,
    out_dir: Path | None = None,
    tracker_db: Path | None = None,
    settle: Callable[[RunSummary, Path | None], Awaitable[None]] | None = None,
) -> None:
    """Drive the graph, appending filtered events to the run's log.

    The keyword arguments are multi-user mode's; left out, the run reads the
    profile the request named and writes where it always has.
    """
    started = time.monotonic()

    def emit(event: NodeEvent) -> None:
        run.history.append(event)

    try:
        run.status = "running"
        options = RunOptions(
            out_dir=str(out_dir or Path("out") / "api" / run.run_id),
            tracker_db=str(tracker_db) if tracker_db else None,
            strict=request.strict,
            use_judge=request.use_judge,
            write_cover_letter=request.write_cover_letter,
            max_pages=request.max_pages,
            layout=request.layout,
            summary=request.summary,
        )
        graph = graph_factory()
        state_in = initial_state(request.jd, profile_path or request.profile, options)

        final: dict[str, Any] = {}
        async for event in graph.astream_events(
            state_in, {"recursion_limit": 120}, version="v2"
        ):
            kind, name = event["event"], event.get("name", "")
            elapsed = round(time.monotonic() - started, 1)

            if name in GRAPH_NODES and kind == "on_chain_start":
                emit(NodeEvent(kind="node_start", name=name, elapsed_s=elapsed))
            elif name in GRAPH_NODES and kind == "on_chain_end":
                emit(NodeEvent(kind="node_end", name=name, elapsed_s=elapsed))
                output = (event.get("data") or {}).get("output")
                if isinstance(output, dict):
                    final.update(output)
            elif kind == "on_chat_model_start":
                # The reason for using astream_events at all: without this the
                # page is a frozen spinner for the twenty seconds a tailoring
                # call takes.
                emit(NodeEvent(kind="model_start", name="model", detail=name, elapsed_s=elapsed))
            elif kind == "on_chat_model_end":
                emit(NodeEvent(kind="model_end", name="model", detail=name, elapsed_s=elapsed))

        # `astream_events` yields deltas; the authoritative final state is
        # whatever the graph settled on, so read it back rather than trusting
        # the accumulation above.
        summary = summarise_state(run.run_id, "done", final)
        pdf = final.get("pdf_path")
        pdf_path = Path(pdf) if pdf else None
        if settle is not None:
            # Before anyone can be told it is done -- and a summary on the run
            # *says* done, which is why it is not attached until after -- so the
            # Download click that follows always finds the PDF where it is served.
            await settle(summary, pdf_path)
        run.summary, run.pdf_path, run.status = summary, pdf_path, "done"
        emit(NodeEvent(kind="status", name="done", elapsed_s=round(time.monotonic() - started, 1)))

    except Exception as exc:  # noqa: BLE001 - the page must be told, not left hanging
        logger.exception("run %s failed", run.run_id)
        run.status = "failed"
        run.summary = RunSummary(run_id=run.run_id, status="failed", errors=[str(exc)])
        emit(NodeEvent(kind="error", name="failed", detail=str(exc)))


async def _execute_for_user(
    run: Run,
    request: RunRequest,
    graph_factory,
    multi: MultiUser,
    slots: asyncio.Semaphore,
    snapshot: Path,
) -> None:
    """A multi-user run: wait for a free slot, run on the snapshot, keep the result.

    Everything the run writes lands beside its snapshot and is deleted after.
    The database keeps the outcome and the PDF; the disk of a free instance is
    not somewhere anything should have to survive.
    """

    async def settle(summary: RunSummary, path: Path | None) -> None:
        pdf = await asyncio.to_thread(path.read_bytes) if path and path.is_file() else None
        await asyncio.to_thread(
            multi.db.finish_run,
            run.run_id,
            status="done",
            company=summary.company,
            title=summary.title,
            overall_fit=summary.overall_fit,
            pdf=pdf,
        )

    try:
        if slots.locked():
            run.history.append(
                NodeEvent(kind="status", name="queued",
                          detail="waiting for another resume to finish")
            )
        async with slots:
            await _execute(
                run, request, graph_factory,
                profile_path=snapshot,
                out_dir=snapshot.parent / "out",
                tracker_db=snapshot.parent / "applications.db",
                settle=settle,
            )
        if run.status == "failed":
            await asyncio.to_thread(multi.db.finish_run, run.run_id, status="failed")
    except Exception:  # noqa: BLE001 - the run itself has already said what happened
        logger.exception("run %s: could not record its outcome", run.run_id)
    finally:
        multi.workspaces.discard_run(run.run_id)


async def _chat_turn(turn, profile, chats: Registry, model_factory) -> None:
    """Run one chat turn, appending to its own event log.

    Two shapes, decided before the call: advice streams prose as it arrives,
    extraction is a single structured call that ends in a proposal. Streaming
    partial JSON would show the user something they should never see.
    """
    try:
        if turn.intent == "extract":
            turn.emit("status", text="reading what you wrote…")
            proposal = extract(
                turn.message, profile, llm=model_factory() if model_factory else None
            )
            turn.proposal = proposal
            turn.reply = proposal.reply
            turn.emit("proposal")
        else:
            model = model_factory() if model_factory else None
            async for fragment in advise(
                turn.message,
                profile,
                chats.conversation(turn.profile).transcript()[:-2],
                llm=model,
            ):
                turn.reply += fragment
                turn.emit("token", text=fragment)

        turn.status = "done"
    except Exception as exc:  # noqa: BLE001 - the page must be told, not left waiting
        logger.exception("chat turn %s failed", turn.turn_id)
        turn.status = "failed"
        turn.error = str(exc)
        turn.emit("error", text=str(exc))


async def _chat_sse(turn) -> AsyncIterator[str]:
    """Same index-read design as `_sse`, terminating with the turn.

    Polled rather than pushed, at the same interval: a token arriving up to a
    tenth of a second late still reads as typing, and a poll needs no condition
    variable to get right.
    """
    sent = 0
    while True:
        while sent < len(turn.events):
            yield f"data: {ChatEvent(**turn.events[sent]).model_dump_json()}\n\n"
            sent += 1

        if turn.status in ("done", "failed"):
            yield "event: end\ndata: {}\n\n"
            return

        await asyncio.sleep(POLL_INTERVAL_S)


def _turn_view(turn) -> ChatTurnView:
    proposal = turn.proposal
    return ChatTurnView(
        turn_id=turn.turn_id,
        status=turn.status,
        intent=turn.intent,
        message=turn.message,
        reply=turn.reply,
        items=[item.model_dump() for item in (proposal.items if proposal else [])],
        flagged=len(proposal.flagged()) if proposal else 0,
        applied=turn.applied,
        error=turn.error,
    )


async def _sse(run: Run) -> AsyncIterator[str]:
    """Serialise a run's events as Server-Sent Events.

    SSE is a text format -- `data: <json>` followed by a blank line -- so it
    needs no library.

    Reads the history by index rather than consuming a queue. That is what makes
    a refresh mid-run replay everything and then continue, instead of showing an
    empty log because the previous connection drained the events.
    """
    sent = 0
    while True:
        while sent < len(run.history):
            yield f"data: {run.history[sent].model_dump_json()}\n\n"
            sent += 1

        if run.status in ("done", "failed"):
            yield "event: end\ndata: {}\n\n"
            return

        await asyncio.sleep(POLL_INTERVAL_S)
