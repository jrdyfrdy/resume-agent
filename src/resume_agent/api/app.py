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

**The run registry is an in-process dict.** This is a localhost, single-user
tool; anything else would be inventing a deployment story the spec does not ask
for. It is the first thing to replace if this ever runs anywhere real.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

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
    "parse_jd", "retrieve", "score", "select", "tailor", "verify", "render",
    "compile", "inspect", "fix_latex", "shrink_budget", "note_overfull",
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
    summary: RunSummary | None = None
    pdf_path: Path | None = None
    task: asyncio.Task | None = None
    # The append-only event log. The SSE endpoint reads it by index rather than
    # consuming a queue, which is what makes a browser refresh reconnect to a
    # run in progress and replay what it missed -- a queue would have been
    # drained by the connection that dropped.
    history: list[NodeEvent] = field(default_factory=list)


def create_app(graph_factory=build_graph, chat_model_factory=None) -> FastAPI:
    """Build the app.

    `graph_factory` and `chat_model_factory` are injectable so the tests can
    drive every endpoint with fakes -- no API key, no network, no compiler.
    """
    app = FastAPI(title="resume-agent", docs_url="/api/docs")
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

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return page

    @app.get("/api/profile", response_model=ProfileSummary)
    async def profile_summary(profile: str = DEFAULT_PROFILE) -> ProfileSummary:
        directory = resolve_profile_dir(profile)
        try:
            loaded = load_profile(directory)
        except ProfileLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        )

    @app.get("/api/profiles", response_model=list[ProfileOption])
    async def profiles() -> list[ProfileOption]:
        """Every knowledge base on disk that a run could point at."""
        return discover_profiles()

    @app.get("/api/profile/detail", response_model=ProfileDetail)
    async def profile_detail(profile: str = DEFAULT_PROFILE) -> ProfileDetail:
        directory = resolve_profile_dir(profile)
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
    async def profile_files(profile: str = DEFAULT_PROFILE) -> ProfileFileList:
        directory = resolve_profile_dir(profile)
        try:
            return ProfileFileList(profile=profile, files=relative_profile_files(directory))
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/profile/file", response_model=ProfileFile)
    async def profile_file(path: str, profile: str = DEFAULT_PROFILE) -> ProfileFile:
        directory = resolve_profile_dir(profile)
        try:
            return ProfileFile(
                profile=profile, path=path, text=read_profile_file(directory, path)
            )
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/profile/file", response_model=SaveResult)
    async def save_profile_file(request: SaveFileRequest) -> SaveResult:
        directory = resolve_writable_profile_dir(request.profile)
        try:
            resolve_editable_path(directory, request.path)
        except ProfileWriteError as exc:
            # A path the editor should never have sent: a bug, not bad YAML.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            backup = write_profile_file(directory, request.path, request.text)
        except ProfileWriteError as exc:
            # Invalid content is an ordinary part of editing YAML, so it is a
            # result rather than an error status. The file on disk is unchanged.
            return SaveResult(ok=False, error=str(exc))

        logger.info("saved %s in %s (backup: %s)", request.path, request.profile, backup)
        return SaveResult(ok=True, backup=str(backup))

    # -- structured editing ------------------------------------------------
    #
    # The form endpoints are a thin layer over `forms.py`, which is itself a layer
    # over `writer.py`. Nothing here writes: it translates `ProfileWriteError`
    # into a status code and lets the writer keep its guarantees.

    @app.get("/api/profile/form", response_model=FormDocument)
    async def profile_form(path: str, profile: str = DEFAULT_PROFILE) -> FormDocument:
        directory = resolve_profile_dir(profile)
        try:
            document = read_document(directory, path)
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FormDocument(**document, skill_usage=skill_usage(directory))

    @app.put("/api/profile/form", response_model=SaveResult)
    async def save_profile_form(request: SaveFormRequest) -> SaveResult:
        directory = resolve_writable_profile_dir(request.profile)
        try:
            resolve_editable_path(directory, request.path)
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            backup = write_document(directory, request.path, request.data)
        except ProfileWriteError as exc:
            # Same contract as the text editor: content that does not validate
            # is a result, not an error status, and the file is unchanged.
            return SaveResult(ok=False, error=str(exc))

        logger.info("saved %s in %s (backup: %s)", request.path, request.profile, backup)
        return SaveResult(ok=True, backup=str(backup))

    @app.post("/api/profile/entry", response_model=ProfileFile, status_code=201)
    async def add_entry(request: CreateEntryRequest) -> ProfileFile:
        directory = resolve_writable_profile_dir(request.profile)
        try:
            relative = create_entry(directory, request.role, request.name)
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ProfileFile(
            profile=request.profile, path=relative, text=read_profile_file(directory, relative)
        )

    @app.delete("/api/profile/file", response_model=SaveResult)
    async def remove_file(request: DeleteFileRequest) -> SaveResult:
        directory = resolve_writable_profile_dir(request.profile)
        try:
            backup = delete_document(directory, request.path)
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SaveResult(ok=True, backup=str(backup))

    @app.post("/api/profile/create", response_model=ProfileFileList, status_code=201)
    async def create_profile(request: CreateProfileRequest) -> ProfileFileList:
        """Copy an existing profile to a new directory.

        The answer to "where does my knowledge base go", made a button: the
        first step becomes a working profile rather than seven empty files.
        """
        source = resolve_profile_dir(request.source)
        target = Path.cwd() / request.name
        try:
            scaffold_profile(source, target)
            return ProfileFileList(
                profile=request.name, files=relative_profile_files(target)
            )
        except ProfileWriteError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/applications", response_model=list[ApplicationSummary])
    async def applications(limit: int = 20) -> list[ApplicationSummary]:
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
    async def start_chat(request: ChatRequest) -> ChatCreated:
        directory = resolve_profile_dir(request.profile)
        try:
            load_profile(directory)
        except ProfileLoadError as exc:
            # A broken profile is exactly when you most want to ask about it, so
            # this says which file rather than failing silently.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        turn = chats.start(request.profile, request.message)
        asyncio.create_task(_chat_turn(turn, directory, chats, chat_model_factory))
        return ChatCreated(turn_id=turn.turn_id, intent=turn.intent)

    @app.get("/api/chat/{turn_id}/events")
    async def stream_chat(turn_id: str) -> StreamingResponse:
        turn = chats.get(turn_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="no such turn")
        return StreamingResponse(
            _chat_sse(turn),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/chat/{turn_id}", response_model=ChatTurnView)
    async def get_chat_turn(turn_id: str) -> ChatTurnView:
        turn = chats.get(turn_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="no such turn")
        return _turn_view(turn)

    @app.post("/api/chat/apply", response_model=ChatTurnView)
    async def apply_proposal(request: ApplyProposalRequest) -> ChatTurnView:
        directory = resolve_writable_profile_dir(request.profile)
        turn = chats.get(request.turn_id)
        if turn is None or turn.proposal is None:
            raise HTTPException(status_code=404, detail="no proposal for that turn")

        try:
            turn.applied = apply_proposal_items(turn.proposal, directory, request.accept)
        except (ProfileWriteError, ValueError) as exc:
            # Includes the whole-directory validation refusing the write, which
            # is a normal outcome worth reading rather than a crash.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        logger.info("chat applied %d item(s) to %s", len(turn.applied), request.profile)
        return _turn_view(turn)

    # -- running -----------------------------------------------------------

    @app.post("/api/runs", response_model=RunCreated, status_code=202)
    async def start_run(request: RunRequest) -> RunCreated:
        # Checked here rather than inside the run: an unknown profile should be
        # a 400 on the request that named it, not an error buried in a stream
        # the page is still waiting to connect to.
        resolve_profile_dir(request.profile)

        run = Run(run_id=uuid.uuid4().hex[:12])
        runs[run.run_id] = run
        run.task = asyncio.create_task(_execute(run, request, graph_factory))
        return RunCreated(run_id=run.run_id)

    @app.get("/api/runs/{run_id}", response_model=RunSummary)
    async def get_run(run_id: str) -> RunSummary:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        if run.summary is not None:
            return run.summary
        return RunSummary(run_id=run_id, status=run.status)

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str) -> StreamingResponse:
        run = runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        return StreamingResponse(
            _sse(run),
            media_type="text/event-stream",
            # Without these a proxy or the browser will happily buffer the whole
            # stream and deliver it at the end, which looks exactly like the
            # feature not working.
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{run_id}/pdf")
    async def get_pdf(run_id: str) -> FileResponse:
        run = runs.get(run_id)
        if run is None or run.pdf_path is None or not run.pdf_path.is_file():
            raise HTTPException(status_code=404, detail="no PDF for this run")
        return FileResponse(run.pdf_path, media_type="application/pdf", filename="resume.pdf")

    return app


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


async def _execute(run: Run, request: RunRequest, graph_factory) -> None:
    """Drive the graph, appending filtered events to the run's log."""
    started = time.monotonic()

    def emit(event: NodeEvent) -> None:
        run.history.append(event)

    try:
        run.status = "running"
        options = RunOptions(
            out_dir=str(Path("out") / "api" / run.run_id),
            strict=request.strict,
            use_judge=request.use_judge,
            write_cover_letter=request.write_cover_letter,
        )
        graph = graph_factory()
        state_in = initial_state(request.jd, request.profile, options)

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
        run.summary = summarise_state(run.run_id, "done", final)
        pdf = final.get("pdf_path")
        run.pdf_path = Path(pdf) if pdf else None
        run.status = "done"
        emit(NodeEvent(kind="status", name="done", elapsed_s=round(time.monotonic() - started, 1)))

    except Exception as exc:  # noqa: BLE001 - the page must be told, not left hanging
        logger.exception("run %s failed", run.run_id)
        run.status = "failed"
        run.summary = RunSummary(run_id=run.run_id, status="failed", errors=[str(exc)])
        emit(NodeEvent(kind="error", name="failed", detail=str(exc)))


async def _chat_turn(turn, profile_dir: Path, chats: Registry, model_factory) -> None:
    """Run one chat turn, appending to its own event log.

    Two shapes, decided before the call: advice streams prose as it arrives,
    extraction is a single structured call that ends in a proposal. Streaming
    partial JSON would show the user something they should never see.
    """
    try:
        profile = load_profile(profile_dir)

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
