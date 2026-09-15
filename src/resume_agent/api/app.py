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
    CreateProfileRequest,
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
    SaveResult,
    build_profile_detail,
    summarise_state,
)
from resume_agent.graph.build import build_graph, initial_state
from resume_agent.graph.state import RunOptions
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


def create_app(graph_factory=build_graph) -> FastAPI:
    """Build the app.

    `graph_factory` is injectable so the tests can drive every endpoint with a
    fake graph -- no API key, no network, no compiler.
    """
    app = FastAPI(title="resume-agent", docs_url="/api/docs")
    runs: dict[str, Run] = {}

    # -- read-only ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

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
        directory = resolve_profile_dir(request.profile)
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
