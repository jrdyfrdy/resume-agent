"""Request and response shapes for the API. Spec 8's M9.

Separate from the graph's own models because the wire is a boundary: the browser
gets a small, stable, JSON-shaped view, not `AgentState` with its Pydantic
objects and its internals. When the graph's state changes -- and it has, in
every milestone since M5 -- the page should not have to.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from resume_agent.latex.context import format_date_range, format_year_month
from resume_agent.models.profile import Bullet, Profile

RunStatus = Literal["queued", "running", "paused", "done", "failed"]


def _dates(start: str, end: str | None) -> str:
    """A date range for the browser.

    `format_date_range` defaults to a LaTeX en-dash (`--`), which is right for
    the template and wrong here: in HTML it prints as two literal hyphens, which
    is exactly what the Browse view was showing.
    """
    return format_date_range(start, end, dash="–")


class RunRequest(BaseModel):
    """What the page posts to start a run."""

    model_config = ConfigDict(extra="forbid")

    jd: str = Field(min_length=1, description="The job posting, pasted in.")
    profile: str = "profile.example"
    strict: bool = False
    use_judge: bool = True
    write_cover_letter: bool = True
    # Mirrors RunOptions. Bounded here as well as there because this one arrives
    # from a browser, and `Literal`/`Field` is the cheapest place to refuse a
    # value that could never be right.
    max_pages: int = Field(default=1, ge=1, le=2)
    layout: Literal["auto", "student", "experienced"] = "auto"
    summary: bool = False


class RunCreated(BaseModel):
    run_id: str


class NodeEvent(BaseModel):
    """One line in the live log.

    Deliberately flat and small. It is serialised to SSE on every node
    transition, and the page renders it without knowing anything about
    LangGraph.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["node_start", "node_end", "model_start", "model_end", "status", "error"]
    name: str
    detail: str | None = None
    elapsed_s: float | None = None


class ProfileSummary(BaseModel):
    """What the page shows before a run starts, so it is clear what is loaded."""

    model_config = ConfigDict(extra="forbid")

    name: str
    experience: int
    projects: int
    bullets: int
    skills: int
    has_credentials: bool
    has_compiler: bool
    # Which provider is configured, and the variable its key is read from, so
    # the page can name the right one. Telling a DeepSeek user to set
    # ANTHROPIC_API_KEY is worse than saying nothing. The key itself never
    # crosses this boundary -- only the name of the variable.
    provider: str
    credentials_var: str
    # True on the public demo, where the profile is read-only and runs are
    # rate limited. The page uses it to say so rather than offering controls
    # whose endpoints have been removed.
    demo: bool = False
    runs_left_today: int | None = None


class ProfileOption(BaseModel):
    """One knowledge base the UI can point a run at.

    Discovered by looking for `identity.yaml`, because that is the one file
    every profile must have and the thing `load_profile` fails on first.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    is_example: bool
    # A profile that exists but does not load is worth showing with its error
    # rather than hiding: "my profile vanished from the list" is a much harder
    # thing to debug than "my profile is listed and says what is wrong with it".
    loads: bool
    error: str | None = None
    bullets: int = 0


class BulletView(BaseModel):
    """One bullet as the Profile tab shows it.

    `metrics` is stringified because the page renders it either way, and a wire
    type of `int | float | str` only creates work for the reader.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    canonical: str
    skills: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    metrics: dict[str, str] = Field(default_factory=dict)
    confidence: str = "verified"
    evidence: str | None = None


class EntryView(BaseModel):
    """Any bullet-carrying entry, flattened into something renderable."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["experience", "project", "leadership", "publication"]
    heading: str
    subheading: str
    tech: list[str] = Field(default_factory=list)
    bullets: list[BulletView] = Field(default_factory=list)


class ProfileDetail(BaseModel):
    """Everything the Profile tab needs to show what is actually loaded.

    The point of the tab is that the knowledge base is the input the whole tool
    runs on, and until now it was invisible from the UI -- you could not tell
    whether a bullet you had written was being read at all.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    person: str
    email: str
    location: str
    entries: list[EntryView] = Field(default_factory=list)
    skills_by_category: dict[str, list[str]] = Field(default_factory=dict)
    narratives: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class ProfileFileList(BaseModel):
    """Which files of a profile the editor may open."""

    model_config = ConfigDict(extra="forbid")

    profile: str
    files: list[str] = Field(default_factory=list)


class ProfileFile(BaseModel):
    """One file's text, exactly as it sits on disk."""

    model_config = ConfigDict(extra="forbid")

    profile: str
    path: str
    text: str


class SaveFileRequest(BaseModel):
    """An edited file on its way back.

    No length cap: a profile file is small, and truncating someone's career data
    to satisfy a limit would be worse than any request this guards against.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str
    path: str
    text: str


class SaveResult(BaseModel):
    """What happened to a save.

    Invalid YAML is a **normal** outcome of editing, not an exceptional one, so
    it comes back 200 with `ok=False` and the loader's own message. The 4xx
    codes are reserved for requests that could never succeed -- an unknown
    profile, a path outside it, a file that does not exist. That split keeps the
    editor's error handling honest: one branch renders a validation message, the
    other is a bug.
    """

    model_config = ConfigDict(extra="forbid")

    ok: bool
    error: str | None = None
    # Where the previous text went. `profile/` is gitignored, so this is the
    # only undo that exists, and the editor says so out loud after a save.
    backup: str | None = None


class FormDocument(BaseModel):
    """One file, as a form: what controls to draw, the values, what to suggest.

    `spec` and `suggestions` are loose on purpose. The field table in
    `kb/forms.py` is the single source of truth for both, and mirroring its
    shape into a Pydantic model here would mean editing two files to add a
    control.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    role: str
    kind: str
    spec: list[dict[str, Any]] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    suggestions: dict[str, list[str]] = Field(default_factory=dict)
    # How many bullets and entries name each skill, so the skills form can warn
    # before removing a row that other files depend on.
    skill_usage: dict[str, int] = Field(default_factory=dict)


class SaveFormRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    path: str
    data: dict[str, Any]


class CreateEntryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    role: Literal["experience", "project", "leadership", "publication"]
    name: str = Field(min_length=1, max_length=200)


class DeleteFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    path: str


class CreateProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Constrained here rather than in the handler: this becomes a directory name
    # on disk, so a slash or a dot-dot must never reach the filesystem layer.
    name: str = Field(default="profile", pattern=r"^[A-Za-z0-9._-]{1,64}$")
    source: str = "profile.example"
    # "empty" is the default because a profile you mean to use should not begin
    # as someone else's career; "example" stays for trying the tool out.
    mode: Literal["empty", "example"] = "empty"
    # Only used by "empty" -- the copy takes its identity from the source.
    display_name: str = Field(default="", max_length=200)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    message: str = Field(min_length=1, max_length=20_000)


class ChatCreated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: str
    # Which of the two things this message was taken to be, so the page can say
    # "reading what you wrote" rather than showing a cursor that never types.
    intent: str


class ChatEvent(BaseModel):
    """One line of a chat turn's stream.

    Separate from `NodeEvent` rather than widening its `Literal`: a run reports
    discrete stage boundaries, a chat turn reports prose arriving a fragment at
    a time, and `extra="forbid"` makes a union of the two awkward to read.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["token", "status", "proposal", "error", "done"]
    text: str = ""


class ChatTurnView(BaseModel):
    """A settled turn, with whatever it produced."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    status: str
    intent: str
    message: str
    reply: str = ""
    # Present only for a dictation turn. Held in memory until accepted -- a
    # proposal is a suggestion, not a change.
    items: list[dict] = Field(default_factory=list)
    flagged: int = 0
    applied: list[str] = Field(default_factory=list)
    error: str | None = None


class ApplyProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str
    turn_id: str
    accept: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    """The finished run, as the page needs it."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus
    company: str | None = None
    title: str | None = None
    recommendation: str | None = None
    overall_fit: float | None = None
    page_count: int | None = None
    line_budget: int | None = None
    layout_attempts: int = 0

    bullets: list[str] = Field(default_factory=list)
    # Surfaced rather than buried: a dropped bullet means the resume is weaker
    # than it could have been, and the page should say so plainly.
    dropped_bullets: list[str] = Field(default_factory=list)
    must_have_gaps: list[str] = Field(default_factory=list)
    cover_letter: str | None = None
    cover_letter_words: int | None = None

    has_pdf: bool = False
    errors: list[str] = Field(default_factory=list)


class FileVersion(BaseModel):
    """One saved state of a profile file. Multi-user mode's replacement for the
    backups folder, which on a server is a path nobody can open."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    deleted: bool
    text: str


class ApplicationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None
    applied_on: str
    company: str
    title: str
    overall_fit: float | None
    outcome: str | None
    # Multi-user mode only, where history comes from the accounts database and
    # a past run's PDF can still be downloaded by its id.
    run_id: str | None = None
    has_pdf: bool = False


def build_profile_detail(name: str, profile: Profile) -> ProfileDetail:
    """Flatten a loaded `Profile` into something the browser can render.

    Experience and projects become one list because the Profile tab shows them
    the same way -- a heading, a date range, some technologies, and the bullets
    underneath. Keeping them apart on the wire would only push the job of
    telling them apart into JavaScript.
    """
    entries: list[EntryView] = []

    for role in profile.experience:
        entries.append(
            EntryView(
                id=role.id,
                kind="experience",
                heading=f"{role.title}, {role.org}",
                subheading=f"{_dates(role.start, role.end)} · {role.location}",
                tech=list(role.tech),
                bullets=[_bullet_view(b) for b in role.bullets],
            )
        )

    for project in profile.projects:
        entries.append(
            EntryView(
                id=project.id,
                kind="project",
                heading=project.name,
                subheading=" · ".join(
                    part
                    for part in (_dates(project.start, project.end), project.role)
                    if part
                ),
                tech=list(project.tech),
                bullets=[_bullet_view(b) for b in project.bullets],
            )
        )

    for role in profile.leadership:
        entries.append(
            EntryView(
                id=role.id,
                kind="leadership",
                heading=f"{role.title}, {role.org}",
                subheading=f"{_dates(role.start, role.end)} · {role.location}",
                tech=list(role.tech),
                bullets=[_bullet_view(b) for b in role.bullets],
            )
        )

    for paper in profile.publications:
        entries.append(
            EntryView(
                id=paper.id,
                kind="publication",
                heading=paper.title,
                # One date, not a range -- it was published, it did not run.
                subheading=f"{paper.venue} · {format_year_month(paper.start)}",
                tech=list(paper.tech),
                bullets=[_bullet_view(b) for b in paper.bullets],
            )
        )

    by_category: dict[str, list[str]] = {}
    for skill in profile.skills:
        by_category.setdefault(skill.category, []).append(skill.canonical)
    for names in by_category.values():
        names.sort()

    return ProfileDetail(
        name=name,
        person=profile.identity.name,
        email=profile.identity.email,
        location=profile.identity.location,
        entries=entries,
        skills_by_category=by_category,
        narratives=[n.name for n in profile.narratives],
        education=[f"{e.degree} — {e.institution}" for e in profile.education],
        certifications=[f"{c.name} — {c.issuer}" for c in profile.certifications],
    )


def _bullet_view(bullet: Bullet) -> BulletView:
    return BulletView(
        id=bullet.id,
        canonical=bullet.canonical,
        skills=list(bullet.skills),
        themes=list(bullet.themes),
        # Stringified here rather than in the page: the grounding gate compares
        # these as text anyway, and showing them is the point -- these are the
        # only numbers a tailored bullet is allowed to contain.
        metrics={key: str(value) for key, value in bullet.metrics.items()},
        confidence=bullet.confidence,
        evidence=bullet.evidence,
    )


def summarise_state(run_id: str, status: RunStatus, state: dict[str, Any]) -> RunSummary:
    """Flatten a finished `AgentState` into the wire shape."""
    job = state.get("job_spec")
    fit = state.get("fit_report")
    letter = state.get("cover_letter")

    return RunSummary(
        run_id=run_id,
        status=status,
        company=job.company if job else None,
        title=job.title if job else None,
        recommendation=fit.recommendation if fit else None,
        overall_fit=fit.overall_fit if fit else None,
        page_count=state.get("page_count"),
        line_budget=state.get("line_budget"),
        layout_attempts=state.get("layout_attempts", 0),
        bullets=[t.text for t in state.get("tailored", [])],
        dropped_bullets=list(state.get("dropped_bullets", [])),
        must_have_gaps=[r.text for r in fit.must_have_gaps()] if fit else [],
        cover_letter=letter.body() if letter else None,
        cover_letter_words=letter.word_count if letter else None,
        has_pdf=bool(state.get("pdf_path")),
        errors=list(state.get("errors", [])),
    )
