"""Request and response shapes for the API. Spec 8's M9.

Separate from the graph's own models because the wire is a boundary: the browser
gets a small, stable, JSON-shaped view, not `AgentState` with its Pydantic
objects and its internals. When the graph's state changes -- and it has, in
every milestone since M5 -- the page should not have to.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from resume_agent.latex.context import format_date_range
from resume_agent.models.profile import Bullet, Profile

RunStatus = Literal["queued", "running", "paused", "done", "failed"]


class RunRequest(BaseModel):
    """What the page posts to start a run."""

    model_config = ConfigDict(extra="forbid")

    jd: str = Field(min_length=1, description="The job posting, pasted in.")
    profile: str = "profile.example"
    strict: bool = False
    use_judge: bool = True
    write_cover_letter: bool = True


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
    """A role or a project, flattened into something renderable."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["experience", "project"]
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


class ApplicationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int | None
    applied_on: str
    company: str
    title: str
    overall_fit: float | None
    outcome: str | None


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
                subheading=f"{format_date_range(role.start, role.end)} · {role.location}",
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
                    for part in (format_date_range(project.start, project.end), project.role)
                    if part
                ),
                tech=list(project.tech),
                bullets=[_bullet_view(b) for b in project.bullets],
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
