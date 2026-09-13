"""Request and response shapes for the API. Spec 8's M9.

Separate from the graph's own models because the wire is a boundary: the browser
gets a small, stable, JSON-shaped view, not `AgentState` with its Pydantic
objects and its internals. When the graph's state changes -- and it has, in
every milestone since M5 -- the page should not have to.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
