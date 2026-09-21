"""The graph's state. Spec 4.

    "Use `Annotated[list[X], operator.add]` only for genuinely append-only
     fields. Everything else is last-write-wins, which is what you want for
     things like `line_budget` that get revised in a loop."

That distinction is the one thing to get right here. `critiques`, `errors` and
`dropped_bullets` accumulate across iterations and want the `operator.add`
reducer. `line_budget`, `grounding_attempts` and `layout_attempts` are *revised*
each time round a loop, and giving them an accumulating reducer would turn
`layout_attempts` into `[1, 1, 1]` -- which still evaluates truthy, so the cap
would never fire and the loop would run forever.

Nodes return only the keys they changed (spec 5), and LangGraph merges each
returned dict into the state using these reducers.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict

from resume_agent.models.fit import EvidenceMatch, FitReport
from resume_agent.models.job import JobSpec
from resume_agent.models.letter import CoverLetter
from resume_agent.models.resume import TailoredBullet
from resume_agent.sections import LayoutChoice


class RunOptions(BaseModel):
    """Per-run switches. Spec 4 names this in the state; the fields are ours."""

    model_config = ConfigDict(extra="forbid")

    out_dir: str = "out"
    # Spec 3.1: bullets marked `claim` can be excluded via a strict mode.
    strict: bool = False
    # The grounding judge costs money. Off makes a run free of layer-2 calls
    # while keeping both free deterministic layers -- used by the loop tests.
    use_judge: bool = True
    use_cache: bool = True
    # Top-k per requirement at retrieval (spec 5 says 8). Exposed because a
    # profile larger than the default reach would never have its later
    # bullets scored at all -- and the layout loop can only be exercised by a
    # profile with more content than one page holds.
    retrieval_k: int = 8
    # M6. Off skips the whole subgraph -- useful when iterating on the resume
    # half, since a letter costs a draft call plus a judge call per attempt.
    write_cover_letter: bool = True
    # Spec 5: gate `human_review` behind a flag "so batch runs don't block".
    interactive: bool = False
    # One page is the spec's definition of done and stays the default. Two is
    # reachable for the profiles that genuinely need it -- an academic record
    # with publications, or a fresh graduate whose leadership and coursework are
    # the evidence. The budget for each page count is separately calibrated.
    max_pages: int = 1
    # Section order. "auto" derives it from the profile (see `sections.py`);
    # the two explicit values are for the minority counting gets wrong, such as
    # a career changer whose months of experience are in another field.
    layout: LayoutChoice = "auto"
    # A generated professional summary, gated against the selected bullets.
    # Off by default: it costs four lines of a ~32-line page, and those lines
    # are only worth spending when the summary says something the achievements
    # underneath it do not.
    summary: bool = False


class AgentState(TypedDict, total=False):
    """Everything the graph carries. Spec 4.

    `total=False` because nodes populate it progressively: at START only the
    inputs exist, and requiring every key up front would mean seeding the state
    with a dozen `None`s that say nothing.

    Deliberately absent: `company_brief`, which belongs to the optional
    research node and is still unbuilt. A state field that no node writes is
    scaffolding, and adding one later is a one-line change -- as `cover_letter`
    was when M6 arrived.
    """

    # -- inputs --------------------------------------------------------------
    raw_jd: str
    profile_path: str
    options: RunOptions

    # -- accumulated ---------------------------------------------------------
    job_spec: JobSpec | None
    # Deduplicated bullet ids from retrieval, handed to scoring. A real
    # state field rather than a side channel: LangGraph merges only declared
    # keys, so an undeclared one is silently dropped between nodes.
    candidates: list[str]
    evidence: list[EvidenceMatch]
    fit_report: FitReport | None
    selected: list[str]  # bullet ids
    tailored: list[TailoredBullet]
    cover_letter: CoverLetter | None
    letter_verified: bool
    review_action: str

    # -- artifacts -----------------------------------------------------------
    # Empty when no summary was asked for, or when the gate rejected every
    # attempt -- the renderer treats both the same way and prints no section.
    summary_text: str
    tex_source: str | None
    pdf_path: str | None
    compile_log: str | None
    page_count: int | None
    out_dir: str | None

    # -- control -------------------------------------------------------------
    # Last-write-wins on purpose. See the module docstring.
    line_budget: int
    grounding_attempts: int
    layout_attempts: int
    letter_attempts: int
    revision_rounds: int

    # Genuinely append-only.
    dropped_bullets: Annotated[list[str], operator.add]
    critiques: Annotated[list[str], operator.add]
    letter_critiques: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
