"""Pydantic models for the career knowledge base (spec sections 3.1 and 3.2).

The whole project rests on this schema. Resumes are not stored as prose here --
they are stored as *atomic evidence units*: one achievement per `Bullet`, each
carrying the structured facts that later milestones use to prove the generated
text did not invent anything.

Three fields do the heavy lifting downstream:

* `canonical` -- the ground-truth sentence. M4 may rephrase it, never contradict it.
* `metrics`   -- the only numbers a generated bullet is allowed to contain.
* `skills`    -- feeds retrieval (M1) and doubles as the technology allow-list (M4).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# --- Shared scalar types ---------------------------------------------------

Confidence = Literal["verified", "approximate", "claim"]
Seniority = Literal["intern", "junior", "mid", "senior", "staff", "lead"]
SkillLevel = Literal["expert", "working", "familiar"]

# Dates are YYYY-MM strings rather than `datetime.date`. A resume never shows a
# day, and keeping the on-disk form identical to the in-memory form means the
# YAML stays hand-editable without a serialisation round-trip surprising anyone.
YearMonth = Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]

# Spec 3.1 only shows integer metrics, but real achievements carry values like
# "99.95%" or "3 regions". Widening the type now avoids reshaping the data at M4,
# where `allowed_number_forms()` will normalise whatever is in here. (Plan D2.)
MetricValue = int | float | str


class _Strict(BaseModel):
    """Base for every profile model: unknown YAML keys are an error, not a shrug.

    A typo like `cannonical:` would otherwise be silently dropped and the bullet
    would render empty. Failing at load is much easier to debug.
    """

    model_config = ConfigDict(extra="forbid")


# --- Evidence units --------------------------------------------------------


class Bullet(_Strict):
    """One atomic, tagged achievement. Spec 3.1."""

    id: str
    canonical: str
    skills: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    seniority_signal: Seniority | None = None
    evidence: str | None = None
    confidence: Confidence = "verified"


class ExperienceEntry(_Strict):
    """A job. Spec 3.1."""

    id: str
    type: Literal["experience"]
    org: str
    title: str
    location: str
    start: YearMonth
    # `None` means "still there" and renders as "Present". Chosen over a magic
    # "present" string so the field keeps a single validated type. (Plan D1.)
    end: YearMonth | None = None
    tech: list[str] = Field(default_factory=list)
    bullets: list[Bullet]

    @property
    def label(self) -> str:
        """What this entry is called in a list: the employer."""
        return self.org


class ProjectEntry(_Strict):
    """A project. Same bullet shape as experience, plus links. Spec 3.2."""

    id: str
    type: Literal["project"]
    name: str
    role: str | None = None
    start: YearMonth
    end: YearMonth | None = None
    tech: list[str] = Field(default_factory=list)
    repo_url: str | None = None
    live_url: str | None = None
    bullets: list[Bullet]

    @property
    def label(self) -> str:
        return self.name


class LeadershipEntry(_Strict):
    """A society role, committee, or volunteer post. Same shape as a job.

    Deliberately *not* a bespoke record. Because it carries `bullets` it is an
    entry, and being an entry is what earns it the whole pipeline for free: the
    cross-file validators below, retrieval, the knapsack, the grounding gate,
    the form editor and the chat. A separate shape would have to re-implement
    each of those or silently skip them.

    It is a distinct type rather than an `ExperienceEntry` with a flag because
    the two are ordered differently on the page -- see `layout/stage.py`.
    """

    id: str
    type: Literal["leadership"]
    org: str
    title: str
    location: str
    start: YearMonth
    end: YearMonth | None = None
    tech: list[str] = Field(default_factory=list)
    bullets: list[Bullet]

    @property
    def label(self) -> str:
        return self.org


class PublicationEntry(_Strict):
    """A paper, article or talk. Carries bullets, so it is an entry too."""

    id: str
    type: Literal["publication"]
    title: str
    venue: str
    # A publication has one date, but it is called `start` so the shared
    # date-range and recency helpers need no special case. `end` stays None and
    # the renderer prints the single date.
    start: YearMonth
    end: YearMonth | None = None
    url: str | None = None
    tech: list[str] = Field(default_factory=list)
    bullets: list[Bullet]

    @property
    def label(self) -> str:
        return self.title


# Every entry type defines `label`, the name it goes by in a list. Per type, on
# purpose: the alternative was an isinstance chain in each consumer, and the one
# in `kb/index.py` fell through to `.name` for leadership entries -- which have
# no `name` -- so a fresh graduate's profile could not be indexed at all.
#
# Anything that carries bullets. Used as the argument type wherever code walks
# `Profile.entries()` without caring which section it came from -- which, after
# the shared `id`/`start`/`end`/`tech`/`bullets` fields, is nearly everywhere.
Entry = ExperienceEntry | ProjectEntry | LeadershipEntry | PublicationEntry


class EducationEntry(_Strict):
    institution: str
    degree: str
    location: str
    start: YearMonth
    end: YearMonth | None = None
    gpa: str | None = None
    coursework: list[str] = Field(default_factory=list)
    # Dean's list, scholarships, latin honours. Printed under the degree in the
    # student layout, dropped in the experienced one, where the degree alone is
    # all the space education earns.
    honors: list[str] = Field(default_factory=list)


# --- Supporting collections ------------------------------------------------


class Link(_Strict):
    label: str
    url: str


class Identity(_Strict):
    name: str
    email: str
    phone: str
    location: str
    links: list[Link] = Field(default_factory=list)
    work_authorization: str | None = None


class Skill(_Strict):
    """One entry in the canonical skill vocabulary. Spec 3.2.

    `aliases` serves double duty: query expansion at retrieval time (M1) and the
    allow-list for the fabrication check (M4).
    """

    canonical: str
    aliases: list[str] = Field(default_factory=list)
    category: str
    level: SkillLevel
    first_used: YearMonth | None = None


class Narrative(_Strict):
    """One markdown file from `narratives/`. Spec 3.2.

    Prose, not schema. These are the raw material for cover letters -- the
    *why* and *how* of working a certain way -- and imposing structure on
    them would get in the way of writing them honestly.

    They are still subject to the fabrication gate downstream: a narrative
    that names a technology absent from skills.yaml produces letters that
    fail verification. See profile.example/narratives/README.md.
    """

    name: str  # the filename stem, e.g. "how_i_learn"
    content: str


class Certification(_Strict):
    name: str
    issuer: str
    issued: YearMonth
    credential_url: str | None = None


class Award(_Strict):
    """A placing, scholarship or honour. One line, no bullets.

    Distinct from `Certification`: a certification is issued on passing
    something and can expire; an award is placing in something. They print
    side by side under Education in the student layout, which is why both
    carry `issuer` and a single date.
    """

    name: str
    issuer: str
    received: YearMonth
    # "Valid thru Nov. 2027", "GWA 1.70" -- a short qualifier printed in
    # parentheses. Free text because the variety here is genuinely unbounded.
    detail: str | None = None


# --- The whole profile -----------------------------------------------------


class Profile(_Strict):
    """Everything in `profile/`, validated and cross-checked."""

    identity: Identity
    education: list[EducationEntry] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    leadership: list[LeadershipEntry] = Field(default_factory=list)
    publications: list[PublicationEntry] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    narratives: list[Narrative] = Field(default_factory=list)

    # -- derived views ------------------------------------------------------

    def entries(self) -> list[Entry]:
        """Everything that carries bullets, in section order.

        The single most important line in this file for the new types. Every
        cross-file validator below iterates this, and so do retrieval,
        selection, `chat/extract.py` and `kb/forms.py`. A bullet-carrying type
        left out of here is not "unsupported" -- it is worse than that, because
        it loads fine and silently skips every integrity check.
        """
        return [*self.experience, *self.projects, *self.leadership, *self.publications]

    def all_bullets(self) -> list[Bullet]:
        return [b for entry in self.entries() for b in entry.bullets]

    def bullet_by_id(self, bullet_id: str) -> Bullet:
        for bullet in self.all_bullets():
            if bullet.id == bullet_id:
                return bullet
        raise KeyError(f"no bullet with id {bullet_id!r}")

    def narrative(self, name: str) -> Narrative | None:
        return next((n for n in self.narratives if n.name == name), None)

    def skill_vocabulary(self) -> set[str]:
        """Every accepted way of naming a technology, lowercased.

        This is the set M4's verifier checks generated bullets against, so it is
        built here once rather than being reconstructed by each consumer.
        """
        vocab: set[str] = set()
        for skill in self.skills:
            vocab.add(skill.canonical.lower())
            vocab.update(alias.lower() for alias in skill.aliases)
        return vocab

    # -- cross-file integrity checks ---------------------------------------
    #
    # These run at load. Each one catches a data bug that would otherwise show up
    # much later as a confusing retrieval miss or a blank line in the PDF.

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Profile:
        self._reject_duplicates([e.id for e in self.entries()], "entry")
        self._reject_duplicates([b.id for b in self.all_bullets()], "bullet")
        return self

    @model_validator(mode="after")
    def _bullet_ids_are_namespaced(self) -> Profile:
        """`exp_acme_be.b1` must live inside the entry `exp_acme_be`.

        Bullet ids end up in `run.json` and in the tracker (spec 5, `finalize`).
        Making them self-describing means a bullet id alone tells you which job
        it came from, without a lookup.
        """
        for entry in self.entries():
            prefix = entry.id + "."
            for bullet in entry.bullets:
                if not bullet.id.startswith(prefix):
                    raise ValueError(
                        f"bullet id {bullet.id!r} must start with {prefix!r} "
                        f"(it lives in entry {entry.id!r})"
                    )
        return self

    @model_validator(mode="after")
    def _skills_resolve_to_vocabulary(self) -> Profile:
        """Every named technology must exist in skills.yaml.

        An unknown skill string is precisely the signal M4 treats as fabrication,
        so catching it at load costs nothing and turns a silent retrieval miss
        into a loud error. (Plan D5.)
        """
        vocab = self.skill_vocabulary()
        unknown: list[str] = []
        for entry in self.entries():
            for name in entry.tech:
                if name.lower() not in vocab:
                    unknown.append(f"{entry.id}.tech: {name!r}")
            for bullet in entry.bullets:
                for name in bullet.skills:
                    if name.lower() not in vocab:
                        unknown.append(f"{bullet.id}.skills: {name!r}")
        if unknown:
            raise ValueError(
                "these skills are not in skills.yaml (add them there, or fix the typo):\n  "
                + "\n  ".join(unknown)
            )
        return self

    @staticmethod
    def _reject_duplicates(values: list[str], label: str) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(f"duplicate {label} ids: {sorted(duplicates)}")
