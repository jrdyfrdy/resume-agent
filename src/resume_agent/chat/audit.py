"""What is weak about a knowledge base, computed rather than opined.

Every other surface in this project judges a profile **against a posting**:
`analyze` needs a job description, the eval judge needs a `JobSpec`. Nothing
answers "what should I improve about this, in general" -- which is the question
you actually have while writing it, before any particular job is in view.

CLAUDE.md rule 2: the LLM judges, Python counts. Everything here is arithmetic
over the profile, so the chat can cite facts instead of forming impressions --
"four of your achievements record no numbers, and here they are" rather than
"consider adding more metrics". The model's job is to explain and prioritise
what this module found, never to invent findings of its own.

The findings are not a style checklist. Each one names a way the pipeline will
actually treat the profile worse, and says which part does it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from resume_agent.graph.nodes.select import MAX_BULLETS_PER_THEME
from resume_agent.models.profile import Profile

# Spec 5 / `tailor_bullets.md` rule 4. A bullet that opens with one of these is
# not wrong, but the rewriter cannot fix it without adding facts it is forbidden
# to add -- so a weak opener in the source survives all the way to the page.
WEAK_OPENERS = (
    "responsible for",
    "worked on",
    "helped with",
    "helped to",
    "assisted with",
    "involved in",
    "tasked with",
    "duties included",
    "participated in",
)

# "Utilized" is the one the prompt calls out by name; the rest are the same
# habit. Matched as the first word only.
WEAK_FIRST_WORDS = ("utilized", "utilised", "leveraged", "spearheaded")

# An entry with this many achievements or fewer has little for selection to
# choose between, so it tends to be represented by whatever it happens to have.
THIN_ENTRY_BULLETS = 1


@dataclass(frozen=True)
class Finding:
    """One thing worth fixing, and what it costs to leave alone."""

    kind: str
    subjects: list[str]
    detail: str


@dataclass(frozen=True)
class ProfileAudit:
    """Facts about a profile. No judgement, no prose, no model."""

    bullets: int = 0
    entries: int = 0
    bullets_without_metrics: list[str] = field(default_factory=list)
    theme_counts: dict[str, int] = field(default_factory=dict)
    crowded_themes: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    unused_skills: list[str] = field(default_factory=list)
    weak_openers: list[str] = field(default_factory=list)
    thin_entries: list[str] = field(default_factory=list)
    narratives: int = 0

    def findings(self) -> list[Finding]:
        """The facts, ordered by how much they cost, with the cost named.

        Ordering is deliberate and mirrors `report.py`: what is holding the
        resume back comes before what is merely untidy.
        """
        found: list[Finding] = []

        if self.bullets_without_metrics:
            found.append(Finding(
                "no_metrics",
                self.bullets_without_metrics,
                "These achievements record no numbers, so no rewrite of them can ever "
                "contain one -- the grounding check allows only figures that trace back "
                "to this list. They are capped at qualitative no matter which job they "
                "are tailored for.",
            ))
        if self.crowded_themes:
            found.append(Finding(
                "crowded_themes",
                self.crowded_themes,
                f"At most {MAX_BULLETS_PER_THEME} achievements per theme reach a page. "
                "Anything beyond that in these themes can never be selected alongside "
                "the rest, however good it is.",
            ))
        if self.thin_entries:
            found.append(Finding(
                "thin_entry",
                self.thin_entries,
                "These roles have almost nothing to choose between, so whichever "
                "achievement exists is the one that represents them -- relevant or not.",
            ))
        if self.weak_openers:
            found.append(Finding(
                "weak_opener",
                self.weak_openers,
                "These start with a phrase that describes a duty rather than an "
                "outcome. The rewriter cannot fix that without inventing a result, "
                "which it is forbidden to do, so the weak opening survives to the page.",
            ))
        if self.claims:
            found.append(Finding(
                "unverified",
                self.claims,
                "Marked as claims, so they disappear entirely when you run with "
                "strict. Worth either substantiating or accepting as strict-only.",
            ))
        if self.unused_skills:
            found.append(Finding(
                "unused_skill",
                self.unused_skills,
                "Declared in your skills but named by no achievement. They still help "
                "a posting match you, but nothing on the page will ever evidence them.",
            ))
        return found

    def as_dict(self) -> dict:
        return asdict(self)


def audit_profile(profile: Profile) -> ProfileAudit:
    """Count everything worth counting about a profile."""
    all_bullets = profile.all_bullets()

    theme_counts: dict[str, int] = {}
    for bullet in all_bullets:
        for theme in bullet.themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    named = {name.lower() for bullet in all_bullets for name in bullet.skills}
    named |= {name.lower() for entry in profile.entries() for name in entry.tech}

    return ProfileAudit(
        bullets=len(all_bullets),
        entries=len(profile.entries()),
        bullets_without_metrics=[b.id for b in all_bullets if not b.metrics],
        theme_counts=dict(sorted(theme_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        crowded_themes=[t for t, n in theme_counts.items() if n > MAX_BULLETS_PER_THEME],
        claims=[b.id for b in all_bullets if b.confidence == "claim"],
        unused_skills=_unused_skills(profile, named),
        weak_openers=[b.id for b in all_bullets if has_weak_opener(b.canonical)],
        thin_entries=[e.id for e in profile.entries() if len(e.bullets) <= THIN_ENTRY_BULLETS],
        narratives=len(profile.narratives),
    )


def has_weak_opener(canonical: str) -> bool:
    """Does this sentence open with a duty rather than an outcome?

    Also catches a leading gerund ("Building the..."), which the prompt forbids
    in a rewrite -- but a rewrite inherits its shape from the source, so the
    place to fix it is here.
    """
    text = canonical.strip().lower()
    if not text:
        return False
    if any(text.startswith(phrase) for phrase in WEAK_OPENERS):
        return True

    first = re.split(r"[^a-z]+", text, maxsplit=1)[0]
    if first in WEAK_FIRST_WORDS:
        return True
    # A gerund opener, minus the words that are ordinary nouns here.
    return first.endswith("ing") and first not in {"engineering", "consulting", "marketing"}


def _unused_skills(profile: Profile, named: set[str]) -> list[str]:
    """Skills declared but evidenced nowhere.

    A skill counts as used if the achievement named its canonical form *or* any
    of its aliases -- matching `Profile.skill_vocabulary`, which is
    case-insensitive and alias-aware, so this does not report a skill merely
    written a different way.
    """
    unused = []
    for skill in profile.skills:
        forms = {skill.canonical.lower()} | {alias.lower() for alias in skill.aliases}
        if not (forms & named):
            unused.append(skill.canonical)
    return unused
