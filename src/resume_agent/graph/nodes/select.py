"""Choose which achievements go on the page. Spec 5, `select_content`.

    "**No LLM here.** A greedy knapsack:
       maximize sum(relevance x requirement_weight x recency_decay)
       subject to:
         sum(estimated_lines) <= line_budget
         every experience entry that appears gets >= 2 bullets
         >= 1 bullet per covered must-have requirement (if evidence exists)
         <= 3 bullets sharing the same theme
         chronological order preserved within sections"

**Why this is not an LLM call.** It is the clearest instance of CLAUDE.md rule 2
in the project. A model asked to "pick the best 18 bullets under a 32-line
budget" will produce a plausible list that quietly violates the budget, because
it is counting lines by vibe. Here the budget is arithmetic, the constraints are
checkable, and `test_selection.py` proves them with Hypothesis over generated
profiles rather than over the three examples I happened to imagine.

Greedy rather than optimal, per spec 5: "greedy is within a few percent of
optimal here and much easier to debug". The debuggability is not a throwaway --
`SelectionResult.rejected` can name the constraint that excluded each bullet
only because the algorithm makes one decision at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

from resume_agent.latex.metrics import estimate_lines
from resume_agent.models.fit import COVERED_THRESHOLD, EvidenceMatch, SelectionResult
from resume_agent.models.job import JobSpec
from resume_agent.models.profile import Bullet, ExperienceEntry, Profile, ProjectEntry

# Spec 5: "<= 3 bullets sharing the same theme". Diversity constraint -- five
# performance bullets for a job that also wants API design reads as one-note.
MAX_BULLETS_PER_THEME = 3

# Spec 5: "every experience entry that appears gets >= 2 bullets". A job with a
# single bullet under it looks like an afterthought; better to omit the role or
# give it room.
MIN_BULLETS_PER_EXPERIENCE = 2

# Recency decay. Spec 5 asks for the term but not its shape, so: an achievement
# loses half its weight every six years, floored so that genuinely relevant old
# work never falls to nothing. Six years is long enough that a normal career
# history stays in play and short enough that a decade-old bullet has to be
# clearly more relevant to beat a recent one.
RECENCY_HALF_LIFE_YEARS = 6.0
RECENCY_FLOOR = 0.25


@dataclass(frozen=True)
class ScoredBullet:
    """A candidate with everything selection needs, precomputed."""

    bullet_id: str
    entry_id: str
    entry_type: str  # "experience" | "project"
    score: float
    lines: int
    themes: tuple[str, ...]
    confidence: str
    # Requirement texts this bullet is the best available evidence for.
    covers_must_haves: frozenset[str]


def recency_decay(
    end: str | None,
    today: str,
    half_life_years: float = RECENCY_HALF_LIFE_YEARS,
) -> float:
    """How much a YYYY-MM end date is discounted for age. Current roles score 1.0."""
    if end is None:
        return 1.0
    end_year, end_month = (int(part) for part in end.split("-"))
    now_year, now_month = (int(part) for part in today.split("-"))
    years_ago = max(0.0, (now_year - end_year) + (now_month - end_month) / 12.0)
    decayed = 0.5 ** (years_ago / half_life_years)
    return max(RECENCY_FLOOR, decayed)


def _entry_of(profile: Profile, bullet_id: str) -> ExperienceEntry | ProjectEntry:
    for entry in profile.entries():
        if any(b.id == bullet_id for b in entry.bullets):
            return entry
    raise KeyError(f"no entry contains bullet {bullet_id!r}")


def score_bullets(
    job: JobSpec,
    matches: list[EvidenceMatch],
    profile: Profile,
    today: str,
) -> list[ScoredBullet]:
    """Collapse per-requirement matches into one score per bullet.

    A bullet can be evidence for several requirements. Its score is the **sum**
    over requirements of `relevance x weight x recency`, which is what makes a
    single achievement that answers three requirements beat one that answers a
    single requirement slightly better -- the property that gets a broad,
    load-bearing bullet onto the page.
    """
    weight_of = {r.text: r.weight for r in job.requirements}
    must_have_texts = {r.text for r in job.requirements if r.is_must_have}

    totals: dict[str, float] = {}
    covers: dict[str, set[str]] = {}

    for match in matches:
        weight = weight_of.get(match.requirement_text)
        if weight is None:
            continue  # a requirement that is not in this posting; ignore rather than crash
        entry = _entry_of(profile, match.bullet_id)
        decay = recency_decay(entry.end, today)
        totals[match.bullet_id] = totals.get(match.bullet_id, 0.0) + (
            match.relevance * weight * decay
        )
        if match.requirement_text in must_have_texts and match.relevance >= COVERED_THRESHOLD:
            covers.setdefault(match.bullet_id, set()).add(match.requirement_text)

    scored: list[ScoredBullet] = []
    for bullet_id, total in totals.items():
        entry = _entry_of(profile, bullet_id)
        bullet: Bullet = next(b for b in entry.bullets if b.id == bullet_id)
        scored.append(
            ScoredBullet(
                bullet_id=bullet_id,
                entry_id=entry.id,
                entry_type=entry.type,
                score=total,
                lines=estimate_lines(bullet.canonical),
                themes=tuple(bullet.themes),
                confidence=bullet.confidence,
                covers_must_haves=frozenset(covers.get(bullet_id, set())),
            )
        )

    # Ties broken by id so selection is deterministic; two bullets with equal
    # scores must not swap between runs, or the golden .tex snapshot flaps.
    scored.sort(key=lambda s: (-s.score, s.bullet_id))
    return scored


def select_content(
    job: JobSpec,
    matches: list[EvidenceMatch],
    profile: Profile,
    line_budget: int,
    *,
    today: str = "2026-09",
    strict: bool = False,
) -> SelectionResult:
    """Greedy knapsack under every constraint in spec 5."""
    candidates = score_bullets(job, matches, profile, today)
    rejected: dict[str, str] = {}

    if strict:
        # Spec 3.1: "Bullets marked `claim` can be excluded via a strict mode."
        kept = []
        for candidate in candidates:
            if candidate.confidence == "claim":
                rejected[candidate.bullet_id] = "strict mode excludes confidence: claim"
            else:
                kept.append(candidate)
        candidates = kept

    selected: list[ScoredBullet] = []
    used_lines = 0
    theme_counts: dict[str, int] = {}

    def would_break_theme_cap(candidate: ScoredBullet) -> str | None:
        for theme in candidate.themes:
            if theme_counts.get(theme, 0) >= MAX_BULLETS_PER_THEME:
                return f"already {MAX_BULLETS_PER_THEME} bullets with theme {theme!r}"
        return None

    def take(candidate: ScoredBullet) -> None:
        nonlocal used_lines
        selected.append(candidate)
        used_lines += candidate.lines
        for theme in candidate.themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

    # --- pass 1: guarantee must-have coverage --------------------------------
    #
    # Spec 5 requires ">= 1 bullet per covered must-have requirement (if
    # evidence exists)". Done first, because a pure score-ordered greedy pass
    # can spend the whole budget before reaching the only bullet that answers a
    # must-have -- and a resume missing the employer's non-negotiable is worse
    # than one missing a higher-scoring but optional achievement.
    satisfied: set[str] = set()
    for candidate in candidates:
        unmet = candidate.covers_must_haves - satisfied
        if not unmet:
            continue
        if used_lines + candidate.lines > line_budget:
            rejected[candidate.bullet_id] = "no budget left for this must-have bullet"
            continue
        if (reason := would_break_theme_cap(candidate)) is not None:
            rejected[candidate.bullet_id] = reason
            continue
        take(candidate)
        satisfied |= unmet

    # --- pass 2: fill the remaining budget by score --------------------------
    chosen_ids = {c.bullet_id for c in selected}
    for candidate in candidates:
        if candidate.bullet_id in chosen_ids:
            continue
        if used_lines + candidate.lines > line_budget:
            rejected.setdefault(candidate.bullet_id, "would exceed the line budget")
            continue
        if (reason := would_break_theme_cap(candidate)) is not None:
            rejected.setdefault(candidate.bullet_id, reason)
            continue
        take(candidate)
        chosen_ids.add(candidate.bullet_id)

    # --- pass 3: enforce the minimum-bullets-per-experience rule -------------
    #
    # Spec 5: "every experience entry that appears gets >= 2 bullets". Applied
    # last, as a repair: try to promote another bullet from the same entry, and
    # if none fits, drop the lone bullet entirely. Dropping is the honest
    # resolution -- a job heading with one line under it reads worse than the
    # job not appearing.
    selected, used_lines, repair_rejections = _enforce_minimum_experience_bullets(
        selected, candidates, used_lines, line_budget, theme_counts
    )
    rejected.update(repair_rejections)

    # A bullet can be rejected in an earlier pass and then legitimately promoted
    # in pass 3: dropping a thin entry frees both budget and theme slots, so a
    # candidate previously blocked by the theme cap can become admissible. It
    # must not appear in both lists -- `rejected` means "did not make the page".
    # (Found by Hypothesis, not by inspection.)
    for candidate in selected:
        rejected.pop(candidate.bullet_id, None)

    # --- ordering -------------------------------------------------------------
    ordered = _order_for_render(selected, profile)

    return SelectionResult(
        selected_bullet_ids=[c.bullet_id for c in ordered],
        total_estimated_lines=used_lines,
        line_budget=line_budget,
        rejected=rejected,
    )


def _enforce_minimum_experience_bullets(
    selected: list[ScoredBullet],
    candidates: list[ScoredBullet],
    used_lines: int,
    line_budget: int,
    theme_counts: dict[str, int],
) -> tuple[list[ScoredBullet], int, dict[str, str]]:
    """Make sure no experience entry appears with fewer than the minimum bullets."""
    rejections: dict[str, str] = {}
    chosen_ids = {c.bullet_id for c in selected}

    while True:
        counts: dict[str, int] = {}
        for candidate in selected:
            if candidate.entry_type == "experience":
                counts[candidate.entry_id] = counts.get(candidate.entry_id, 0) + 1

        thin = [entry_id for entry_id, n in counts.items() if n < MIN_BULLETS_PER_EXPERIENCE]
        if not thin:
            return selected, used_lines, rejections

        entry_id = thin[0]

        promoted = None
        for candidate in candidates:
            if candidate.entry_id != entry_id or candidate.bullet_id in chosen_ids:
                continue
            if used_lines + candidate.lines > line_budget:
                continue
            if any(theme_counts.get(t, 0) >= MAX_BULLETS_PER_THEME for t in candidate.themes):
                continue
            promoted = candidate
            break

        if promoted is not None:
            selected.append(promoted)
            chosen_ids.add(promoted.bullet_id)
            used_lines += promoted.lines
            for theme in promoted.themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1
            continue

        # Nothing can be promoted, so the entry has to go.
        for candidate in [c for c in selected if c.entry_id == entry_id]:
            selected.remove(candidate)
            chosen_ids.discard(candidate.bullet_id)
            used_lines -= candidate.lines
            for theme in candidate.themes:
                theme_counts[theme] = max(0, theme_counts.get(theme, 0) - 1)
            rejections[candidate.bullet_id] = (
                f"entry {entry_id} could not reach {MIN_BULLETS_PER_EXPERIENCE} bullets"
            )


def _order_for_render(
    selected: list[ScoredBullet],
    profile: Profile,
) -> list[ScoredBullet]:
    """Spec 5: "chronological order preserved within sections".

    Selection ranks by score, but a resume is read chronologically -- a 2021
    bullet above a 2025 one under the same job reads as a typo. Entries come out
    reverse-chronologically, and bullets keep their authored order within an
    entry, which is the order the profile's owner chose.
    """
    entry_order = {entry.id: index for index, entry in enumerate(profile.experience)}
    offset = len(entry_order)
    for index, entry in enumerate(profile.projects):
        entry_order[entry.id] = offset + index

    bullet_order: dict[str, int] = {}
    for entry in profile.entries():
        for index, bullet in enumerate(entry.bullets):
            bullet_order[bullet.id] = index

    return sorted(
        selected,
        key=lambda c: (entry_order.get(c.entry_id, 9999), bullet_order.get(c.bullet_id, 9999)),
    )
