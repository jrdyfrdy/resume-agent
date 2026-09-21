"""Which sections a resume has, and what order they go in.

A fresh graduate's resume leads with education, because a degree with honours is
the strongest thing on it, and gives space to society roles and publications
because there is not yet a work history to spend the page on. Six years later the
same person's resume leads with work, and that society role is not worth the four
lines it costs. Same tool, same profile schema, different arithmetic.

**This is arithmetic, not judgment** -- CLAUDE.md rule 2. "Do you have
professional experience yet" is months of non-internship work and whether your
studies have finished. Asking a model to make that call would be slower, cost
money, and give a different answer on Tuesday.

**What this module decides and what it does not.** It decides section *order*.
It does not decide what appears: a section with no selected bullets does not
render, and which bullets survive is the knapsack's business in
`graph/nodes/select.py`. So an experienced profile's leadership section vanishes
because its bullets lost to better evidence, not because a flag switched it off.
Order is a layout decision; presence is a budget one, and keeping them apart is
what stops this module from growing into a second selection algorithm.
"""

from __future__ import annotations

import re
from typing import Literal

from resume_agent.models.profile import Profile

Stage = Literal["student", "early", "experienced"]
LayoutChoice = Literal["auto", "student", "experienced"]

# Titles that are experience on the page but not evidence you have outgrown the
# student layout -- which is the only question being asked here. An internship
# still prints under Experience; it just does not move education off the top.
INTERN_TITLE = re.compile(
    r"\b(intern|internship|trainee|apprentice|working[ -]student|co-?op|"
    r"practicum|ojt|on[- ]the[- ]job)\b",
    re.IGNORECASE,
)

# Below this, education and leadership still outrank work history. Two years is
# the point at which someone has a track record to talk about instead of a
# degree -- and it is a constant here rather than a magic number inline so the
# one place to argue with it is obvious.
EARLY_CAREER_MONTHS = 24

# Section keys, in the order each stage wants them. The renderer walks these; it
# never hardcodes an order of its own.
STUDENT_ORDER: tuple[str, ...] = (
    "summary",
    "education",
    "experience",
    "projects",
    "leadership",
    "publications",
    "skills",
)

EXPERIENCED_ORDER: tuple[str, ...] = (
    "summary",
    "experience",
    "projects",
    "skills",
    "publications",
    "education",
    "leadership",
)


def this_month() -> str:
    """Today as `YYYY-MM`. The default "now" for stage decisions."""
    from datetime import UTC, datetime  # noqa: PLC0415 - only needed here

    return datetime.now(UTC).strftime("%Y-%m")


def month_index(year_month: str) -> int:
    """`YYYY-MM` as a count of months, for subtraction.

    Avoids date arithmetic entirely: these values are already validated as
    `YYYY-MM` by `YearMonth`, and turning them into datetimes to get a month
    difference back out is a round trip that can only introduce bugs.
    """
    year, month = year_month.split("-")
    return int(year) * 12 + int(month)


def professional_months(profile: Profile, today: str) -> int:
    """Months of non-internship work, counting overlaps once.

    Merged rather than summed: two concurrent roles are one stretch of career,
    and summing them would let someone hold a part-time job alongside a
    full-time one and appear to have twice the experience.
    """
    spans: list[tuple[int, int]] = []
    for job in profile.experience:
        if INTERN_TITLE.search(job.title):
            continue
        start = month_index(job.start)
        end = month_index(job.end or today)
        if end > start:
            spans.append((start, end))

    total = 0
    furthest = None
    for start, end in sorted(spans):
        # Each span contributes only the part not already covered by an earlier
        # one, which is what makes overlapping roles count once.
        begin = start if furthest is None else max(start, furthest)
        if end > begin:
            total += end - begin
        furthest = end if furthest is None else max(furthest, end)
    return total


def still_studying(profile: Profile, today: str) -> bool:
    """True while any degree is unfinished.

    An open-ended education entry (`end: null`) counts as ongoing, matching how
    `end` already reads as "Present" everywhere else.
    """
    return any(
        entry.end is None or month_index(entry.end) > month_index(today)
        for entry in profile.education
    )


def career_stage(
    profile: Profile, today: str, *, override: LayoutChoice = "auto"
) -> Stage:
    """`student`, `early` or `experienced`.

    The override exists because counting is wrong for a real minority of people
    -- someone changing career after a decade elsewhere, someone back from a
    PhD -- and for them the computed answer is confidently wrong rather than
    uncertain. `early` is not offered as an override: it differs from `student`
    only in emphasis, and anyone who cares enough to override wants one end or
    the other.
    """
    if override != "auto":
        return override

    if still_studying(profile, today) or not profile.experience:
        return "student"

    months = professional_months(profile, today)
    if months == 0:
        return "student"
    return "early" if months < EARLY_CAREER_MONTHS else "experienced"


def section_order(stage: Stage) -> tuple[str, ...]:
    """`early` shares the student order: still leading with education."""
    return EXPERIENCED_ORDER if stage == "experienced" else STUDENT_ORDER
