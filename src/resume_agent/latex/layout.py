"""How many lines of content fit on one page. Spec 5, `select_content`.

Spec 5 constrains selection with `sum(estimated_lines) <= line_budget` and never
says what the budget is. It cannot be one number: fixed furniture consumes the
page before any bullet does, so a resume with four jobs and one project has a
smaller bullet budget than one with two jobs and three projects.

    line_budget(shape) = TOTAL_PAGE_LINES - overhead(shape)

Every constant below was measured by compiling, not estimated -- same discipline
as `CHARS_PER_LINE` in metrics.py.

---------------------------------------------------------------------------
Derived by scripts/calibrate_line_budget.py on 2026-09-05, against
jake_resume.tex.j2 with tectonic 0.17.0, by appending one-line bullets to a
profile until `page_count` became 2, across six profile shapes.

Measured capacities (bullet lines that fit on one page):

    shape                 exp  prj  edu  skill_lines   capacity
    baseline                2    3    1            6         32
    one_less_project        2    2    1            6         34
    no_projects             2    0    1            6         38
    no_education            2    3    0            6         37
    one_less_experience     1    3    1            6         35
    one_less_skill_row      2    3    1            4         34

Solving those six equations gives the constants below, and the resulting
additive model reproduces **all six measurements exactly, with zero error**.
`test_layout.py` pins that; if it ever fails, re-run the calibration script
rather than adjusting a constant by hand.

A section heading genuinely costs 0 lines. That looks like a mistake and is not:
the template's `\titleformat` wraps each heading in `\vspace{-4pt}` before and
`\vspace{-5pt}` after the rule, so a heading occupies almost exactly the
whitespace it displaces. It is left at 0 because that is what was measured.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from resume_agent.latex.metrics import estimate_lines

# Total content capacity of one page, before any furniture is subtracted.
TOTAL_PAGE_LINES = 55

# A two-row tabular: title/dates, then org/location.
EXPERIENCE_SUBHEADING_LINES = 3

# A one-row tabular: name + tech stack, then dates.
PROJECT_HEADING_LINES = 2

# Institution/location + degree/dates, plus the section's own whitespace.
EDUCATION_ENTRY_LINES = 5

# See the note above -- measured, not forgotten.
SECTION_HEADING_LINES = 0

# The fraction the layout loop cuts the budget by when a PDF comes back at two
# pages (spec 5, `inspect_output`: "pages > 1 -> reduce line_budget by 8%").
LAYOUT_RETRY_SHRINK = 0.08


def skill_row_lines(rows: list[dict]) -> int:
    """How many rendered lines the Technical Skills block occupies.

    Computed from the actual text rather than assumed to be one line per
    category, because a long category wraps -- in profile.example the Practices
    row runs to two lines and the other four fit on one, which is exactly the
    six lines the calibration measured.

    Takes the rows produced by `latex.context.group_skills_by_category`.
    """
    total = 0
    for row in rows:
        # The bold label is part of the line's width, so it is part of the estimate.
        rendered = f"{row['label']}: {', '.join(row['names'])}"
        total += estimate_lines(rendered)
    return total


def line_budget(
    *,
    experience_entries: int,
    project_entries: int,
    education_entries: int,
    skill_lines: int,
    sections: int = 0,
) -> int:
    """Lines available for bullets, given a profile's shape.

    Never returns below zero: a profile whose furniture alone overflows the page
    has no bullet budget, and the caller should hear that as "0" rather than as
    a negative number that silently passes a `<=` check.
    """
    overhead = (
        EXPERIENCE_SUBHEADING_LINES * experience_entries
        + PROJECT_HEADING_LINES * project_entries
        + EDUCATION_ENTRY_LINES * education_entries
        + SECTION_HEADING_LINES * sections
        + skill_lines
    )
    return max(0, TOTAL_PAGE_LINES - overhead)


def shrink_budget(budget: int, shrink: float = LAYOUT_RETRY_SHRINK) -> int:
    """The reduced budget for a layout retry (spec 5).

    Always cuts by at least one line: an 8% reduction of a small budget rounds
    to zero change, which would make M5's layout loop spin without converging.
    """
    reduced = int(budget * (1.0 - shrink))
    return max(0, min(reduced, budget - 1)) if budget > 0 else 0
