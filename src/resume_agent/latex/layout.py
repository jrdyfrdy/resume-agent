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

# ---------------------------------------------------------------------------
# Student sections, measured the same way on 2026-09-16 against profile.example
# with leadership and publication entries added, one filler bullet each:
#
#     probe               capacity   implied cost
#     base (no new)             32   --
#     + 1 leadership            28   4
#     + 2 leadership            26   6      -> 2 per entry, 2 for the section
#     + 1 publication (short)   29   3
#     + 2 publications (short)  28   4
#     + 1 publication (wraps)   28   4      -> 2 per entry, 2 for the section
#     base over two pages       86   --     -> a second page adds 54
#
# A leadership entry uses the same `\resumeSubheading` macro as a job but
# measures 2 rather than the 3 charged for experience. That is not a
# contradiction: the experience constant was solved jointly with five other
# terms against six equations, and this one is a direct marginal measurement.
# Both reproduce their own measurements exactly, which is the property that
# matters.
LEADERSHIP_SUBHEADING_LINES = 2

# Publication headings wrap (`\resumeWrappingHeading`), so a long title costs a
# line more than a short one. Charged at the wrapping price because paper titles
# are long far more often than not, and over-charging underfills the page --
# much the better failure, per the calibration script's own note.
PUBLICATION_HEADING_LINES = 2

# What a `\resumeItemListStart`/`End` block costs beyond the items inside it.
#
# Charged once for the education extras (GPA, honours, coursework) and once for
# the awards/certifications beneath them, because in the experienced layout
# neither list exists at all and the education entry is just its heading. Found
# by compiling: the student fixture's true one-page capacity measured 17 bullet
# lines against a model that predicted 19, and the two missing lines are exactly
# these two wrappers.
ITEM_LIST_LINES = 1

# What a leadership or publications section costs beyond its entries. Unlike
# `SECTION_HEADING_LINES`, which genuinely measured 0 for the four original
# sections, these sit at the end of the document where the list wrapper's
# trailing space is not absorbed by a following section.
EXTRA_SECTION_LINES = 2

# Extra capacity of each page after the first. Larger than a first page because
# the name/contact block prints once at the top of the document -- which is
# exactly why this is not `TOTAL_PAGE_LINES * pages`.
ADDITIONAL_PAGE_LINES = 54

# Lines held back for a professional summary when one will be written.
#
# Reserved rather than measured, and that asymmetry is the point: the summary is
# generated *after* selection but prints *above* it, so if its cost were counted
# afterwards the discovery would be that a bullet the knapsack already committed
# to no longer fits. Three lines of prose plus the section's own furniture; the
# prompt is told the limit so the model writes to it.
SUMMARY_LINES = 4

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


def total_page_lines(pages: int = 1) -> int:
    """Content capacity for a resume of `pages` pages, before furniture.

    Deliberately not a multiplication. The heading block -- name, phone, email,
    links -- prints once at the top of the document, so page two carries more
    content lines than page one does. Both numbers are measured.
    """
    if pages < 1:
        raise ValueError(f"a resume has at least one page, got {pages}")
    return TOTAL_PAGE_LINES + ADDITIONAL_PAGE_LINES * (pages - 1)


def item_lines(texts: list[str]) -> int:
    """Rendered lines for a list of `\\resumeItem` strings.

    Education extras (GPA, honours, coursework) and the awards/certifications
    under them are `\\resumeItem`s -- structurally the same thing as a bullet --
    so they are measured from their text rather than charged a fixed constant.
    A three-word honour and a wrapping coursework list do not cost the same, and
    a constant would be wrong for both.
    """
    return sum(estimate_lines(text) for text in texts)


def line_budget(
    *,
    experience_entries: int,
    project_entries: int,
    education_entries: int,
    skill_lines: int,
    sections: int = 0,
    leadership_entries: int = 0,
    publication_entries: int = 0,
    education_extra_lines: int = 0,
    credential_lines: int = 0,
    summary_lines: int = 0,
    pages: int = 1,
) -> int:
    """Lines available for bullets, given a profile's shape.

    Never returns below zero: a profile whose furniture alone overflows the page
    has no bullet budget, and the caller should hear that as "0" rather than as
    a negative number that silently passes a `<=` check.

    `summary_lines` is *reserved*, not measured: the summary is written after
    selection but prints above it, so its cost has to come out of the budget
    before the knapsack commits. Measuring it afterwards would mean discovering
    that a chosen bullet no longer fits.
    """
    overhead = (
        EXPERIENCE_SUBHEADING_LINES * experience_entries
        + PROJECT_HEADING_LINES * project_entries
        + EDUCATION_ENTRY_LINES * education_entries
        + LEADERSHIP_SUBHEADING_LINES * leadership_entries
        + PUBLICATION_HEADING_LINES * publication_entries
        + SECTION_HEADING_LINES * sections
        # Charged once per section that is actually present, not per entry.
        + EXTRA_SECTION_LINES * bool(leadership_entries)
        + EXTRA_SECTION_LINES * bool(publication_entries)
        # Each list pays for its own wrapper, once, only when it has contents.
        + education_extra_lines
        + ITEM_LIST_LINES * bool(education_extra_lines)
        + credential_lines
        + ITEM_LIST_LINES * bool(credential_lines)
        + summary_lines
        + skill_lines
    )
    return max(0, total_page_lines(pages) - overhead)


def shrink_budget(budget: int, shrink: float = LAYOUT_RETRY_SHRINK) -> int:
    """The reduced budget for a layout retry (spec 5).

    Always cuts by at least one line: an 8% reduction of a small budget rounds
    to zero change, which would make M5's layout loop spin without converging.
    """
    reduced = int(budget * (1.0 - shrink))
    return max(0, min(reduced, budget - 1)) if budget > 0 else 0
