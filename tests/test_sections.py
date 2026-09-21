"""Career stage, section order, and the furniture the student sections cost.

The stage is the one decision in this feature that is *not* a judgment call, so
it is tested as arithmetic: given these dates and these titles, this answer. If
any of this ever needs a model to decide it, something has gone wrong.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from resume_agent.analyze import budget_for_profile
from resume_agent.kb.loader import load_profile
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.latex.layout import line_budget, total_page_lines
from resume_agent.models.profile import Bullet, ExperienceEntry
from resume_agent.sections import (
    EXPERIENCED_ORDER,
    STUDENT_ORDER,
    career_stage,
    professional_months,
    section_order,
    still_studying,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STUDENT_PROFILE = REPO_ROOT / "tests" / "fixtures" / "profile.student"
EXAMPLE_PROFILE = REPO_ROOT / "profile.example"

TODAY = "2026-09"


@pytest.fixture
def student():
    return load_profile(STUDENT_PROFILE)


@pytest.fixture
def veteran():
    return load_profile(EXAMPLE_PROFILE)


def _job(title: str, start: str, end: str | None, entry_id: str = "exp_probe"):
    return ExperienceEntry(
        id=entry_id, type="experience", org="Probe Co", title=title,
        location="Somewhere", start=start, end=end,
        bullets=[Bullet(id=f"{entry_id}.b1", canonical="Did a thing.")],
    )


# ===========================================================================
# The stage
# ===========================================================================


def test_a_fresh_graduate_with_only_internships_is_a_student(student) -> None:
    """The fixture's two roles are both internships, so the arithmetic that
    matters is that they contribute zero professional months -- not that they
    are absent from the page, because they are on it."""
    assert professional_months(student, TODAY) == 0
    assert career_stage(student, TODAY) == "student"


def test_internships_still_render_as_experience(student) -> None:
    """The distinction is about what the page is *shaped* like, not about
    hiding anything: an internship is experience and prints as experience."""
    context = build_resume_context(student, today=TODAY)
    assert [row["title"] for row in context["experience"]] == [
        "Accessibility QA Intern",
        "Digitisation Intern",
    ]


def test_someone_still_studying_is_a_student_whatever_their_job(veteran) -> None:
    profile = copy.deepcopy(veteran)
    profile.education[0].end = "2030-06"

    assert still_studying(profile, TODAY)
    assert career_stage(profile, TODAY) == "student"


def test_two_years_of_real_work_is_the_boundary(student) -> None:
    profile = copy.deepcopy(student)
    profile.education[0].end = "2024-06"          # finished studying

    profile.experience = [*student.experience, _job("Software Engineer", "2024-10", "2026-08")]
    assert professional_months(profile, TODAY) == 22
    assert career_stage(profile, TODAY) == "early"

    profile.experience[-1] = _job("Software Engineer", "2024-01", "2026-08")
    assert professional_months(profile, TODAY) == 31
    assert career_stage(profile, TODAY) == "experienced"


def test_overlapping_roles_are_counted_once(student) -> None:
    """Summing spans would let a part-time job held alongside a full-time one
    read as twice the experience."""
    profile = copy.deepcopy(student)
    profile.education[0].end = "2024-06"
    profile.experience = [
        _job("Software Engineer", "2024-01", "2026-01", "exp_a"),
        _job("Contract Engineer", "2024-06", "2025-06", "exp_b"),
    ]

    assert professional_months(profile, TODAY) == 24   # not 24 + 12


def test_an_open_ended_role_counts_up_to_today(student) -> None:
    profile = copy.deepcopy(student)
    profile.education[0].end = "2024-06"
    profile.experience = [_job("Software Engineer", "2026-03", None)]

    assert professional_months(profile, TODAY) == 6


def test_the_override_wins_over_the_arithmetic(student, veteran) -> None:
    """For the minority counting gets wrong: a career changer's months are real
    but in another field, and a returning PhD's are real but old."""
    assert career_stage(student, TODAY, override="experienced") == "experienced"
    assert career_stage(veteran, TODAY, override="student") == "student"
    assert career_stage(veteran, TODAY, override="auto") == "experienced"


# ===========================================================================
# Order -- and the fact that order is all this decides
# ===========================================================================


def test_a_student_leads_with_education_and_a_veteran_does_not() -> None:
    assert STUDENT_ORDER.index("education") < STUDENT_ORDER.index("experience")
    assert EXPERIENCED_ORDER.index("experience") < EXPERIENCED_ORDER.index("education")
    assert section_order("early") == STUDENT_ORDER


def test_the_rendered_section_order_follows_the_stage(student) -> None:
    tex = render_template(
        "jake_resume.tex.j2", build_resume_context(student, today=TODAY)
    )
    assert re.findall(r"\\section\{([^}]*)\}", tex) == [
        "Education, Awards \\& Certifications",
        "Experience",
        "Projects",
        "Leadership Experience",
        "Publications",
        "Technical Skills",
    ]


def test_an_experienced_profile_drops_the_student_credentials(veteran) -> None:
    """GPA, coursework and awards are what education earns while it is the best
    evidence you have. Once there is a work history they are lines taken from
    something stronger."""
    context = build_resume_context(veteran, today=TODAY)

    assert context["stage"] == "experienced"
    assert context["credentials"] == []
    assert context["education"][0]["gpa"] is None
    assert context["education"][0]["coursework"] == []


def test_a_section_with_no_surviving_bullets_does_not_render(student) -> None:
    """Order is this feature's decision; presence is the knapsack's. A section
    whose every bullet lost the budget must not print as a bare heading."""
    keep = [b.id for b in student.all_bullets() if not b.id.startswith("ldr_")]
    tex = render_template(
        "jake_resume.tex.j2",
        build_resume_context(student, today=TODAY, selected_ids=keep),
    )

    assert "Leadership Experience" not in tex
    assert "Publications" in tex


def test_the_summary_prints_only_when_there_is_one(student) -> None:
    context = build_resume_context(student, today=TODAY, summary="A summary.")
    assert "summary" in context["sections"]
    assert "A summary." in render_template("jake_resume.tex.j2", context)

    without = render_template(
        "jake_resume.tex.j2", build_resume_context(student, today=TODAY)
    )
    assert "\\section{Summary}" not in without


# ===========================================================================
# The budget
# ===========================================================================


BASE_SHAPE = dict(
    experience_entries=2, project_entries=3, education_entries=1, skill_lines=6
)

# Measured on 2026-09-16 by compiling; see the table in latex/layout.py. Pinned
# here for the same reason the original six are: a constant edited by hand
# instead of re-measured is the failure this catches.
@pytest.mark.parametrize(
    "extra,measured",
    [
        ({}, 32),
        ({"leadership_entries": 1}, 28),
        ({"leadership_entries": 2}, 26),
        ({"pages": 2}, 86),
    ],
)
def test_the_budget_model_reproduces_its_measurements(extra: dict, measured: int) -> None:
    assert line_budget(**BASE_SHAPE, **extra) == measured


@pytest.mark.parametrize(
    "extra,measured",
    [({"publication_entries": 1}, 29), ({"publication_entries": 2}, 28)],
)
def test_publications_are_charged_at_the_wrapping_price(extra: dict, measured: int) -> None:
    """Deliberately an over-estimate: a short title costs a line less than the
    model charges, because paper titles usually wrap and under-charging spills
    onto a second page."""
    assert line_budget(**BASE_SHAPE, **extra) <= measured


def test_a_second_page_is_not_twice_a_first() -> None:
    """The name and contact block prints once, so page two holds more."""
    assert total_page_lines(2) > total_page_lines(1) * 2 - total_page_lines(1)
    assert total_page_lines(2) != total_page_lines(1) * 2

    with pytest.raises(ValueError, match="at least one page"):
        total_page_lines(0)


def test_the_student_shape_matches_its_compiled_capacity(student) -> None:
    """Pinned against a real compile, not against the arithmetic.

    This is the measurement the unit tests above could not have caught: the
    model predicted 19 bullet lines for this shape and the PDF came back at two
    pages, because nothing was charging for the two `\\resumeItemListStart`
    wrappers the student layout adds. Re-measure with a compile before changing
    it -- a constant adjusted to make this pass is the bug it exists to catch.
    """
    assert budget_for_profile(student, today=TODAY) == 17


def test_the_student_layout_costs_more_furniture_than_the_experienced_one(student) -> None:
    """GPA, honours, coursework and the awards under them are printed lines, so
    the stage is part of the page's shape rather than a presentation detail."""
    as_student = budget_for_profile(student, today=TODAY, stage="student")
    as_veteran = budget_for_profile(student, today=TODAY, stage="experienced")

    assert as_student < as_veteran


def test_reserving_the_summary_takes_lines_from_bullets(student) -> None:
    """Reserved before selection, not measured after: the summary prints above
    the bullets, so discovering its cost afterwards would mean a bullet the
    knapsack already committed to no longer fits."""
    plain = budget_for_profile(student, today=TODAY)
    reserved = budget_for_profile(student, today=TODAY, reserve_summary=True)

    assert reserved == plain - 4


# ===========================================================================
# Skill categories
# ===========================================================================


def test_an_uncurated_category_prints_instead_of_refusing() -> None:
    """This used to raise, telling you to add a label to a dict in
    `latex/context.py`. That is a reasonable instruction for whoever wrote the
    loader and a useless one for somebody adding a skill through the web form,
    whose profile would stop rendering until they edited Python."""
    from resume_agent.latex.context import group_skills_by_category
    from resume_agent.models.profile import Skill

    rows = group_skills_by_category([
        Skill(canonical="Python", category="language", level="working"),
        Skill(canonical="Fusion 360", category="mechanical_cad", level="familiar"),
    ])

    assert [row["label"] for row in rows] == ["Languages", "Mechanical Cad"]


def test_curated_categories_print_before_uncurated_ones() -> None:
    """The curated order is a judgment about what a reader should see first, so
    an unknown bucket goes after it rather than wherever it happened to land."""
    from resume_agent.latex.context import group_skills_by_category
    from resume_agent.models.profile import Skill

    rows = group_skills_by_category([
        Skill(canonical="Fusion 360", category="mechanical_cad", level="familiar"),
        Skill(canonical="Python", category="language", level="working"),
    ])

    assert [row["label"] for row in rows] == ["Languages", "Mechanical Cad"]


# ===========================================================================
# Typography that belongs to one output and not the other
# ===========================================================================


def test_the_latex_dash_stays_the_default() -> None:
    """`--` is LaTeX's en-dash and the golden snapshot is byte-compared, so the
    default has to keep producing it."""
    from resume_agent.latex.context import format_date_range

    assert format_date_range("2023-06", None) == "June 2023 -- Present"
    assert format_date_range("2023-06", "2024-01") == "June 2023 -- Jan. 2024"


def test_the_browser_gets_a_real_en_dash() -> None:
    """The API reused the LaTeX formatter for the Browse view, so dates rendered
    as "Jan. 2025 -- Apr. 2025" -- two literal hyphens, because `--` is
    typography only once TeX has read it."""
    from resume_agent.api.models import build_profile_detail

    detail = build_profile_detail("x", load_profile(STUDENT_PROFILE))
    subheadings = " ".join(entry.subheading for entry in detail.entries)

    assert "\u2013" in subheadings
    assert "--" not in subheadings
