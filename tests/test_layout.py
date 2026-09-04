"""The calibrated line budget.

The load-bearing test is `test_model_reproduces_every_calibration_measurement`.
`layout.py`'s constants were solved from six compiled profile shapes; if the
additive model stops reproducing those measurements, a constant has been edited
by hand instead of re-derived, and the budget silently stops describing the page.
"""

from __future__ import annotations

import pytest

from resume_agent.latex.context import group_skills_by_category
from resume_agent.latex.layout import (
    EDUCATION_ENTRY_LINES,
    EXPERIENCE_SUBHEADING_LINES,
    PROJECT_HEADING_LINES,
    TOTAL_PAGE_LINES,
    line_budget,
    shrink_budget,
    skill_row_lines,
)
from resume_agent.models.profile import Profile

# (label, experience, projects, education, skill_lines, measured capacity)
# Straight from scripts/calibrate_line_budget.py, 2026-09-05, tectonic 0.17.0.
CALIBRATION_MEASUREMENTS = [
    ("baseline", 2, 3, 1, 6, 32),
    ("one_less_project", 2, 2, 1, 6, 34),
    ("no_projects", 2, 0, 1, 6, 38),
    ("no_education", 2, 3, 0, 6, 37),
    ("one_less_experience", 1, 3, 1, 6, 35),
    ("one_less_skill_row", 2, 3, 1, 4, 34),
]


@pytest.mark.parametrize(
    ("label", "experience", "projects", "education", "skill_lines", "measured"),
    CALIBRATION_MEASUREMENTS,
)
def test_model_reproduces_every_calibration_measurement(
    label: str, experience: int, projects: int, education: int, skill_lines: int, measured: int
) -> None:
    """Zero error on all six shapes. Re-run the script if this ever fails."""
    predicted = line_budget(
        experience_entries=experience,
        project_entries=projects,
        education_entries=education,
        skill_lines=skill_lines,
    )
    assert predicted == measured, f"{label}: predicted {predicted}, measured {measured}"


def test_constants_are_the_measured_values() -> None:
    """Pinned so a casual edit has to be deliberate."""
    assert TOTAL_PAGE_LINES == 55
    assert EXPERIENCE_SUBHEADING_LINES == 3
    assert PROJECT_HEADING_LINES == 2
    assert EDUCATION_ENTRY_LINES == 5


def test_budget_shrinks_as_furniture_grows() -> None:
    """More jobs means less room for bullets. The whole reason this is a function."""
    two_jobs = line_budget(
        experience_entries=2, project_entries=0, education_entries=1, skill_lines=6
    )
    five_jobs = line_budget(
        experience_entries=5, project_entries=0, education_entries=1, skill_lines=6
    )
    assert five_jobs < two_jobs


def test_budget_never_goes_negative() -> None:
    """A profile whose furniture alone overflows should report 0, not a negative."""
    assert (
        line_budget(
            experience_entries=100, project_entries=100, education_entries=100, skill_lines=100
        )
        == 0
    )


def test_example_profile_skill_rows_measure_six_lines(example_profile: Profile) -> None:
    """The calibration assumed 6; if the example's skills change, this catches it."""
    rows = group_skills_by_category(example_profile.skills)
    assert skill_row_lines(rows) == 6


def test_skill_rows_account_for_wrapping() -> None:
    """A long category wraps; assuming one line per row would over-budget the page."""
    short = [{"label": "Languages", "names": ["Python", "Go"]}]
    long = [{"label": "Practices", "names": [f"Practice number {i}" for i in range(20)]}]
    assert skill_row_lines(short) == 1
    assert skill_row_lines(long) > 1


# --- the layout retry shrink (used by M5) -----------------------------------


def test_shrink_reduces_the_budget() -> None:
    assert shrink_budget(100) == 92  # spec 5: "reduce line_budget by 8%"


def test_shrink_always_removes_at_least_one_line() -> None:
    """8% of a small budget rounds to nothing, which would make M5's loop spin."""
    for budget in range(1, 13):
        assert shrink_budget(budget) < budget


def test_shrink_of_zero_is_zero() -> None:
    assert shrink_budget(0) == 0
