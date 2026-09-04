r"""Measure how many lines of content fit on one page of this template.

Run with:  uv run python scripts/calibrate_line_budget.py

Why this script exists
----------------------
Spec 5 constrains selection with `sum(estimated_lines) <= line_budget` but never
says what the budget is -- and it cannot be a single number, because fixed
furniture eats the page before any bullet does. A section heading, a job
subheading and a project heading all consume height that bullets then cannot
have, so a resume with four jobs and one project has a smaller bullet budget
than one with two jobs and three projects.

What this measures
------------------
Two things, both by rendering and counting pages, never by estimating:

1. `TOTAL_PAGE_LINES` -- total bullet-line capacity of one page for a *known*
   profile shape. Found by appending one-line bullets to profile.example until
   `page_count` becomes 2.

2. The per-element overhead of each piece of furniture. Found by removing one
   element (a project, a whole section, a skill row) and re-measuring how many
   extra bullet lines then fit. The difference is that element's cost in lines.

Solving for both gives `line_budget(shape)` in latex/layout.py, which is what
M3's knapsack spends and what M5's layout loop shrinks by 8% when the PDF comes
back at two pages.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resume_agent.kb.loader import load_profile  # noqa: E402
from resume_agent.latex.compile import compile_tex  # noqa: E402
from resume_agent.latex.context import build_resume_context  # noqa: E402
from resume_agent.latex.env import render_template  # noqa: E402
from resume_agent.latex.inspect import page_count  # noqa: E402
from resume_agent.latex.metrics import CHARS_PER_LINE  # noqa: E402
from resume_agent.models.profile import Bullet, Profile  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "profile.example"
WORK_DIR = REPO_ROOT / "out" / "line_budget_calibration"

# A bullet of exactly one rendered line, so that "how many bullets fit" and
# "how many lines fit" are the same question.
ONE_LINE_TEXT = "x" * (CHARS_PER_LINE - 20)


@dataclass
class ShapeMeasurement:
    label: str
    experience_entries: int
    project_entries: int
    education_entries: int
    skill_rows: int
    baseline_bullet_lines: int  # lines already used by the profile's own bullets
    extra_one_line_bullets: int  # how many more fit before page 2
    total_content_lines: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_content_lines = self.baseline_bullet_lines + self.extra_one_line_bullets


def count_existing_bullet_lines(profile: Profile) -> int:
    """Lines the profile's real bullets already occupy, by the M0 estimator."""
    from resume_agent.latex.metrics import estimate_lines

    return sum(estimate_lines(b.canonical) for b in profile.all_bullets())


def with_extra_bullets(profile: Profile, count: int) -> Profile:
    """A copy of `profile` with `count` one-line filler bullets appended.

    Appended to the first experience entry so they land inside an existing
    itemize rather than creating new furniture, which would confound the
    measurement with the very overhead we are trying to isolate.
    """
    clone = copy.deepcopy(profile)
    entry = clone.experience[0]
    for index in range(count):
        entry.bullets.append(
            Bullet(id=f"{entry.id}.filler{index}", canonical=ONE_LINE_TEXT, confidence="verified")
        )
    return clone


def renders_to_one_page(profile: Profile, job_name: str) -> bool:
    tex = render_template("jake_resume.tex.j2", build_resume_context(profile))
    result = compile_tex(tex, WORK_DIR, job_name=job_name)
    if not result.ok or result.pdf_path is None:
        raise RuntimeError(f"probe did not compile:\n{result.log[-2000:]}")
    return page_count(result.pdf_path) == 1


def max_extra_bullets(profile: Profile, label: str, ceiling: int = 40) -> int:
    """Largest number of one-line bullets that can be added and still fit.

    A linear scan rather than a binary search: page breaks are monotonic in
    content but the compile is ~1s, the range is small, and a linear walk cannot
    be fooled by a non-monotonic step the way a bisection silently can.
    """
    fitting = 0
    for count in range(ceiling + 1):
        if renders_to_one_page(with_extra_bullets(profile, count), f"{label}_{count}"):
            fitting = count
        else:
            break
    return fitting


def measure_shape(profile: Profile, label: str) -> ShapeMeasurement:
    from resume_agent.latex.context import group_skills_by_category

    return ShapeMeasurement(
        label=label,
        experience_entries=len(profile.experience),
        project_entries=len(profile.projects),
        education_entries=len(profile.education),
        skill_rows=len(group_skills_by_category(profile.skills)),
        baseline_bullet_lines=count_existing_bullet_lines(profile),
        extra_one_line_bullets=max_extra_bullets(profile, label),
    )


def drop_last_project(profile: Profile) -> Profile:
    clone = copy.deepcopy(profile)
    clone.projects = clone.projects[:-1]
    return clone


def drop_all_projects(profile: Profile) -> Profile:
    clone = copy.deepcopy(profile)
    clone.projects = []
    return clone


def drop_education(profile: Profile) -> Profile:
    clone = copy.deepcopy(profile)
    clone.education = []
    return clone


def drop_last_experience(profile: Profile) -> Profile:
    clone = copy.deepcopy(profile)
    clone.experience = clone.experience[:-1]
    return clone


def drop_one_skill_category(profile: Profile) -> Profile:
    """Remove every skill in one category, which removes one printed row."""
    clone = copy.deepcopy(profile)
    clone.skills = [s for s in clone.skills if s.category != "practice"]
    # Bullets reference practice skills by name, so drop those references too or
    # the cross-file validator (correctly) rejects the profile.
    vocabulary = {s.canonical.lower() for s in clone.skills}
    vocabulary |= {a.lower() for s in clone.skills for a in s.aliases}
    for entry in clone.entries():
        entry.tech = [t for t in entry.tech if t.lower() in vocabulary]
        for bullet in entry.bullets:
            bullet.skills = [s for s in bullet.skills if s.lower() in vocabulary]
    return clone


def main() -> None:
    profile = load_profile(PROFILE_DIR)

    print("Measuring one-page capacity by compiling. This takes a minute.\n")

    shapes = [
        measure_shape(profile, "baseline"),
        measure_shape(drop_last_project(profile), "one_less_project"),
        measure_shape(drop_all_projects(profile), "no_projects"),
        measure_shape(drop_education(profile), "no_education"),
        measure_shape(drop_last_experience(profile), "one_less_experience"),
        measure_shape(drop_one_skill_category(profile), "one_less_skill_row"),
    ]

    print(f"{'shape':<20} {'exp':>4} {'prj':>4} {'edu':>4} {'skills':>7} "
          f"{'own':>5} {'extra':>6} {'TOTAL':>6}")
    for shape in shapes:
        print(
            f"{shape.label:<20} {shape.experience_entries:>4} {shape.project_entries:>4} "
            f"{shape.education_entries:>4} {shape.skill_rows:>7} "
            f"{shape.baseline_bullet_lines:>5} {shape.extra_one_line_bullets:>6} "
            f"{shape.total_content_lines:>6}"
        )

    baseline, one_less_project, no_projects, no_education, one_less_exp, one_less_skill = shapes

    project_heading_cost = one_less_project.total_content_lines - baseline.total_content_lines
    all_projects_cost = no_projects.total_content_lines - baseline.total_content_lines
    education_cost = no_education.total_content_lines - baseline.total_content_lines
    experience_heading_cost = one_less_exp.total_content_lines - baseline.total_content_lines
    skill_row_cost = one_less_skill.total_content_lines - baseline.total_content_lines

    # Removing every project also removes the "Projects" section heading, so the
    # section's own cost is whatever is left after accounting for the three
    # project headings it contained.
    section_heading_cost = all_projects_cost - (project_heading_cost * baseline.project_entries)

    print()
    print("Measured line costs (each is 'lines freed when this element is removed'):")
    print(f"  one-page capacity, baseline shape : {baseline.total_content_lines} bullet lines")
    print(f"  experience subheading             : {experience_heading_cost}")
    print(f"  project heading                   : {project_heading_cost}")
    print(f"  section heading                   : {section_heading_cost}")
    print(f"  education section (heading+entry) : {education_cost}")
    print(f"  one skill row                     : {skill_row_cost}")
    print()
    print("Put these into latex/layout.py. Round section/skill costs UP when in")
    print("doubt: over-estimating overhead under-fills the page, which is a much")
    print("better failure than spilling to a second page.")


if __name__ == "__main__":
    main()
