"""Flatten a validated `Profile` into the dict the resume template renders.

Deliberately a separate module from `env.py` and from the template itself.

Templates are a bad place for logic: a date-formatting bug inside a `.tex.j2`
surfaces as a LaTeX error twenty steps downstream, and you cannot unit-test it
without compiling. Everything here is pure Python with no I/O and no Jinja, so
the ordering, grouping and date rules are testable on their own.

Nothing in this module escapes anything. Escaping happens at interpolation time
via the `tex` filter, exactly once, in the template -- see plan D4 and the
enforcement test in `tests/test_env.py`.
"""

from __future__ import annotations

from typing import Any

from resume_agent.models.profile import (
    EducationEntry,
    ExperienceEntry,
    Profile,
    ProjectEntry,
    Skill,
)

# Month abbreviations in the style Jake's Resume uses ("Aug. 2018 -- May 2021").
# A hand-rolled table rather than `strftime("%b")` because that is locale
# dependent, and a resume rendered on a machine with a German locale should not
# say "Mai".
_MONTHS = {
    "01": "Jan.",
    "02": "Feb.",
    "03": "Mar.",
    "04": "Apr.",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "Aug.",
    "09": "Sept.",
    "10": "Oct.",
    "11": "Nov.",
    "12": "Dec.",
}

# Which skill categories print, in which order, under which heading. Categories
# absent from the profile are simply skipped; a category present in the profile
# but missing from this table would be dropped silently, so `build_resume_context`
# asserts that cannot happen.
SKILL_CATEGORY_LABELS: dict[str, str] = {
    "language": "Languages",
    "framework": "Frameworks",
    "data": "Data",
    "infrastructure": "Infrastructure",
    "practice": "Practices",
}


def format_year_month(value: str) -> str:
    """`2023-06` -> `June 2023`."""
    year, month = value.split("-")
    return f"{_MONTHS[month]} {year}"


def format_date_range(start: str, end: str | None) -> str:
    """`("2023-06", None)` -> `June 2023 -- Present`.

    The separator is a LaTeX en-dash (`--`), which is why this returns a string
    destined for the template rather than a pair of dates: the dash is
    typography, and putting it here keeps it out of the template's control flow.
    """
    tail = format_year_month(end) if end else "Present"
    return f"{format_year_month(start)} -- {tail}"


def build_display_name_index(skills: list[Skill]) -> dict[str, str]:
    """Map every accepted spelling of a skill to its canonical display form.

    Bullets and `tech:` lists may refer to a skill by any alias and in any case
    (`postgres`, `PostgreSQL`, `rds postgres`); the resume must print one
    consistent name. This is the alias table doing its third job, after
    retrieval expansion and the fabrication allow-list.
    """
    index: dict[str, str] = {}
    for skill in skills:
        index[skill.canonical.lower()] = skill.canonical
        for alias in skill.aliases:
            index[alias.lower()] = skill.canonical
    return index


def resolve_tech_names(names: list[str], display_index: dict[str, str]) -> list[str]:
    """Canonicalise a `tech:` list for display, preserving the author's order.

    Order is preserved rather than sorted because the order a `tech:` list is
    written in is itself a signal -- the first two entries are the ones the
    author considers load-bearing for that entry.
    """
    return [display_index.get(name.lower(), name) for name in names]


def group_skills_by_category(skills: list[Skill]) -> list[dict[str, Any]]:
    """Bucket skills into the printable category rows of the Technical Skills section."""
    unknown = {s.category for s in skills} - set(SKILL_CATEGORY_LABELS)
    if unknown:
        raise ValueError(
            f"skills.yaml uses categories with no display label: {sorted(unknown)}. "
            f"Add them to SKILL_CATEGORY_LABELS in latex/context.py."
        )
    rows: list[dict[str, Any]] = []
    for category, label in SKILL_CATEGORY_LABELS.items():
        names = [s.canonical for s in skills if s.category == category]
        if names:
            rows.append({"label": label, "names": names})
    return rows


# --- per-section shaping ---------------------------------------------------


def _education_row(entry: EducationEntry) -> dict[str, Any]:
    return {
        "institution": entry.institution,
        "location": entry.location,
        "degree": entry.degree,
        "dates": format_date_range(entry.start, entry.end),
    }


def _bullet_texts(
    entry: ExperienceEntry | ProjectEntry,
    overrides: dict[str, str] | None,
    allowed_ids: set[str] | None,
) -> list[str]:
    """The lines to print under one entry.

    Three modes, in one place so experience and projects cannot drift apart:

    * no arguments -- every bullet, as written. This is M0's behaviour and what
      `resume-agent build` still does.
    * `allowed_ids` -- only the bullets selection chose (M3).
    * `overrides` -- the verified rewrite in place of the canonical text (M4).

    Bullets keep their authored order regardless, because a resume is read
    chronologically even though selection ranks by score.
    """
    texts = []
    for bullet in entry.bullets:
        if allowed_ids is not None and bullet.id not in allowed_ids:
            continue
        texts.append((overrides or {}).get(bullet.id, bullet.canonical))
    return texts


def _experience_row(
    entry: ExperienceEntry,
    overrides: dict[str, str] | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    return {
        "title": entry.title,
        "dates": format_date_range(entry.start, entry.end),
        "org": entry.org,
        "location": entry.location,
        "bullets": _bullet_texts(entry, overrides, allowed_ids),
    }


def _project_row(
    entry: ProjectEntry,
    display_index: dict[str, str],
    overrides: dict[str, str] | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": entry.name,
        "tech": ", ".join(resolve_tech_names(entry.tech, display_index)),
        "dates": format_date_range(entry.start, entry.end),
        "bullets": _bullet_texts(entry, overrides, allowed_ids),
    }


def _sort_key_most_recent_first(entry: ExperienceEntry | ProjectEntry) -> tuple[str, str]:
    """Reverse-chronological by end date, then start date.

    A current role (`end is None`) sorts first: "9999-99" beats every real
    YYYY-MM string lexicographically, and YYYY-MM strings sort correctly as
    plain text, which is one of the reasons the schema stores them that way.
    """
    return (entry.end or "9999-99", entry.start)


def build_resume_context(
    profile: Profile,
    *,
    selected_ids: list[str] | None = None,
    tailored_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Everything `jake_resume.tex.j2` needs, and nothing else.

    `selected_ids` restricts output to what M3's knapsack chose; `tailored_text`
    substitutes M4's verified rewrites. Passing neither renders the whole
    profile verbatim, which is what M0's `build` command does.
    """
    display_index = build_display_name_index(profile.skills)
    allowed = set(selected_ids) if selected_ids is not None else None

    experience = sorted(profile.experience, key=_sort_key_most_recent_first, reverse=True)
    projects = sorted(profile.projects, key=_sort_key_most_recent_first, reverse=True)
    education = sorted(
        profile.education,
        key=lambda e: (e.end or "9999-99", e.start),
        reverse=True,
    )

    experience_rows = [_experience_row(e, tailored_text, allowed) for e in experience]
    project_rows = [_project_row(p, display_index, tailored_text, allowed) for p in projects]

    # An entry whose every bullet was dropped must not print as a bare heading.
    # This also keeps the rendered page honest with the line budget selection
    # spent, which charges for furniture only when the furniture appears.
    if allowed is not None:
        experience_rows = [row for row in experience_rows if row["bullets"]]
        project_rows = [row for row in project_rows if row["bullets"]]

    return {
        "identity": {
            "name": profile.identity.name,
            "phone": profile.identity.phone,
            "email": profile.identity.email,
            "location": profile.identity.location,
            "links": [{"label": link.label, "url": link.url} for link in profile.identity.links],
        },
        "education": [_education_row(e) for e in education],
        "experience": experience_rows,
        "projects": project_rows,
        "skill_groups": group_skills_by_category(profile.skills),
    }
