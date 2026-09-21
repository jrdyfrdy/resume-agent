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
    Entry,
    ExperienceEntry,
    LeadershipEntry,
    Profile,
    ProjectEntry,
    PublicationEntry,
    Skill,
)
from resume_agent.sections import Stage, career_stage, section_order, this_month

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
# Insertion order is print order. The buckets below "Infrastructure" were added
# for student and engineering resumes, where the skills section carries more of
# the weight and "Frameworks" is too coarse a bucket for a CAD package, an RPA
# tool and a soldering iron.
SKILL_CATEGORY_LABELS: dict[str, str] = {
    "language": "Languages",
    "ai_data": "AI & Data Science",
    "framework": "Frameworks",
    "web": "Web Development",
    "database": "Databases",
    "data": "Data",
    "infrastructure": "Infrastructure",
    "automation": "Automation",
    "hardware": "Hardware",
    "dev_tools": "Developer Tools",
    "design": "Design & Modelling",
    "practice": "Practices",
}


def category_label(category: str) -> str:
    """The heading a skill category prints under.

    Unknown categories are humanised rather than refused. This used to raise,
    telling you to add a label to a dict in this file -- which is a fine
    instruction for the person who wrote the loader and a useless one for
    somebody adding a skill through the web form, whose profile would simply
    stop rendering until they edited Python. A category nobody curated still
    prints; it just prints last, in the order it was first seen.
    """
    known = SKILL_CATEGORY_LABELS.get(category)
    if known:
        return known
    return category.replace("_", " ").replace("-", " ").strip().title()


def format_year_month(value: str) -> str:
    """`2023-06` -> `June 2023`."""
    year, month = value.split("-")
    return f"{_MONTHS[month]} {year}"


def format_date_range(start: str, end: str | None, *, dash: str = "--") -> str:
    """`("2023-06", None)` -> `June 2023 -- Present`.

    The default separator is a LaTeX en-dash (`--`), which is why this returns a
    string destined for the template rather than a pair of dates: the dash is
    typography, and putting it here keeps it out of the template's control flow.

    `dash` exists because the API reuses this for the browser, where `--` is not
    typography -- it is two hyphens, printed literally, which is what the Browse
    view was showing. The LaTeX form stays the default so the golden `.tex`
    snapshot cannot move by accident; HTML callers pass a real en-dash.
    """
    tail = format_year_month(end) if end else "Present"
    return f"{format_year_month(start)} {dash} {tail}"


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
    """Bucket skills into the printable category rows of the Technical Skills section.

    Curated categories print first, in the order `SKILL_CATEGORY_LABELS` lists
    them, because that order is a judgment about what a reader should see first.
    Anything else follows in the order it appears in skills.yaml.
    """
    rows: list[dict[str, Any]] = []
    for category in SKILL_CATEGORY_LABELS:
        names = [s.canonical for s in skills if s.category == category]
        if names:
            rows.append({"label": category_label(category), "names": names})

    seen: list[str] = []
    for skill in skills:
        if skill.category not in SKILL_CATEGORY_LABELS and skill.category not in seen:
            seen.append(skill.category)
    for category in seen:
        rows.append({
            "label": category_label(category),
            "names": [s.canonical for s in skills if s.category == category],
        })
    return rows


# --- per-section shaping ---------------------------------------------------


def _education_row(entry: EducationEntry, *, compact: bool) -> dict[str, Any]:
    """`compact` drops everything but the degree itself.

    Once there is a work history, the reader wants to know where you studied and
    nothing more -- a GPA and a coursework list at that point are lines taken
    from achievements that are better evidence. While you are a student they are
    the opposite: they may be the strongest thing on the page.
    """
    return {
        "institution": entry.institution,
        "location": entry.location,
        "degree": entry.degree,
        "dates": format_date_range(entry.start, entry.end),
        "gpa": None if compact else entry.gpa,
        "honors": [] if compact else list(entry.honors),
        "coursework": [] if compact else list(entry.coursework),
    }


def credential_rows(profile: Profile) -> list[dict[str, Any]]:
    """Awards and certifications as printable one-liners, newest first.

    Composed here rather than in the template because each is exactly one
    printed line and `latex/layout.py` has to measure that line to budget for
    it. Two places composing the same string would drift, and the symptom would
    be a page that overflows for reasons the budget cannot see.
    """
    rows = [
        {
            "date": award.received,
            "text": f"{award.name} — {award.issuer}"
            + (f" ({award.detail})" if award.detail else ""),
        }
        for award in profile.awards
    ] + [
        {"date": cert.issued, "text": f"{cert.name} — {cert.issuer}"}
        for cert in profile.certifications
    ]
    return sorted(rows, key=lambda row: row["date"], reverse=True)


def _bullet_texts(
    entry: Entry,
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


def _leadership_row(
    entry: LeadershipEntry,
    overrides: dict[str, str] | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Identical shape to an experience row: same two-line heading in the PDF,
    which is also why `layout.py` charges it the same overhead."""
    return {
        "title": entry.title,
        "dates": format_date_range(entry.start, entry.end),
        "org": entry.org,
        "location": entry.location,
        "bullets": _bullet_texts(entry, overrides, allowed_ids),
    }


def _publication_row(
    entry: PublicationEntry,
    display_index: dict[str, str],
    overrides: dict[str, str] | None = None,
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    """A publication prints one date, not a range -- it was published, it did
    not run from one month to another."""
    return {
        "title": entry.title,
        "venue": entry.venue,
        "dates": format_year_month(entry.start),
        "bullets": _bullet_texts(entry, overrides, allowed_ids),
    }


def _sort_key_most_recent_first(entry: Entry) -> tuple[str, str]:
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
    stage: Stage | None = None,
    today: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Everything `jake_resume.tex.j2` needs, and nothing else.

    `selected_ids` restricts output to what M3's knapsack chose; `tailored_text`
    substitutes M4's verified rewrites. Passing neither renders the whole
    profile verbatim, which is what M0's `build` command does.

    `stage` decides the **order** of the sections and whether education prints
    in full or compact. It does not decide what appears -- an entry with no
    surviving bullets is dropped below regardless, and that is the knapsack's
    decision, not this module's. Left as None it is computed from the profile.
    """
    today = today or this_month()
    stage = stage or career_stage(profile, today)
    compact_education = stage == "experienced"

    display_index = build_display_name_index(profile.skills)
    allowed = set(selected_ids) if selected_ids is not None else None

    experience = sorted(profile.experience, key=_sort_key_most_recent_first, reverse=True)
    projects = sorted(profile.projects, key=_sort_key_most_recent_first, reverse=True)
    leadership = sorted(profile.leadership, key=_sort_key_most_recent_first, reverse=True)
    publications = sorted(profile.publications, key=_sort_key_most_recent_first, reverse=True)
    education = sorted(
        profile.education,
        key=lambda e: (e.end or "9999-99", e.start),
        reverse=True,
    )

    experience_rows = [_experience_row(e, tailored_text, allowed) for e in experience]
    project_rows = [_project_row(p, display_index, tailored_text, allowed) for p in projects]
    leadership_rows = [_leadership_row(e, tailored_text, allowed) for e in leadership]
    publication_rows = [
        _publication_row(p, display_index, tailored_text, allowed) for p in publications
    ]

    # An entry whose every bullet was dropped must not print as a bare heading.
    # This also keeps the rendered page honest with the line budget selection
    # spent, which charges for furniture only when the furniture appears.
    if allowed is not None:
        experience_rows = [row for row in experience_rows if row["bullets"]]
        project_rows = [row for row in project_rows if row["bullets"]]
        leadership_rows = [row for row in leadership_rows if row["bullets"]]
        publication_rows = [row for row in publication_rows if row["bullets"]]

    return {
        # The template walks this and emits each named section in turn, so
        # nothing downstream hardcodes an order of its own.
        "sections": list(section_order(stage)),
        "stage": stage,
        "identity": {
            "name": profile.identity.name,
            "phone": profile.identity.phone,
            "email": profile.identity.email,
            "location": profile.identity.location,
            "links": [{"label": link.label, "url": link.url} for link in profile.identity.links],
        },
        "summary": (summary or "").strip(),
        "education": [_education_row(e, compact=compact_education) for e in education],
        # Awards and certifications print under Education, as they do on a real
        # student resume. Suppressed once compact: they are student credentials,
        # and a work history is stronger evidence than any of them.
        "credentials": [] if compact_education else credential_rows(profile),
        "experience": experience_rows,
        "projects": project_rows,
        "leadership": leadership_rows,
        "publications": publication_rows,
        "skill_groups": group_skills_by_category(profile.skills),
    }
