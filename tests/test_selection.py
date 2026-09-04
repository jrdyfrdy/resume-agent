"""The selection knapsack, under property-based tests.

M3's definition of done: "selection respects every constraint in §5 under
property-based tests."

Why Hypothesis rather than examples: the five constraints in spec §5 interact.
The theme cap can block the bullet that satisfies a must-have; the
minimum-bullets-per-experience rule can force a repair that frees budget and
changes what else fits. Example-based tests would confirm the three scenarios I
thought of. Hypothesis generates profiles, scores and budgets until it finds the
one I did not -- and then shrinks it to the smallest case that still fails.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from resume_agent.graph.nodes.select import (
    MAX_BULLETS_PER_THEME,
    MIN_BULLETS_PER_EXPERIENCE,
    RECENCY_FLOOR,
    recency_decay,
    score_bullets,
    select_content,
)
from resume_agent.latex.metrics import estimate_lines
from resume_agent.models.fit import EvidenceMatch
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.profile import (
    Bullet,
    ExperienceEntry,
    Identity,
    Profile,
    ProjectEntry,
    Skill,
)

THEMES = ["performance", "backend", "data", "cost", "reliability", "frontend"]


# --- generators -------------------------------------------------------------


@st.composite
def bullets(draw: st.DrawFn, entry_id: str, count: int) -> list[Bullet]:
    result = []
    for index in range(count):
        result.append(
            Bullet(
                id=f"{entry_id}.b{index}",
                canonical=draw(st.text(alphabet="abcdefg ", min_size=10, max_size=200)),
                themes=draw(st.lists(st.sampled_from(THEMES), min_size=0, max_size=2, unique=True)),
                confidence=draw(st.sampled_from(["verified", "approximate", "claim"])),
            )
        )
    return result


@st.composite
def profiles(draw: st.DrawFn) -> Profile:
    """A structurally valid profile with no skill vocabulary to satisfy.

    Bullets are generated with empty `skills`/`tech` so the cross-file validator
    has nothing to reject -- this suite is about selection, and a generator that
    kept tripping schema validation would test the schema instead.
    """
    n_exp = draw(st.integers(min_value=0, max_value=3))
    n_prj = draw(st.integers(min_value=0, max_value=3))

    experience = []
    for index in range(n_exp):
        entry_id = f"exp_{index}"
        experience.append(
            ExperienceEntry(
                id=entry_id,
                type="experience",
                org=f"Org {index}",
                title="Engineer",
                location="Remote",
                start="2020-01",
                end=draw(st.sampled_from([None, "2022-06", "2024-03"])),
                bullets=draw(bullets(entry_id, draw(st.integers(min_value=1, max_value=4)))),
            )
        )

    projects = []
    for index in range(n_prj):
        entry_id = f"prj_{index}"
        projects.append(
            ProjectEntry(
                id=entry_id,
                type="project",
                name=f"Project {index}",
                start="2021-01",
                end="2022-01",
                bullets=draw(bullets(entry_id, draw(st.integers(min_value=1, max_value=3)))),
            )
        )

    return Profile(
        identity=Identity(name="A", email="a@example.com", phone="1", location="Remote"),
        experience=experience,
        projects=projects,
        skills=[Skill(canonical="Python", category="language", level="expert")],
    )


@st.composite
def job_and_matches(draw: st.DrawFn, profile: Profile) -> tuple[JobSpec, list[EvidenceMatch]]:
    all_bullets = profile.all_bullets()
    n_req = draw(st.integers(min_value=1, max_value=4))

    requirements = [
        Requirement(
            text=f"requirement {index}",
            category="practice",
            weight=draw(st.integers(min_value=1, max_value=5)),
            is_must_have=draw(st.booleans()),
        )
        for index in range(n_req)
    ]
    job = JobSpec.from_fields(
        JobSpecFields(
            company="C",
            title="T",
            seniority="mid",
            domain="d",
            requirements=requirements,
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )

    matches = []
    for requirement in requirements:
        for bullet in all_bullets:
            if draw(st.booleans()):
                matches.append(
                    EvidenceMatch(
                        bullet_id=bullet.id,
                        requirement_text=requirement.text,
                        relevance=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
                        rationale="r",
                    )
                )
    return job, matches


@st.composite
def selection_inputs(draw: st.DrawFn) -> tuple[JobSpec, list[EvidenceMatch], Profile, int]:
    profile = draw(profiles())
    job, matches = draw(job_and_matches(profile))
    budget = draw(st.integers(min_value=0, max_value=40))
    return job, matches, profile, budget


SETTINGS = settings(
    max_examples=250,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
)


# --- the constraints from spec 5, one property each -------------------------


@SETTINGS
@given(selection_inputs())
def test_never_exceeds_the_line_budget(inputs) -> None:
    """Spec 5: `sum(estimated_lines) <= line_budget`. The load-bearing one."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    assert result.total_estimated_lines <= budget
    recomputed = sum(
        estimate_lines(profile.bullet_by_id(bid).canonical) for bid in result.selected_bullet_ids
    )
    assert recomputed == result.total_estimated_lines, "reported line count does not match reality"


@SETTINGS
@given(selection_inputs())
def test_experience_entries_get_at_least_two_bullets(inputs) -> None:
    """Spec 5: "every experience entry that appears gets >= 2 bullets"."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    counts: dict[str, int] = {}
    for bullet_id in result.selected_bullet_ids:
        for entry in profile.experience:
            if any(b.id == bullet_id for b in entry.bullets):
                counts[entry.id] = counts.get(entry.id, 0) + 1

    for entry_id, count in counts.items():
        assert count >= MIN_BULLETS_PER_EXPERIENCE, (
            f"{entry_id} appears with only {count} bullet(s)"
        )


@SETTINGS
@given(selection_inputs())
def test_no_more_than_three_bullets_share_a_theme(inputs) -> None:
    """Spec 5: "<= 3 bullets sharing the same theme"."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    counts: dict[str, int] = {}
    for bullet_id in result.selected_bullet_ids:
        for theme in profile.bullet_by_id(bullet_id).themes:
            counts[theme] = counts.get(theme, 0) + 1

    for theme, count in counts.items():
        assert count <= MAX_BULLETS_PER_THEME, f"{count} bullets share theme {theme!r}"


@SETTINGS
@given(selection_inputs())
def test_chronological_order_preserved_within_entries(inputs) -> None:
    """Spec 5: "chronological order preserved within sections"."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    authored = {}
    for entry in profile.entries():
        for index, bullet in enumerate(entry.bullets):
            authored[bullet.id] = (entry.id, index)

    positions = [authored[bid] for bid in result.selected_bullet_ids]
    for (entry_a, index_a), (entry_b, index_b) in zip(positions, positions[1:], strict=False):
        if entry_a == entry_b:
            assert index_a < index_b, "bullets within an entry came out re-ordered"


@SETTINGS
@given(selection_inputs())
def test_output_is_a_subset_of_the_input_with_no_duplicates(inputs) -> None:
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    known = {b.id for b in profile.all_bullets()}
    assert set(result.selected_bullet_ids) <= known, "selected a bullet that does not exist"
    assert len(result.selected_bullet_ids) == len(set(result.selected_bullet_ids))


@SETTINGS
@given(selection_inputs())
def test_selection_is_deterministic(inputs) -> None:
    """Two runs on identical input must agree, or the golden snapshot flaps."""
    job, matches, profile, budget = inputs
    first = select_content(job, matches, profile, budget)
    second = select_content(job, matches, profile, budget)
    assert first.selected_bullet_ids == second.selected_bullet_ids


@SETTINGS
@given(selection_inputs())
def test_strict_mode_excludes_claims(inputs) -> None:
    """Spec 3.1: "Bullets marked `claim` can be excluded via a strict mode"."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget, strict=True)

    for bullet_id in result.selected_bullet_ids:
        assert profile.bullet_by_id(bullet_id).confidence != "claim"


@SETTINGS
@given(selection_inputs())
def test_every_rejection_has_a_reason(inputs) -> None:
    """`rejected` is how you answer "why is my best bullet missing?"."""
    job, matches, profile, budget = inputs
    result = select_content(job, matches, profile, budget)

    for bullet_id, reason in result.rejected.items():
        assert reason, f"{bullet_id} was rejected with an empty reason"
        assert bullet_id not in result.selected_bullet_ids


# --- degenerate inputs ------------------------------------------------------


def _tiny_case(budget: int):
    profile = Profile(
        identity=Identity(name="A", email="a@example.com", phone="1", location="Remote"),
        experience=[
            ExperienceEntry(
                id="exp_1",
                type="experience",
                org="Org",
                title="Engineer",
                location="Remote",
                start="2020-01",
                end=None,
                bullets=[
                    Bullet(id="exp_1.b1", canonical="short one", themes=["backend"]),
                    Bullet(id="exp_1.b2", canonical="short two", themes=["backend"]),
                ],
            )
        ],
        skills=[Skill(canonical="Python", category="language", level="expert")],
    )
    job = JobSpec.from_fields(
        JobSpecFields(
            company="C",
            title="T",
            seniority="mid",
            domain="d",
            requirements=[
                Requirement(text="python", category="language", weight=5, is_must_have=True)
            ],
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )
    matches = [
        EvidenceMatch(
            bullet_id="exp_1.b1", requirement_text="python", relevance=0.9, rationale="r"
        ),
        EvidenceMatch(
            bullet_id="exp_1.b2", requirement_text="python", relevance=0.8, rationale="r"
        ),
    ]
    return job, matches, profile, budget


def test_zero_budget_selects_nothing() -> None:
    job, matches, profile, _ = _tiny_case(0)
    result = select_content(job, matches, profile, 0)
    assert result.selected_bullet_ids == []
    assert result.total_estimated_lines == 0


def test_budget_for_one_bullet_drops_the_lone_experience() -> None:
    """A job heading with one bullet under it reads worse than no job at all."""
    job, matches, profile, _ = _tiny_case(1)
    result = select_content(job, matches, profile, 1)
    assert result.selected_bullet_ids == []
    assert any("could not reach" in reason for reason in result.rejected.values())


def test_budget_for_two_bullets_keeps_both() -> None:
    job, matches, profile, _ = _tiny_case(2)
    result = select_content(job, matches, profile, 2)
    assert result.selected_bullet_ids == ["exp_1.b1", "exp_1.b2"]


def test_no_matches_selects_nothing() -> None:
    job, _, profile, _ = _tiny_case(30)
    result = select_content(job, [], profile, 30)
    assert result.selected_bullet_ids == []


def test_matches_for_unknown_requirements_are_ignored() -> None:
    """A stale EvidenceMatch must not crash selection."""
    job, matches, profile, _ = _tiny_case(30)
    matches.append(
        EvidenceMatch(
            bullet_id="exp_1.b1",
            requirement_text="not in this posting",
            relevance=1.0,
            rationale="r",
        )
    )
    result = select_content(job, matches, profile, 30)
    assert result.selected_bullet_ids


# --- recency decay ----------------------------------------------------------


def test_current_role_is_not_decayed() -> None:
    assert recency_decay(None, "2026-09") == 1.0


def test_decay_halves_at_the_half_life() -> None:
    assert recency_decay("2020-09", "2026-09") == 0.5


def test_decay_is_floored() -> None:
    """Genuinely relevant old work must not fall to nothing."""
    assert recency_decay("1995-01", "2026-09") == RECENCY_FLOOR


def test_more_recent_scores_higher() -> None:
    assert recency_decay("2024-01", "2026-09") > recency_decay("2018-01", "2026-09")


def test_a_bullet_answering_two_requirements_outscores_one_answering_one() -> None:
    """Scores sum across requirements, which is what surfaces broad bullets."""
    job, _, profile, _ = _tiny_case(30)
    job.requirements.append(
        Requirement(text="second", category="practice", weight=5, is_must_have=False)
    )
    matches = [
        EvidenceMatch(
            bullet_id="exp_1.b1", requirement_text="python", relevance=0.8, rationale="r"
        ),
        EvidenceMatch(
            bullet_id="exp_1.b1", requirement_text="second", relevance=0.8, rationale="r"
        ),
        EvidenceMatch(
            bullet_id="exp_1.b2", requirement_text="python", relevance=0.9, rationale="r"
        ),
    ]
    scored = {s.bullet_id: s.score for s in score_bullets(job, matches, profile, "2026-09")}
    assert scored["exp_1.b1"] > scored["exp_1.b2"]
