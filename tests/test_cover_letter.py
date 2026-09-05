"""The cover letter subgraph. M6's definition of done.

Spec 8: "<= 320 words, no claim absent from the KB, consistent with resume
bullets."

The three clauses map to the three checks, and two of them cost nothing:

    <= 320 words              -> verify_length          free, counted in Python
    no claim absent from KB   -> verify_letter_grounding free, reuses M4
    consistent with resume    -> verify_consistency      one cheap judge call

Everything here runs against fakes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.graph.nodes.cover_letter import (
    MAX_LETTER_ATTEMPTS,
    build_cover_letter_subgraph,
    route_after_letter_verify,
    verify_consistency,
    verify_cover_letter,
    verify_length,
    verify_letter_grounding,
)
from resume_agent.graph.state import AgentState, RunOptions
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.letter import (
    MAX_LETTER_WORDS,
    ConsistencyVerdict,
    CoverLetter,
    CoverLetterFields,
    count_words,
)
from resume_agent.models.profile import Profile

from .conftest import PROFILE_EXAMPLE

# A letter that is truthful against profile.example: every number and every
# technology below appears in that profile's bullets.
GOOD = CoverLetterFields(
    hook=(
        "Your posting describes a platform where latency and cost are the product, "
        "not an afterthought."
    ),
    proof_one=(
        "At my current role I cut p95 checkout latency from 820ms to 50ms with a Redis "
        "read-through cache, and removed N+1 queries across 12 endpoints."
    ),
    proof_two=(
        "Separately I repartitioned an events table on user_id, which took the slowest "
        "tenant report from 41s to well under a second."
    ),
    why_this_company=(
        "The role is explicitly about making expensive things cheap, which is the "
        "work I keep choosing."
    ),
    close="I would like to talk about what your slowest query looks like.",
    bullet_ids_used=["exp_halvorsen_bright.b1", "exp_halvorsen_bright.b3"],
)


def letter(**overrides: Any) -> CoverLetter:
    return CoverLetter.from_fields(GOOD.model_copy(update=overrides))


class FakeConsistencyJudge(BaseChatModel):
    verdict: ConsistencyVerdict = ConsistencyVerdict(verdict="consistent", reason="fine")
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-consistency"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> ConsistencyVerdict:
                fake.call_count += 1
                return fake.verdict

        return _Runnable()


class FakeLetterWriter(BaseChatModel):
    """Returns a scripted sequence of drafts, one per attempt."""

    drafts: list[CoverLetterFields] = []
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-letter-writer"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> CoverLetterFields:
                index = min(fake.call_count, len(fake.drafts) - 1)
                fake.call_count += 1
                return fake.drafts[index]

        return _Runnable()


def letter_state(**overrides: Any) -> AgentState:
    job = JobSpec.from_fields(
        JobSpecFields(
            company="Acme Corp",
            title="Backend Engineer",
            seniority="mid",
            domain="payments",
            requirements=[
                Requirement(text="caching", category="practice", weight=5, is_must_have=True)
            ],
            responsibilities=[],
            ats_keywords=["Redis"],
            culture_signals=["small team"],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )
    state: AgentState = {
        "raw_jd": "raw",
        "profile_path": str(PROFILE_EXAMPLE),
        "options": RunOptions(use_judge=False),
        "job_spec": job,
        "evidence": [],
        "tailored": [],
        "letter_attempts": 0,
        "letter_critiques": [],
        "letter_verified": False,
        "errors": [],
    }
    state.update(overrides)
    return state


# ===========================================================================
# Narratives -- the input the letter is built from
# ===========================================================================


def test_narratives_load_from_markdown(example_profile: Profile) -> None:
    assert {n.name for n in example_profile.narratives} == {
        "hardest_bug",
        "how_i_learn",
        "values",
        "why_backend",
    }
    assert example_profile.narrative("values") is not None
    assert example_profile.narrative("nope") is None


def test_narratives_readme_is_not_loaded(example_profile: Profile) -> None:
    """README.md documents the directory for a human.

    Feeding it to a cover-letter prompt would be feeding the model instructions
    written for the profile's owner.
    """
    assert "readme" not in {n.name.lower() for n in example_profile.narratives}


def test_narratives_introduce_no_ungrounded_technology(example_profile: Profile) -> None:
    """The rule the narratives README states, enforced.

    A narrative naming a technology absent from skills.yaml produces letters
    that fail the grounding check and get retried to the cap -- so the example
    narratives must not contain that trap.
    """
    from resume_agent.grounding.vocabulary import unsupported_technologies

    corpus = " ".join(b.canonical for b in example_profile.all_bullets())
    for narrative in example_profile.narratives:
        unsupported = unsupported_technologies(
            narrative.content, corpus, example_profile.skill_vocabulary(), prose=True
        )
        assert not unsupported, f"{narrative.name} introduces {sorted(unsupported)}"


# ===========================================================================
# DoD 1: <= 320 words
# ===========================================================================


def test_word_count_is_computed_not_returned() -> None:
    """CLAUDE.md rule 2. A model asked how many words it wrote will guess."""
    assert "word_count" not in CoverLetterFields.model_fields
    assert letter().word_count == count_words(letter().body())


def test_the_limit_is_320() -> None:
    assert MAX_LETTER_WORDS == 320


def test_a_letter_within_the_limit_passes() -> None:
    assert verify_length(letter()).passed


def test_an_overlong_letter_is_rejected_with_a_countable_critique() -> None:
    long_letter = letter(close="word " * 400)
    result = verify_length(long_letter)

    assert result.passed is False
    assert result.check == "length"
    # The critique must say how much to cut, not merely that it is too long.
    assert str(long_letter.word_count) in result.critique
    assert "320" in result.critique


def test_word_count_ignores_paragraph_breaks() -> None:
    assert count_words("one two\n\nthree") == 3
    assert count_words("   ") == 0


# ===========================================================================
# DoD 2: no claim absent from the KB
# ===========================================================================


def test_a_truthful_letter_passes_grounding(example_profile: Profile) -> None:
    result = verify_letter_grounding(letter(), example_profile)
    assert result.passed, f"rejected a truthful letter: {result.critique}"


def test_an_invented_number_is_caught(example_profile: Profile) -> None:
    """The same rule as the resume: every figure traces to recorded data."""
    fabricated = letter(proof_one="I cut latency by 94% and saved 7919 engineer-hours.")
    result = verify_letter_grounding(fabricated, example_profile)

    assert result.passed is False
    assert result.check == "grounding"
    assert "7919" in result.critique


def test_an_unrecorded_technology_is_caught(example_profile: Profile) -> None:
    fabricated = letter(proof_two="I designed the Cassandra cluster that backs the product.")
    result = verify_letter_grounding(fabricated, example_profile)

    assert result.passed is False
    assert result.check == "grounding"
    assert "cassandra" in result.critique.lower()


def test_grounding_allows_numbers_from_any_bullet(example_profile: Profile) -> None:
    """A letter draws on several achievements at once.

    Unlike a rewritten bullet -- checked against its own source alone -- the
    letter's allowed set is the union across the whole knowledge base.
    """
    across_two = letter(
        proof_one="I cut latency from 820ms to 50ms.",
        proof_two="I cut monthly AWS spend 38%.",  # a different bullet entirely
    )
    assert verify_letter_grounding(across_two, example_profile).passed


# ===========================================================================
# DoD 3: consistent with the resume's bullets
# ===========================================================================


def test_consistency_judge_sees_the_actual_resume_bullets() -> None:
    """What makes "consistent with resume bullets" checkable rather than aspirational."""
    judge = FakeConsistencyJudge(
        verdict=ConsistencyVerdict(
            verdict="inconsistent",
            reason="'led the migration' conflicts with 'Collaborated with two engineers'.",
        )
    )
    result = verify_consistency(letter(), ["Collaborated with two engineers"], llm=judge)

    assert result.passed is False
    assert result.check == "consistency"
    assert "Collaborated" in result.critique
    assert judge.call_count == 1


def test_a_consistent_letter_passes() -> None:
    judge = FakeConsistencyJudge()
    assert verify_consistency(letter(), ["Cut p95 latency"], llm=judge).passed


def test_no_resume_bullets_means_nothing_to_contradict() -> None:
    """An empty resume is not a licence to invent a verdict."""
    judge = FakeConsistencyJudge()
    assert verify_consistency(letter(), [], llm=judge).passed
    assert judge.call_count == 0


# ===========================================================================
# Layer ordering
# ===========================================================================


def test_free_checks_run_before_the_judge(example_profile: Profile) -> None:
    """An overlong letter must never reach a paid call."""
    judge = FakeConsistencyJudge()
    result = verify_cover_letter(
        letter(close="word " * 400), example_profile, ["a bullet"], llm=judge
    )

    assert result.check == "length"
    assert judge.call_count == 0, "spent money judging a letter a word count already rejected"


def test_grounding_runs_before_the_judge(example_profile: Profile) -> None:
    judge = FakeConsistencyJudge()
    result = verify_cover_letter(
        letter(proof_one="saved 7919 hours"), example_profile, ["a bullet"], llm=judge
    )
    assert result.check == "grounding"
    assert judge.call_count == 0


def test_use_judge_false_keeps_the_free_layers(example_profile: Profile) -> None:
    judge = FakeConsistencyJudge()
    assert verify_cover_letter(
        letter(), example_profile, ["a bullet"], use_judge=False, llm=judge
    ).passed
    assert judge.call_count == 0

    # ...but the free layers still bite.
    rejected = verify_cover_letter(
        letter(proof_one="saved 7919 hours"),
        example_profile,
        ["a bullet"],
        use_judge=False,
        llm=judge,
    )
    assert rejected.passed is False


# ===========================================================================
# The subgraph and its cap
# ===========================================================================


def test_subgraph_compiles_with_its_retry_cycle() -> None:
    mermaid = build_cover_letter_subgraph().get_graph().draw_mermaid()
    assert "draft" in mermaid and "verify" in mermaid
    assert "give_up" in mermaid


def test_a_good_draft_is_accepted_first_time() -> None:
    writer = FakeLetterWriter(drafts=[GOOD])
    graph = build_cover_letter_subgraph(draft_llm=writer)

    final = graph.invoke(letter_state())

    assert final["letter_verified"] is True
    assert final["cover_letter"] is not None
    assert writer.call_count == 1, "retried a letter that already passed"


def test_a_rejected_draft_is_retried_and_can_recover() -> None:
    """The critique goes back to the writer, and a fixed draft is accepted."""
    bad = GOOD.model_copy(update={"proof_one": "I saved 7919 engineer-hours."})
    writer = FakeLetterWriter(drafts=[bad, GOOD])
    graph = build_cover_letter_subgraph(draft_llm=writer)

    final = graph.invoke(letter_state())

    assert final["letter_verified"] is True
    assert "7919" not in final["cover_letter"].body()
    assert writer.call_count == 2


def test_an_unfixable_letter_is_abandoned_not_shipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The cap behaviour, and the reason it differs from a dropped bullet.

    A bullet can be dropped and leave a slightly weaker resume. A letter is one
    artifact -- so at the cap there is simply no letter, loudly, and the resume
    still ships. Shipping an unverified letter would defeat the whole verifier.
    """
    bad = GOOD.model_copy(update={"proof_one": "I saved 7919 engineer-hours."})
    writer = FakeLetterWriter(drafts=[bad])
    graph = build_cover_letter_subgraph(draft_llm=writer)

    with caplog.at_level(logging.ERROR):
        final = graph.invoke(letter_state())

    assert final["cover_letter"] is None, "shipped a letter that never passed verification"
    assert final.get("errors"), "abandoning the letter left no trace in errors"
    assert any("ABANDONING" in record.getMessage() for record in caplog.records)


def test_letter_attempts_are_capped() -> None:
    bad = GOOD.model_copy(update={"proof_one": "I saved 7919 engineer-hours."})
    writer = FakeLetterWriter(drafts=[bad])
    build_cover_letter_subgraph(draft_llm=writer).invoke(letter_state())

    assert writer.call_count == MAX_LETTER_ATTEMPTS + 1, (
        f"made {writer.call_count} attempts, expected {MAX_LETTER_ATTEMPTS + 1}"
    )


def test_invented_bullet_ids_are_dropped_from_the_letter() -> None:
    """A claimed source that does not exist is a claim attached to nothing."""
    lying = GOOD.model_copy(update={"bullet_ids_used": ["exp_halvorsen_bright.b1", "nope.b9"]})
    writer = FakeLetterWriter(drafts=[lying])

    final = build_cover_letter_subgraph(draft_llm=writer).invoke(letter_state())

    assert final["cover_letter"].bullet_ids_used == ["exp_halvorsen_bright.b1"]


# --- routing ---------------------------------------------------------------


def test_route_retries_while_unverified() -> None:
    assert route_after_letter_verify({"letter_verified": False, "letter_attempts": 1}) == "draft"


def test_route_stops_when_verified() -> None:
    assert route_after_letter_verify({"letter_verified": True, "letter_attempts": 1}) == "done"


def test_route_gives_up_at_the_cap() -> None:
    state = {"letter_verified": False, "letter_attempts": MAX_LETTER_ATTEMPTS + 1}
    assert route_after_letter_verify(state) == "give_up"


# --- the template ----------------------------------------------------------


def test_cover_letter_template_escapes_every_interpolation() -> None:
    """Same mechanical guard the resume template has (plan D4)."""
    import re

    path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "resume_agent"
        / "templates"
        / "cover_letter.tex.j2"
    )
    expressions = re.findall(r"\\VAR\{([^}]*)\}", path.read_text(encoding="utf-8"))
    assert expressions
    unescaped = [e.strip() for e in expressions if not re.search(r"\|\s*tex\s*$", e)]
    assert not unescaped, f"unescaped interpolations: {unescaped}"


# ===========================================================================
# End to end, through the parent graph
# ===========================================================================


@pytest.mark.skipif(
    __import__("resume_agent.latex.compile", fromlist=["find_compiler"]).find_compiler() is None,
    reason="no LaTeX compiler",
)
def test_full_graph_produces_a_compiled_cover_letter(tmp_path: Path) -> None:
    """The subgraph really runs inside the parent, and the letter really compiles.

    Everything downstream of the letter is real: the LaTeX template, tectonic,
    and finalize writing both artifacts side by side into the run directory.
    """
    from resume_agent.graph.build import build_graph, initial_state
    from resume_agent.jd_cache import JobSpecCache, jd_cache_key
    from resume_agent.llm import PARSE_MODEL, prompt_version
    from tests.test_graph import (
        FakeFixer,
        FakeJudge,
        FakeScorer,
        FakeWriter,
        make_job,
    )

    raw_jd = "Backend engineer. We need caching and Redis experience. Python required."
    JobSpecCache().put(
        jd_cache_key(raw_jd, PARSE_MODEL, prompt_version("parse_jd")), make_job()
    )

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(),
        judge_llm=FakeJudge(),
        fix_llm=FakeFixer(),
        letter_llm=FakeLetterWriter(drafts=[GOOD]),
        letter_judge_llm=FakeConsistencyJudge(),
    )
    final = graph.invoke(
        initial_state(
            raw_jd,
            PROFILE_EXAMPLE,
            RunOptions(out_dir=str(tmp_path), use_judge=False, write_cover_letter=True),
        ),
        {"recursion_limit": 80},
    )

    assert final["cover_letter"] is not None
    assert final["cover_letter"].word_count <= MAX_LETTER_WORDS

    run_dir = Path(final["out_dir"])
    assert (run_dir / "resume.pdf").is_file()
    assert (run_dir / "cover_letter.pdf").is_file(), "the letter did not compile"
    assert (run_dir / "cover_letter.tex").is_file()

    import json

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["cover_letter"]["word_count"] <= MAX_LETTER_WORDS
    assert record["prompt_versions"]["write_cover_letter"]
