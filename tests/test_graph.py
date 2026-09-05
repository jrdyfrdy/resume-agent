"""The graph, and both feedback loops. M5's definition of done.

Spec 8:

    "inject a deliberate LaTeX syntax error -> auto-repaired within 3
     iterations; force a 2-page overflow -> converges to 1 page; both loops
     respect their caps and fail gracefully at the limit."

Every test here drives the **real compiled graph** -- real routing functions,
real edges, real tectonic -- with fake models standing in for the LLM calls.
That combination is deliberate: the routing and the compiler are what M5
actually builds, and faking the models makes the loops deterministic and free,
so a failure here means the graph is wrong rather than that a model had an off
day.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from resume_agent.graph.build import (
    MAX_LAYOUT_ATTEMPTS,
    build_graph,
    initial_state,
    route_after_inspect,
    route_after_verify,
)
from resume_agent.graph.nodes.score import ScoredMatch, ScoringResult
from resume_agent.graph.nodes.tailor import TailoringResult
from resume_agent.graph.nodes.verify import MAX_GROUNDING_ATTEMPTS
from resume_agent.graph.state import AgentState, RunOptions
from resume_agent.kb.loader import load_profile
from resume_agent.latex.compile import find_compiler
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.profile import Profile
from resume_agent.models.resume import JudgeVerdict, TailoredBulletFields

from .conftest import PROFILE_EXAMPLE

requires_latex = pytest.mark.skipif(find_compiler() is None, reason="no LaTeX compiler")

RAW_JD = "Backend engineer. We need caching and Redis experience. Python required."

# The corruption injected into the rendered source. Deliberately at the document
# level rather than in bullet text: the escaper turns every stray backslash in
# profile content into literal characters, so content cannot produce this. A
# template bug can.
BROKEN_MARKER = "\\thisIsNotACommand"


def make_job() -> JobSpec:
    return JobSpec.from_fields(
        JobSpecFields(
            company="Acme Corp",
            title="Backend Engineer",
            seniority="mid",
            domain="payments",
            requirements=[
                Requirement(text="caching", category="practice", weight=5, is_must_have=True),
                Requirement(text="Python", category="language", weight=5, is_must_have=True),
            ],
            responsibilities=[],
            ats_keywords=["Redis", "Python"],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        RAW_JD,
    )


# --- fakes ------------------------------------------------------------------


class _Structured(BaseChatModel):
    """Base for the structured-output fakes: canned payload, counts calls."""

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _payload(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> Any:
                fake.call_count += 1
                return fake._payload()

        return _Runnable()


def _ids_in_prompt(messages: Any, marker: str) -> list[str]:
    """Pull the bullet ids out of the prompt the node actually built.

    Fakes that return a fixed list drift from what the graph fed them -- the
    node then discards every unrecognised id and the test silently exercises a
    smaller pipeline than it thinks. Reading the ids back makes the fake behave
    like a real model: it answers about what it was shown.
    """
    text = "\n".join(str(getattr(message, "content", message)) for message in (messages or []))
    ids = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            ids.append(stripped[len(marker) :].strip())
    return ids


class FakeScorer(_Structured):
    """Scores every candidate the node put in front of it, at 0.9."""

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, messages: Any, **_kw: Any) -> ScoringResult:
                fake.call_count += 1
                return ScoringResult(
                    matches=[
                        ScoredMatch(
                            bullet_id=bullet_id,
                            requirement_text="caching",
                            relevance=0.9,
                            rationale="direct evidence",
                        )
                        for bullet_id in _ids_in_prompt(messages, "- id:")
                    ]
                )

        return _Runnable()


class FakeWriter(_Structured):
    """Rewrites each bullet as its own canonical text -- so always grounded.

    With `corrupt=True` it appends a number that traces to no metric, which the
    verifier must reject on every attempt.
    """

    profile_path: str = str(PROFILE_EXAMPLE)
    corrupt: bool = False

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, messages: Any, **_kw: Any) -> TailoringResult:
                fake.call_count += 1
                profile = load_profile(Path(fake.profile_path))
                bullets = []
                # The tailoring prompt heads each achievement with "### <id>".
                for bullet_id in _ids_in_prompt(messages, "### "):
                    text = profile.bullet_by_id(bullet_id).canonical
                    if fake.corrupt:
                        text = f"{text} across 7919 regions"
                    bullets.append(TailoredBulletFields(source_id=bullet_id, text=text))
                return TailoringResult(bullets=bullets)

        return _Runnable()


class FakeJudge(_Structured):
    def _payload(self) -> Any:
        return JudgeVerdict(verdict="supported", reason="fine")


class FakeFixer(BaseChatModel):
    """Plain-text stand-in for `fix_latex`. Returns whatever it is told to."""

    replacement: str = ""
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-fixer"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def invoke(self, _messages: Any, **_kw: Any) -> AIMessage:
        self.call_count += 1
        return AIMessage(content=self.replacement)


# --- fixtures ---------------------------------------------------------------


def seeded(profile_dir: Path, out_dir: Path, **options: Any) -> AgentState:
    """A start state with the JD parse pre-cached.

    `parse_jd` is the only node with no LLM injection point, because it is the
    graph's entry. Its cache is the documented way to avoid re-paying for a
    posting already parsed, so priming it here uses the real mechanism rather
    than a test-only backdoor.
    """
    from resume_agent.jd_cache import JobSpecCache, jd_cache_key
    from resume_agent.llm import PARSE_MODEL, prompt_version

    JobSpecCache().put(jd_cache_key(RAW_JD, PARSE_MODEL, prompt_version("parse_jd")), make_job())
    return initial_state(
        RAW_JD,
        profile_dir,
        RunOptions(
            out_dir=str(out_dir),
            use_judge=False,
            # These tests exercise the layout loop; M6's subgraph has its own
            # file. Left on, every one of them would need a letter writer too.
            write_cover_letter=False,
            **options,
        ),
    )


@pytest.fixture
def oversized_profile_dir(tmp_path: Path) -> Path:
    """A profile with more content than one page can hold.

    `profile.example` cannot overflow -- fitting on one page is M0's definition
    of done, so it is sized to fit by construction. Forcing the layout loop
    needs a profile that genuinely does not fit: two extra jobs, giving 35 lines
    of content against a measured 26-line capacity.
    """
    target = tmp_path / "profile.big"
    shutil.copytree(PROFILE_EXAMPLE, target)

    base = (PROFILE_EXAMPLE / "experience" / "halvorsen_bright.yaml").read_text(encoding="utf-8")
    for index, org in [(1, "Extra One Systems"), (2, "Extra Two Labs")]:
        text = (
            base.replace("exp_halvorsen_bright", f"exp_extra_{index}")
            .replace("Halvorsen & Bright R&D", org)
            .replace("end: null", f"end: 202{index}-01")
        )
        (target / "experience" / f"extra_{index}.yaml").write_text(text, encoding="utf-8")

    # Give every bullet in the whole profile a theme of its own.
    #
    # Selection's diversity constraint (<= 3 bullets sharing a theme) otherwise
    # binds long before the line budget does. Measured: 15 bullets / 25 lines
    # selected no matter how large the budget, against a 26-line capacity -- so
    # the page could never overflow and the layout loop would never run. Making
    # themes unique leaves the budget as the only binding constraint, which is
    # the one this test is about.
    counter = itertools.count()
    for path in [*(target / "experience").glob("*.yaml"), *(target / "projects").glob("*.yaml")]:
        text = re.sub(
            r"themes: \[([^\]]*)\]",
            lambda _match: f"themes: [theme_{next(counter)}]",
            path.read_text(encoding="utf-8"),
        )
        path.write_text(text, encoding="utf-8")
    return target


def corrupting_render_factory(real_render, fixer: FakeFixer | None = None):
    """Wrap `render_latex` so it emits a document that cannot compile."""

    def corrupting_render(state: AgentState) -> dict:
        result = real_render(state)
        if fixer is not None:
            # The fixer's job is to hand back exactly this clean version.
            fixer.replacement = result["tex_source"]
        broken = result["tex_source"].replace(
            "\\begin{document}", "\\begin{document}\n" + BROKEN_MARKER
        )
        return {"tex_source": broken}

    return corrupting_render


# ===========================================================================
# Structure
# ===========================================================================


def test_graph_compiles_and_contains_every_node() -> None:
    nodes = set(build_graph().get_graph().nodes)
    for name in [
        "parse_jd", "retrieve", "score", "select", "tailor", "verify",
        "render", "compile", "inspect", "fix_latex", "shrink_budget",
        "note_overfull", "finalize",
    ]:  # fmt: skip
        assert name in nodes, f"missing node {name}"


def test_mermaid_shows_both_cycles() -> None:
    """The diagram must show the loops, or it is not describing this system."""
    mermaid = build_graph().get_graph().draw_mermaid()

    assert "verify" in mermaid and "tailor" in mermaid  # grounding cycle
    assert "fix_latex --> compile" in mermaid  # repair cycle
    assert "shrink_budget --> select" in mermaid  # page-overflow cycle
    assert "note_overfull --> tailor" in mermaid  # overfull cycle


def test_caps_are_the_documented_values() -> None:
    assert MAX_GROUNDING_ATTEMPTS == 2
    assert MAX_LAYOUT_ATTEMPTS == 3


# ===========================================================================
# Routing, tested directly -- these functions are pure
# ===========================================================================


def _state(**overrides: Any) -> AgentState:
    base = initial_state(RAW_JD, PROFILE_EXAMPLE, RunOptions())
    base.update(overrides)
    return base


def test_route_compile_error_goes_to_fix_latex() -> None:
    state = _state(compile_log="! Undefined control sequence.\nl.7 \\nope", pdf_path=None)
    assert route_after_inspect(state) == "fix_latex"


def test_route_two_pages_goes_to_reselect() -> None:
    state = _state(compile_log="ok", pdf_path="x.pdf", page_count=2)
    assert route_after_inspect(state) == "reselect"


def test_route_overfull_goes_to_retailor() -> None:
    state = _state(
        compile_log="Overfull \\hbox (12.0pt too wide) in paragraph at lines 1--2",
        pdf_path="x.pdf",
        page_count=1,
    )
    assert route_after_inspect(state) == "retailor"


def test_route_clean_goes_to_finalize() -> None:
    state = _state(compile_log="fine", pdf_path="x.pdf", page_count=1)
    assert route_after_inspect(state) == "layout_ok"


def test_route_skips_the_letter_when_disabled() -> None:
    state = _state(compile_log="fine", pdf_path="x.pdf", page_count=1)
    state["options"] = RunOptions(write_cover_letter=False)
    assert route_after_inspect(state) == "skip_letter"


def test_route_respects_the_layout_cap() -> None:
    """At the cap, a still-broken document finalizes rather than looping again."""
    state = _state(
        compile_log="! Undefined control sequence.",
        pdf_path=None,
        layout_attempts=MAX_LAYOUT_ATTEMPTS,
    )
    assert route_after_inspect(state) == "layout_ok"


def test_route_after_verify_loops_while_bullets_are_unverified() -> None:
    state = _state(selected=["a", "b"], tailored=[], grounding_attempts=0)
    assert route_after_verify(state) == "tailor"


def test_route_after_verify_stops_at_the_grounding_cap() -> None:
    state = _state(selected=["a"], tailored=[], grounding_attempts=MAX_GROUNDING_ATTEMPTS + 1)
    assert route_after_verify(state) == "render"


def test_route_after_verify_ignores_dropped_bullets() -> None:
    """A dropped bullet is settled -- it must not keep the cycle spinning."""
    state = _state(selected=["a"], tailored=[], dropped_bullets=["a"], grounding_attempts=1)
    assert route_after_verify(state) == "render"


# ===========================================================================
# M5 DoD 1: a LaTeX syntax error is auto-repaired
# ===========================================================================


@requires_latex
def test_injected_latex_error_is_repaired(
    example_profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec 8: "inject a deliberate LaTeX syntax error -> auto-repaired within 3 iterations".

    Nothing about the outcome is faked: tectonic really fails, `inspect` really
    finds the `!` line, routing really sends it to `fix_latex`, and the repaired
    document really has to compile before the run can finish.
    """
    from resume_agent.graph import build as build_module

    fixer = FakeFixer()
    monkeypatch.setattr(
        build_module,
        "render_latex",
        corrupting_render_factory(build_module.render_latex, fixer),
    )

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(),
        judge_llm=FakeJudge(),
        fix_llm=fixer,
    )
    final = graph.invoke(seeded(PROFILE_EXAMPLE, tmp_path), {"recursion_limit": 60})

    assert fixer.call_count >= 1, "the repair node never ran"
    assert final.get("page_count") == 1, "the document never compiled cleanly after repair"
    assert final.get("layout_attempts", 0) <= MAX_LAYOUT_ATTEMPTS
    assert BROKEN_MARKER not in (final.get("tex_source") or "")


@requires_latex
def test_unrepairable_latex_stops_at_the_cap(
    example_profile: Profile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Spec 8: "both loops respect their caps and fail gracefully at the limit".

    A fixer that never fixes anything must stop after 3 attempts and finalize
    with what it has -- not spin, and not raise.
    """
    from resume_agent.graph import build as build_module

    monkeypatch.setattr(
        build_module, "render_latex", corrupting_render_factory(build_module.render_latex)
    )

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(),
        judge_llm=FakeJudge(),
        fix_llm=FakeFixer(replacement=""),  # returns nothing, forever
    )

    with caplog.at_level(logging.ERROR):
        final = graph.invoke(seeded(PROFILE_EXAMPLE, tmp_path), {"recursion_limit": 100})

    assert final.get("out_dir"), "did not reach finalize -- the cap failed to fire"
    assert final.get("layout_attempts", 0) <= MAX_LAYOUT_ATTEMPTS + 1
    assert any("cap" in record.getMessage().lower() for record in caplog.records)


# ===========================================================================
# M5 DoD 2: a two-page overflow converges to one page
# ===========================================================================


@requires_latex
def test_two_page_overflow_converges_to_one_page(
    oversized_profile_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Spec 8: "force a 2-page overflow -> converges to 1 page".

    The starting budget is measured, not guessed. Compiling this profile at a
    range of budgets shows where the page actually breaks:

        budget 26 -> 16 bullets, 26 est. lines -> 1 page
        budget 30 -> 17 bullets, 28 est. lines -> 1 page
        budget 34 -> 20 bullets, 34 est. lines -> 2 pages   <- breaks here
        budget 38 -> 21 bullets, 35 est. lines -> 2 pages

    So 40 genuinely overflows, and the 8% shrink walks it back 40 -> 36 -> 33,
    reaching one page in two iterations -- inside the cap of three.

    Nothing is faked about the outcome. tectonic compiles, pypdf counts the
    pages, and the loop keeps cutting until the real page count is 1.
    """
    profile = load_profile(oversized_profile_dir)
    assert len(profile.all_bullets()) == 21, "fixture changed; re-measure the budgets above"

    starting_budget = 40
    state = seeded(oversized_profile_dir, tmp_path, retrieval_k=30)
    state["line_budget"] = starting_budget

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(profile_path=str(oversized_profile_dir)),
        judge_llm=FakeJudge(),
        fix_llm=FakeFixer(),
    )

    with caplog.at_level(logging.WARNING):
        final = graph.invoke(state, {"recursion_limit": 100})

    assert final.get("page_count") == 1, (
        f"did not converge: {final.get('page_count')} pages after "
        f"{final.get('layout_attempts')} attempts"
    )
    assert final["line_budget"] < starting_budget, "the budget was never reduced"
    assert final.get("layout_attempts", 0) <= MAX_LAYOUT_ATTEMPTS
    assert any("cutting the line budget" in r.getMessage() for r in caplog.records)


@requires_latex
def test_a_clean_run_reaches_finalize_with_artifacts(
    example_profile: Profile, tmp_path: Path
) -> None:
    """The happy path, end to end, with a real PDF and a real run.json."""

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(),
        judge_llm=FakeJudge(),
        fix_llm=FakeFixer(),
    )
    final = graph.invoke(seeded(PROFILE_EXAMPLE, tmp_path), {"recursion_limit": 60})

    run_dir = Path(final["out_dir"])
    assert (run_dir / "resume.pdf").is_file()
    assert (run_dir / "resume.tex").is_file()

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["job_spec"]["company"] == "Acme Corp"
    assert record["selected_bullet_ids"]
    # Spec 5: run.json must answer "which bullets did I actually send?".
    assert record["prompt_versions"]["tailor_bullets"]
    assert record["models"]["generation"]


# ===========================================================================
# M5 DoD 3: the grounding cycle drops rather than ships
# ===========================================================================


@requires_latex
def test_ungroundable_bullets_are_dropped_and_the_run_continues(
    example_profile: Profile, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A writer that always fabricates must produce a resume WITHOUT those bullets.

    Not a crash, not a hang, and above all not the fabricated text on the page.
    """

    graph = build_graph(
        score_llm=FakeScorer(),
        tailor_llm=FakeWriter(corrupt=True),
        judge_llm=FakeJudge(),
        fix_llm=FakeFixer(),
    )

    with caplog.at_level(logging.ERROR):
        final = graph.invoke(seeded(PROFILE_EXAMPLE, tmp_path), {"recursion_limit": 100})

    assert final.get("dropped_bullets"), "nothing dropped despite constant fabrication"
    assert final.get("tailored") == [], "shipped a bullet that never passed verification"
    assert any("DROPPING bullet" in r.getMessage() for r in caplog.records)
    # 7919 was the invented number; it must appear nowhere in the output.
    assert "7919" not in (final.get("tex_source") or "")
