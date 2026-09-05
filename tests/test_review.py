"""Human-in-the-loop and checkpoint durability. M7's definition of done.

Spec 8: "`--interactive` pauses, survives process restart via checkpointer,
resumes correctly."

The load-bearing test is `test_a_paused_run_survives_a_fresh_connection`. A
pytest process cannot usefully fork itself, but it can do the thing that
actually matters: build a **completely new graph object with a new SqliteSaver
on a new sqlite3 connection** to the same file, and resume from that. Nothing is
shared but the file on disk -- which is exactly what a second `resume-agent`
invocation has.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from resume_agent.graph.checkpoint import open_checkpointer, thread_config, thread_id_for
from resume_agent.graph.nodes.review import (
    MAX_REVISION_ROUNDS,
    build_review_payload,
    human_review,
    note_revision_cap,
    route_after_review,
)
from resume_agent.graph.state import AgentState, RunOptions
from resume_agent.models.fit import FitReport
from resume_agent.models.job import JobSpec, JobSpecFields, Requirement
from resume_agent.models.resume import TailoredBullet

from .conftest import PROFILE_EXAMPLE

RAW_JD = "Backend engineer wanted. Caching and Redis."


def make_state(**overrides: Any) -> AgentState:
    job = JobSpec.from_fields(
        JobSpecFields(
            company="Acme Corp",
            title="Backend Engineer",
            seniority="mid",
            domain="payments",
            requirements=[
                Requirement(text="kubernetes", category="tool", weight=5, is_must_have=True)
            ],
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        RAW_JD,
    )
    state: AgentState = {
        "raw_jd": RAW_JD,
        "profile_path": str(PROFILE_EXAMPLE),
        "options": RunOptions(interactive=False),
        "job_spec": job,
        "fit_report": FitReport(overall_fit=0.5, gaps=job.requirements, recommendation="stretch"),
        "tailored": [TailoredBullet(source_id="a.b1", text="Did a thing", estimated_lines=1)],
        "dropped_bullets": [],
        "critiques": [],
        "errors": [],
        "revision_rounds": 0,
        "page_count": 1,
    }
    state.update(overrides)
    return state


# ===========================================================================
# Thread ids -- stable across processes, which is the whole point
# ===========================================================================


def test_thread_id_is_stable_for_the_same_inputs() -> None:
    """A uuid would be correct once and useless the moment the process exits."""
    assert thread_id_for(RAW_JD, "profile.example") == thread_id_for(RAW_JD, "profile.example")


def test_thread_id_changes_with_the_posting() -> None:
    assert thread_id_for(RAW_JD, "p") != thread_id_for("A different posting", "p")


def test_thread_id_changes_with_the_profile() -> None:
    assert thread_id_for(RAW_JD, "profile.example") != thread_id_for(RAW_JD, "profile")


def test_thread_id_ignores_the_jd_path() -> None:
    """Derived from content, so renaming the file does not orphan a paused run."""
    assert thread_id_for(RAW_JD, "profile") == thread_id_for(RAW_JD + "\n", "profile")


def test_checkpointer_creates_its_tables(tmp_path: Path) -> None:
    saver, connection = open_checkpointer(tmp_path / "cp.db")
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "checkpoints" in tables
        assert isinstance(saver, SqliteSaver)
    finally:
        connection.close()


# ===========================================================================
# The review node
# ===========================================================================


def test_non_interactive_runs_do_not_pause() -> None:
    """Spec 5: gate it "so batch runs don't block"."""
    assert human_review(make_state())["review_action"] == "approve"


def test_review_payload_contains_everything_spec_5_lists() -> None:
    payload = build_review_payload(make_state())
    for key in [
        "recommendation", "gaps", "selected_bullets", "cover_letter", "pdf_path",
    ]:  # fmt: skip
        assert key in payload


def test_review_payload_separates_must_have_gaps() -> None:
    """What decides whether to send this at all, so it is not buried in the list."""
    payload = build_review_payload(make_state())
    assert payload["must_have_gaps"] == ["kubernetes"]


def test_review_payload_is_json_serialisable() -> None:
    """It crosses a checkpointer; a Pydantic model handed over whole would not survive."""
    import json

    json.dumps(build_review_payload(make_state()))


# --- routing ---------------------------------------------------------------


def test_approve_goes_to_finalize() -> None:
    assert route_after_review({"review_action": "approve"}) == "approve"


def test_revise_goes_back_to_tailoring() -> None:
    assert route_after_review({"review_action": "revise", "revision_rounds": 1}) == "revise"


def test_revision_rounds_are_capped() -> None:
    """CLAUDE.md rule 7. Spec 5 gives the action but no cap."""
    state = {"review_action": "revise", "revision_rounds": MAX_REVISION_ROUNDS}
    assert route_after_review(state) == "cap"


def test_hitting_the_cap_is_recorded() -> None:
    result = note_revision_cap({})
    assert result["review_action"] == "approve"
    assert result["errors"]


# ===========================================================================
# M7 DoD: pause, survive a restart, resume correctly
# ===========================================================================


def _review_only_graph(checkpointer):
    """A minimal graph around the real `human_review` node.

    Deliberately not the full agent: this test is about the checkpointer and the
    interrupt, and running parse/score/tailor would drag four fake models into a
    test whose subject is persistence.
    """
    builder = StateGraph(AgentState)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", lambda state: {"out_dir": "done"})
    builder.add_edge(START, "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"approve": "finalize", "revise": "finalize", "cap": "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer)


def test_interactive_run_pauses(tmp_path: Path) -> None:
    saver, connection = open_checkpointer(tmp_path / "cp.db")
    try:
        graph = _review_only_graph(saver)
        config = thread_config("t1")

        result = graph.invoke(make_state(options=RunOptions(interactive=True)), config)

        assert "__interrupt__" in result, "an interactive run did not pause"
        payload = result["__interrupt__"][0].value
        assert payload["company"] == "Acme Corp"
        assert graph.get_state(config).next, "the graph does not know where to resume"
    finally:
        connection.close()


def test_a_paused_run_survives_a_fresh_connection(tmp_path: Path) -> None:
    """M7's definition of done, as directly as a single process can test it.

    The second half shares **nothing** with the first but the file on disk: new
    sqlite3 connection, new SqliteSaver, new StateGraph, new compiled graph.
    That is precisely the situation a second `resume-agent` invocation is in.
    """
    db = tmp_path / "cp.db"
    config = thread_config("t-restart")

    # --- "process" 1: start and pause ------------------------------------
    saver_a, connection_a = open_checkpointer(db)
    graph_a = _review_only_graph(saver_a)
    paused = graph_a.invoke(make_state(options=RunOptions(interactive=True)), config)
    assert "__interrupt__" in paused
    connection_a.close()
    del graph_a, saver_a

    # --- "process" 2: reopen and resume -----------------------------------
    saver_b, connection_b = open_checkpointer(db)
    try:
        graph_b = _review_only_graph(saver_b)

        # The state is still there, and still knows it is waiting.
        snapshot = graph_b.get_state(config)
        assert snapshot.next == ("human_review",)
        assert snapshot.values["job_spec"].company == "Acme Corp", "state did not persist"

        final = graph_b.invoke(Command(resume={"action": "approve"}), config)

        assert final["review_action"] == "approve"
        assert final["out_dir"] == "done", "the run did not continue past the pause"
    finally:
        connection_b.close()


def test_resuming_with_revise_feeds_the_notes_back(tmp_path: Path) -> None:
    """A human note becomes a critique -- the same shape a verifier objection has."""
    db = tmp_path / "cp.db"
    config = thread_config("t-revise")

    saver_a, connection_a = open_checkpointer(db)
    graph_a = _review_only_graph(saver_a)
    graph_a.invoke(make_state(options=RunOptions(interactive=True)), config)
    connection_a.close()

    saver_b, connection_b = open_checkpointer(db)
    try:
        graph_b = _review_only_graph(saver_b)
        final = graph_b.invoke(
            Command(resume={"action": "revise", "notes": "lead with the Kafka work"}),
            config,
        )

        assert final["review_action"] == "revise"
        assert final["revision_rounds"] == 1
        assert any("lead with the Kafka work" in c for c in final["critiques"])
        # The previous rewrites are stale once a revision is requested.
        assert final["tailored"] == []
    finally:
        connection_b.close()


def test_two_postings_do_not_share_a_paused_run(tmp_path: Path) -> None:
    """Distinct thread ids, so resuming one cannot pick up the other's state."""
    db = tmp_path / "cp.db"
    saver, connection = open_checkpointer(db)
    try:
        graph = _review_only_graph(saver)
        first = thread_config(thread_id_for("posting one", "profile.example"))
        second = thread_config(thread_id_for("posting two", "profile.example"))

        graph.invoke(make_state(options=RunOptions(interactive=True)), first)

        assert graph.get_state(first).next, "the first run should be paused"
        assert not graph.get_state(second).next, "an unrelated posting appeared paused"
    finally:
        connection.close()


def test_checkpoint_database_is_a_real_file(tmp_path: Path) -> None:
    """InMemorySaver would pass every other test here and fail the milestone."""
    db = tmp_path / "cp.db"
    saver, connection = open_checkpointer(db)
    try:
        graph = _review_only_graph(saver)
        graph.invoke(
            make_state(options=RunOptions(interactive=True)), thread_config("t-file")
        )
    finally:
        connection.close()

    assert db.is_file() and db.stat().st_size > 0

    # Readable by a plain sqlite3 client -- nothing in-process required.
    with sqlite3.connect(db) as raw:
        count = raw.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    assert count > 0, "nothing was actually written to disk"
