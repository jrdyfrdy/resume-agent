"""The graph. Spec 2, and the only place edges are defined (CLAUDE.md rule 6).

    parse_jd -> retrieve -> score -> select -> tailor -> verify
                                       ^          ^        |
                                       |          +--------+  grounding cycle, cap 2
                                       |                      then drop the bullet
                                       |
                render -> compile -> inspect                  layout cycle, cap 3
                             ^          |                     then fail with the
                             +-- fix ---+                      best artifact so far
                                        -> finalize

Nodes import nothing from each other; every routing decision is a function in
this file. That is what makes the topology readable as one artifact and lets
`graph.get_graph().draw_mermaid()` produce a diagram that is actually true.

**Two cycles, two caps** (CLAUDE.md rule 7):

* grounding, 2 retries -- then the offending bullet is dropped and the run
  continues without it. Never ship an unverified claim.
* layout, 3 retries -- then stop and hand back the best artifact produced so
  far, with the failure recorded in `errors`. Never spin.

The counters live in the state as plain ints (last-write-wins), which is exactly
why spec 4 warns against giving them an `operator.add` reducer: an accumulating
`layout_attempts` would become `[1, 1, 1]`, stay truthy forever, and the cap
would never fire.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from resume_agent.analyze import budget_for_profile
from resume_agent.graph.nodes.finalize import finalize
from resume_agent.graph.nodes.fix_latex import fix_latex
from resume_agent.graph.nodes.parse_jd import parse_job_description
from resume_agent.graph.nodes.render import compile_pdf, inspect_pdf, render_latex
from resume_agent.graph.nodes.retrieve import retrieve_evidence
from resume_agent.graph.nodes.score import build_fit_report, score_fit
from resume_agent.graph.nodes.select import select_content
from resume_agent.graph.nodes.tailor import tailor_bullets
from resume_agent.graph.nodes.verify import (
    MAX_GROUNDING_ATTEMPTS,
    drop_unverified,
    verify_grounding,
)
from resume_agent.graph.state import AgentState, RunOptions
from resume_agent.kb.index import ProfileIndex
from resume_agent.kb.loader import load_profile
from resume_agent.kb.retriever import HybridRetriever
from resume_agent.latex.inspect import find_overfull_boxes, first_latex_error
from resume_agent.latex.layout import shrink_budget

logger = logging.getLogger(__name__)

# Spec 5: "layout cycle (`inspect_output` -> `select_content` or `render_latex`),
# max 3 retries".
MAX_LAYOUT_ATTEMPTS = 3


# ===========================================================================
# Nodes -- thin (state) -> dict wrappers over the tested M2-M4 functions.
# ===========================================================================


def _profile(state: AgentState):
    return load_profile(Path(state["profile_path"]))


def node_parse_jd(state: AgentState) -> dict:
    job, cache_hit = parse_job_description(state["raw_jd"], use_cache=state["options"].use_cache)
    logger.info("parse_jd: %s at %s (%s)", job.title, job.company, "cached" if cache_hit else "new")
    return {"job_spec": job}


def node_retrieve(state: AgentState) -> dict:
    profile = _profile(state)
    profile_dir = Path(state["profile_path"])
    index, _ = ProfileIndex.open(profile, profile_dir)
    try:
        retriever = HybridRetriever(index, profile)
        candidates = retrieve_evidence(state["job_spec"], retriever, k=state["options"].retrieval_k)
    finally:
        index.close()
    logger.info("retrieve: %d candidate bullets", len(candidates.bullet_ids))
    return {"candidates": candidates.bullet_ids}


def node_score(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    profile = _profile(state)
    candidates = state.get("candidates", [])
    matches = score_fit(
        state["job_spec"], candidates, profile, llm=llm, use_cache=state["options"].use_cache
    )
    return {"evidence": matches, "fit_report": build_fit_report(state["job_spec"], matches)}


def node_select(state: AgentState) -> dict:
    """The knapsack. Re-entered by the layout loop with a smaller budget."""
    profile = _profile(state)
    budget = state.get("line_budget") or budget_for_profile(profile)

    result = select_content(
        state["job_spec"],
        state.get("evidence", []),
        profile,
        budget,
        strict=state["options"].strict,
    )
    logger.info(
        "select: %d bullets, %d/%d lines",
        len(result.selected_bullet_ids),
        result.total_estimated_lines,
        budget,
    )
    return {
        "selected": result.selected_bullet_ids,
        "line_budget": budget,
        # Re-entering selection invalidates the previous rewrites, and resets the
        # grounding budget: these are different bullets facing the gate fresh.
        "tailored": [],
        "grounding_attempts": 0,
    }


def node_tailor(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """One rewriting pass. The *cycle* is an edge, not a loop in here."""
    profile = _profile(state)
    already_done = {t.source_id for t in state.get("tailored", [])}
    pending = [
        profile.bullet_by_id(bid)
        for bid in state.get("selected", [])
        if bid not in already_done and bid not in state.get("dropped_bullets", [])
    ]
    if not pending:
        return {}

    # Critiques accumulate in state across the cycle; group them by bullet so
    # each rewrite sees only the objections raised against it.
    by_bullet: dict[str, list[str]] = {}
    for entry in state.get("critiques", []):
        bullet_id, _, text = entry.partition(": ")
        by_bullet.setdefault(bullet_id, []).append(text)

    fresh = tailor_bullets(
        state["job_spec"],
        pending,
        profile,
        critiques=by_bullet,
        llm=llm,
        use_cache=state["options"].use_cache,
    )
    return {"tailored": [*state.get("tailored", []), *fresh]}


def node_verify(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """The fabrication gate. Rejected bullets go back round; at the cap they go."""
    profile = _profile(state)
    vocabulary = profile.skill_vocabulary()
    attempts = state.get("grounding_attempts", 0)

    kept, critiques, dropped = [], [], []
    for candidate in state.get("tailored", []):
        source = profile.bullet_by_id(candidate.source_id)
        result = verify_grounding(
            candidate, source, vocabulary, judge=state["options"].use_judge, llm=llm
        )
        if result.passed:
            kept.append(candidate)
            continue

        if attempts >= MAX_GROUNDING_ATTEMPTS:
            # Cap reached. Spec 5: drop it, loudly. Not shipped.
            drop_unverified(source, [result.critique or "rejected"])
            dropped.append(candidate.source_id)
        else:
            critiques.append(f"{candidate.source_id}: {result.critique}")

    return {
        "tailored": kept,
        "grounding_attempts": attempts + 1,
        "critiques": critiques,
        "dropped_bullets": dropped,
    }


def node_fix_latex(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    return fix_latex(state, llm=llm)


# ===========================================================================
# Routing -- every conditional edge's decision function.
# ===========================================================================


def route_after_verify(state: AgentState) -> str:
    """Grounding cycle. Back to tailoring while anything is still unverified."""
    profile = _profile(state)
    verified = {t.source_id for t in state.get("tailored", [])}
    dropped = set(state.get("dropped_bullets", []))
    outstanding = [
        bid for bid in state.get("selected", []) if bid not in verified and bid not in dropped
    ]

    if outstanding and state.get("grounding_attempts", 0) <= MAX_GROUNDING_ATTEMPTS:
        logger.info("grounding cycle: %d bullet(s) still unverified", len(outstanding))
        return "tailor"

    del profile
    return "render"


def route_after_inspect(state: AgentState) -> str:
    """Layout cycle. Spec 5's routing table, in order of severity.

    compile error   -> fix_latex
    pages > 1       -> shrink the budget, reselect
    overfull hboxes -> re-tailor the offending bullets shorter
    clean           -> forward
    """
    attempts = state.get("layout_attempts", 0)
    log = state.get("compile_log") or ""

    if attempts >= MAX_LAYOUT_ATTEMPTS:
        logger.error(
            "layout cycle hit its cap of %d attempts; finalizing the best artifact so far",
            MAX_LAYOUT_ATTEMPTS,
        )
        return "finalize"

    if state.get("pdf_path") is None or first_latex_error(log):
        return "fix_latex"

    if (state.get("page_count") or 1) > 1:
        return "reselect"

    if find_overfull_boxes(log):
        return "retailor"

    return "finalize"


def node_shrink_budget(state: AgentState) -> dict:
    """Spec 5: "pages > 1 -> reduce `line_budget` by 8%, back to `select_content`"."""
    current = state.get("line_budget", 0)
    reduced = shrink_budget(current)
    logger.warning(
        "layout: %s pages, cutting the line budget %d -> %d",
        state.get("page_count"),
        current,
        reduced,
    )
    return {"line_budget": reduced, "layout_attempts": state.get("layout_attempts", 0) + 1}


def node_note_overfull(state: AgentState) -> dict:
    """Spec 5: "overfull hboxes -> back to `tailor_bullets` with 'shorten bullet X'"."""
    boxes = find_overfull_boxes(state.get("compile_log") or "")
    logger.warning("layout: %d overfull hbox(es); re-tailoring shorter", len(boxes))
    return {
        "tailored": [],
        "grounding_attempts": 0,
        "layout_attempts": state.get("layout_attempts", 0) + 1,
        "critiques": [
            f"content ran into the margin by {box.points_over:.1f}pt; "
            f"shorten the affected bullet by roughly 15 characters"
            for box in boxes[:3]
        ],
    }


# ===========================================================================
# Assembly
# ===========================================================================


def build_graph(
    *,
    score_llm: BaseChatModel | None = None,
    tailor_llm: BaseChatModel | None = None,
    judge_llm: BaseChatModel | None = None,
    fix_llm: BaseChatModel | None = None,
):
    """Assemble and compile the graph.

    The `*_llm` parameters exist so the loop tests can drive the real routing
    with fake models -- proving the cycles converge without spending money or
    depending on a model behaving a particular way. Nothing else injects them.
    """
    builder = StateGraph(AgentState)

    builder.add_node("parse_jd", node_parse_jd)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("score", lambda state: node_score(state, llm=score_llm))
    builder.add_node("select", node_select)
    builder.add_node("tailor", lambda state: node_tailor(state, llm=tailor_llm))
    builder.add_node("verify", lambda state: node_verify(state, llm=judge_llm))
    builder.add_node("render", render_latex)
    builder.add_node("compile", compile_pdf)
    builder.add_node("inspect", inspect_pdf)
    builder.add_node("fix_latex", lambda state: node_fix_latex(state, llm=fix_llm))
    builder.add_node("shrink_budget", node_shrink_budget)
    builder.add_node("note_overfull", node_note_overfull)
    builder.add_node("finalize", finalize)

    # --- the straight line ---------------------------------------------------
    builder.add_edge(START, "parse_jd")
    builder.add_edge("parse_jd", "retrieve")
    builder.add_edge("retrieve", "score")
    builder.add_edge("score", "select")
    builder.add_edge("select", "tailor")
    builder.add_edge("tailor", "verify")

    # --- grounding cycle (spec 2), cap 2 ------------------------------------
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"tailor": "tailor", "render": "render"},
    )

    builder.add_edge("render", "compile")
    builder.add_edge("compile", "inspect")

    # --- layout cycle (spec 2), cap 3 ---------------------------------------
    builder.add_conditional_edges(
        "inspect",
        route_after_inspect,
        {
            "fix_latex": "fix_latex",
            "reselect": "shrink_budget",
            "retailor": "note_overfull",
            "finalize": "finalize",
        },
    )
    # A repaired document goes straight back to the compiler -- the compiler is
    # the only thing whose opinion of the repair counts.
    builder.add_edge("fix_latex", "compile")
    builder.add_edge("shrink_budget", "select")
    builder.add_edge("note_overfull", "tailor")

    builder.add_edge("finalize", END)

    return builder.compile()


def initial_state(
    raw_jd: str,
    profile_path: str | Path,
    options: RunOptions | None = None,
) -> AgentState:
    """A state seeded with the inputs and zeroed counters.

    The counters must start at 0 rather than be absent: `route_after_inspect`
    compares them against a cap, and a missing key would make the first
    comparison depend on a `.get` default living far from the cap itself.
    """
    return {
        "raw_jd": raw_jd,
        "profile_path": str(profile_path),
        "options": options or RunOptions(),
        "line_budget": 0,
        "grounding_attempts": 0,
        "layout_attempts": 0,
        "tailored": [],
        "selected": [],
        "evidence": [],
        "candidates": [],
        "dropped_bullets": [],
        "critiques": [],
        "errors": [],
    }
