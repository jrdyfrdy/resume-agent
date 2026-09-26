"""The graph. Spec 2, and the only place edges are defined (CLAUDE.md rule 6).

    parse_jd -> retrieve -> score -> select -> tailor -> verify
                                       ^          ^        |
                                       |          +--------+  grounding cycle, cap 2
                                       |                      then drop the bullet
                                       |
                render -> compile -> inspect                  layout cycle, cap 3
                             ^          |                     then fail with the
                             +-- fix ---+                      best artifact so far
                                        v
                              cover_letter -> finalize        (a SUBGRAPH, spec 10;
                                                               its own retry cycle,
                                                               cap 2, invisible here)

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
from resume_agent.graph.nodes.cover_letter import build_cover_letter_subgraph
from resume_agent.graph.nodes.finalize import finalize
from resume_agent.graph.nodes.fix_latex import fix_latex
from resume_agent.graph.nodes.parse_jd import parse_job_description
from resume_agent.graph.nodes.render import compile_pdf, inspect_pdf, render_latex
from resume_agent.graph.nodes.retrieve import retrieve_evidence
from resume_agent.graph.nodes.review import human_review, note_revision_cap, route_after_review
from resume_agent.graph.nodes.score import build_fit_report, score_fit
from resume_agent.graph.nodes.select import select_content
from resume_agent.graph.nodes.summary import write_summary
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
from resume_agent.sections import Stage, career_stage, this_month

logger = logging.getLogger(__name__)

# Spec 5: "layout cycle (`inspect_output` -> `select_content` or `render_latex`),
# max 3 retries".
MAX_LAYOUT_ATTEMPTS = 3


# ===========================================================================
# Nodes -- thin (state) -> dict wrappers over the tested M2-M4 functions.
# ===========================================================================


def _profile(state: AgentState):
    return load_profile(Path(state["profile_path"]))


def _stage(state: AgentState) -> Stage:
    """The section order this run renders with.

    Resolved once here rather than recomputed per node: the budget and the
    render have to agree about the shape of the page, and a stage derived
    twice from a "today" that crossed a month boundary mid-run would not.
    """
    return career_stage(
        _profile(state), this_month(), override=state["options"].layout
    )


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
    scoring: dict = {}
    matches = score_fit(
        state["job_spec"],
        candidates,
        profile,
        llm=llm,
        use_cache=state["options"].use_cache,
        details=scoring,
    )
    return {
        "evidence": matches,
        "fit_report": build_fit_report(state["job_spec"], matches),
        "scoring": scoring,
    }


def node_select(state: AgentState) -> dict:
    """The knapsack. Re-entered by the layout loop with a smaller budget."""
    profile = _profile(state)
    options = state["options"]
    budget = state.get("line_budget") or budget_for_profile(
        profile,
        pages=options.max_pages,
        stage=_stage(state),
        reserve_summary=options.summary,
    )

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


def node_summary(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Write the opening summary, if this run asked for one.

    Placed after verify and before render because it draws on the *tailored*
    text: the summary should describe the sentences that will actually appear
    under it, not the canonical ones they were rewritten from.
    """
    if not state["options"].summary:
        return {"summary_text": ""}

    tailored = {t.source_id: t.text for t in state.get("tailored", [])}
    dropped = set(state.get("dropped_bullets", []))
    surviving = [bid for bid in state.get("selected", []) if bid not in dropped]

    job = state.get("job_spec")
    text, complaints = write_summary(
        _profile(state),
        getattr(job, "title", "") or "",
        surviving,
        tailored,
        llm=llm,
    )
    if text:
        logger.info("summary: %d chars", len(text))
    return {"summary_text": text, "errors": complaints if not text else []}


def route_after_inspect(state: AgentState) -> str:
    """Layout cycle. Spec 5's routing table, in order of severity.

    compile error   -> fix_latex
    over max_pages  -> shrink the budget, reselect
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
        return _after_layout(state)

    if state.get("pdf_path") is None or first_latex_error(log):
        return "fix_latex"

    if (state.get("page_count") or 1) > state["options"].max_pages:
        return "reselect"

    if find_overfull_boxes(log):
        return "retailor"

    return _after_layout(state)


def _after_layout(state: AgentState) -> str:
    """Where the run goes once the layout loop is done with it.

    Both exits from the layout loop come through here -- the clean one and the
    one that gave up at the cap -- so that `--no-cover-letter` cannot be honoured
    on one path and silently ignored on the other. (It was, briefly: the cap
    branch returned before the option was consulted.)

    Spec 2 puts the letter after this loop rather than beside it because the
    letter quotes the resume's verified bullets, and those are not settled until
    the layout stops changing them.
    """
    if not state["options"].write_cover_letter:
        return "skip_letter"
    return "layout_ok"


def route_after_cover_letter(state: AgentState) -> str:
    """Always forward, to the review gate.

    A conditional edge with one destination looks redundant, and is: it is here
    so that a future branch after the letter is a change to this function rather
    than a change to the graph's shape.

    The key names the *outcome*, not the destination. Naming destinations is how
    the layout loop ended up with an edge labelled "finalize" pointing at
    `cover_letter` when a node was inserted -- and the diagram is generated, so
    a stale label is visible to everyone.
    """
    return "review"


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
    checkpointer=None,
    score_llm: BaseChatModel | None = None,
    tailor_llm: BaseChatModel | None = None,
    judge_llm: BaseChatModel | None = None,
    summary_llm: BaseChatModel | None = None,
    fix_llm: BaseChatModel | None = None,
    letter_llm: BaseChatModel | None = None,
    letter_judge_llm: BaseChatModel | None = None,
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
    builder.add_node("summary", lambda state: node_summary(state, llm=summary_llm))
    builder.add_node("render", render_latex)
    builder.add_node("compile", compile_pdf)
    builder.add_node("inspect", inspect_pdf)
    builder.add_node("fix_latex", lambda state: node_fix_latex(state, llm=fix_llm))
    builder.add_node("shrink_budget", node_shrink_budget)
    builder.add_node("note_overfull", node_note_overfull)
    # A compiled subgraph added as a single node (spec 10). Its draft/verify
    # retry cycle is internal -- the parent calls one node and either gets a
    # letter or does not.
    builder.add_node(
        "cover_letter",
        build_cover_letter_subgraph(draft_llm=letter_llm, judge_llm=letter_judge_llm),
    )
    builder.add_node("human_review", human_review)
    builder.add_node("revision_cap", note_revision_cap)
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
        {"tailor": "tailor", "render": "summary"},
    )

    builder.add_edge("summary", "render")
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
            # Spec 2 puts the cover letter after the layout loop settles: it
            # quotes the resume's verified bullets, so it cannot be written
            # until they have stopped changing.
            "layout_ok": "cover_letter",
            "skip_letter": "human_review",
        },
    )
    # A repaired document goes straight back to the compiler -- the compiler is
    # the only thing whose opinion of the repair counts.
    builder.add_edge("fix_latex", "compile")
    builder.add_edge("shrink_budget", "select")
    builder.add_edge("note_overfull", "tailor")

    builder.add_conditional_edges(
        "cover_letter",
        route_after_cover_letter,
        {"review": "human_review"},
    )

    # --- human-in-the-loop (spec 5), revise capped at 3 rounds ---------------
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"approve": "finalize", "revise": "tailor", "cap": "revision_cap"},
    )
    builder.add_edge("revision_cap", "finalize")

    builder.add_edge("finalize", END)

    # A checkpointer is what makes a paused run survive the process that
    # started it (spec 8's M7). Without one, `interrupt` still pauses -- but
    # only until the interpreter exits.
    return builder.compile(checkpointer=checkpointer)


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
        "letter_critiques": [],
        "letter_attempts": 0,
        "letter_verified": False,
        "revision_rounds": 0,
        "review_action": "",
        "errors": [],
    }
