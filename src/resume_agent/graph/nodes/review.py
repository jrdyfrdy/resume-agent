"""The human-in-the-loop pause. Spec 5, `human_review`.

    decision = interrupt({
        "fit_report": ..., "selected_bullets": ..., "cover_letter": ...,
        "pdf_path": ..., "gaps": ...,
    })

    Resume with `Command(resume={"action": "approve" | "revise", "notes": "..."})`.
    Requires a checkpointer and a stable `thread_id`. Gate behind `--interactive`
    so batch runs don't block.

**How `interrupt` actually behaves**, because it is easy to write code that
assumes otherwise: it raises out of the node, LangGraph persists the state to
the checkpointer, and `invoke` returns. On resume, **the node runs again from
the top** and the `interrupt(...)` call returns the value supplied by
`Command(resume=...)`. So everything above the interrupt executes twice, and
anything with a side effect must not live there.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from resume_agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# CLAUDE.md rule 7. Spec 5 gives the revise action but no cap; without one a
# caller that always answers "revise" loops forever. Three rounds is already
# more than an interactive session wants, so this is a safety net against a
# script rather than a limit on anyone's patience.
MAX_REVISION_ROUNDS = 3


def build_review_payload(state: AgentState) -> dict[str, Any]:
    """Exactly what spec 5 says to put in front of the human.

    Serialisable by construction: it crosses a checkpointer, so a Pydantic model
    handed over whole would have to survive a JSON round trip that nothing
    guarantees.
    """
    fit = state.get("fit_report")
    letter = state.get("cover_letter")

    return {
        "company": state["job_spec"].company if state.get("job_spec") else None,
        "title": state["job_spec"].title if state.get("job_spec") else None,
        "recommendation": fit.recommendation if fit else None,
        "overall_fit": fit.overall_fit if fit else None,
        # Spec 5 lists gaps separately from the fit report even though it
        # contains them -- because the gaps are the thing you actually read
        # before deciding whether to send this.
        "gaps": [r.text for r in fit.gaps] if fit else [],
        "must_have_gaps": [r.text for r in fit.must_have_gaps()] if fit else [],
        "selected_bullets": [t.text for t in state.get("tailored", [])],
        "dropped_bullets": state.get("dropped_bullets", []),
        "cover_letter": letter.body() if letter else None,
        "cover_letter_words": letter.word_count if letter else None,
        "pdf_path": state.get("pdf_path"),
        "page_count": state.get("page_count"),
        "revision_round": state.get("revision_rounds", 0),
    }


def human_review(state: AgentState) -> dict:
    """Pause for a human decision, or pass straight through.

    Gated on `--interactive` (spec 5) so that batch runs -- and every test that
    is not about this node -- do not block forever waiting for an answer nobody
    is there to give.
    """
    if not state["options"].interactive:
        return {"review_action": "approve"}

    # Nothing with a side effect above this line: on resume the node re-runs
    # from the top and everything here happens a second time.
    decision = interrupt(build_review_payload(state))

    action = (decision or {}).get("action", "approve")
    notes = (decision or {}).get("notes", "")
    logger.info("human_review: %s%s", action, f" -- {notes}" if notes else "")

    if action != "revise":
        return {"review_action": "approve"}

    return {
        "review_action": "revise",
        "revision_rounds": state.get("revision_rounds", 0) + 1,
        # Fed back as critiques so the tailoring node treats them exactly like a
        # verifier objection. A human note and a machine critique are the same
        # kind of thing to the writer: a reason the last attempt was not right.
        "critiques": [f"reviewer: {notes}"] if notes else ["reviewer asked for a revision"],
        # The previous rewrites are stale now.
        "tailored": [],
        "grounding_attempts": 0,
    }


def route_after_review(state: AgentState) -> str:
    """approve -> finalize; revise -> back to tailoring, until the cap."""
    if state.get("review_action") != "revise":
        return "approve"

    if state.get("revision_rounds", 0) >= MAX_REVISION_ROUNDS:
        logger.error(
            "human_review: %d revision rounds requested, which is the cap; finalizing "
            "the current draft. Edit the profile or the prompts rather than asking again.",
            state.get("revision_rounds", 0),
        )
        return "cap"

    return "revise"


def note_revision_cap(state: AgentState) -> dict:
    """Record hitting the revision cap, then continue to finalize."""
    return {
        "review_action": "approve",
        "errors": [
            f"reached the {MAX_REVISION_ROUNDS}-round revision cap; finalized the "
            f"current draft without further changes"
        ],
    }
