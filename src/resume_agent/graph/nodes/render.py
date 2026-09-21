"""The three artifact nodes: render, compile, inspect. Spec 5.

Thin `(state) -> dict` wrappers over the M0 machinery, which already does the
work and is already tested. They live here rather than in `latex/` because a
node knows about `AgentState` and `latex/` deliberately does not -- keeping the
LaTeX layer usable by `resume-agent build`, which has no graph at all.

Spec 5: each node "returns only the keys it changed".
"""

from __future__ import annotations

import logging
from pathlib import Path

from resume_agent.graph.state import AgentState
from resume_agent.kb.loader import load_profile
from resume_agent.latex.compile import compile_tex
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.latex.inspect import find_overfull_boxes, inspect_output
from resume_agent.sections import career_stage, this_month

logger = logging.getLogger(__name__)

RESUME_TEMPLATE = "jake_resume.tex.j2"


def render_latex(state: AgentState) -> dict:
    """Selected + tailored bullets -> LaTeX source. Spec 5, `render_latex`."""
    profile = load_profile(Path(state["profile_path"]))

    # A bullet that failed grounding was dropped, so it has no rewrite. Rendering
    # its canonical text instead would put unverified-but-truthful content on the
    # page; rendering nothing is what the drop meant. Selection ids are filtered
    # to those that actually survived tailoring.
    tailored_text = {t.source_id: t.text for t in state.get("tailored", [])}
    surviving = [bid for bid in state.get("selected", []) if bid in tailored_text]

    tex_source = render_template(
        RESUME_TEMPLATE,
        build_resume_context(
            profile,
            selected_ids=surviving,
            tailored_text=tailored_text,
            # Resolved from the same options the budget used, so the page the
            # knapsack costed and the page that renders are the same shape.
            stage=career_stage(profile, this_month(), override=state["options"].layout),
            summary=state.get("summary_text") or "",
        ),
    )
    return {"tex_source": tex_source}


def compile_pdf(state: AgentState) -> dict:
    """LaTeX source -> PDF + log. Spec 5, `compile_pdf`.

    Never raises: "a failed compile is data for the next node, not an
    exception". `inspect_output` is what reads that data.
    """
    options = state["options"]
    out_dir = Path(options.out_dir) / "work"

    result = compile_tex(state["tex_source"] or "", out_dir, job_name="resume")
    return {
        "compile_log": result.log,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
    }


def inspect_pdf(state: AgentState) -> dict:
    """Read the environment's verdict. Spec 5, `inspect_output`.

    This is the node that makes the project agentic rather than a chatbot with
    extra steps: `page_count` and the overfull-box list are ground truth from
    outside the model, and every routing decision downstream is made from them.
    """
    from resume_agent.latex.compile import CompileResult

    log = state.get("compile_log") or ""
    pdf_path = state.get("pdf_path")

    report = inspect_output(
        CompileResult(
            ok=pdf_path is not None,
            pdf_path=Path(pdf_path) if pdf_path else None,
            log=log,
            compiler=None,
            returncode=None,
        )
    )

    boxes = find_overfull_boxes(log)
    logger.info(
        "inspect: pages=%s overfull=%d error=%s",
        report.page_count,
        len(boxes),
        (report.first_error or "none").splitlines()[0] if report.first_error else "none",
    )
    return {"page_count": report.page_count}
