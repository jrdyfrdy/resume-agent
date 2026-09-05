"""Repair a document that failed to compile. Spec 5, `inspect_output` routing.

    "compile error -> `fix_latex` node (LLM sees the error + the offending
     template region)"

This is the node that makes the compile loop a genuine agent loop rather than a
retry: the model is shown a real error message produced by real software, and
its output is judged by running that software again. Nothing here trusts the
model's opinion of whether it fixed anything.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.graph.state import AgentState
from resume_agent.latex.inspect import first_latex_error
from resume_agent.llm import PARSE_MODEL, build_chat_model, load_prompt

logger = logging.getLogger(__name__)

PROMPT_NAME = "fix_latex"

# How much of the source to show around the failing line. The whole document is
# only ~200 lines, so this is generous enough to always include the preamble
# rules the prompt tells the model not to touch.
CONTEXT_LINES = 40


def _offending_region(tex_source: str, error: str | None) -> str:
    """The source around the failing line, with line numbers.

    LaTeX reports `l.<n>`, and handing the model a numbered window lets it match
    that number to a line instead of guessing.
    """
    lines = tex_source.splitlines()
    numbered = [f"{index:4d} | {line}" for index, line in enumerate(lines, start=1)]

    line_number = None
    if error:
        for token in error.split():
            if token.startswith("l.") and token[2:].isdigit():
                line_number = int(token[2:])
                break

    if line_number is None:
        return "\n".join(numbered)

    start = max(0, line_number - CONTEXT_LINES)
    end = min(len(numbered), line_number + CONTEXT_LINES)
    return "\n".join(numbered[start:end])


def fix_latex(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Ask a model to repair the LaTeX, and hand the result back for recompiling."""
    tex_source = state.get("tex_source") or ""
    error = first_latex_error(state.get("compile_log") or "")

    logger.warning(
        "fix_latex: attempting repair (layout attempt %d). First error: %s",
        state.get("layout_attempts", 0) + 1,
        (error or "unknown").splitlines()[0] if error else "unknown",
    )

    llm = llm or build_chat_model(model=PARSE_MODEL)

    # Plain text out, not structured: the deliverable is a whole LaTeX document,
    # and wrapping it in a JSON string field only adds an escaping round-trip
    # that can itself corrupt backslashes.
    response = llm.invoke(
        [
            SystemMessage(content=load_prompt(PROMPT_NAME)),
            HumanMessage(
                content=(
                    f"<error>\n{error or 'unknown error'}\n</error>\n\n"
                    f"<source_region>\n{_offending_region(tex_source, error)}\n"
                    f"</source_region>\n\n"
                    f"<full_source>\n{tex_source}\n</full_source>"
                )
            ),
        ]
    )

    repaired = response.content if isinstance(response.content, str) else str(response.content)
    repaired = _strip_code_fence(repaired).strip()

    if not repaired:
        # An empty repair would blank the document and the next compile would
        # fail differently, wasting an iteration of a capped loop.
        logger.error("fix_latex: model returned nothing; keeping the original source")
        return {
            "layout_attempts": state.get("layout_attempts", 0) + 1,
            "errors": ["fix_latex returned an empty document"],
        }

    return {
        "tex_source": repaired,
        "layout_attempts": state.get("layout_attempts", 0) + 1,
    }


def _strip_code_fence(text: str) -> str:
    """Remove a ```latex fence if the model wrapped its answer in one.

    The prompt asks for bare source, but a fenced response is a common and
    harmless deviation, and letting ``` reach the compiler turns a successful
    repair into a fresh syntax error.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
