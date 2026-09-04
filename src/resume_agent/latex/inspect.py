r"""Read the environment's answer back: how many pages, what overflowed, what broke.

Spec 5 (`inspect_output`).

This module is the reason the project is agentic rather than a chatbot with extra
steps. Most LLM pipelines have no ground truth -- the model says it is done and
you believe it. Here, `pypdf` reports the page count and the LaTeX log reports
every line that did not fit, and neither of them is negotiable. M5's routing
decisions are made from these three functions:

    pages > 1          -> cut the line budget, reselect content
    overfull hboxes    -> shorten the specific bullets named
    a `!` error line   -> repair the LaTeX

CLAUDE.md rule 2: the LLM judges, Python counts. Counting pages is Python's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from resume_agent.latex.compile import CompileResult

# TeX writes, e.g.:
#   Overfull \hbox (12.34567pt too wide) in paragraph at lines 128--130
#   Overfull \hbox (5.0pt too wide) detected at line 99
# The "too wide" amount is the useful part: it says how much to cut, and 0.5pt of
# overflow is invisible while 30pt is a bullet hanging into the margin.
_OVERFULL_RE = re.compile(
    r"Overfull \\hbox \((?P<points>[\d.]+)pt too wide\)"
    r"(?P<tail>[^\n]*)",
)
_LINE_REF_RE = re.compile(r"lines? (?P<line>\d+)")


@dataclass(frozen=True)
class OverfullBox:
    points_over: float
    line_number: int | None
    raw: str


@dataclass(frozen=True)
class InspectionReport:
    page_count: int | None  # None when there is no PDF to count
    overfull_boxes: list[OverfullBox]
    first_error: str | None

    @property
    def is_clean(self) -> bool:
        """One page, nothing overflowing, nothing broken. Spec 9's deterministic checks."""
        return self.page_count == 1 and not self.overfull_boxes and self.first_error is None


def page_count(pdf_path: Path) -> int:
    """Number of pages in a PDF."""
    return len(PdfReader(str(pdf_path)).pages)


def find_overfull_boxes(log: str) -> list[OverfullBox]:
    """Every `Overfull \\hbox` warning in a compiler log, in order.

    Underfull boxes are deliberately ignored: they mean TeX stretched the spacing
    to fill a line, which is a typographic nit, not something that runs into the
    margin. Only overfull boxes are visible defects.
    """
    boxes: list[OverfullBox] = []
    for match in _OVERFULL_RE.finditer(log):
        tail = match.group("tail")
        line_match = _LINE_REF_RE.search(tail)
        boxes.append(
            OverfullBox(
                points_over=float(match.group("points")),
                line_number=int(line_match.group("line")) if line_match else None,
                raw=match.group(0).strip(),
            )
        )
    return boxes


def first_latex_error(log: str, context_lines: int = 4) -> str | None:
    """The first `! ...` block in the log, with the lines that follow it.

    Spec 5: "extract the first `! ` line, which is where LaTeX actually tells you
    what's wrong". The following lines matter as much as the first one -- the
    `l.<n>` line names the source line, and without it the message is unactionable.

    Only the *first* error is returned. After one failure LaTeX's recovery makes
    every later message suspect, so a repair loop that tried to fix all of them at
    once would mostly be chasing noise.
    """
    lines = log.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("!"):
            block = lines[index : index + 1 + context_lines]
            return "\n".join(block).rstrip()
    return None


def inspect_output(result: CompileResult) -> InspectionReport:
    """Everything the layout loop needs to know about one compile."""
    pages: int | None = None
    if result.pdf_path is not None and result.pdf_path.is_file():
        try:
            pages = page_count(result.pdf_path)
        except Exception as exc:  # noqa: BLE001 - a corrupt PDF is data, not a crash
            # A PDF that exists but cannot be parsed is a compile failure by
            # another name. Record it in the report rather than propagating, so
            # the caller keeps its single "look at the report" code path.
            return InspectionReport(
                page_count=None,
                overfull_boxes=find_overfull_boxes(result.log),
                first_error=f"! pypdf could not read {result.pdf_path}: {exc}",
            )

    return InspectionReport(
        page_count=pages,
        overfull_boxes=find_overfull_boxes(result.log),
        first_error=first_latex_error(result.log),
    )
