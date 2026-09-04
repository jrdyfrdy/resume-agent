"""How much vertical space a piece of text will take. Spec 5 (`select_content`).

M3's selection knapsack budgets content in lines, and this is where the line
count comes from. CLAUDE.md rule 2 -- "the LLM judges; Python counts" -- means
this must never become a model call: it is arithmetic with a measured constant.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Measured, not guessed. Derived by scripts/calibrate_chars_per_line.py on
# 2026-09-05 against jake_resume.tex.j2 (article, letterpaper, 11pt, fullpage
# margins), using the concatenated `canonical` bullet text of profile.example as
# the probe corpus and tectonic 0.17.0 as the engine.
#
# Two independent measurements, and they agree:
#
#   1. Wrapping. 5 probes per length, 60-140 characters, one \resumeItem per
#      page in the production itemize nesting, line counts read back from the
#      PDF with pypdf:
#
#          109 chars  #####  5/5 probes on one line
#          110 chars  ####.  4/5
#          112 chars  ###..  3/5
#          115 chars  #....  1/5
#          120 chars  .....  0/5
#
#   2. Natural width. \settowidth on the same probes reports \linewidth =
#      507.095pt inside that nesting, giving N * linewidth / width = 112 chars.
#      Slightly higher than the wrapping figure, exactly as expected: natural
#      width ignores the fact that TeX breaks at word boundaries, so it is an
#      upper bound on what actually fits.
#
# 109 is the conservative end of that range -- the largest length where *every*
# probe still fit. Erring low over-estimates line counts, which makes the M3
# budget err towards a resume that fits rather than one that overflows.
#
# Note this is well above the "~95" that spec 5 offers as a ballpark. The spec
# says to calibrate it and hardcode what you measure, so: 109.
#
# Re-derive after any change to the template's geometry, font, or list nesting:
#     uv run python scripts/calibrate_chars_per_line.py
# ---------------------------------------------------------------------------
CHARS_PER_LINE = 109


def estimate_lines(text: str, chars_per_line: int = CHARS_PER_LINE) -> int:
    """How many rendered lines `text` will occupy as a resume bullet.

    Deliberately the crude `ceil(len / width)` model from spec 5 rather than a
    real line-breaking simulation. It is within a character or two of the truth
    for resume prose, it is trivially testable, and -- most importantly -- the
    compile loop has a real oracle: if the estimate is wrong the PDF comes back
    at two pages and `inspect_output` says so. An estimator does not need to be
    exact when something downstream is checking its work.

    Empty text still occupies a line: an empty bullet is a rendered bullet.
    """
    if chars_per_line <= 0:
        raise ValueError(f"chars_per_line must be positive, got {chars_per_line}")
    return max(1, math.ceil(len(text) / chars_per_line))
