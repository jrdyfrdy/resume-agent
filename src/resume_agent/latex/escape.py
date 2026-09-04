r"""Turn arbitrary profile text into something safe to paste into LaTeX.

Spec 6.2. Every value interpolated into a template goes through `latex_escape`,
registered as the Jinja filter `tex` in `latex/env.py`.

Why this is more than a `str.replace` chain
-------------------------------------------
The obvious implementation is a loop of `str.replace` calls, and it has a famous
bug: the replacements themselves contain backslashes and braces, so a later pass
re-escapes what an earlier pass just wrote. `"&"` becomes `r"\&"` becomes
`r"\textbackslash{}\&"`. The usual advice is "escape the backslash first", which
works but leaves the correctness of the function resting on the ordering of a
dict literal.

A single `re.sub` over a character class sidesteps the problem entirely: the scan
moves left to right through the *input* and never revisits text it has already
emitted, so ordering cannot matter. That is the whole reason for the regex here.
"""

from __future__ import annotations

import re

# The ten characters from spec 6.2, plus three additions (see below).
LATEX_ESCAPES: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # --- documented deviation from spec 6.2 -------------------------------
    # Jake's Resume loads no `fontenc`, so it typesets in TeX's default OT1
    # encoding. In OT1 these three characters are *not* errors -- they are
    # silently wrong, which is worse: "<" prints as an inverted exclamation
    # mark, ">" as an inverted question mark, and "|" as an em-dash. A resume
    # saying "p99 <100ms" would render as "p99 !100ms" and compile cleanly.
    # Escaping them costs nothing and closes a silent-corruption hole.
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
}

# `re.escape` on the joined keys keeps the character class literal -- several of
# these ("\", "^", "$") are regex metacharacters in their own right.
_ESCAPE_PATTERN = re.compile("|".join(re.escape(char) for char in LATEX_ESCAPES))


def latex_escape(value: object) -> str:
    """Escape `value` for safe interpolation into a LaTeX document.

    Non-string input is coerced with `str()` rather than rejected: `metrics`
    dicts hold ints and floats (see `MetricValue`), and a filter that raised on
    an int would push type-juggling into the templates, which is exactly where
    it is hardest to see.
    """
    text = value if isinstance(value, str) else str(value)
    return _ESCAPE_PATTERN.sub(lambda match: LATEX_ESCAPES[match.group()], text)
