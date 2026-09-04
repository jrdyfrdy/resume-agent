r"""The Jinja/LaTeX environment, and the guard that makes explicit escaping safe.

Plan D4: escaping is written out in the template (`| tex` on every value) rather
than hooked into Jinja's `finalize` and applied invisibly. Explicit reads better,
but it has one failure mode -- forgetting the filter on a single field, which
produces a PDF that is silently wrong or a compile error a hundred lines away.

`test_every_interpolation_is_escaped` closes that hole mechanically. It is the
test that earns the readability of the explicit style.
"""

from __future__ import annotations

import re

import jinja2
import pytest

from resume_agent.latex.env import make_latex_env, render_template

# `variable_end_string` is a single "}", so an interpolation can never contain
# one -- which is what makes this crude regex a complete parse.
_VAR_RE = re.compile(r"\\VAR\{([^}]*)\}")
_ENDS_WITH_TEX_FILTER = re.compile(r"\|\s*tex\s*$")

# Expressions allowed to skip the `tex` filter. Empty today, and every future
# entry needs a comment saying why the value cannot contain LaTeX specials.
ESCAPE_EXEMPT: set[str] = set()


def test_every_interpolation_is_escaped(template_source: str) -> None:
    expressions = _VAR_RE.findall(template_source)
    assert expressions, "found no interpolations at all -- is the regex still right?"

    unescaped = [
        expr.strip()
        for expr in expressions
        if expr.strip() not in ESCAPE_EXEMPT and not _ENDS_WITH_TEX_FILTER.search(expr)
    ]
    assert not unescaped, (
        "these template interpolations do not end in the `tex` filter:\n  "
        + "\n  ".join(unescaped)
        + "\nAdd `| tex`, or add the expression to ESCAPE_EXEMPT with a reason."
    )


def test_ats_lines_are_present_in_the_template(template_source: str) -> None:
    """CLAUDE.md rule 4. These lines must survive every future edit of the template.

    They are wrapped in an `\\ifdefined\\pdfgentounicode` guard because tectonic's
    engine is XeTeX and has no such primitive, but they must still be there for
    the pdfTeX rungs of the fallback chain. The property they protect is checked
    on the real PDF in test_render_compile.py.
    """
    assert r"\input{glyphtounicode}" in template_source
    assert r"\pdfgentounicode=1" in template_source


def test_geometry_is_not_a_template_variable(template_source: str) -> None:
    """CLAUDE.md rule 3: the agent controls content length, never page geometry.

    Enforced by not exposing it. If margins, font size or \\vspace ever become
    interpolated values, a future tailoring prompt will discover it and start
    shrinking the page to fit instead of cutting content.
    """
    geometry_markers = [
        r"\addtolength",
        r"\vspace",
        "documentclass",
        "oddsidemargin",
        "textheight",
        "textwidth",
    ]
    for expression in _VAR_RE.findall(template_source):
        for marker in geometry_markers:
            assert marker not in expression, (
                f"template interpolates page geometry: \\VAR{{{expression}}}"
            )


# --- the environment itself ------------------------------------------------


def test_custom_delimiters_do_not_collide_with_latex(tmp_path) -> None:
    r"""Default Jinja would mangle `{}` and `%`; the spec 6.1 delimiters must not."""
    template = tmp_path / "probe.tex.j2"
    template.write_text(
        # A line of plausible LaTeX containing every construct that would break
        # under default delimiters: {{ }}, {% %}, and a bare % comment.
        "% a LaTeX comment with 100% signs\n"
        r"\newcommand{\x}[1]{{#1}}" + "\n" + r"\resumeItem{\VAR{value | tex}}" + "\n",
        encoding="utf-8",
    )
    env = make_latex_env(tmp_path)
    out = render_template("probe.tex.j2", {"value": "AT&T"}, env=env)

    assert r"\newcommand{\x}[1]{{#1}}" in out
    assert "% a LaTeX comment with 100% signs" in out
    assert r"\resumeItem{AT\&T}" in out


def test_tex_filter_is_registered() -> None:
    assert "tex" in make_latex_env().filters


def test_undefined_variables_fail_loudly(tmp_path) -> None:
    """StrictUndefined: a typo'd context key must stop the render.

    The default `Undefined` renders empty, which produces `\\resumeItem{}` --
    valid LaTeX that compiles to a blank bullet. Failing here instead means the
    error names the variable rather than showing up as a mystery gap in the PDF.
    """
    (tmp_path / "probe.tex.j2").write_text(r"\VAR{nope | tex}", encoding="utf-8")
    with pytest.raises(jinja2.UndefinedError):
        render_template("probe.tex.j2", {}, env=make_latex_env(tmp_path))


def test_trailing_newline_is_kept(tmp_path) -> None:
    """The golden snapshot compares bytes, so this must not drift."""
    (tmp_path / "probe.tex.j2").write_text("hello\n", encoding="utf-8")
    assert render_template("probe.tex.j2", {}, env=make_latex_env(tmp_path)) == "hello\n"


def test_rendered_resume_has_no_leftover_jinja(rendered_tex: str) -> None:
    assert r"\VAR{" not in rendered_tex
    assert r"\BLOCK{" not in rendered_tex
