r"""The Jinja2 environment configured for LaTeX. Spec 6.1.

Jinja's default delimiters (`{{ }}`, `{% %}`) and LaTeX both want braces and
percent signs, and the collision is not merely cosmetic -- `{%` is a legal
sequence in TeX and `{{` appears in nested macro arguments, so a default-config
Jinja environment will happily mangle a working `.tex` file.

The fix from spec 6.1 is to move Jinja's markers somewhere LaTeX never goes:

    \VAR{name}            interpolation
    \BLOCK{for b in ...}  statements
    \#{...}               comments
    %% statement          line statement
    %# comment            line comment

The result reads like LaTeX macros, which means your editor's syntax
highlighting and brace matching keep working on the template.

One consequence worth internalising: `variable_end_string` is a single `}`, so a
`\VAR{...}` expression ends at the *first* closing brace. Keep the expressions
inside them simple -- `\VAR{x | tex}` is fine, a dict literal is not.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jinja2

from resume_agent.latex.escape import latex_escape

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def make_latex_env(template_dir: Path | None = None) -> jinja2.Environment:
    """Build the LaTeX-flavoured Jinja environment from spec 6.1."""
    env = jinja2.Environment(
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        lstrip_blocks=True,
        # HTML escaping would be actively harmful here -- LaTeX escaping is a
        # different alphabet, and it is applied explicitly via the `tex` filter.
        autoescape=False,
        # Without this Jinja strips the file's final newline, and the golden
        # snapshot test compares bytes.
        keep_trailing_newline=True,
        loader=jinja2.FileSystemLoader(template_dir or TEMPLATE_DIR),
        # An undefined variable should stop the render, not silently produce an
        # empty macro argument that fails much later inside tectonic.
        undefined=jinja2.StrictUndefined,
    )
    # Registered under a short name because it appears on every single
    # interpolation in the template; `\VAR{x | latex_escape}` would be noise.
    env.filters["tex"] = latex_escape
    return env


def render_template(
    template_name: str,
    context: Mapping[str, Any],
    env: jinja2.Environment | None = None,
) -> str:
    """Render `template_name` to a LaTeX source string."""
    env = env or make_latex_env()
    return env.get_template(template_name).render(**context)
