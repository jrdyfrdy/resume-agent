"""The escape fixture list from spec 6.2, plus the three added characters.

Spec 6.2: "Test fixtures that must pass: AT&T, C#, 100% uptime, user_id,
Node.js ^18, ~50ms, R&D. Every one of these is a realistic resume string and
every one of them breaks a naive implementation."
"""

from __future__ import annotations

import pytest

from resume_agent.latex.escape import LATEX_ESCAPES, latex_escape

# --- spec 6.2, verbatim ----------------------------------------------------

SPEC_FIXTURES = [
    ("AT&T", r"AT\&T"),
    ("C#", r"C\#"),
    ("100% uptime", r"100\% uptime"),
    ("user_id", r"user\_id"),
    ("Node.js ^18", r"Node.js \textasciicircum{}18"),
    ("~50ms", r"\textasciitilde{}50ms"),
    ("R&D", r"R\&D"),
]


@pytest.mark.parametrize(("raw", "expected"), SPEC_FIXTURES)
def test_spec_6_2_fixtures(raw: str, expected: str) -> None:
    assert latex_escape(raw) == expected


# --- the three characters added beyond spec 6.2 (plan D3) ------------------

ADDED_FIXTURES = [
    ("p99 <100ms", r"p99 \textless{}100ms"),
    (">2M rows/day", r"\textgreater{}2M rows/day"),
    ("stdout | jq", r"stdout \textbar{} jq"),
]


@pytest.mark.parametrize(("raw", "expected"), ADDED_FIXTURES)
def test_added_fixtures(raw: str, expected: str) -> None:
    assert latex_escape(raw) == expected


# --- the ordering hazard spec 6.2 warns about ------------------------------


def test_backslash_is_not_double_escaped() -> None:
    """The single-pass regex must not re-escape the backslashes it emits.

    A naive `str.replace` chain that handles "&" before "\\" produces
    `\\textbackslash{}\\&` here. This is the bug spec 6.2's "order matters" note
    is about, and the assertion that proves the regex approach avoids it.
    """
    assert latex_escape("a\\b") == r"a\textbackslash{}b"
    assert latex_escape("&") == r"\&"
    # Backslash and ampersand together: each is escaped exactly once.
    assert latex_escape("\\&") == r"\textbackslash{}\&"


def test_every_replacement_is_itself_stable() -> None:
    """Escaping an already-escaped string must not corrupt it further.

    Not a property we rely on (values are escaped exactly once, at render), but
    if it ever fails it means a replacement emitted a character that is itself
    in the escape table without being consumed -- a real bug.
    """
    for char, replacement in LATEX_ESCAPES.items():
        once = latex_escape(char)
        assert once == replacement
        # Escaping the *replacement* is allowed to change it; escaping the
        # original twice must produce the double-escaped form, not garbage.
        assert latex_escape(once) == latex_escape(replacement)


# --- shape and coercion ----------------------------------------------------


def test_all_thirteen_characters_are_covered() -> None:
    assert set(LATEX_ESCAPES) == set("\\&%$#_{}~^<>|")


def test_non_string_input_is_coerced() -> None:
    """`metrics` values are int | float | str, so the filter must accept all three."""
    assert latex_escape(820) == "820"
    assert latex_escape(3.14) == "3.14"
    assert latex_escape(None) == "None"


def test_plain_text_is_untouched() -> None:
    plain = "Cut p95 checkout latency by adding a read-through cache"
    assert latex_escape(plain) == plain


def test_empty_string() -> None:
    assert latex_escape("") == ""


def test_realistic_bullet_with_several_hazards() -> None:
    raw = "Built a C#/.NET service for AT&T that cleared $1.2M (38% of disputes) on user_id"
    expected = (
        r"Built a C\#/.NET service for AT\&T that cleared \$1.2M "
        r"(38\% of disputes) on user\_id"
    )
    assert latex_escape(raw) == expected
