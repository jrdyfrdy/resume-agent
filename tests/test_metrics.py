"""The line-estimation model that M3's budget will be built on.

CLAUDE.md rule 2: this is arithmetic, and arithmetic gets tests.
"""

from __future__ import annotations

import pytest

from resume_agent.latex.metrics import CHARS_PER_LINE, estimate_lines
from resume_agent.models.profile import Profile


def test_calibrated_constant_is_the_measured_value() -> None:
    """Pinned so a casual edit to the constant has to be deliberate.

    If the template's geometry or font changes, re-run
    `scripts/calibrate_chars_per_line.py` and update both places.
    """
    assert CHARS_PER_LINE == 109


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        (0, 1),  # an empty bullet still occupies a line
        (1, 1),
        (108, 1),
        (109, 1),  # exactly the calibrated width
        (110, 2),  # one character over
        (218, 2),
        (219, 3),
    ],
)
def test_estimate_lines_boundaries(length: int, expected: int) -> None:
    assert estimate_lines("x" * length) == expected


def test_chars_per_line_is_overridable() -> None:
    assert estimate_lines("x" * 100, chars_per_line=50) == 2


def test_non_positive_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        estimate_lines("text", chars_per_line=0)


def test_example_bullets_are_at_most_two_lines(example_profile: Profile) -> None:
    """A sanity check on the example data, not on the estimator.

    A three-line bullet on a one-page resume is a content problem. M0 has no
    selection step to catch it, so the assertion lives here.
    """
    for bullet in example_profile.all_bullets():
        lines = estimate_lines(bullet.canonical)
        assert lines <= 2, f"{bullet.id} estimates at {lines} lines: {bullet.canonical!r}"
