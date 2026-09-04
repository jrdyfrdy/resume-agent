"""Reading the compiler's feedback back out of the log and the PDF.

These three functions are what M5's layout loop routes on, so they are parsed
from checked-in fixture logs rather than only from live compiles -- a live
tectonic run produces no overfull boxes and no errors on a healthy template,
which would leave the interesting branches untested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_agent.latex.compile import CompileResult
from resume_agent.latex.inspect import (
    find_overfull_boxes,
    first_latex_error,
    inspect_output,
    page_count,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OVERFULL_LOG = (FIXTURES / "overfull.log").read_text(encoding="utf-8")
ERROR_LOG = (FIXTURES / "error.log").read_text(encoding="utf-8")


# --- overfull boxes --------------------------------------------------------


def test_finds_every_overfull_hbox() -> None:
    boxes = find_overfull_boxes(OVERFULL_LOG)
    assert [box.points_over for box in boxes] == [12.34567, 5.0, 0.51234]


def test_extracts_line_numbers_where_present() -> None:
    boxes = find_overfull_boxes(OVERFULL_LOG)
    assert boxes[0].line_number == 128  # "in paragraph at lines 128--130"
    assert boxes[1].line_number == 99  # "detected at line 99"
    assert boxes[2].line_number == 204


def test_underfull_boxes_are_ignored() -> None:
    """Underfull means loose spacing, not text running into the margin."""
    assert all("Underfull" not in box.raw for box in find_overfull_boxes(OVERFULL_LOG))


def test_overfull_vbox_is_ignored() -> None:
    """Only \\hbox matters here: a vbox overflow is a page-height problem, and
    page height is already covered by the page count."""
    assert all("vbox" not in box.raw for box in find_overfull_boxes(OVERFULL_LOG))


def test_clean_log_has_no_overfull_boxes() -> None:
    assert find_overfull_boxes("Output written on resume.pdf (1 page).") == []


# --- errors ----------------------------------------------------------------


def test_first_error_includes_the_source_line_reference() -> None:
    error = first_latex_error(ERROR_LOG)
    assert error is not None
    assert error.startswith("! Undefined control sequence.")
    # The `l.<n>` line is what makes the message actionable; without it a repair
    # node has no idea where to look.
    assert "l.7 \\pdfglyphtounicode" in error


def test_only_the_first_error_is_returned() -> None:
    """After one failure LaTeX's recovery makes later messages unreliable."""
    error = first_latex_error(ERROR_LOG)
    assert error is not None
    assert "Something's wrong" not in error


def test_clean_log_has_no_error() -> None:
    assert first_latex_error(OVERFULL_LOG) is None


# --- page count and the combined report ------------------------------------


def test_page_count_of_a_real_pdf(compiled_resume: CompileResult) -> None:
    pytest.importorskip("pypdf")
    if compiled_resume.pdf_path is None:
        pytest.skip("no compiler available")
    assert page_count(compiled_resume.pdf_path) == 1


def test_inspect_output_with_no_pdf() -> None:
    """A failed compile still produces a usable report rather than an exception."""
    result = CompileResult(
        ok=False, pdf_path=None, log=ERROR_LOG, compiler="tectonic", returncode=1
    )
    report = inspect_output(result)
    assert report.page_count is None
    assert report.first_error is not None
    assert not report.is_clean


def test_inspect_output_reports_a_corrupt_pdf_instead_of_raising(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "resume.pdf"
    fake_pdf.write_bytes(b"this is not a PDF")
    result = CompileResult(ok=True, pdf_path=fake_pdf, log="", compiler="tectonic", returncode=0)
    report = inspect_output(result)
    assert report.page_count is None
    assert report.first_error is not None
    assert "pypdf could not read" in report.first_error


def test_is_clean_requires_all_three_conditions() -> None:
    result = CompileResult(
        ok=True, pdf_path=None, log=OVERFULL_LOG, compiler="tectonic", returncode=0
    )
    report = inspect_output(result)
    assert report.overfull_boxes  # the fixture has three
    assert not report.is_clean
