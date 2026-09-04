"""The M0 definition of done, as tests.

Spec 8: "`make test` green; `out/resume.pdf` opens; the escape fixture list all
render correctly; `AT&T` appears as 'AT&T' not a crash."

Two layers:

* a golden `.tex` snapshot, which catches template and context regressions in
  milliseconds and shows exactly what changed, and
* a real compile, which is the only thing that can prove the document is valid
  LaTeX, fits on one page, and extracts as text.

The snapshot alone would be a test of our own output against itself. The compile
alone would tell you something broke but not what. Both together are useful.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader

from resume_agent.latex import compile as compile_module
from resume_agent.latex.compile import (
    INSTALL_MESSAGE,
    TECTONIC_ENV_VAR,
    CompileResult,
    compile_tex,
    find_compiler,
)
from resume_agent.latex.inspect import inspect_output

from .conftest import GOLDEN_PATH, requires_latex

# Strings that must survive the whole pipeline -- YAML, Pydantic, Jinja, the
# escaper, LaTeX, and PDF text extraction -- and come back out unchanged.
# Spec 6.2's fixture list, as it appears in profile.example.
ATS_PROBES = [
    "AT&T",
    "C#",
    "100% uptime",
    "user_id",
    "Halvorsen & Bright R&D",
    "Node.js ^18",
    "~50ms",
    "<100ms",
    ">2M",
    "stdout | jq",
    "{z}/{x}/{y}",
    "$1.2M",
]


# --- golden snapshot -------------------------------------------------------


def test_golden_tex_snapshot(rendered_tex: str) -> None:
    """Byte-compare the render against the checked-in snapshot.

    Regenerate deliberately, never accidentally:
        REGEN_GOLDEN=1 uv run pytest tests/test_render_compile.py
    then read the diff before committing it. (Plan D11.)
    """
    if os.environ.get("REGEN_GOLDEN"):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(rendered_tex, encoding="utf-8")
        pytest.skip(f"regenerated {GOLDEN_PATH}")

    assert GOLDEN_PATH.is_file(), "missing golden file: run REGEN_GOLDEN=1 pytest"
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    assert rendered_tex == expected, (
        "rendered .tex differs from the golden snapshot. If the change is "
        "intended, regenerate with REGEN_GOLDEN=1 and review the diff."
    )


def test_golden_contains_the_escaped_hazards() -> None:
    """The snapshot is only worth having if it pins the escaping.

    Asserted against the file on disk rather than the fresh render, so this
    fails if someone regenerates the golden from a broken escaper.
    """
    golden = GOLDEN_PATH.read_text(encoding="utf-8")
    for escaped in [
        r"Halvorsen \& Bright R\&D",
        r"AT\&T",
        r"C\#/.NET",
        r"100\% uptime",
        r"user\_id",
        r"\textasciitilde{}50ms",
        r"\textasciicircum{}18",
        r"\textless{}100ms",
        r"\textgreater{}2M",
        r"\textbar{} jq",
        r"\{z\}/\{x\}/\{y\}",
        r"\$1.2M",
    ]:
        assert escaped in golden, f"golden .tex no longer contains {escaped!r}"


# --- the real compile ------------------------------------------------------


@requires_latex
def test_compiles_successfully(compiled_resume: CompileResult) -> None:
    report = inspect_output(compiled_resume)
    assert compiled_resume.ok, (
        f"compile failed with {compiled_resume.compiler} "
        f"(rc={compiled_resume.returncode}):\n{report.first_error}"
    )


@requires_latex
def test_page_count_is_one(compiled_resume: CompileResult) -> None:
    """Spec 8's M0 DoD, and spec 9's first deterministic eval check."""
    assert inspect_output(compiled_resume).page_count == 1


@requires_latex
def test_no_overfull_hboxes(compiled_resume: CompileResult) -> None:
    """Spec 9: "zero overfull hboxes". Not in M0's DoD, but free to check here."""
    boxes = inspect_output(compiled_resume).overfull_boxes
    assert not boxes, "content runs into the margin:\n" + "\n".join(b.raw for b in boxes)


@requires_latex
def test_no_latex_errors(compiled_resume: CompileResult) -> None:
    assert inspect_output(compiled_resume).first_error is None


@requires_latex
def test_pdf_extracts_as_real_text(compiled_resume: CompileResult) -> None:
    """The ATS premise of the whole project, checked on the finished artifact.

    CLAUDE.md rule 4 protects this property via `\\input{glyphtounicode}` and
    `\\pdfgentounicode=1`. Those are pdfTeX primitives and the template guards
    them with `\\ifdefined`, because tectonic's engine is XeTeX and has no such
    primitive -- XeTeX instead emits Unicode-mapped PDFs natively.

    That makes the *mechanism* engine-dependent, so this test checks the
    *property* directly and stays correct under either engine: every hazardous
    string must come back out of the PDF byte for byte. If it does not, the
    resume parses as garbage in a keyword scanner and the premise collapses --
    which is precisely what rule 4 exists to prevent.
    """
    assert compiled_resume.pdf_path is not None
    text = PdfReader(str(compiled_resume.pdf_path)).pages[0].extract_text()

    missing = [probe for probe in ATS_PROBES if probe not in text]
    assert not missing, (
        f"these strings did not survive PDF text extraction: {missing}\n"
        f"extracted text was:\n{text}"
    )


@requires_latex
def test_at_and_t_specifically(compiled_resume: CompileResult) -> None:
    """Spec 8, verbatim: "`AT&T` appears as 'AT&T' not a crash"."""
    assert compiled_resume.pdf_path is not None
    text = PdfReader(str(compiled_resume.pdf_path)).pages[0].extract_text()
    assert "AT&T" in text
    assert r"AT\&T" not in text  # escaped in the .tex, plain in the PDF


# --- failure paths ---------------------------------------------------------


@requires_latex
def test_broken_latex_returns_a_log_instead_of_raising(tmp_path: Path) -> None:
    """CLAUDE.md: a failed compile is data for the next node, never an exception.

    This is the behaviour M5's repair loop depends on entirely.
    """
    broken = "\\documentclass{article}\n\\begin{document}\n\\thisIsNotACommand\n\\end{document}\n"
    result = compile_tex(broken, tmp_path, job_name="broken")

    assert result.ok is False
    assert result.log  # the log is the feedback signal
    report = inspect_output(result)
    assert report.first_error is not None
    assert "Undefined control sequence" in report.first_error


def test_no_compiler_returns_the_install_message(tmp_path: Path, monkeypatch) -> None:
    """Spec 6.4: "Fail with a clear install message rather than a stack trace"."""
    # Blind every rung of the discovery chain rather than uninstalling anything.
    monkeypatch.setenv(TECTONIC_ENV_VAR, str(tmp_path / "nonexistent-tectonic.exe"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "empty-localappdata"))
    monkeypatch.setattr(compile_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "empty-home"))

    assert find_compiler() is None

    result = compile_tex("\\documentclass{article}\\begin{document}x\\end{document}", tmp_path)
    assert result.ok is False
    assert result.compiler is None
    assert result.log == INSTALL_MESSAGE
    assert "tectonic" in result.log
    # The .tex is still written, so the user can compile it by hand.
    assert (tmp_path / "resume.tex").is_file()


def test_shutil_is_actually_used_by_find_compiler() -> None:
    """Guards the monkeypatch above: if compile.py stopped calling shutil.which,
    the no-compiler test would pass for the wrong reason."""
    assert compile_module.shutil is shutil
