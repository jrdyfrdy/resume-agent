r"""Measure how many characters of resume prose fit on one line of this template.

Run with:  uv run python scripts/calibrate_chars_per_line.py

Why this script exists
----------------------
M3's selection knapsack budgets content in *lines*, and it gets there from
`estimated_lines = ceil(len(text) / chars_per_line)` (spec 5). That constant is a
property of the template's geometry and font, so it has to be measured against
the real template, not guessed. Spec 5 suggests ~95 for Jake's Resume at 11pt;
this script is how we find out what it actually is.

Method
------
1. Render the real template through Jinja and keep its preamble verbatim, so the
   probe document has identical margins, font size and macros to production.
2. Reproduce the exact typesetting context of a bullet: a `\resumeItem` nested
   inside `\resumeItemListStart` inside `\resumeSubHeadingListStart`. The
   itemize nesting changes `\linewidth`, so measuring outside it would be wrong.
3. For each candidate length N, emit several probes -- each N characters taken
   from a different offset in the concatenated `canonical` strings of
   profile.example. Real resume prose, so the average character advance width is
   representative; several offsets, so one unluckily-placed long word does not
   decide the answer.
4. One probe per page, compile once, then count the text lines pypdf finds on
   each page. The largest N where *every* probe still fits on one line is the
   answer (conservative on purpose: under-estimating chars_per_line
   over-estimates line counts, which keeps the budget safe).
5. Cross-check with TeX's own opinion. `\settowidth` measures the unbroken
   natural width of a probe, and `\the\linewidth` is the space available, so
   N * linewidth / probewidth is an independent estimate of the same constant.
   If the two methods disagree badly, something about the probe is unrepresentative.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Make `src/` importable when run as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resume_agent.kb.loader import load_profile  # noqa: E402
from resume_agent.latex.compile import compile_tex  # noqa: E402
from resume_agent.latex.context import build_resume_context  # noqa: E402
from resume_agent.latex.env import render_template  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO_ROOT / "profile.example"
WORK_DIR = REPO_ROOT / "out" / "calibration"

MIN_LENGTH = 60
MAX_LENGTH = 140
PROBES_PER_LENGTH = 5

# The space is optional because TeX swallows the one that terminates
# `\the\probewidth`, so the log actually reads "...ptlinewidth=...".
_TYPEOUT_RE = re.compile(r"CALIB probe=(\d+) width=([\d.]+)pt\s*linewidth=([\d.]+)pt")


@dataclass(frozen=True)
class CalibrationResult:
    chars_per_line: int  # largest N where every probe fit on one line
    first_wrapping_length: int  # smallest N where any probe wrapped
    linewidth_pt: float
    width_ratio_estimate: int  # the independent \settowidth cross-check
    measurements: dict[int, int]  # N -> how many of its probes fit on one line


def build_corpus(profile_dir: Path) -> str:
    """All canonical bullet text, joined -- the prose the constant must describe."""
    profile = load_profile(profile_dir)
    return " ".join(bullet.canonical for bullet in profile.all_bullets())


def probe_strings(corpus: str, length: int, count: int) -> list[str]:
    """`count` slices of exactly `length` characters, spread across the corpus.

    Slices are cut at exact character counts, mid-word if that is where the count
    lands. That is deliberate: the model being calibrated is
    `ceil(len(text) / chars_per_line)`, which knows nothing about words, so the
    measurement should not either.
    """
    usable = len(corpus) - length
    if usable <= 0:
        raise ValueError(f"corpus too short ({len(corpus)}) to slice {length} characters")
    step = max(1, usable // count)
    return [corpus[offset : offset + length] for offset in range(0, step * count, step)][:count]


def build_probe_document(preamble: str, probes: list[tuple[int, str]]) -> str:
    r"""One page per probe, each a `\resumeItem` in its production context."""
    from resume_agent.latex.escape import latex_escape

    parts = [preamble, r"\newlength{\probewidth}", r"\begin{document}"]
    for index, (length, text) in enumerate(probes):
        escaped = latex_escape(text)
        parts.extend(
            [
                r"\resumeSubHeadingListStart",
                # The outer itemize needs an \item before a nested list may
                # start; in production \resumeSubheading provides it. An empty
                # one is used here instead, because it contributes no text to
                # the page while leaving the list nesting -- and therefore
                # \linewidth -- identical to production.
                r"\item[]",
                r"\resumeItemListStart",
                rf"\resumeItem{{{escaped}}}",
                # Measured inside the same itemize nesting, so \linewidth is the
                # width a real bullet actually gets.
                rf"\settowidth{{\probewidth}}{{\small {escaped}}}",
                rf"\typeout{{CALIB probe={length} width=\the\probewidth "
                rf"linewidth=\the\linewidth}}",
                r"\resumeItemListEnd",
                r"\resumeSubHeadingListEnd",
            ]
        )
        if index != len(probes) - 1:
            parts.append(r"\newpage")
    parts.append(r"\end{document}")
    return "\n".join(parts) + "\n"


def count_text_lines(page_text: str) -> int:
    """How many visual lines of text pypdf sees on a probe page.

    Each page holds exactly one bullet and nothing else, so every non-blank line
    of extracted text is a wrapped line of that bullet.
    """
    return len([line for line in page_text.splitlines() if line.strip()])


def measure(profile_dir: Path = PROFILE_DIR) -> CalibrationResult:
    corpus = build_corpus(profile_dir)

    # Take the preamble from a real render so the probe document cannot drift
    # away from the template it is supposed to describe.
    profile = load_profile(profile_dir)
    rendered = render_template("jake_resume.tex.j2", build_resume_context(profile))
    preamble = rendered[: rendered.index(r"\begin{document}")]

    probes: list[tuple[int, str]] = []
    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        probes.extend((length, text) for text in probe_strings(corpus, length, PROBES_PER_LENGTH))

    document = build_probe_document(preamble, probes)
    result = compile_tex(document, WORK_DIR, job_name="calibration")
    if not result.ok:
        raise RuntimeError(f"calibration probe document did not compile:\n{result.log[-3000:]}")

    reader = PdfReader(str(result.pdf_path))
    if len(reader.pages) != len(probes):
        raise RuntimeError(f"expected {len(probes)} probe pages, got {len(reader.pages)}")

    fits: dict[int, int] = {}
    for (length, _), page in zip(probes, reader.pages, strict=True):
        if count_text_lines(page.extract_text()) == 1:
            fits[length] = fits.get(length, 0) + 1
    for length in range(MIN_LENGTH, MAX_LENGTH + 1):
        fits.setdefault(length, 0)

    all_fit = [n for n in sorted(fits) if fits[n] == PROBES_PER_LENGTH]
    any_wrapped = [n for n in sorted(fits) if fits[n] < PROBES_PER_LENGTH]
    chars_per_line = max(all_fit) if all_fit else MIN_LENGTH
    first_wrapping = min(any_wrapped) if any_wrapped else MAX_LENGTH

    # Independent estimate from TeX's own width arithmetic.
    linewidth_pt = 0.0
    ratios: list[float] = []
    for match in _TYPEOUT_RE.finditer(result.log):
        length, width, linewidth = int(match.group(1)), float(match.group(2)), float(match.group(3))
        linewidth_pt = linewidth
        if width > 0:
            ratios.append(length * linewidth / width)
    width_ratio_estimate = int(sum(ratios) / len(ratios)) if ratios else 0

    return CalibrationResult(
        chars_per_line=chars_per_line,
        first_wrapping_length=first_wrapping,
        linewidth_pt=linewidth_pt,
        width_ratio_estimate=width_ratio_estimate,
        measurements=fits,
    )


def main() -> None:
    result = measure()

    print("corpus     : profile.example canonical bullet text")
    print(f"probes     : {PROBES_PER_LENGTH} per length, {MIN_LENGTH}..{MAX_LENGTH} characters")
    print(f"linewidth  : {result.linewidth_pt}pt (inside the bullet itemize nesting)")
    print()
    print("  chars  probes fitting on one line")
    for length in sorted(result.measurements):
        fitting = result.measurements[length]
        if fitting not in (0, PROBES_PER_LENGTH) or length in (
            result.chars_per_line,
            result.chars_per_line + 1,
            result.first_wrapping_length,
        ):
            bar = "#" * fitting + "." * (PROBES_PER_LENGTH - fitting)
            print(f"  {length:5d}  {bar}  {fitting}/{PROBES_PER_LENGTH}")
    print()
    print(f"largest length where ALL probes fit on one line : {result.chars_per_line}")
    print(f"smallest length where ANY probe wrapped         : {result.first_wrapping_length}")
    print(f"cross-check (N * linewidth / settowidth)        : {result.width_ratio_estimate}")
    print()
    print(f"=> CHARS_PER_LINE = {result.chars_per_line}")


if __name__ == "__main__":
    main()
