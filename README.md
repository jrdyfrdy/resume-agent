# resume-agent

A LangGraph agent that takes a job description and produces a truthful, one-page,
ATS-clean LaTeX resume and a matching cover letter, grounded entirely in a
structured career knowledge base.

Design: [`RESUME_AGENT_SPEC.md`](RESUME_AGENT_SPEC.md).
Standing rules for contributors (human or otherwise): [`CLAUDE.md`](CLAUDE.md).

**Status: M0 complete.** The deterministic LaTeX pipeline works end to end with
no LLM anywhere in it: `profile.example/` → Pydantic → Jinja2 → `.tex` →
tectonic → a one-page PDF. Retrieval (M1), JD parsing (M2), selection (M3),
tailoring and verification (M4) and the graph itself (M5) are not built yet.

---

## Setup

Requires **Python 3.12** (managed by `uv`) and a **LaTeX compiler**.

```bash
uv sync
```

### LaTeX compiler

`resume-agent` looks for one of these, in this order (spec §6.4):

1. **tectonic** — recommended. A single self-contained binary that downloads only
   the packages the document needs.
2. **latexmk** — from a full TeX Live or MiKTeX install.
3. **docker** — runs the `texlive/texlive` image. Correct, but slow.

Windows, no admin required:

```bash
curl.exe -sSL -o tectonic.zip https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip
```

Extract `tectonic.exe` into `%LOCALAPPDATA%\Programs\tectonic\` — the compiler
resolver checks there explicitly, so it does not need to be on `PATH`. On macOS,
`brew install tectonic`.

Installed somewhere unusual? Point at it directly:

```bash
set RESUME_AGENT_TECTONIC=C:\path\to\tectonic.exe
```

If nothing is found, `resume-agent` prints install instructions and exits 3 — it
never raises a stack trace at you.

---

## Use

```bash
uv run resume-agent build --profile profile.example --out out/
```

Writes `out/resume.tex`, `out/resume.pdf` and `out/compile.log`, then reports the
compiler used, the page count and any overfull hboxes.

Exit codes: `0` success · `1` compile failed · `2` profile invalid · `3` no compiler.

---

## Development

There is a `Makefile`, but `make` is not standard on Windows. The underlying
commands:

| Task | `make` | direct |
|---|---|---|
| Install deps | `make setup` | `uv sync` |
| Run tests | `make test` | `uv run pytest -v` |
| Tests, no compiler needed | `make test-fast` | `uv run pytest -m "not latex"` |
| Lint | `make lint` | `uv run ruff check .` |
| Build the example resume | `make build` | `uv run resume-agent build --profile profile.example --out out/` |
| Re-derive `CHARS_PER_LINE` | `make calibrate` | `uv run python scripts/calibrate_chars_per_line.py` |
| Regenerate the golden `.tex` | `make golden` | `REGEN_GOLDEN=1 uv run pytest tests/test_render_compile.py::test_golden_tex_snapshot` |

---

## What lives where

```
profile.example/          fake profile; what tests load. `profile/` is real data and is gitignored.
src/resume_agent/
  models/profile.py       the evidence-unit schema (spec §3.1, §3.2)
  kb/loader.py            YAML -> Pydantic, with cross-file integrity checks
  latex/escape.py         latex_escape() -- one regex pass, 13 characters
  latex/env.py            the Jinja environment with \VAR{} / \BLOCK{} delimiters
  latex/context.py        Profile -> template dict (dates, ordering, skill grouping)
  latex/compile.py        the tectonic -> latexmk -> docker fallback chain
  latex/inspect.py        page count, overfull hboxes, first LaTeX error
  latex/metrics.py        CHARS_PER_LINE, measured not guessed
  templates/              jake_resume.tex.j2 (Jake's Resume, MIT, adapted)
scripts/                  calibrate_chars_per_line.py
tests/                    escape fixtures, golden .tex, a real compile
```

---

## Two things worth knowing before you edit

**The escaper is enforced, not trusted.** Every value interpolated into the
template is written `\VAR{x | tex}`. `tests/test_env.py` parses the template and
fails if any interpolation is missing the filter — that mechanical check is what
makes the explicit style safe.

**`CHARS_PER_LINE` is measured.** It is 109 for this template, derived by
compiling probe bullets at every length from 60 to 140 and reading back from the
PDF where wrapping starts. The spec's ballpark was ~95. Re-run `make calibrate`
after any change to the template's geometry, font or list nesting.

---

## Licence note

`src/resume_agent/templates/jake_resume.tex.j2` is adapted from
[jakegut/resume](https://github.com/jakegut/resume), MIT licensed,
© 2020 Jake Gutierrez. The licence text is reproduced in the template header.
