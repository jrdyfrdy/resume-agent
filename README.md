# resume-agent

A LangGraph agent that takes a job description and produces a truthful, one-page,
ATS-clean LaTeX resume and a matching cover letter, grounded entirely in a
structured career knowledge base.

Design: [`RESUME_AGENT_SPEC.md`](RESUME_AGENT_SPEC.md).
Standing rules for contributors (human or otherwise): [`CLAUDE.md`](CLAUDE.md).

**Status: M0-M3 complete.**

* **M0** — the deterministic LaTeX pipeline: `profile.example/` → Pydantic →
  Jinja2 → `.tex` → tectonic → a one-page PDF.
* **M1** — the knowledge base and hybrid retrieval: SQLite + a `sqlite-vec`
  vector index, BM25 and dense retrieval fused with Reciprocal Rank Fusion, and
  query expansion through the `skills.yaml` alias table. No LLM.
* **M2** — job description parsing: a posting becomes a validated `JobSpec` via
  structured output, cached on a hash of the posting, the model and the prompt.
  **This is the first milestone that calls a model.**

* **M3** — scoring, selection and `analyze`: batched relevance scoring in one
  LLM call, a deterministic `FitReport`, and a greedy knapsack that picks what
  fits on one page under every constraint in spec §5.

Tailoring and verification (M4) and the graph itself (M5) are not built yet.
**There is no web UI** — that is M9.

---

## Setup

Requires **Python 3.12** (managed by `uv`) and a **LaTeX compiler**.

```bash
uv sync
```

The first `search` or `index` run downloads a 67 MB ONNX embedding model
(`BAAI/bge-small-en-v1.5`) once. It is cached per-user, not in a temp
directory, so it survives reboots: `%LOCALAPPDATA%/resume-agent/fastembed` on
Windows, `~/.cache/resume-agent/fastembed` elsewhere. After that everything is
offline -- no API key, no per-query cost. Override the location with the
`RESUME_AGENT_MODEL_CACHE` environment variable.

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

### Parsing a job description

Needs `ANTHROPIC_API_KEY`; everything else in this project works without one.

```bash
uv run resume-agent parse-jd --jd evals/datasets/jds/mid.txt
```

Prints the requirements sorted by weight, with **inferred priorities shown
separately** — things the posting keeps circling back to without ever listing as
a requirement. Add `--json` for the raw `JobSpec`.

Parses are cached under `.cache/jd/` on `sha256(posting + model + prompt)`, so
re-running while you iterate downstream costs nothing. The prompt hash is in the
key deliberately: editing `prompts/parse_jd.md` invalidates the parses it
produced rather than silently serving stale ones.

Model is `claude-opus-5` (`PARSE_MODEL` in `llm.py`), about /usr/bin/bash.05–0.07 per
posting.

### Analysing a posting  ← the one you'll use daily

```bash
uv run resume-agent analyze --jd evals/datasets/jds/mid.txt
```

Answers "should I apply, and what am I missing?". Prints the recommendation
first, then **gaps before coverage** — what you lack is actionable, what you have
is reassurance. Must-have gaps are separated from nice-to-haves, and inferred
requirements are marked so you never think you failed to meet something the
posting never asked for.

It also shows what selection chose for the resume and, for everything it didn't,
the constraint that excluded it — because "why is my best bullet missing?" is
the first question anyone asks of a selector.

`--strict` drops bullets marked `confidence: claim`. `--json` emits the raw
`FitReport`. Needs `ANTHROPIC_API_KEY`; costs roughly $0.20 per new posting and
nothing on a re-run.

### Searching the knowledge base

```bash
uv run resume-agent search "kubernetes" --explain
```

`--explain` shows where each retriever placed a bullet next to the fused score,
and `--mode bm25` / `--mode dense` run one retriever alone. That comparison is
the point: querying `kubernetes` against the example profile, BM25 returns the
single bullet that literally says Kubernetes, while dense returns a full ranking
in which a DuckDB linter scores 0.69. Fusion keeps the exact match on top
without throwing away the semantic recall that finds "reduce cloud spend" →
"Cut monthly AWS spend 38%".

The index lives in `.index/` (gitignored), is keyed on a hash of the profile's
YAML, and rebuilds itself when that changes — `resume-agent index --rebuild`
forces it.

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
| Build the search index | `make index` | `uv run resume-agent index --profile profile.example` |
| Inspect retrieval | `make search Q="redis"` | `uv run resume-agent search "redis" --explain` |
| Parse a JD | `make parse-jd` | `uv run resume-agent parse-jd --jd evals/datasets/jds/mid.txt` |
| Analyse a JD | `make analyze` | `uv run resume-agent analyze --jd evals/datasets/jds/mid.txt` |
| Re-derive the line budget | `make calibrate-budget` | `uv run python scripts/calibrate_line_budget.py` |
| Regenerate JD snapshots | `make snapshots` | `REGEN_SNAPSHOTS=1 uv run pytest tests/test_parse_jd.py -m llm` |
| Re-derive `CHARS_PER_LINE` | `make calibrate` | `uv run python scripts/calibrate_chars_per_line.py` |
| Regenerate the golden `.tex` | `make golden` | `REGEN_GOLDEN=1 uv run pytest tests/test_render_compile.py::test_golden_tex_snapshot` |

---

## What lives where

```
profile.example/          fake profile; what tests load. `profile/` is real data and is gitignored.
src/resume_agent/
  models/profile.py       the evidence-unit schema (spec §3.1, §3.2)
  kb/loader.py            YAML -> Pydantic, with cross-file integrity checks
  kb/tokenize.py          technology-aware tokenizer ("C#" stays "c#")
  kb/embeddings.py        local ONNX embeddings behind LangChain's interface
  kb/index.py             SQLite + sqlite-vec; staleness via a content hash
  kb/retriever.py         query expansion, BM25, dense, RRF (k=60)
  models/job.py           JobSpec (spec §4); source_hash is computed, not generated
  llm.py                  model id, effort, prompt loading + versioning
  jd_cache.py             on-disk parse cache
  prompts/parse_jd.md     the JD-parsing prompt, versioned
  graph/nodes/parse_jd.py parse_job_description()
  graph/nodes/retrieve.py per-requirement retrieval + dedupe (no LLM)
  graph/nodes/score.py    one batched scoring call; FitReport built in Python
  graph/nodes/select.py   the greedy knapsack (no LLM)
  latex/layout.py         line budget, measured by compiling six profile shapes
  analyze.py / report.py  the analyze pipeline and its human-readable output
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

## Three things worth knowing before you edit

**The escaper is enforced, not trusted.** Every value interpolated into the
template is written `\VAR{x | tex}`. `tests/test_env.py` parses the template and
fails if any interpolation is missing the filter — that mechanical check is what
makes the explicit style safe.

**Retrieval quality is tested without labels.** Most of the retrieval suite
asserts invariants rather than opinions: every bullet must retrieve itself,
`k8s` and `kubernetes` must return identical results, and RRF's arithmetic is
pinned exactly. On top of that sits a small hand-written relevance set in
`tests/fixtures/retrieval_expectations.yaml` — including queries for things the
profile genuinely lacks, which must *not* come back looking answered.

**Both layout constants are measured, not guessed.** `CHARS_PER_LINE = 109`
comes from compiling probe bullets at lengths 60–140 and reading back where
wrapping starts. The one-page line budget comes from compiling six different
profile shapes and solving for each element's cost; the resulting additive model
reproduces all six measurements with zero error, and `test_layout.py` pins that.

**`CHARS_PER_LINE` is measured.** It is 109 for this template, derived by
compiling probe bullets at every length from 60 to 140 and reading back from the
PDF where wrapping starts. The spec's ballpark was ~95. Re-run `make calibrate`
after any change to the template's geometry, font or list nesting.

---

## Licence note

`src/resume_agent/templates/jake_resume.tex.j2` is adapted from
[jakegut/resume](https://github.com/jakegut/resume), MIT licensed,
© 2020 Jake Gutierrez. The licence text is reproduced in the template header.
