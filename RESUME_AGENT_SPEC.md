# Resume & Cover Letter Tailoring Agent — Technical Spec

**Status:** design doc / build contract
**Owner:** you
**Purpose:** given a job description, produce a truthful, one-page, ATS-clean LaTeX resume and a matching cover letter, grounded entirely in a structured career knowledge base.

---

## 1. What makes this hard (and therefore worth building)

The naive version is one prompt: *"here's my resume, here's a JD, rewrite it."* That version invents metrics, drifts into generic language, silently overflows to two pages, and gives you no way to know whether it got better or worse when you change the prompt.

The real problem decomposes into four sub-problems, and each one maps to a skill you want on your CV as an AI engineer:

| Sub-problem | What it really is | Skill it demonstrates |
|---|---|---|
| "Which of my 40 achievements matter for *this* job?" | Retrieval + relevance scoring | RAG, hybrid search, reranking |
| "Which 18 of those fit on one page?" | Constrained selection under a budget | Knowing when *not* to use an LLM |
| "Rewrite them without lying" | Grounded generation + verification | Guardrails, LLM-as-judge, structured output |
| "Make it actually compile to one page" | Self-correcting loop with a real oracle | LangGraph cycles, tool feedback, retry limits |

That last one is the interesting bit. Most agent demos have no ground truth — the LLM says it's done and you take its word. Here you have `pdfinfo` telling you the page count and a LaTeX log telling you what broke. **That is a real environment signal**, which makes this a genuinely agentic system rather than a chatbot with extra steps.

**Design principle that runs through the whole thing:** the LLM makes *judgments* (is this bullet relevant? does this phrasing match the JD's register?). Deterministic code does *arithmetic and verification* (does this fit in 43 lines? is this number present in the source data? did it compile?). Every time you're tempted to ask the model to count something, don't.

---

## 2. System overview

```
                         ┌──────────────────────┐
   job description  ───► │   parse_jd           │  JD text → JobSpec (Pydantic)
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  company_research    │  optional web search → CompanyBrief
                         └──────────┬───────────┘
                                    ▼
   career KB (YAML) ────► ┌──────────────────────┐
   + hybrid index         │  retrieve_evidence   │  per-requirement top-k evidence units
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  score_fit           │  FitReport: covered / partial / gaps
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  select_content      │  knapsack under line budget (pure Python)
                         └──────────┬───────────┘
                                    ▼
              ┌────────► ┌──────────────────────┐
              │          │  tailor_bullets      │  grounded rewrite, structured out
              │          └──────────┬───────────┘
              │                     ▼
              │          ┌──────────────────────┐
              └──────────│  verify_grounding    │  fabrication gate (code + judge)
                 critique└──────────┬───────────┘
                                    ▼
              ┌────────► ┌──────────────────────┐
              │          │  render_latex        │  Jinja2 → .tex
              │          └──────────┬───────────┘
              │                     ▼
              │          ┌──────────────────────┐
              │          │  compile_pdf         │  tectonic → .pdf + log
              │          └──────────┬───────────┘
              │                     ▼
              │          ┌──────────────────────┐
              └──────────│  inspect_output      │  pages? overfull hbox? errors?
                 reflow  └──────────┬───────────┘
                 or fix              │ ok
                                    ▼
                         ┌──────────────────────┐
                         │  write_cover_letter  │  subgraph
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  human_review        │  interrupt() → approve / edit
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │  finalize            │  PDFs + tracker row + run metadata
                         └──────────────────────┘
```

Two feedback cycles, both with hard iteration caps:
- **grounding cycle** (`verify_grounding` → `tailor_bullets`), max 2 retries
- **layout cycle** (`inspect_output` → `select_content` or `render_latex`), max 3 retries

---

## 3. The career knowledge base

This is the foundation. Get the data model wrong and everything downstream is mush.

### 3.1 Atomic evidence units

Do **not** store your resume as prose. Store it as atomic, tagged achievements. Source of truth is YAML on disk, git-versioned, hand-editable.

```yaml
# profile/experience/acme_backend.yaml
id: exp_acme_be
type: experience
org: Acme Corp
title: Backend Engineer
location: Remote
start: 2023-06
end: 2025-01
tech: [python, fastapi, postgresql, redis, aws, docker]
bullets:
  - id: exp_acme_be.b1
    canonical: >
      Cut p95 API latency from 820ms to 190ms by adding Redis read-through
      caching and eliminating N+1 queries across 12 endpoints
    skills: [redis, caching, sql-optimization, performance, fastapi]
    themes: [performance, backend, cost]
    metrics:
      latency_before_ms: 820
      latency_after_ms: 190
      endpoints_touched: 12
    seniority_signal: mid
    evidence: "PR #4412; Grafana p95 dashboard, Nov 2024"
    confidence: verified   # verified | approximate | claim
```

Field-by-field rationale:

- **`canonical`** — the ground truth sentence. The agent may rephrase it; it may never contradict it.
- **`metrics`** — structured numbers. **The hard rule: any digit in a generated bullet must trace to a value in `metrics`.** This single constraint kills 90% of resume hallucination, and it's checkable with a regex, not a judge.
- **`skills`** — used for retrieval and for the vocabulary allow-list.
- **`themes`** — coarse buckets for diversity constraints during selection (don't pick five performance bullets for a job that also wants API design).
- **`confidence`** — lets you mark soft claims. Bullets marked `claim` can be excluded via a strict mode.
- **`evidence`** — for you, not the model. It's what you'll cite when an interviewer asks "how'd you measure that?" Fill it in honestly; it will make you a better candidate independent of the tool.

### 3.2 Other collections

```
profile/
├── identity.yaml           name, email, phone, links, location, work auth
├── education.yaml
├── experience/*.yaml
├── projects/*.yaml         same bullet shape; add repo_url, live_url, role
├── skills.yaml             canonical skill vocabulary + aliases
├── certifications.yaml
└── narratives/             raw material for cover letters (markdown)
    ├── why_ai_engineering.md
    ├── how_i_learn.md
    ├── hardest_bug.md
    └── values.md
```

`skills.yaml` deserves attention — it's the alias table that makes retrieval work:

```yaml
skills:
  - canonical: kubernetes
    aliases: [k8s, container orchestration, eks, gke]
    category: infrastructure
    level: working        # expert | working | familiar
    first_used: 2024-02
  - canonical: langgraph
    aliases: [lang graph, langgraph agents]
    category: ai-frameworks
    level: working
    first_used: 2026-06
```

The alias table serves double duty: it expands queries at retrieval time, and it acts as the **allow-list for the fabrication check** — if the generated bullet says "Kafka" and Kafka isn't in your vocabulary, that's a fabrication.

### 3.3 Indexing

Load YAML → validate with Pydantic → write to:
- **SQLite** for structured queries and the application tracker
- **Vector index** (`chromadb` or `sqlite-vec`) over `canonical + skills + themes` for semantic recall

Retrieval is **hybrid**: BM25 (via `rank_bm25`, it's fine at this scale) + dense, fused with Reciprocal Rank Fusion. Do not skip BM25. Job descriptions are full of exact tokens — "Terraform", "gRPC", "PySpark" — and dense embeddings are notoriously mushy about exact technology names. RRF with `k=60`:

```python
score(doc) = Σ_over_retrievers 1 / (60 + rank_in_that_retriever)
```

Your KB is maybe 200 documents. Everything fits in memory. Resist the urge to reach for a vector DB service.

---

## 4. Data models (Pydantic v2)

Sketch — Claude Code will flesh these out.

```python
class Requirement(BaseModel):
    text: str
    category: Literal["language","framework","tool","domain","practice","soft"]
    weight: int = Field(ge=1, le=5)          # how load-bearing in the JD
    is_must_have: bool

class JobSpec(BaseModel):
    company: str
    title: str
    seniority: Literal["intern","junior","mid","senior","staff","lead","unknown"]
    domain: str
    requirements: list[Requirement]
    responsibilities: list[str]
    ats_keywords: list[str]      # verbatim terms worth mirroring exactly
    culture_signals: list[str]
    tone: Literal["formal","conversational","startup","academic"]
    red_flags: list[str]         # unpaid overtime hints, 15 "must haves" for a junior role, etc.
    source_hash: str             # sha256 of raw JD → cache key

class EvidenceMatch(BaseModel):
    bullet_id: str
    requirement_text: str
    relevance: float             # 0-1, from the scoring node
    rationale: str

class FitReport(BaseModel):
    overall_fit: float
    covered: list[EvidenceMatch]
    partial: list[EvidenceMatch]
    gaps: list[Requirement]      # honest: no evidence exists
    recommendation: Literal["strong_apply","apply","stretch","skip"]

class TailoredBullet(BaseModel):
    source_id: str               # MUST reference a real bullet id
    text: str
    keywords_used: list[str]
    metrics_used: list[str]      # keys from source metrics
    estimated_lines: int

class ResumeDraft(BaseModel):
    sections: list[Section]
    total_estimated_lines: int
```

### Graph state

```python
class AgentState(TypedDict):
    # inputs
    raw_jd: str
    profile_path: str
    options: RunOptions

    # accumulated
    job_spec: JobSpec | None
    company_brief: CompanyBrief | None
    evidence: list[EvidenceMatch]
    fit_report: FitReport | None
    selected: list[str]                       # bullet ids
    tailored: list[TailoredBullet]
    resume_draft: ResumeDraft | None
    cover_letter: str | None

    # artifacts
    tex_path: str | None
    pdf_path: str | None
    compile_log: str | None
    page_count: int | None

    # control
    line_budget: int
    grounding_attempts: int
    layout_attempts: int
    critiques: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
```

Use `Annotated[list[X], operator.add]` only for genuinely append-only fields. Everything else is last-write-wins, which is what you want for things like `line_budget` that get revised in a loop.

---

## 5. Node contracts

Each node is a pure-ish function `(state) -> dict` returning only the keys it changed.

### `parse_jd`
Structured output via `llm.with_structured_output(JobSpec)`. Cache on `sha256(raw_jd)` — you will re-run this graph many times on the same JD while debugging, and you shouldn't pay for it twice. Prompt should push the model to distinguish stated requirements from *inferred* priorities (a JD that mentions "on-call" three times cares about ops maturity even if it never says so).

### `company_research` *(optional, flag-gated)*
Web search (Tavily or DuckDuckGo) for: what the company does, recent news, engineering blog, tech stack, funding stage. Output a `CompanyBrief` with a `sources` list. Feeds the cover letter's hook. Cap at 3 searches. Skip by default; enable with `--research`.

### `retrieve_evidence`
For each requirement: expand query using the alias table, run hybrid retrieval, take top-8. Deduplicate across requirements. This is a plain function, no LLM.

### `score_fit`
One LLM call, batched: given all requirements and all retrieved candidates, emit relevance scores + rationales. Batching matters — 20 separate calls is slow and expensive and produces inconsistent scoring because the model has no comparative context. One call with everything in view produces better-calibrated scores.

Then compute `FitReport` deterministically from the scores.

**Ship the FitReport as a standalone CLI command.** `resume-agent analyze --jd job.txt` telling you "you cover 7/9 must-haves, you're missing Kubernetes and Kafka, recommendation: apply" is genuinely useful on day one, before any of the LaTeX machinery exists.

### `select_content`
**No LLM here.** A greedy knapsack:

```
maximize Σ (relevance × requirement_weight × recency_decay)
subject to:
  Σ estimated_lines ≤ line_budget
  every experience entry that appears gets ≥ 2 bullets
  ≥ 1 bullet per covered must-have requirement (if evidence exists)
  ≤ 3 bullets sharing the same theme
  chronological order preserved within sections
```

`estimated_lines` = `ceil(len(text) / chars_per_line)` where `chars_per_line` is calibrated empirically for the template (~95 for Jake's Resume body text at 11pt with default margins). Calibrate it once by rendering known strings and measuring; hardcode the constant with a comment explaining how you got it.

Start greedy. If you want to be fancy later, `pulp` turns this into a proper ILP in about 30 lines — but greedy is within a few percent of optimal here and much easier to debug.

### `tailor_bullets`
For each selected bullet, one structured call (batch by section to cut latency). The prompt gets:
- the `canonical` text
- the allowed `metrics` dict
- the allowed skill vocabulary
- the JD's `ats_keywords`
- a target character count derived from the line budget
- any `critiques` from a previous verification failure

Rules stated explicitly in the system prompt:
1. You may rephrase. You may not add facts.
2. Every number must come from the provided metrics dict.
3. Every technology named must appear in the provided vocabulary.
4. Start with a strong past-tense verb. No "Responsible for". No "Utilized".
5. Mirror the JD's exact terminology where it is *truthfully* interchangeable with the source's terminology.
6. Stay under {n} characters.

### `verify_grounding`
Two layers, cheap first:

**Deterministic (runs always, free):**
```python
nums_generated = set(re.findall(r'\d+(?:\.\d+)?%?', tailored.text))
nums_allowed   = allowed_number_forms(source.metrics)   # 820 → {"820", "820ms", "0.82s"}
assert nums_generated <= nums_allowed

tech_generated = extract_tech_tokens(tailored.text)     # capitalized + known-tech regex
assert tech_generated <= global_skill_vocabulary
```

**LLM judge (runs only on bullets that pass layer 1):** "Given the source fact and the rewritten bullet, does the rewrite assert anything not supported by the source? Answer with `supported` / `unsupported` + reason." Use a cheap model. This catches semantic inflation that regex can't — "led a team of engineers" when the source says "collaborated with two engineers".

On failure: append the critique to `state["critiques"]`, increment `grounding_attempts`, route back to `tailor_bullets`. After 2 failures, **drop the bullet and log it loudly**. Never ship an unverified claim because the retry budget ran out.

Write a test that feeds the verifier a deliberately fabricated bullet and asserts it gets caught. That test is the most important one in the repo.

### `render_latex`
Jinja2 → `.tex`. See §6 for the landmines.

### `compile_pdf`
Shell out to `tectonic`. Capture stdout/stderr into `compile_log`. Never raise — a failed compile is data for the next node, not an exception.

### `inspect_output`
- Page count via `pypdf`
- `grep 'Overfull \\hbox' compile_log` → count and extract the offending line numbers
- Compile errors → extract the first `! ` line, which is where LaTeX actually tells you what's wrong

Routes:
- compile error → `fix_latex` node (LLM sees the error + the offending template region)
- pages > 1 → reduce `line_budget` by 8%, back to `select_content`
- overfull hboxes → back to `tailor_bullets` with "shorten bullet X by ~15 chars"
- clean → forward

### `write_cover_letter` *(subgraph)*
Inputs: `JobSpec`, `CompanyBrief`, top-3 `EvidenceMatch`, `narratives/`.
Structure: hook → proof ¶1 → proof ¶2 → why-this-company → close.
Constraints: ≤ 320 words. No "I am writing to apply for". No adjective without evidence behind it. **Consistency check: every claim must also be supported by the same KB, and must not contradict the resume's selected bullets.** Renders to its own LaTeX template.

### `human_review`
```python
decision = interrupt({
    "fit_report": state["fit_report"],
    "selected_bullets": state["tailored"],
    "cover_letter": state["cover_letter"],
    "pdf_path": state["pdf_path"],
    "gaps": state["fit_report"].gaps,
})
```
Resume with `Command(resume={"action": "approve" | "revise", "notes": "..."})`. Requires a checkpointer and a stable `thread_id`. Gate behind `--interactive` so batch runs don't block.

### `finalize`
Copy artifacts to `out/{company}_{role}_{date}/`, write `run.json` (JobSpec, fit report, bullet IDs used, model versions, token cost, git SHA of the profile), insert a tracker row.

That `run.json` is your future dataset. Six months from now you'll want to ask "which bullets appear in applications that got callbacks?" — you can only answer that if you logged it from the start.

---

## 6. LaTeX: where projects like this actually die

Read this section twice. These four issues account for most of the pain.

### 6.1 Jinja2 and LaTeX both want `{}` and `%`

Default Jinja delimiters (`{{ }}`, `{% %}`) collide catastrophically with LaTeX. Configure a custom environment:

```python
LATEX_JINJA = jinja2.Environment(
    block_start_string=r"\BLOCK{",   block_end_string="}",
    variable_start_string=r"\VAR{",  variable_end_string="}",
    comment_start_string=r"\#{",     comment_end_string="}",
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
    loader=jinja2.FileSystemLoader("src/resume_agent/templates"),
)
```

Now your template reads `\VAR{name}` and `\BLOCK{for b in bullets}` — valid-looking LaTeX that your editor won't fight you about.

### 6.2 Escaping

`& % $ # _ { } ~ ^ \` all need escaping. Write `latex_escape()`, register it as a Jinja filter, and apply it to **every** interpolated value.

```python
_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
```
Order matters — backslash first, or you'll double-escape your own replacements.

Test fixtures that must pass: `AT&T`, `C#`, `100% uptime`, `user_id`, `Node.js ^18`, `~50ms`, `R&D`. Every one of these is a realistic resume string and every one of them breaks a naive implementation.

### 6.3 Jake's Resume specifics

The template is MIT-licensed (`github.com/jakegut/resume`). Fetch it and adapt rather than reproducing from memory. What matters:

- Custom macros to preserve: `\resumeSubheading{title}{dates}{org}{location}`, `\resumeItem{text}`, `\resumeSubHeadingListStart/End`, `\resumeItemListStart/End`, `\resumeProjectHeading{}{}`
- Requires `fontawesome5` — tectonic will fetch it
- **Critical for ATS:** keep `\input{glyphtounicode}` and `\pdfgentounicode=1`. This makes the PDF's ligatures and glyphs extract as real text. Drop it and your beautifully typeset resume parses as garbage in a keyword scanner. This is the single highest-leverage line in the file.
- Margins are already aggressive. **Rule: the agent controls content length, never the geometry.** If it starts shrinking margins to fit, it will produce something that looks desperate and prints badly. Put this in the prompt and enforce it by not exposing geometry as a template variable at all.

### 6.4 Compiler choice

`tectonic` — single self-contained binary, fetches only the packages it needs, reproducible, no 5GB TeX Live install. Install via `cargo`, Homebrew, or the release binaries.

Fallback chain, detected at runtime: `tectonic` → `latexmk -pdf` → Docker `texlive/texlive`. Fail with a clear install message rather than a stack trace.

---

## 7. Repository layout

```
resume-agent/
├── pyproject.toml
├── CLAUDE.md                        # standing instructions for Claude Code
├── README.md
├── .env.example
├── Makefile                         # setup / test / eval / run
├── profile/                         # YOUR data — gitignored or private submodule
│   ├── identity.yaml
│   ├── education.yaml
│   ├── skills.yaml
│   ├── experience/
│   ├── projects/
│   └── narratives/
├── profile.example/                 # fake profile, committed, used by tests
├── src/resume_agent/
│   ├── config.py                    # pydantic-settings
│   ├── models/
│   │   ├── profile.py  job.py  resume.py  state.py
│   ├── kb/
│   │   ├── loader.py                # YAML → Pydantic → SQLite
│   │   ├── index.py                 # embeddings + BM25
│   │   └── retriever.py             # hybrid + RRF
│   ├── graph/
│   │   ├── build.py                 # StateGraph assembly, the only place edges live
│   │   ├── state.py
│   │   └── nodes/
│   │       ├── parse_jd.py  research.py  retrieve.py  score.py
│   │       ├── select.py  tailor.py  verify.py
│   │       ├── render.py  compile.py  inspect.py  fix_latex.py
│   │       ├── cover_letter.py  review.py  finalize.py
│   ├── latex/
│   │   ├── env.py  escape.py  compile.py  inspect.py  metrics.py
│   ├── templates/
│   │   ├── jake_resume.tex.j2
│   │   └── cover_letter.tex.j2
│   ├── prompts/                     # versioned .md files, NOT inline strings
│   ├── tracker/
│   │   ├── db.py  models.py
│   └── cli.py                       # typer
├── tests/
│   ├── test_escape.py               # the fixture list from §6.2
│   ├── test_selection.py            # budget constraints hold
│   ├── test_verifier.py             # fabrication is caught
│   ├── test_render_compile.py       # golden .tex + real PDF
│   └── test_graph_smoke.py          # fake LLM, full traversal
├── evals/
│   ├── datasets/jds/*.txt
│   ├── judges/
│   └── run_eval.py
└── out/
```

Two structural decisions worth defending:

**Prompts live in files, not in Python string literals.** You will iterate on prompts far more than on code. Files mean clean diffs, easy A/B, and the ability to load a prompt by version in an eval.

**`build.py` is the only place edges are defined.** Nodes never know about each other. This keeps the graph readable as a single artifact and makes it trivial to draw with `graph.get_graph().draw_mermaid_png()`.

---

## 8. Milestones

Each has a hard definition of done. Do not start N+1 until N's DoD passes.

**M0 — Skeleton + LaTeX pipeline, zero LLM.**
`profile.example/` → Pydantic → Jinja → `.tex` → tectonic → 1-page PDF.
*DoD:* `make test` green; `out/resume.pdf` opens; the escape fixture list all render correctly; `AT&T` appears as "AT&T" not a crash.
This milestone is deliberately LLM-free. If the deterministic half doesn't work, adding a language model on top just makes the failures harder to read.

**M1 — KB + hybrid retrieval.**
*DoD:* `resume-agent search "kubernetes"` returns the right bullets; BM25-only and dense-only both return sensible results before you fuse them.

**M2 — JD parsing.**
*DoD:* 3 saved JDs (junior, mid, senior) parse to valid `JobSpec`; snapshot tests; cache hit on second run.

**M3 — Scoring + selection + `analyze` command.**
*DoD:* `resume-agent analyze --jd x.txt` prints a fit report; selection respects every constraint in §5 under property-based tests.
**This is the first genuinely useful deliverable.** Ship it, use it for a week on real postings, and let that tell you whether the scoring is calibrated before you build on top of it.

**M4 — Tailoring + verifier.**
*DoD:* `test_verifier.py` catches an injected fabrication; verified bullets never contain out-of-vocabulary numbers.

**M5 — Graph wiring + the compile loop.**
*DoD:* inject a deliberate LaTeX syntax error → auto-repaired within 3 iterations; force a 2-page overflow → converges to 1 page; both loops respect their caps and fail gracefully at the limit.

**M6 — Cover letter subgraph.** *DoD:* ≤320 words, no claim absent from the KB, consistent with resume bullets.

**M7 — HITL + tracker + CLI polish.** *DoD:* `--interactive` pauses, survives process restart via checkpointer, resumes correctly.

**M8 — Eval harness.** *DoD:* 15 JDs, deterministic + judge scores, `make eval` produces a table; a deliberately worsened prompt shows a measurable score drop.

**M9 — API + UI.** FastAPI + SSE streaming of node events; minimal frontend.

---

## 9. Evaluation

This is what turns the project from "a thing I built" into "a thing I can talk about in an interview for 20 minutes."

**Deterministic checks** (run on every JD in the eval set, no LLM, no cost):
- compiles without error
- exactly 1 page
- zero overfull hboxes
- zero fabricated numbers (verifier layer 1)
- must-have keyword coverage ≥ 70% where evidence exists
- no bullet exceeds the character cap

**LLM-as-judge rubric** (1–5, one judge call per resume, with the JD in context):
relevance to JD · specificity of claims · ATS keyword alignment · tone match · absence of filler

**Human spot-check:** 3 resumes per eval run, reviewed by you. Judges drift; your eye is the calibration set.

Wire it to LangSmith datasets and `evaluate()`. Then make prompt changes gated: a PR that drops mean judge score by >0.3 doesn't merge. This is the habit that separates AI engineers from people who tweak prompts until the vibes improve.

---

## 10. Concept → milestone map

You said you're upskilling for AI engineering roles. Here's where each concept actually lands, so the project doubles as a curriculum:

| Concept | Where it shows up |
|---|---|
| Pydantic structured output | M2 `parse_jd`, M4 `tailor_bullets` |
| Prompt templating & versioning | M2 onward, `prompts/` |
| Embeddings + vector store | M1 |
| Hybrid retrieval + RRF | M1 |
| Document loaders / chunking decisions | M1 (and the lesson: your chunks are *already* atomic, so don't chunk) |
| `StateGraph`, reducers, conditional edges | M5 |
| Cycles with termination conditions | M5 (both loops) |
| Checkpointers & durable state | M7 |
| `interrupt()` / human-in-the-loop | M7 |
| Subgraphs | M6 |
| Streaming (`astream_events`) | M9 |
| Tool use with real environment feedback | M5 (compiler as tool) |
| Guardrails & verification | M4 |
| LLM-as-judge | M8 |
| Eval datasets & regression gating | M8 |
| Tracing, cost tracking, caching | throughout |

Note what's *not* here: no multi-agent supervisor, no autonomous planner, no ReAct loop over 40 tools. Those are fun but they're not what makes this work. A well-constrained graph with two feedback loops and a real verifier is a better engineering story than a swarm of agents that occasionally produces something plausible. If you want to add a supervisor pattern later, do it as an explicit experiment with the eval harness measuring whether it actually helped.

---

## 11. Stretch ideas, roughly by value

1. **Recruiter simulator.** An adversarial node that screens the finished resume cold — 8 seconds, reject/advance, reasons. Feed its objections back as critiques. This is the highest-value addition and it's a great demo.
2. **Multi-variant A/B.** Generate two versions with different emphasis; the tracker records which one you sent; correlate with callbacks. Real data on your own job search.
3. **Gap → learning plan.** The `FitReport` already knows what you're missing. Turn it into a ranked study list across all the jobs you're targeting — "Kubernetes appears in 8 of your 12 target roles."
4. **JD ingestion from URL.** httpx + readability. Careful with sites that require auth or forbid scraping; check robots.txt and ToS.
5. **Interview prep pack.** Same fit report → likely questions + your evidence for each.
6. **Local model for the tailoring node.** Ollama + a 7B for rewriting; keep the frontier model for parsing and judging. Measures cost/quality tradeoff, which is a real production skill.
7. **Portfolio site generation** from the same KB, so the resume and the website never drift apart.

Deliberately *not* recommended: auto-submitting applications through Workday/Greenhouse. It violates most sites' terms, it's detectable, and mass-applying is not actually the bottleneck in a job search.

---

## 12. Honest caveats

**On tailoring vs. lying.** The whole architecture is built around the assumption that tailoring means *emphasis and vocabulary alignment*, never invention. The verifier isn't a nice-to-have; it's the thing that makes the tool ethical to use. Keep it strict even when it's annoying, because the failure mode — getting caught having claimed experience you don't have — is far worse than a slightly weaker resume.

**On ATS.** Modern ATS parse PDFs reasonably well; the "you must submit .docx" advice is mostly outdated, and `\pdfgentounicode=1` handles the rest. Keyword stuffing is also largely counterproductive — a human reads the resume after the filter, and stuffed resumes read badly. The goal is honest alignment: using *their* word for the thing you actually did.

**On cost.** Cache JD parses by hash. `tailor_bullets` is the expensive node; batch it by section. A full run should land in the low tens of cents. If it doesn't, you're re-parsing something you already parsed.

**On the profile as the real bottleneck.** The agent can only surface what you've written down. Most of the value of this project will come from the afternoon you spend actually filling in `profile/` with honest, metric-bearing achievements — including the `evidence` field. Budget real time for that; it's not setup work, it's the substance.
