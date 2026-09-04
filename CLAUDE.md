# CLAUDE.md — standing instructions for this repo

## What this project is

A LangGraph agent that takes a job description and produces a truthful, one-page,
ATS-clean LaTeX resume (Jake's Resume template) plus a matching cover letter,
grounded entirely in a structured career knowledge base in `profile/`.

Full design: `RESUME_AGENT_SPEC.md`. Read it before making architectural decisions.
If something here contradicts the spec, the spec wins — flag the conflict.

## Non-negotiable rules

1. **Never fabricate resume content.** Generated bullets may only rephrase a
   `canonical` source string. Every number must trace to a key in that bullet's
   `metrics` dict. Every technology named must exist in `skills.yaml`. If a
   verification check is inconvenient, make the generation better — do not
   loosen the check.
2. **The LLM judges; Python counts.** Never ask a model to count lines, sum
   scores, check page counts, or enforce a budget. Those are deterministic
   functions with tests.
3. **The agent controls content length, never page geometry.** Margins, font
   size and `\vspace` are not template variables. Do not expose them.
4. **Keep `\input{glyphtounicode}` and `\pdfgentounicode=1`** in the resume
   template. Without them the PDF does not extract as text and the whole
   ATS premise collapses.
5. **Prompts live in `src/resume_agent/prompts/*.md`**, never as inline Python
   string literals. They are versioned artifacts.
6. **All graph edges are defined in `src/resume_agent/graph/build.py`.** Nodes
   import nothing from each other.
7. **Every loop has a hard iteration cap** and a defined behaviour at the cap.
   Grounding: 2 retries then drop the bullet with a loud log. Layout: 3 retries
   then fail with the best artifact so far and a clear message.

## Stack

- Python 3.12, `uv` for dependency management
- `langgraph` 1.x (`StateGraph`, `interrupt`, `Command` from `langgraph.types`)
- `langchain-core` + a provider package for structured output
- `pydantic` v2 everywhere data crosses a boundary
- `jinja2` with **custom LaTeX delimiters** (`\VAR{}`, `\BLOCK{}`) — see spec §6.1
- `tectonic` for compilation; `pypdf` for inspection
- `chromadb` or `sqlite-vec` + `rank_bm25` for hybrid retrieval
- `typer` for CLI, `pytest` for tests, `ruff` for lint
- `langsmith` for tracing (optional, env-gated)

LangGraph 1.x moves fast. Before writing graph code, check the current API at
`docs.langchain.com/oss/python/langgraph`. Do not rely on remembered signatures
for `interrupt`, `Command`, streaming, or checkpointer imports.

## Working style

- **Plan before building.** For any milestone, propose the file list and the
  node contracts, and wait for approval before writing code.
- **One milestone at a time.** Do not start M(n+1) until M(n)'s definition of
  done passes. Do not "helpfully" scaffold future milestones.
- **Tests are part of done**, not a follow-up. Every node gets at least a
  happy-path test with a fake LLM.
- **Run the code.** Don't report a milestone complete without executing the
  tests and, where relevant, actually compiling a PDF and checking its page count.
- **Commit per milestone** with a message describing the DoD that now passes.
- **Ask rather than assume** on anything involving my actual career data,
  the profile schema, or a change to the graph topology.

## Things that will waste our time if you do them

- Adding a multi-agent supervisor, a ReAct loop, or an autonomous planner.
  The topology in the spec is deliberate.
- Reaching for a hosted vector DB. The knowledge base is ~200 documents.
- Chunking the knowledge base. The bullets are already atomic; that's the point.
- Inventing dependencies. If you want a library not listed above, propose it
  and say why.
- Editing anything under `profile/` — that is my real data. Use
  `profile.example/` for tests and development.
- Silently swallowing a LaTeX compile failure. The log is the feedback signal;
  capture it, never `except: pass` it.

## Definitions of done

Live in `RESUME_AGENT_SPEC.md` §8. Quote the relevant one when you claim a
milestone is finished, and show the command output that proves it.
