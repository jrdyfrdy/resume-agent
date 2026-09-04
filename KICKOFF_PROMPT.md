# Claude Code kickoff

## How to use these three files

```bash
mkdir resume-agent && cd resume-agent && git init
# copy RESUME_AGENT_SPEC.md, CLAUDE.md and this file into the repo root
claude
```

Then press `Shift+Tab` to enter **plan mode** and paste the prompt below.
Plan mode matters here: you want to review the file layout and node contracts
before any code exists, because refactoring a graph is much more annoying than
refactoring a function.

---

## Prompt for message 1 (M0)

> I'm building a LangGraph agent that tailors a LaTeX resume and cover letter to
> a specific job description, grounded in a structured career knowledge base.
> The full design is in `RESUME_AGENT_SPEC.md` and my standing rules are in
> `CLAUDE.md`. Read both before you do anything else.
>
> I'm doing this partly to build the tool and partly to learn LangChain and
> LangGraph properly, so favour explicit, readable constructions over clever
> abstractions, and add a short comment wherever you make a non-obvious design
> choice explaining the reasoning. I'd rather understand every line than have
> fewer of them.
>
> Today we're doing **M0 only**: the deterministic LaTeX pipeline, with no LLM
> anywhere in it. Definition of done is in spec §8.
>
> Scope for M0:
> 1. `pyproject.toml` with `uv`, Python 3.12, and only the deps M0 actually
>    needs (pydantic, jinja2, typer, pypdf, pyyaml, pytest, ruff).
> 2. Pydantic models for the profile schema exactly as described in spec §3.1
>    and §3.2 — evidence units with `canonical`, `metrics`, `skills`, `themes`,
>    `confidence`, `evidence`.
> 3. A `profile.example/` directory: one fake person, two jobs, three projects,
>    a `skills.yaml` with aliases, and enough bullets to exercise selection later.
>    Deliberately include strings that stress the escaper — an employer with an
>    ampersand, a `C#` skill, a `100% uptime` claim, a `user_id` mention.
> 4. `latex/escape.py` with `latex_escape()` and `latex/env.py` with the custom
>    Jinja environment from spec §6.1.
> 5. `templates/jake_resume.tex.j2` — fetch the real Jake's Resume template from
>    `github.com/jakegut/resume` (MIT), then adapt it to the custom Jinja
>    delimiters. Keep `\input{glyphtounicode}` and `\pdfgentounicode=1`.
> 6. `latex/compile.py` — tectonic with a runtime fallback chain and a clear
>    install message if nothing is available. Never raise on a compile failure;
>    return the log.
> 7. `latex/inspect.py` — page count via pypdf, overfull hbox extraction from
>    the log, first LaTeX error extraction.
> 8. `latex/metrics.py` — empirically calibrate `chars_per_line` for this
>    template by rendering test strings and measuring where wrapping occurs.
>    Hardcode the constant with a comment showing how you derived it.
> 9. CLI: `resume-agent build --profile profile.example --out out/` producing a
>    one-page PDF from all example content.
> 10. Tests: the full escape fixture list from spec §6.2, plus a golden `.tex`
>     snapshot, plus an end-to-end test that actually compiles and asserts
>     `page_count == 1`.
>
> Start by giving me the plan: the file tree, the public signature of every
> function you'll write, and any decisions where the spec left you a choice.
> Don't write code until I approve the plan.

---

## Prompts for later milestones

Keep them in this shape. The pattern is: name the milestone, restate its DoD,
list the scope, ask for a plan first.

**M1 — knowledge base and retrieval**

> M0 is done and committed. Now M1: load `profile.example/` into SQLite plus a
> vector index, and build hybrid retrieval (BM25 + dense, fused with RRF, k=60)
> with query expansion through the `skills.yaml` alias table. Add
> `resume-agent search "<query>"` so I can inspect results by hand. DoD in spec
> §8. Show me the plan first, and include how you'll test retrieval quality
> without a labelled dataset.

**M2 — JD parsing**

> M2: `parse_jd` node producing a `JobSpec` via structured output, cached on
> `sha256(raw_jd)`. Save three real job descriptions to `evals/datasets/jds/`
> (I'll paste them) and snapshot-test the parses. The prompt should distinguish
> stated requirements from inferred priorities — see spec §5.

**M3 — scoring, selection and the `analyze` command**

> M3: batched relevance scoring in one LLM call, deterministic `FitReport`
> assembly, and the greedy knapsack selection from spec §5 with every constraint
> enforced. Then `resume-agent analyze --jd x.txt` printing the fit report.
> Property-based tests on the selection constraints. This is the first thing
> I'll use daily, so the CLI output needs to be genuinely readable.

**M4 — tailoring and the verifier**

> M4: `tailor_bullets` and `verify_grounding` per spec §5. The deterministic
> layer runs first and free; the judge only sees bullets that pass it. Write
> `test_verifier.py` with deliberately fabricated bullets — an invented metric,
> an out-of-vocabulary technology, and a semantic inflation ("led a team" from
> "collaborated with two engineers") — and assert all three are caught.

**M5 — graph assembly and the feedback loops**

> M5: wire everything into a `StateGraph` in `graph/build.py`. Both cycles from
> spec §2 with their iteration caps and cap behaviours. Check the current
> LangGraph 1.x API at docs.langchain.com before writing this. Then prove the
> loops work: inject a LaTeX syntax error and show it auto-repairs; force a
> two-page overflow and show it converges to one page. Also generate the mermaid
> diagram of the graph and save it to the README.

---

## Two habits worth keeping

**Ask for the reasoning, not just the code.** Every few milestones, ask "walk me
through why the selection node is deterministic instead of an LLM call, and what
would break if we changed it." You're building this to learn; the explanation is
half the deliverable, and it's also literally the interview answer.

**Use `/clear` between milestones.** Each one is self-contained and the spec is
on disk. Carrying M2's context into M5 mostly just crowds out room for the code.
