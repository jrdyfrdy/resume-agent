# JD parse snapshots

One committed `JobSpec` JSON per job description in `evals/datasets/jds/`.

These are **checked in on purpose**. They let `test_parse_jd.py` assert M2's
definition of done — "3 saved JDs parse to valid `JobSpec`" — on a machine with
no API key and in CI, without spending money on every test run. The snapshot
tests that read them skip cleanly when a file is absent.

## Generating them

Needs `ANTHROPIC_API_KEY`. Roughly $0.20 for all three on Claude Opus 5.

```bash
REGEN_SNAPSHOTS=1 uv run pytest tests/test_parse_jd.py -m llm
```

## What is and isn't asserted

The snapshot tests check **structure and a few load-bearing judgements** —
seniority per level, that the junior posting produces red flags, that the senior
posting doesn't produce invented ones, that the mid posting yields inferred
priorities.

They deliberately do **not** assert byte equality against a fresh parse. The
model is not deterministic, and a test demanding identical output would fail for
reasons that say nothing about correctness. A snapshot here is a record of what
the parser produced, not a contract that it must produce it again exactly.

A snapshot goes stale if the JD text changes — `test_snapshot_is_a_valid_jobspec`
catches that by comparing `source_hash` against the current file.
