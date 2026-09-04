# Job description dataset

Three postings at junior / mid / senior level, used by M2's snapshot tests and
(from M8) by the eval harness.

## ⚠️ These three are SYNTHETIC

`junior.txt`, `mid.txt` and `senior.txt` were written to exercise the parser,
not copied from real postings. They are deliberately loaded with the things
`prompts/parse_jd.md` has to get right:

| File | What it is there to test |
|---|---|
| `junior.txt` | 16 "requirements" for a 2-years-experience role, and an explicit hours disclaimer — the `red_flags` case, and spec §5's own example of a posting that over-asks. Startup `tone`. |
| `mid.txt` | On-call, incidents, postmortems, SLOs and toil recur in five separate places while never appearing as a single stated requirement — the `is_inferred` case, straight from spec §5 ("a JD that mentions on-call three times cares about ops maturity even if it never says so"). Formal `tone`. |
| `senior.txt` | A clean, well-written posting: `red_flags` should come back **empty**. Tests that the parser does not invent concerns to look useful. Conversational `tone`. |

Real postings are messier than these — bullet soup, buried must-haves,
boilerplate legal text, inconsistent capitalisation. **Snapshots taken against
synthetic JDs are testing an easier problem than the real thing.**

## Swapping in real postings

Drop the real text into the same three filenames and regenerate:

```bash
uv run resume-agent parse-jd --jd evals/datasets/jds/junior.txt
REGEN_SNAPSHOTS=1 uv run pytest tests/test_parse_jd.py -k snapshot
```

Then delete this warning section. Nothing else needs to change — the tests read
whatever is in these files, and the parse cache is keyed on a hash of the
content, so new text simply misses the cache and re-parses.
