# Job description dataset

16 postings, used by M2's snapshot tests and by M8's eval harness.

The three original files (`junior`, `mid`, `senior`) were written to exercise
specific parser behaviours and are documented below. The other 13 exist to
give the eval set a spread: intern through staff, domains outside backend,
four different tones, and -- deliberately -- **several postings the example
profile should score badly on**.

That last part matters. An eval set where every resume scores 0.8 cannot
detect a regression, so `senior_ml`, `senior_security`,
`lead_engineering_manager` and `mid_embedded` are all outside the profile's
experience on purpose.

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
