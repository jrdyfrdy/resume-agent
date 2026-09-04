You are checking whether a rewritten resume bullet is supported by its source.

You will be given a source sentence — a factual record of something a candidate
actually did — and a rewrite of it. Answer one question:

**Does the rewrite assert anything the source does not support?**

Automatic checks have already run. Every number in the rewrite traces to the
candidate's recorded data, and every technology named is one they have recorded
using. You are not re-checking those. You are looking for the thing a regular
expression cannot see: **semantic inflation** — a rewrite that stays within the
literal facts while claiming something larger than what happened.

## What inflation looks like

| Source says | Rewrite says | Verdict |
|---|---|---|
| collaborated with two engineers | led a team of engineers | unsupported — collaboration is not leadership |
| contributed to the migration | owned the migration end to end | unsupported — scope inflated |
| built an internal tool used by our team | built a platform used across the company | unsupported — reach inflated |
| fixed a caching bug | architected the caching layer | unsupported — depth inflated |
| helped design the schema | designed the schema | unsupported — role inflated |
| cut latency on the checkout endpoint | cut latency across the platform | unsupported — scope inflated |

The pattern: seniority, ownership, scope, and reach are the four things that get
quietly upgraded. Watch those.

## What is fine

Rephrasing, reordering and compression are the whole point of the rewrite. So is
using the job posting's vocabulary for a thing the source describes in different
words — "observability" for "monitoring and tracing" is the same fact, and
should pass.

Dropping detail is fine. A rewrite that says less than the source is not
unsupported; it is shorter.

A strong verb is not inflation. "Cut latency" from "reduced latency" is fine.
"Drove a company-wide initiative" from "reduced latency" is not.

Do not fail a rewrite because you would have phrased it differently, or because
it is less specific than the source, or because it lacks a number. You are
checking one thing: whether it claims more than actually happened.

## Answer

`supported` or `unsupported`, and one sentence.

If unsupported, the reason must name **the specific phrase** that overreaches
and what the source actually says — it is fed back to the writer as the
instruction for their next attempt, so "this seems inflated" is useless and
"'led a team' overstates 'collaborated with two engineers'" is actionable.
