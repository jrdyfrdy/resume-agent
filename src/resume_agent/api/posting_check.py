"""Is the pasted text a job ad at all? Asked before a run spends anything (M11 J2).

A run is a minute of model calls, and on the hosted site one of a person's
runs for the day. Pasting a resume, a cover letter or a stray paragraph into
the posting box used to spend all of that on a parse that could only produce
nonsense. Jev answers this one question in well under a second, for a fraction
of a cent.

**It refuses only the clear cases.** Below `REFUSE_BELOW` the paste is turned
away with a sentence saying what to paste instead. Anything above it goes
ahead, as it always did: a posting judged wrongly as "not a posting" would
cost someone a resume, while a non-posting let through costs one run, which
was the behaviour before this check.

**It is a courtesy, not a gate.** Jev switched off, slow or down means the run
goes ahead, exactly as before (`DecisionsUnavailable`).
"""

from __future__ import annotations

import logging

from resume_agent import decisions

logger = logging.getLogger(__name__)

# Below this probability that the paste is a job posting, it is refused.
REFUSE_BELOW = 0.2

# Enough for any real posting, and well inside Jev's input limit (64k tokens).
# Trimmed rather than refused: a very long paste is not evidence of anything.
MAX_CHARS = 24_000

REFUSAL = (
    "That doesn't look like a job ad, so no resume was made and nothing was used "
    "up. Paste the whole posting: the job title, what the job involves, and what "
    "it asks for."
)


def posting_refusal(text: str) -> str | None:
    """None to go ahead. Otherwise the sentence to show the person."""
    if not decisions.jev_enabled():
        return None
    try:
        decision = decisions.ask(
            text[:MAX_CHARS], {"is_posting": decisions.load_question("is_posting")}
        )
    except decisions.DecisionsUnavailable:
        return None

    probability = decision.answers["is_posting"].noul
    if probability < REFUSE_BELOW:
        logger.info("posting check: refused a paste (p=%.2f)", probability)
        return REFUSAL
    return None
