"""The fabrication gate. Spec 5, `verify_grounding`.

    "Two layers, cheap first. [...] On failure: append the critique, increment
     `grounding_attempts`, route back to `tailor_bullets`. After 2 failures,
     **drop the bullet and log it loudly**. Never ship an unverified claim
     because the retry budget ran out."

Spec 12 on why this module exists at all:

    "The verifier isn't a nice-to-have; it's the thing that makes the tool
     ethical to use. Keep it strict even when it's annoying, because the failure
     mode -- getting caught having claimed experience you don't have -- is far
     worse than a slightly weaker resume."

Layer 1 is two regex checks and costs nothing, so a fabricated metric is caught
before a single token is spent. Layer 2 is a judge on a cheaper model, and only
ever sees rewrites that already passed layer 1.

CLAUDE.md rule 1 governs any future edit here: "If a verification check is
inconvenient, make the generation better -- do not loosen the check."
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.grounding.numbers import unsupported_numbers
from resume_agent.grounding.vocabulary import unsupported_technologies
from resume_agent.llm import build_chat_model, load_prompt
from resume_agent.models.profile import Bullet
from resume_agent.models.resume import JudgeVerdict, TailoredBullet, VerificationResult

logger = logging.getLogger(__name__)

PROMPT_NAME = "verify_grounding"

# CLAUDE.md rule 7 and spec 5: two retries, then the bullet is dropped.
MAX_GROUNDING_ATTEMPTS = 2


# --- layer 1: free ----------------------------------------------------------


def verify_numbers(tailored: TailoredBullet, source: Bullet) -> VerificationResult:
    """Every number in the rewrite must trace to the metrics or the source sentence."""
    unsupported = unsupported_numbers(tailored.text, source.metrics, source.canonical)
    if not unsupported:
        return VerificationResult.ok(source.id)

    allowed = sorted(source.metrics.keys())
    return VerificationResult(
        bullet_id=source.id,
        passed=False,
        layer="numbers",
        critique=(
            f"The number(s) {sorted(unsupported)} appear in your rewrite but are not in "
            f"the source data. Only these metric keys are available: {allowed}. "
            f"Remove the invented figure, or re-express one that is actually recorded. "
            f"Do not compute new numbers from the existing ones."
        ),
    )


def verify_vocabulary(
    tailored: TailoredBullet, source: Bullet, vocabulary: set[str]
) -> VerificationResult:
    """Every technology newly named must exist in skills.yaml."""
    unsupported = unsupported_technologies(tailored.text, source.canonical, vocabulary)
    if not unsupported:
        return VerificationResult.ok(source.id)

    return VerificationResult(
        bullet_id=source.id,
        passed=False,
        layer="vocabulary",
        critique=(
            f"Your rewrite names {sorted(unsupported)}, which the candidate has no "
            f"recorded experience with. Describe only what the source sentence says: "
            f"{source.canonical.strip()!r}"
        ),
    )


# --- layer 2: the judge -----------------------------------------------------


def verify_with_judge(
    tailored: TailoredBullet,
    source: Bullet,
    *,
    llm: BaseChatModel | None = None,
) -> VerificationResult:
    """Catch semantic inflation that no regex can see.

    Spec 5's example: "led a team of engineers" when the source says
    "collaborated with two engineers". Both sentences are numerically and
    technologically clean; only the meaning is wrong.
    """
    llm = llm or build_chat_model("judge")
    structured = llm.with_structured_output(JudgeVerdict)

    verdict = structured.invoke(
        [
            SystemMessage(content=load_prompt(PROMPT_NAME)),
            HumanMessage(
                content=(
                    f"<source>\n{source.canonical.strip()}\n</source>\n\n"
                    f"<rewrite>\n{tailored.text.strip()}\n</rewrite>"
                )
            ),
        ]
    )
    if not isinstance(verdict, JudgeVerdict):
        verdict = JudgeVerdict.model_validate(verdict)

    if verdict.verdict == "supported":
        return VerificationResult.ok(source.id)

    return VerificationResult(
        bullet_id=source.id,
        passed=False,
        layer="judge",
        critique=(
            f"{verdict.reason} Stay within what the source actually says: "
            f"{source.canonical.strip()!r}"
        ),
    )


# --- the gate ---------------------------------------------------------------


def verify_grounding(
    tailored: TailoredBullet,
    source: Bullet,
    vocabulary: set[str],
    *,
    judge: bool = True,
    llm: BaseChatModel | None = None,
) -> VerificationResult:
    """Run the layers cheapest-first and return the first failure.

    Short-circuiting is not just an optimisation: a rewrite with an invented
    metric should never reach a paid call, and the numeric critique is more
    actionable than whatever the judge would say about the same sentence.
    """
    for check in (
        lambda: verify_numbers(tailored, source),
        lambda: verify_vocabulary(tailored, source, vocabulary),
    ):
        result = check()
        if not result.passed:
            return result

    if not judge:
        return VerificationResult.ok(source.id)

    return verify_with_judge(tailored, source, llm=llm)


def drop_unverified(source: Bullet, critiques: list[str]) -> None:
    """Log a bullet being dropped at the retry cap. Loudly, per spec 5.

    `logger.error` rather than a warning, and the full critique history rather
    than just the last one, because this is the event that silently makes a
    resume weaker -- and the only trace of it.
    """
    logger.error(
        "DROPPING bullet %s after %d failed grounding attempts; it will not appear on the "
        "resume. Critiques: %s | Source: %r",
        source.id,
        len(critiques),
        " || ".join(critiques),
        source.canonical.strip(),
    )
