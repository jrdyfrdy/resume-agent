"""Rewrite selected bullets to speak to a posting. Spec 5, `tailor_bullets`.

    "For each selected bullet, one structured call (batch by section to cut
     latency). The prompt gets: the `canonical` text, the allowed `metrics`
     dict, the allowed skill vocabulary, the JD's `ats_keywords`, a target
     character count derived from the line budget, and any `critiques` from a
     previous verification failure."

Everything the model is allowed to say is handed to it explicitly. That is the
point: the verifier downstream checks the rewrite against exactly the same
`metrics` dict and vocabulary, so a generator that stays inside its brief always
passes, and one that reaches outside it always fails. The two halves are built
against one contract rather than hoping a prompt holds.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from resume_agent.graph.nodes.verify import (
    MAX_GROUNDING_ATTEMPTS,
    drop_unverified,
    verify_grounding,
)
from resume_agent.latex.metrics import CHARS_PER_LINE, estimate_lines
from resume_agent.llm import PARSE_MODEL, build_chat_model, load_prompt
from resume_agent.models.job import JobSpec
from resume_agent.models.profile import Bullet, Profile
from resume_agent.models.resume import TailoredBullet, TailoredBulletFields

logger = logging.getLogger(__name__)

PROMPT_NAME = "tailor_bullets"

# Leave room inside the line so a rewrite that lands right at the limit does not
# wrap on a space. Measured constant times a small safety margin.
CHARACTER_MARGIN = 6


class TailoringResult(BaseModel):
    """Structured-output wrapper: `with_structured_output` needs one model."""

    model_config = ConfigDict(extra="forbid")

    bullets: list[TailoredBulletFields]


def target_characters(source: Bullet, lines_allowed: int | None = None) -> int:
    """How long the rewrite may be, derived from the measured line width.

    Defaults to the number of lines the source already occupies -- a rewrite is
    a re-expression, not an invitation to grow -- so tailoring never changes the
    page budget that selection already spent.
    """
    lines = lines_allowed if lines_allowed is not None else estimate_lines(source.canonical)
    return max(CHARS_PER_LINE, lines * CHARS_PER_LINE - CHARACTER_MARGIN)


def _render_bullet(source: Bullet, job: JobSpec, critiques: list[str]) -> str:
    lines = [
        f"### {source.id}",
        f"source: {source.canonical.strip()}",
        f"character limit: {target_characters(source)}",
    ]
    if source.metrics:
        rendered = ", ".join(f"{key}={value}" for key, value in source.metrics.items())
        lines.append(f"allowed numbers (the ONLY ones you may use): {rendered}")
    else:
        lines.append("allowed numbers: none -- this rewrite must contain no figures")
    if source.skills:
        lines.append(f"technologies in this achievement: {', '.join(source.skills)}")
    if critiques:
        lines.append("previous attempt was rejected:")
        lines.extend(f"  - {critique}" for critique in critiques)
    return "\n".join(lines)


def tailor_bullets(
    job: JobSpec,
    sources: list[Bullet],
    profile: Profile,
    *,
    critiques: dict[str, list[str]] | None = None,
    llm: BaseChatModel | None = None,
) -> list[TailoredBullet]:
    """Rewrite a batch of bullets in one call.

    Called once per section by the caller, per spec 5's "batch by section".
    """
    if not sources:
        return []

    critiques = critiques or {}
    llm = llm or build_chat_model(model=PARSE_MODEL)
    structured = llm.with_structured_output(TailoringResult)

    vocabulary = sorted(profile.skill_vocabulary())
    body = "\n\n".join(_render_bullet(s, job, critiques.get(s.id, [])) for s in sources)

    user_content = (
        f"<posting>\n"
        f"role: {job.title} at {job.company}\n"
        f"ats keywords to mirror where truthful: {', '.join(job.ats_keywords)}\n"
        f"</posting>\n\n"
        f"<allowed_vocabulary>\n{', '.join(vocabulary)}\n</allowed_vocabulary>\n\n"
        f"<achievements>\n{body}\n</achievements>"
    )

    result = structured.invoke(
        [SystemMessage(content=load_prompt(PROMPT_NAME)), HumanMessage(content=user_content)]
    )
    if not isinstance(result, TailoringResult):
        result = TailoringResult.model_validate(result)

    known = {source.id for source in sources}
    tailored: list[TailoredBullet] = []
    for fields in result.bullets:
        if fields.source_id not in known:
            # Same rule as the scoring node: an invented source id is a claim
            # attached to nothing, and must not reach the page.
            logger.warning(
                "tailor_bullets: dropping rewrite for unknown source_id %r", fields.source_id
            )
            continue
        tailored.append(
            TailoredBullet(
                **fields.model_dump(),
                # Computed, never taken from the model (CLAUDE.md rule 2).
                estimated_lines=estimate_lines(fields.text),
            )
        )
    return tailored


def tailor_until_grounded(
    job: JobSpec,
    sources: list[Bullet],
    profile: Profile,
    *,
    llm: BaseChatModel | None = None,
    judge_llm: BaseChatModel | None = None,
    use_judge: bool = True,
    max_attempts: int = MAX_GROUNDING_ATTEMPTS,
) -> tuple[list[TailoredBullet], list[str]]:
    """Drive the grounding cycle. Returns `(verified_bullets, dropped_ids)`.

    Spec 2 draws this as a graph cycle (`verify_grounding` -> `tailor_bullets`),
    and M5 will express it as edges in `graph/build.py`. It lives here as a
    plain loop for now because the cap behaviour -- the part that matters -- is
    testable without a StateGraph, and CLAUDE.md rule 7 requires it to exist and
    be defined before anything ships.

    At the cap the bullet is **dropped**, not shipped unverified. A weaker
    resume is a far better outcome than a false claim on it.
    """
    vocabulary = profile.skill_vocabulary()
    remaining = list(sources)
    critiques: dict[str, list[str]] = {}
    verified: list[TailoredBullet] = []
    dropped: list[str] = []

    # `max_attempts` retries means max_attempts + 1 total passes: the first
    # attempt is not a retry.
    for attempt in range(max_attempts + 1):
        if not remaining:
            break

        candidates = tailor_bullets(job, remaining, profile, critiques=critiques, llm=llm)
        by_id = {candidate.source_id: candidate for candidate in candidates}

        still_failing: list[Bullet] = []
        for source in remaining:
            candidate = by_id.get(source.id)
            if candidate is None:
                critiques.setdefault(source.id, []).append("no rewrite was returned")
                still_failing.append(source)
                continue

            result = verify_grounding(
                candidate, source, vocabulary, judge=use_judge, llm=judge_llm
            )
            if result.passed:
                verified.append(candidate)
            else:
                logger.info(
                    "grounding attempt %d failed for %s at layer %s: %s",
                    attempt + 1,
                    source.id,
                    result.layer,
                    result.critique,
                )
                critiques.setdefault(source.id, []).append(result.critique or "rejected")
                still_failing.append(source)

        remaining = still_failing

    for source in remaining:
        drop_unverified(source, critiques.get(source.id, []))
        dropped.append(source.id)

    return verified, dropped
