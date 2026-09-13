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

from resume_agent.cache import ModelListCache, content_key
from resume_agent.latex.metrics import CHARS_PER_LINE, estimate_lines
from resume_agent.llm import (
    build_chat_model,
    load_prompt,
    model_for,
    prompt_version,
    structured_output,
)
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


def tailoring_cache_key(job: JobSpec, bullet_ids: list[str], model: str) -> str:
    """Identity of "these bullets, rewritten for this posting, first attempt"."""
    return content_key(
        job.source_hash,
        "|".join(sorted(bullet_ids)),
        prompt_version(PROMPT_NAME),
        model,
    )


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
    use_cache: bool = True,
    model: str | None = None,
) -> list[TailoredBullet]:
    """Rewrite a batch of bullets in one call.

    Called once per section by the caller, per spec 5's "batch by section".

    **Only the first attempt is cached.** A retry exists precisely because the
    previous output was rejected, and its prompt carries critiques that change
    what a correct answer looks like -- so caching a retry would either serve
    back the rejected text or key on critique strings that never repeat. Caching
    the critique-free first attempt captures nearly all the saving anyway,
    because that is the call every run makes.
    """
    if not sources:
        return []

    # Resolved at call time, not as a default argument: the id is part of the
    # cache key, so freezing it at import would let one provider serve another's
    # rewrites.
    model = model or model_for()

    critiques = critiques or {}
    cache = ModelListCache(TailoredBullet, "tailored")
    cacheable = not any(critiques.get(source.id) for source in sources)
    key = tailoring_cache_key(job, [s.id for s in sources], model)

    if use_cache and cacheable:
        cached = cache.get(key)
        if cached is not None:
            logger.info("tailor_bullets: cache hit (%d bullets)", len(cached))
            return cached

    llm = llm or build_chat_model(model=model)
    structured = structured_output(llm, TailoringResult)

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

    if use_cache and cacheable:
        cache.put(key, tailored)
    return tailored
