"""Score retrieved evidence against a posting's requirements. Spec 5, `score_fit`.

    "One LLM call, batched: given all requirements and all retrieved candidates,
     emit relevance scores + rationales. Batching matters -- 20 separate calls is
     slow and expensive and produces inconsistent scoring because the model has
     no comparative context. One call with everything in view produces
     better-calibrated scores.
     Then compute `FitReport` deterministically from the scores."

Two halves, and the split is the point:

* `score_fit` makes **judgements** -- is this achievement evidence for that
  requirement, and why. One batched call.
* `build_fit_report` does **arithmetic** -- bucketing, weighting, the overall
  number, the recommendation. Pure Python, no model, fully testable.

Asking the model for the summary too would produce an `overall_fit` that drifts
between runs and cannot be unit-tested (CLAUDE.md rule 2).
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from resume_agent.cache import ModelListCache, content_key
from resume_agent.llm import build_chat_model, load_prompt, model_for, prompt_version
from resume_agent.models.fit import (
    COVERED_THRESHOLD,
    PARTIAL_THRESHOLD,
    EvidenceMatch,
    FitReport,
    compute_overall_fit,
    recommend,
)
from resume_agent.models.job import JobSpec
from resume_agent.models.profile import Profile

logger = logging.getLogger(__name__)

PROMPT_NAME = "score_fit"


class ScoredMatch(BaseModel):
    """One (achievement, requirement) judgement, as the model returns it."""

    model_config = ConfigDict(extra="forbid")

    bullet_id: str = Field(description="Exactly one of the bullet IDs provided. Never invented.")
    requirement_text: str = Field(description="Exactly one of the requirement texts provided.")
    relevance: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One concrete sentence naming what supports this score.")


class ScoringResult(BaseModel):
    """The structured-output wrapper. `with_structured_output` needs one model."""

    model_config = ConfigDict(extra="forbid")

    matches: list[ScoredMatch]


def _render_candidates(profile: Profile, bullet_ids: list[str]) -> str:
    lines = []
    for bullet_id in bullet_ids:
        bullet = profile.bullet_by_id(bullet_id)
        detail = bullet.canonical.strip()
        if bullet.skills:
            detail += f"  [skills: {', '.join(bullet.skills)}]"
        lines.append(f"- id: {bullet_id}\n  achievement: {detail}")
    return "\n".join(lines)


def _render_requirements(job: JobSpec) -> str:
    lines = []
    for requirement in job.requirements:
        kind = "MUST-HAVE" if requirement.is_must_have else "nice-to-have"
        origin = " (inferred, not stated in the posting)" if requirement.is_inferred else ""
        lines.append(
            f"- text: {requirement.text}\n"
            f"  category: {requirement.category}, weight: {requirement.weight}/5, "
            f"{kind}{origin}"
        )
    return "\n".join(lines)


def scoring_cache_key(job: JobSpec, bullet_ids: list[str], model: str) -> str:
    """Identity of "this posting, these candidates, this prompt, this model"."""
    return content_key(
        job.source_hash,
        "|".join(sorted(bullet_ids)),
        prompt_version(PROMPT_NAME),
        model,
    )


def score_fit(
    job: JobSpec,
    bullet_ids: list[str],
    profile: Profile,
    *,
    llm: BaseChatModel | None = None,
    use_cache: bool = True,
    model: str | None = None,
) -> list[EvidenceMatch]:
    """One batched call scoring every plausible (bullet, requirement) pair.

    Matches naming a bullet or requirement that was not provided are dropped
    with a warning rather than trusted: an invented bullet id is the retrieval
    equivalent of a fabricated claim, and it would flow into selection as
    evidence for something the candidate never did.

    Cached because this is one of the two calls that dominate a run's cost, and
    M5's layout loop re-runs everything downstream of it several times per
    posting without the scores ever changing.
    """
    if not job.requirements or not bullet_ids:
        return []

    # Resolved at call time, not as a default argument: the id is part of the
    # cache key, so freezing it at import would let one provider serve another's
    # scores.
    model = model or model_for()

    cache = ModelListCache(EvidenceMatch, "scores")
    key = scoring_cache_key(job, bullet_ids, model)
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            logger.info("score_fit: cache hit (%d matches)", len(cached))
            return cached

    llm = llm or build_chat_model(model=model)
    structured = llm.with_structured_output(ScoringResult)

    user_content = (
        f"<requirements>\n{_render_requirements(job)}\n</requirements>\n\n"
        f"<achievements>\n{_render_candidates(profile, bullet_ids)}\n</achievements>"
    )

    result = structured.invoke(
        [SystemMessage(content=load_prompt(PROMPT_NAME)), HumanMessage(content=user_content)]
    )
    if not isinstance(result, ScoringResult):
        result = ScoringResult.model_validate(result)

    known_bullets = set(bullet_ids)
    known_requirements = {r.text for r in job.requirements}

    matches: list[EvidenceMatch] = []
    for scored in result.matches:
        if scored.bullet_id not in known_bullets:
            logger.warning(
                "score_fit: dropping match for unknown bullet id %r -- the model invented it",
                scored.bullet_id,
            )
            continue
        if scored.requirement_text not in known_requirements:
            logger.warning(
                "score_fit: dropping match for unknown requirement %r", scored.requirement_text
            )
            continue
        matches.append(
            EvidenceMatch(
                bullet_id=scored.bullet_id,
                requirement_text=scored.requirement_text,
                relevance=scored.relevance,
                rationale=scored.rationale,
            )
        )

    if use_cache:
        cache.put(key, matches)
    return matches


def build_fit_report(job: JobSpec, matches: list[EvidenceMatch]) -> FitReport:
    """Assemble the report from the scores. No model involved.

    Gaps are derived from the **requirement** side, not the match side. A
    requirement that retrieved nothing produces no matches at all, so a report
    built only from what came back would silently omit exactly the requirements
    the candidate most needs to know about.
    """
    best_by_requirement: dict[str, EvidenceMatch] = {}
    for match in matches:
        current = best_by_requirement.get(match.requirement_text)
        if current is None or match.relevance > current.relevance:
            best_by_requirement[match.requirement_text] = match

    covered: list[EvidenceMatch] = []
    partial: list[EvidenceMatch] = []
    gaps = []

    for requirement in job.requirements:
        best = best_by_requirement.get(requirement.text)
        if best is None or best.relevance < PARTIAL_THRESHOLD:
            gaps.append(requirement)
        elif best.relevance >= COVERED_THRESHOLD:
            covered.append(best)
        else:
            partial.append(best)

    best_relevance = {text: match.relevance for text, match in best_by_requirement.items()}
    overall = compute_overall_fit(job.requirements, best_relevance)

    return FitReport(
        overall_fit=overall,
        covered=sorted(covered, key=lambda m: -m.relevance),
        partial=sorted(partial, key=lambda m: -m.relevance),
        # Gaps sorted by how much they matter: a weight-5 must-have you lack is
        # the first thing to read.
        gaps=sorted(gaps, key=lambda r: (not r.is_must_have, -r.weight)),
        recommendation=recommend(overall, job.requirements, best_relevance),
    )
