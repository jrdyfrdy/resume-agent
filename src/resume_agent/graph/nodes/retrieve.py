"""Find the achievements worth scoring against a posting. Spec 5, `retrieve_evidence`.

    "For each requirement: expand query using the alias table, run hybrid
     retrieval, take top-8. Deduplicate across requirements. This is a plain
     function, no LLM."

The deduplication matters more than it sounds. Twenty requirements retrieving
eight candidates each is 160 rows, but a profile only has so many bullets and
the same three will answer half the requirements. Passing the deduplicated set
to the scoring node is what keeps that single batched call affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

from resume_agent.kb.retriever import HybridRetriever
from resume_agent.models.job import JobSpec, Requirement

DEFAULT_TOP_K = 8


@dataclass(frozen=True)
class RetrievedCandidates:
    """Which bullets were retrieved, and which requirement pulled each in."""

    # requirement text -> bullet ids, best first
    by_requirement: dict[str, list[str]]
    # every bullet id any requirement retrieved, deduplicated, stable order
    bullet_ids: list[str]

    def __len__(self) -> int:
        return len(self.bullet_ids)


def retrieve_evidence(
    job: JobSpec,
    retriever: HybridRetriever,
    k: int = DEFAULT_TOP_K,
) -> RetrievedCandidates:
    """Top-`k` bullets per requirement, plus the deduplicated union."""
    by_requirement: dict[str, list[str]] = {}
    seen: list[str] = []

    for requirement in job.requirements:
        hits = retriever.search(_query_for(requirement), k=k)
        ids = [hit.bullet_id for hit in hits]
        by_requirement[requirement.text] = ids
        for bullet_id in ids:
            if bullet_id not in seen:
                seen.append(bullet_id)

    return RetrievedCandidates(by_requirement=by_requirement, bullet_ids=seen)


def _query_for(requirement: Requirement) -> str:
    """The retrieval query for one requirement.

    Just the requirement text: `HybridRetriever.search` already expands it
    through the skills alias table, so prefixing the category or the word
    "experience" would only dilute the query with tokens no bullet contains.
    """
    return requirement.text
