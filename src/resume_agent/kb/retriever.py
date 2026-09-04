"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion. Spec 3.3.

Why both halves are needed is easiest to see from a real result. Querying
"kubernetes" against `profile.example`, the dense retriever alone returns:

    0.77  exp_halvorsen_bright.b4   the EKS migration      <- correct
    0.69  prj_ledgerlint.b1         a DuckDB linter        <- nothing to do with k8s
    0.66  exp_halvorsen_bright.b2   Kafka consumers        <- nothing to do with k8s

The right answer is first, but the tail is noise: everything vaguely
"infrastructure-shaped" scores in the 0.6s. That is spec 3.3's point --
"dense embeddings are notoriously mushy about exact technology names" -- and it
is why BM25, which either matched the literal token or did not, is not optional.

RRF fuses them without needing the two score scales to be comparable, which is
the trick: BM25 scores are unbounded and corpus-relative, cosine similarities sit
in [-1, 1], and averaging them directly would be meaningless. Ranks are
comparable even when scores are not.

    score(doc) = sum over retrievers of 1 / (k + rank_in_that_retriever)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi

from resume_agent.kb.embeddings import FastEmbedEmbeddings
from resume_agent.kb.index import IndexedBullet, ProfileIndex
from resume_agent.kb.tokenize import tokenize
from resume_agent.models.profile import Bullet, Profile

# Spec 3.3 fixes this at 60. It is the standard RRF constant: large enough that
# the difference between rank 1 and rank 2 (1/61 vs 1/62) is small, so a
# document both retrievers merely like can outrank one that a single retriever
# loves. That damping is the entire point of the fusion.
RRF_K = 60

SearchMode = Literal["hybrid", "bm25", "dense"]


@dataclass(frozen=True)
class RetrievalHit:
    bullet_id: str
    score: float
    bullet: Bullet
    entry_label: str
    entry_id: str
    # Where each retriever placed this bullet (1-based), or None if it did not
    # return it at all. Carried purely so `search --explain` can show the
    # fusion working; nothing downstream depends on them.
    bm25_rank: int | None = None
    dense_rank: int | None = None


# --- query expansion -------------------------------------------------------


def build_alias_groups(profile: Profile) -> list[tuple[str, set[str]]]:
    """Every skill as `(surface_form, all_surface_forms)`, longest form first.

    Sorting by length descending is what makes matching greedy: the alias
    "container orchestration" must be found before the bare word "orchestration"
    would be, or a two-word alias never matches as a unit.
    """
    groups: list[tuple[str, set[str]]] = []
    for skill in profile.skills:
        forms = {skill.canonical.lower(), *(alias.lower() for alias in skill.aliases)}
        for form in forms:
            groups.append((form, forms))
    groups.sort(key=lambda pair: len(pair[0]), reverse=True)
    return groups


def expand_query(query: str, profile: Profile) -> list[str]:
    """Add every alias of any skill the query mentions.

    "k8s" becomes ["k8s", "kubernetes", "container orchestration", "eks", "gke"],
    which is what lets a job description's vocabulary reach a profile written in
    different words. This is the alias table's first job of three -- the other
    two are display naming and the M4 fabrication allow-list.

    Matching is on word boundaries so that "go" does not fire inside "google".
    """
    lowered = query.lower()
    added: list[str] = []
    seen: set[str] = set()

    for form, forms in build_alias_groups(profile):
        if form in seen:
            continue
        if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", lowered):
            for sibling in sorted(forms):
                if sibling not in seen and sibling not in lowered:
                    added.append(sibling)
                seen.add(sibling)
    return [query, *added]


# --- fusion ----------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one, highest fused score first.

    Ranks are 1-based, which is the convention the RRF paper uses and what makes
    `k=60` mean what everyone else means by it.

    Ties are broken by id so the output is deterministic -- without that, two
    documents with identical fused scores could swap places between runs and
    make a snapshot test flap.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


# --- the retriever ---------------------------------------------------------


class HybridRetriever:
    """BM25 + dense over one profile's bullets."""

    def __init__(
        self,
        index: ProfileIndex,
        profile: Profile,
        embeddings: FastEmbedEmbeddings | None = None,
    ) -> None:
        self.index = index
        self.profile = profile
        self.embeddings = embeddings or FastEmbedEmbeddings()

        self._bullets: list[IndexedBullet] = index.bullets()
        self._ids = [b.bullet_id for b in self._bullets]
        # BM25 is rebuilt in memory on construction rather than persisted. At
        # 200 documents this is microseconds, and spec 3.3 is explicit that
        # everything fits in memory at this scale.
        self._bm25 = BM25Okapi([b.tokens for b in self._bullets])

    # -- the two halves, separately, because the DoD requires seeing them -----

    def bm25_ranking(self, query_terms: list[str]) -> list[str]:
        """All bullet ids, best BM25 score first."""
        tokens: list[str] = []
        for term in query_terms:
            tokens.extend(tokenize(term))
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda p: (-p[1], p[0]))
        # A zero score means no query term appeared at all. Keeping those would
        # hand RRF a ranking of pure noise for the long tail.
        return [bullet_id for bullet_id, score in ranked if score > 0.0]

    def dense_ranking(self, query_terms: list[str], k: int) -> list[str]:
        """Bullet ids by cosine similarity, best first."""
        # The expanded terms are folded into one string so the model sees a
        # single richer query rather than several thin ones.
        vector = self.embeddings.embed_query(" ".join(query_terms))
        return [bullet_id for bullet_id, _ in self.index.dense_search(vector, k)]

    # -- the public entry point ---------------------------------------------

    def search(self, query: str, k: int = 8, mode: SearchMode = "hybrid") -> list[RetrievalHit]:
        """Top-`k` bullets for `query`.

        `mode` exists because M1's definition of done requires checking that
        "BM25-only and dense-only both return sensible results before you fuse
        them" -- so each half has to be inspectable on its own, not just in the
        blend.
        """
        query_terms = expand_query(query, self.profile)
        # Fuse over a pool wider than k: a document ranked 9th by one retriever
        # and 2nd by the other should still be able to surface.
        pool = min(len(self._bullets), max(k * 4, 20))

        bm25 = self.bm25_ranking(query_terms) if mode in ("hybrid", "bm25") else []
        dense = self.dense_ranking(query_terms, pool) if mode in ("hybrid", "dense") else []

        if mode == "bm25":
            fused = [(doc_id, 1.0 / (RRF_K + rank)) for rank, doc_id in enumerate(bm25, start=1)]
        elif mode == "dense":
            fused = [(doc_id, 1.0 / (RRF_K + rank)) for rank, doc_id in enumerate(dense, start=1)]
        else:
            fused = reciprocal_rank_fusion([bm25, dense])

        bm25_positions = {doc_id: i for i, doc_id in enumerate(bm25, start=1)}
        dense_positions = {doc_id: i for i, doc_id in enumerate(dense, start=1)}

        hits: list[RetrievalHit] = []
        for bullet_id, score in fused[:k]:
            indexed = self.index.get(bullet_id)
            hits.append(
                RetrievalHit(
                    bullet_id=bullet_id,
                    score=score,
                    bullet=indexed.bullet,
                    entry_label=indexed.entry_label,
                    entry_id=indexed.entry_id,
                    bm25_rank=bm25_positions.get(bullet_id),
                    dense_rank=dense_positions.get(bullet_id),
                )
            )
        return hits
