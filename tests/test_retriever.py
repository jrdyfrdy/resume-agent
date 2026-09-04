"""Hybrid retrieval: query expansion, the two retrievers, and RRF.

Almost everything here is **label-free**. RRF is pure arithmetic and gets exact
assertions; expansion is table-driven and gets equivalence assertions; and
self-retrieval ("can a document find itself?") is a strong correctness signal
that needs no human judgement at all. Opinions about which bullet *should* win a
given query live in test_search_quality.py, separated on purpose.
"""

from __future__ import annotations

import pytest

from resume_agent.kb.retriever import (
    RRF_K,
    HybridRetriever,
    expand_query,
    reciprocal_rank_fusion,
)
from resume_agent.models.profile import Profile

# --- RRF: pure arithmetic, exact assertions --------------------------------


def test_rrf_matches_the_formula_from_spec_3_3() -> None:
    """score(doc) = sum of 1 / (60 + rank), ranks 1-based."""
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "a"]]))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_k_is_60() -> None:
    assert RRF_K == 60


def test_document_ranked_first_by_both_wins() -> None:
    fused = reciprocal_rank_fusion([["top", "x", "y"], ["top", "y", "x"]])
    assert fused[0][0] == "top"


def test_agreement_beats_a_single_strong_opinion() -> None:
    """The damping that makes RRF worth using.

    "both" is 2nd in both rankings; "solo" is 1st in one and absent from the
    other. Consensus should win -- that is the whole reason k is large.
    """
    fused = dict(reciprocal_rank_fusion([["solo", "both"], ["other", "both"]]))
    assert fused["both"] > fused["solo"]


def test_rrf_is_order_invariant_across_retrievers() -> None:
    """Fusing [bm25, dense] must equal fusing [dense, bm25]."""
    a, b = ["x", "y", "z"], ["z", "x", "w"]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([b, a])


def test_rrf_output_is_the_union_of_its_inputs() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert {doc for doc, _ in fused} == {"a", "b", "c"}


def test_rrf_has_no_duplicates() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"], ["a"]])
    assert len([doc for doc, _ in fused]) == len({doc for doc, _ in fused})


def test_rrf_scores_are_non_increasing() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"], ["d", "c", "b", "a"]])
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_breaks_ties_deterministically() -> None:
    """Without a tiebreak, equal-scoring documents could swap between runs."""
    first = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    second = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    assert first == second
    assert [doc for doc, _ in first] == ["a", "b"]  # alphabetical on a tie


def test_rrf_handles_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    assert [doc for doc, _ in reciprocal_rank_fusion([["a"], []])] == ["a"]


# --- query expansion -------------------------------------------------------


def test_expands_an_alias_to_its_canonical(example_profile: Profile) -> None:
    terms = expand_query("k8s", example_profile)
    assert terms[0] == "k8s"
    assert "kubernetes" in terms


def test_expands_a_canonical_to_its_aliases(example_profile: Profile) -> None:
    terms = expand_query("kubernetes", example_profile)
    assert {"k8s", "eks", "gke", "container orchestration"} <= set(terms)


def test_expands_multi_word_aliases(example_profile: Profile) -> None:
    """Longest-match first, or "container orchestration" never matches as a unit."""
    terms = expand_query("we need container orchestration", example_profile)
    assert "kubernetes" in terms


def test_does_not_expand_inside_a_longer_word(example_profile: Profile) -> None:
    """"go" is a skill; "google" must not trigger it."""
    assert "golang" not in expand_query("google cloud", example_profile)


def test_unknown_query_is_returned_unchanged(example_profile: Profile) -> None:
    assert expand_query("underwater basket weaving", example_profile) == [
        "underwater basket weaving"
    ]


def test_expansion_does_not_repeat_a_term_already_present(example_profile: Profile) -> None:
    """Element 0 is the original query; the rest are only what expansion added."""
    terms = expand_query("kubernetes k8s", example_profile)
    assert terms[0] == "kubernetes k8s"

    added = terms[1:]
    assert "kubernetes" not in added, "re-added a term the query already contained"
    assert "k8s" not in added
    assert len(added) == len(set(added)), "expansion produced duplicates"
    assert "container orchestration" in added  # the siblings still arrive


# --- label-free retrieval invariants ---------------------------------------


def test_every_bullet_retrieves_itself(
    retriever: HybridRetriever, example_profile: Profile
) -> None:
    """The strongest correctness signal available without labels.

    Query with a bullet's own canonical text; it must come back first. A
    document that cannot find itself means the index, the embedding or the
    tokenizer is broken, and no amount of tuning will fix what follows.
    """
    for bullet in example_profile.all_bullets():
        hits = retriever.search(bullet.canonical, k=1)
        assert hits, f"{bullet.id} returned nothing for its own text"
        assert hits[0].bullet_id == bullet.id, (
            f"{bullet.id} did not retrieve itself; got {hits[0].bullet_id}"
        )


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_self_retrieval_holds_in_every_mode(
    retriever: HybridRetriever, example_profile: Profile, mode: str
) -> None:
    """M1's DoD: "BM25-only and dense-only both return sensible results"."""
    bullet = example_profile.bullet_by_id("exp_halvorsen_bright.b4")
    hits = retriever.search(bullet.canonical, k=1, mode=mode)
    assert hits and hits[0].bullet_id == bullet.id


def test_alias_and_canonical_return_the_same_results(retriever: HybridRetriever) -> None:
    """Expansion is only worth having if "k8s" and "kubernetes" agree."""
    by_alias = [h.bullet_id for h in retriever.search("k8s", k=5)]
    by_canonical = [h.bullet_id for h in retriever.search("kubernetes", k=5)]
    assert by_alias == by_canonical


def test_neither_retriever_is_degenerate_alone(retriever: HybridRetriever) -> None:
    """The DoD requires each half to work before they are fused."""
    for mode in ("bm25", "dense"):
        hits = retriever.search("kubernetes", k=5, mode=mode)
        assert hits, f"{mode} returned nothing"
        assert len({h.bullet_id for h in hits}) == len(hits), f"{mode} returned duplicates"


def test_bm25_returns_only_literal_matches(retriever: HybridRetriever) -> None:
    """BM25's job is exactness; a zero-score document must not be returned.

    On this corpus exactly one bullet mentions Kubernetes, so BM25 alone should
    return exactly that one -- which is the behaviour that lets it correct the
    dense retriever's mushiness during fusion.
    """
    hits = retriever.search("kubernetes", k=8, mode="bm25")
    assert [h.bullet_id for h in hits] == ["exp_halvorsen_bright.b4"]


def test_dense_returns_a_full_ranking(retriever: HybridRetriever) -> None:
    """Dense always has an opinion about everything -- that is its weakness."""
    hits = retriever.search("kubernetes", k=8, mode="dense")
    assert len(hits) == 8


def test_hybrid_puts_the_literal_match_first(retriever: HybridRetriever) -> None:
    hits = retriever.search("kubernetes", k=5)
    assert hits[0].bullet_id == "exp_halvorsen_bright.b4"
    # And by a clear margin, because it is the only bullet both halves ranked.
    assert hits[0].score > hits[1].score * 1.5


def test_ranks_are_reported_for_explain(retriever: HybridRetriever) -> None:
    top = retriever.search("kubernetes", k=1)[0]
    assert top.bm25_rank == 1
    assert top.dense_rank == 1


def test_k_limits_results(retriever: HybridRetriever) -> None:
    assert len(retriever.search("cache", k=3)) == 3


def test_results_carry_display_context(retriever: HybridRetriever) -> None:
    """`search` has to be readable by a human; ids alone are not."""
    hit = retriever.search("kubernetes", k=1)[0]
    assert hit.entry_label == "Halvorsen & Bright R&D"
    assert hit.entry_id == "exp_halvorsen_bright"
    assert "Kubernetes" in hit.bullet.canonical


def test_empty_query_does_not_crash(retriever: HybridRetriever) -> None:
    assert retriever.search("", k=5, mode="bm25") == []


def test_search_is_deterministic(retriever: HybridRetriever) -> None:
    first = [(h.bullet_id, h.score) for h in retriever.search("caching", k=5)]
    second = [(h.bullet_id, h.score) for h in retriever.search("caching", k=5)]
    assert first == second
