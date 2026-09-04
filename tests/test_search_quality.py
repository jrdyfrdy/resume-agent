"""Retrieval quality, measured without a labelled dataset.

M1's kickoff asked how this gets tested when there is no ground truth to score
against. The answer is three things, in descending order of how much they prove:

1. **Label-free invariants** (test_retriever.py). Self-retrieval, alias
   equivalence, RRF algebra. These are facts, not opinions, and they catch the
   failures that actually happen -- a broken tokenizer, a wrong distance metric,
   an index that never rebuilt.

2. **Paraphrase recall** (here). Take a bullet, describe it by hand in words
   that do *not* appear in it, and require the bullet back in the top 3. This is
   the closest thing to a quality metric available, because the paraphrase
   shares no vocabulary with the target -- so only the dense half can find it,
   and BM25 cannot accidentally pass the test.

3. **Honest gaps** (here). Queries for things the profile genuinely lacks must
   *not* produce a confident match. A tool that always returns something
   confident is useless for M3's `FitReport.gaps`, which has to be able to say
   "no evidence exists".

What none of this measures is whether retrieval is *good* in an absolute sense.
That is what `resume-agent search --explain` is for; spec 9 is explicit that the
human eye is the calibration set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from resume_agent.kb.retriever import HybridRetriever

EXPECTATIONS_PATH = Path(__file__).resolve().parent / "fixtures" / "retrieval_expectations.yaml"
CASES: list[dict[str, Any]] = yaml.safe_load(EXPECTATIONS_PATH.read_text(encoding="utf-8"))["cases"]

POSITIVE_CASES = [c for c in CASES if c.get("must_include_top3")]
GAP_CASES = [c for c in CASES if c.get("expect_no_strong_match")]

# A hybrid score at or above this means "both retrievers agreed", since a
# document found by only one retriever tops out near 1/61 = 0.0164. It is
# therefore the natural line between "real match" and "dense guessed".
STRONG_MATCH_SCORE = 1 / 61 + 1 / 70


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=lambda c: c["query"])
def test_expected_bullet_is_in_top_3(retriever: HybridRetriever, case: dict[str, Any]) -> None:
    hits = retriever.search(case["query"], k=3)
    found = [hit.bullet_id for hit in hits]
    for expected in case["must_include_top3"]:
        assert expected in found, (
            f"query {case['query']!r} ({case.get('note', '')})\n"
            f"  expected {expected} in top 3, got {found}"
        )


@pytest.mark.parametrize("case", GAP_CASES, ids=lambda c: c["query"])
def test_gaps_do_not_produce_a_confident_match(
    retriever: HybridRetriever, case: dict[str, Any]
) -> None:
    """A query with no supporting evidence must not come back looking answered.

    Dense retrieval will always return *something* -- it ranks the whole corpus.
    What must not happen is a high fused score, which would mean BM25 agreed and
    a real match exists.
    """
    hits = retriever.search(case["query"], k=3)
    if not hits:
        return
    assert hits[0].score < STRONG_MATCH_SCORE, (
        f"query {case['query']!r} ({case.get('note', '')}) has no evidence in the profile, "
        f"but returned {hits[0].bullet_id} at score {hits[0].score:.5f}"
    )
    assert hits[0].bm25_rank is None, (
        f"query {case['query']!r} should have no literal match, "
        f"but BM25 ranked {hits[0].bullet_id}"
    )


def test_paraphrase_recall_at_3(retriever: HybridRetriever) -> None:
    """Aggregate recall@3 over every paraphrase case, reported as one number.

    Individually parameterised above so a failure names the query; aggregated
    here so the overall figure is visible and can be watched over time.
    """
    hit_count = 0
    for case in POSITIVE_CASES:
        found = {h.bullet_id for h in retriever.search(case["query"], k=3)}
        if set(case["must_include_top3"]) <= found:
            hit_count += 1

    recall = hit_count / len(POSITIVE_CASES)
    assert recall == 1.0, f"recall@3 fell to {recall:.0%} ({hit_count}/{len(POSITIVE_CASES)})"


def test_expectations_reference_real_bullets(retriever: HybridRetriever) -> None:
    """Guards the fixture itself: a typo'd bullet id would be an unfalsifiable test."""
    known = {b.bullet_id for b in retriever.index.bullets()}
    for case in CASES:
        for bullet_id in case.get("must_include_top3", []):
            assert bullet_id in known, f"{bullet_id} in expectations does not exist"


def test_every_case_declares_an_expectation() -> None:
    for case in CASES:
        assert case.get("must_include_top3") or case.get("expect_no_strong_match"), (
            f"case {case['query']!r} asserts nothing"
        )
