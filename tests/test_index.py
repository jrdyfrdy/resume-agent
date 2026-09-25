"""The SQLite + sqlite-vec index.

The load-bearing test here is `test_sqlite_vec_agrees_with_numpy_cosine`. A
vector store using the wrong distance metric does not raise -- it just returns
slightly worse results forever, which is indistinguishable from "embeddings are
hard" unless something checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from resume_agent.kb.embeddings import DEFAULT_DIMENSIONS, FastEmbedEmbeddings
from resume_agent.kb.index import (
    ProfileIndex,
    build_embed_text,
    collect_indexed_bullets,
    index_path_for,
    profile_content_hash,
)
from resume_agent.models.profile import Bullet, Profile

from .conftest import PROFILE_EXAMPLE

# --- what gets indexed -----------------------------------------------------


def test_one_row_per_bullet(example_profile: Profile) -> None:
    """Spec 3.1: the bullets are already atomic, so they are not chunked."""
    rows = collect_indexed_bullets(example_profile)
    assert len(rows) == len(example_profile.all_bullets()) == 13


def test_embed_text_is_canonical_plus_skills_plus_themes() -> None:
    """Spec 3.3 names exactly these three fields."""
    bullet = Bullet(
        id="x.b1",
        canonical="Cut latency with a cache",
        skills=["redis", "caching"],
        themes=["performance"],
        metrics={"before_ms": 800},
    )
    text = build_embed_text(bullet)
    assert text == "Cut latency with a cache redis caching performance"
    # Metrics are deliberately absent: they are the fabrication allow-list for
    # M4, not retrieval signal, and embedding bare numbers adds noise.
    assert "800" not in text


def test_embed_text_excludes_employer_and_entry_tech(example_profile: Profile) -> None:
    """A bullet must not become findable by a technology it never mentions."""
    rows = {r.bullet_id: r for r in collect_indexed_bullets(example_profile)}
    kafka_bullet = rows["exp_halvorsen_bright.b1"]  # Redis/caching bullet
    assert "kafka" not in kafka_bullet.embed_text.lower()
    assert "halvorsen" not in kafka_bullet.embed_text.lower()


def test_entry_label_is_org_for_jobs_and_name_for_projects(example_profile: Profile) -> None:
    rows = {r.bullet_id: r for r in collect_indexed_bullets(example_profile)}
    assert rows["exp_halvorsen_bright.b1"].entry_label == "Halvorsen & Bright R&D"
    assert rows["prj_ledgerlint.b1"].entry_label == "LedgerLint"


# --- the vector half -------------------------------------------------------


def test_dense_search_returns_k_results(profile_index: ProfileIndex, embeddings) -> None:
    vector = embeddings.embed_query("kubernetes")
    assert len(profile_index.dense_search(vector, 5)) == 5


def test_dense_search_is_ordered_by_similarity(profile_index: ProfileIndex, embeddings) -> None:
    vector = embeddings.embed_query("redis caching")
    scores = [score for _, score in profile_index.dense_search(vector, 10)]
    assert scores == sorted(scores, reverse=True)


def test_sqlite_vec_agrees_with_numpy_cosine(
    profile_index: ProfileIndex, embeddings: FastEmbedEmbeddings
) -> None:
    """Cross-check the vector store against five lines of numpy.

    fastembed returns unit vectors, so cosine similarity is a plain dot product,
    and sqlite-vec's euclidean distance converts exactly:
        ||a - b||^2 = 2 - 2*(a . b)   =>   cos = 1 - d^2/2

    If sqlite-vec were silently using a different metric, the rankings would
    diverge here and nowhere else.
    """
    rows = profile_index.bullets()
    matrix = np.array(embeddings.embed_documents([r.embed_text for r in rows]))
    ids = [r.bullet_id for r in rows]

    for query in ["kubernetes", "redis caching", "reduce cloud cost"]:
        vector = np.array(embeddings.embed_query(query))

        similarities = matrix @ vector
        expected_order = [ids[i] for i in np.argsort(-similarities)[:5]]
        expected_scores = sorted(similarities, reverse=True)[:5]

        actual = profile_index.dense_search(vector.tolist(), 5)
        assert [bullet_id for bullet_id, _ in actual] == expected_order, query
        for (_, actual_score), expected in zip(actual, expected_scores, strict=True):
            assert actual_score == pytest.approx(expected, abs=1e-5), query


def test_embeddings_are_unit_vectors(embeddings: FastEmbedEmbeddings) -> None:
    """The cosine/L2 identity above depends on this, so it is asserted directly."""
    vectors = np.array(embeddings.embed_documents(["a short bullet", "another one"]))
    assert vectors.shape[1] == DEFAULT_DIMENSIONS
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


# --- staleness and rebuilding ----------------------------------------------


def test_index_path_is_keyed_on_profile_directory_name() -> None:
    """A real `profile/` and `profile.example/` must not share a database."""
    assert index_path_for(Path("profile.example")).name == "profile.example.db"
    assert index_path_for(Path("profile")).name == "profile.db"


def test_content_hash_changes_when_yaml_changes(tmp_path: Path) -> None:
    (tmp_path / "identity.yaml").write_text("name: A\n", encoding="utf-8")
    before = profile_content_hash(tmp_path)

    (tmp_path / "identity.yaml").write_text("name: B\n", encoding="utf-8")
    assert profile_content_hash(tmp_path) != before


def test_content_hash_changes_when_a_file_is_added(tmp_path: Path) -> None:
    (tmp_path / "identity.yaml").write_text("name: A\n", encoding="utf-8")
    before = profile_content_hash(tmp_path)

    (tmp_path / "extra.yaml").write_text("name: A\n", encoding="utf-8")
    assert profile_content_hash(tmp_path) != before


def test_content_hash_is_stable_across_calls() -> None:
    assert profile_content_hash(PROFILE_EXAMPLE) == profile_content_hash(PROFILE_EXAMPLE)


def test_open_builds_when_missing(example_profile: Profile, tmp_path: Path, embeddings) -> None:
    db_path = tmp_path / "fresh.db"
    index, rebuilt = ProfileIndex.open(
        example_profile, PROFILE_EXAMPLE, db_path, embeddings=embeddings
    )
    assert rebuilt is True
    assert db_path.is_file()
    index.close()


def test_open_reuses_a_current_index(example_profile: Profile, tmp_path: Path, embeddings) -> None:
    db_path = tmp_path / "reuse.db"
    first, _ = ProfileIndex.open(example_profile, PROFILE_EXAMPLE, db_path, embeddings=embeddings)
    first.close()

    second, rebuilt = ProfileIndex.open(
        example_profile, PROFILE_EXAMPLE, db_path, embeddings=embeddings
    )
    assert rebuilt is False, "a current index was re-embedded unnecessarily"
    second.close()


def test_open_rebuilds_a_corrupt_index(
    example_profile: Profile, tmp_path: Path, embeddings
) -> None:
    """The index is a cache; a truncated file should be replaced, not diagnosed."""
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"not a database")

    index, rebuilt = ProfileIndex.open(
        example_profile, PROFILE_EXAMPLE, db_path, embeddings=embeddings
    )
    assert rebuilt is True
    assert len(index.bullets()) == 13
    index.close()


def test_rebuild_is_deterministic(example_profile: Profile, tmp_path: Path, embeddings) -> None:
    """Same input, same embeddings -- otherwise nothing downstream is reproducible."""
    query = embeddings.embed_query("kubernetes")

    first = ProfileIndex.build(example_profile, PROFILE_EXAMPLE, tmp_path / "a.db", embeddings)
    second = ProfileIndex.build(example_profile, PROFILE_EXAMPLE, tmp_path / "b.db", embeddings)

    assert first.dense_search(query, 8) == second.dense_search(query, 8)
    first.close()
    second.close()


# ===========================================================================
# Every entry type can be indexed
# ===========================================================================


def test_a_profile_with_every_entry_type_can_be_indexed() -> None:
    """Regression. `_entry_label` was an isinstance chain that knew jobs and
    projects, and fell through to `.name` for anything else. A leadership entry
    has no `name`, so a fresh graduate's profile raised AttributeError at
    retrieval -- every run on it would have crashed before any model was
    called. Each entry type now declares its own `label`."""
    from resume_agent.kb.index import collect_indexed_bullets
    from resume_agent.kb.loader import load_profile

    student = load_profile(
        Path(__file__).resolve().parent / "fixtures" / "profile.student"
    )
    rows = collect_indexed_bullets(student)

    labels = {row.entry_id: row.entry_label for row in rows}
    assert labels["ldr_student_radio"] == "Kingsbridge Student Radio"
    assert labels["pub_caption_study"] == (
        "Reading Speed as a Practical Quality Signal for Lecture Captions"
    )
    assert labels["exp_harlow_press"] == "Harlow Press"
    assert labels["prj_caption_lint"] == "CaptionLint"
    assert len(rows) == len(student.all_bullets())
