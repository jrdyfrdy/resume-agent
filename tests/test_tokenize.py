"""The BM25 tokenizer.

Spec 3.3 keeps BM25 in the pipeline specifically to match exact technology
tokens that dense embeddings blur. These tests pin the tokens that would be
destroyed by a naive `\\w+` tokenizer -- if any of them regress, BM25 stops
earning its place.
"""

from __future__ import annotations

import pytest

from resume_agent.kb.tokenize import STOPWORDS, tokenize
from resume_agent.models.profile import Profile

# --- the tokens a naive tokenizer would destroy ----------------------------


@pytest.mark.parametrize(
    ("text", "expected_token"),
    [
        ("C#", "c#"),
        ("C++", "c++"),
        (".NET", ".net"),
        ("Node.js", "node.js"),
        ("CI/CD", "ci/cd"),
        ("user_id", "user_id"),
        ("p95", "p95"),
        ("read-through", "read-through"),
        ("GitHub Actions", "github"),
        ("sql-optimization", "sql-optimization"),
        ("AT&T", "at&t"),
        ("R&D", "r&d"),
    ],
)
def test_technology_names_survive(text: str, expected_token: str) -> None:
    assert expected_token in tokenize(text)


def test_c_sharp_does_not_collide_with_bare_c() -> None:
    """The failure that motivates the whole module.

    Under `\\w+`, "C#" tokenizes to ["c"], so a JD requiring C# would match any
    bullet containing a stray "c". The `#` must stay attached.
    """
    assert tokenize("C#") == ["c#"]
    assert "c" not in tokenize("C#")


def test_dotnet_does_not_collide_with_network() -> None:
    assert ".net" in tokenize(".NET")
    assert "net" not in tokenize("networking")


# --- compound expansion ----------------------------------------------------


def test_compounds_emit_whole_then_parts() -> None:
    """A query for "cd" should still find "ci/cd"."""
    tokens = tokenize("CI/CD")
    assert tokens[0] == "ci/cd"
    assert "ci" in tokens
    assert "cd" in tokens


def test_compound_expansion_can_be_disabled() -> None:
    assert tokenize("CI/CD", expand_compounds=False) == ["ci/cd"]


def test_plain_words_are_not_duplicated() -> None:
    """A non-compound splits into itself; it must be emitted once."""
    assert tokenize("caching") == ["caching"]


def test_c_sharp_is_not_split() -> None:
    """`#` and `+` are not compound separators -- splitting would undo the point."""
    assert tokenize("C#") == ["c#"]
    assert tokenize("C++") == ["c++"]


# --- ordinary behaviour ----------------------------------------------------


def test_lowercases() -> None:
    assert tokenize("PostgreSQL") == ["postgresql"]


def test_drops_stopwords() -> None:
    tokens = tokenize("the cache is in the queue")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "cache" in tokens
    assert "queue" in tokens


def test_stopword_list_stays_small() -> None:
    """An aggressive stopword list breaks queries like "not null"; BM25's IDF
    already discounts common words at this corpus size."""
    assert len(STOPWORDS) < 40


def test_term_frequency_is_preserved() -> None:
    """BM25 is a bag-of-words model; deduplicating here would distort scoring."""
    assert tokenize("cache cache cache").count("cache") == 3


def test_punctuation_and_empty_input() -> None:
    assert tokenize("") == []
    assert tokenize("--- , . ; ---") == []


def test_trailing_punctuation_is_not_captured() -> None:
    assert tokenize("Redis, Kafka.") == ["redis", "kafka"]


def test_numbers_and_units() -> None:
    tokens = tokenize("cut 820ms to 50ms across 12 endpoints")
    assert "820ms" in tokens
    assert "50ms" in tokens
    assert "12" in tokens


# --- against the real corpus -----------------------------------------------


def test_every_skill_name_tokenizes_to_something(example_profile: Profile) -> None:
    """A skill that tokenizes to nothing can never be matched by BM25."""
    for skill in example_profile.skills:
        for surface in [skill.canonical, *skill.aliases]:
            assert tokenize(surface), f"{surface!r} produced no tokens"


def test_example_bullets_tokenize_to_useful_terms(example_profile: Profile) -> None:
    bullet = example_profile.bullet_by_id("exp_northwind_data.b1")
    tokens = tokenize(bullet.canonical)
    assert "c#" in tokens
    assert ".net" in tokens
    # "AT&T" survives whole. Without "&" as a token character it would split
    # into ["at", "t"], and "at" is a stopword -- leaving the single letter "t".
    assert "at&t" in tokens
    assert "t" not in tokens
