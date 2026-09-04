r"""Tokenizer for the BM25 half of retrieval.

Spec 3.3 is explicit about why BM25 is here at all:

    "Do not skip BM25. Job descriptions are full of exact tokens -- 'Terraform',
    'gRPC', 'PySpark' -- and dense embeddings are notoriously mushy about exact
    technology names."

Which means the tokenizer is not an incidental detail; it is the thing that
decides whether BM25 can do the one job it was brought in for. The obvious
implementation, `re.findall(r"\w+", text.lower())`, quietly destroys exactly the
tokens that matter:

    C#        -> ["c"]                 collides with C, C++, and the letter c
    .NET      -> ["net"]               collides with "network", "net new"
    Node.js   -> ["node", "js"]
    CI/CD     -> ["ci", "cd"]
    user_id   -> ["user", "id"]        (\w keeps _, but most tokenizers split it)
    p95       -> ["p95"]               this one survives, by luck

A JD asking for "C#" would then match every bullet containing the word "c", and
the exact-token advantage BM25 was supposed to provide is gone.

So: keep `# . + / - _` *inside* tokens, and additionally emit the sub-parts of
compound tokens so that a query for "cd" still finds "ci/cd". Emitting both the
whole and the parts costs nothing at this corpus size and makes the matcher
forgiving in the direction that helps.
"""

from __future__ import annotations

import re

# A token may start with a dot -- ".NET" is the reason -- and may then contain
# the punctuation that appears *inside* real technology names. Trailing
# punctuation is excluded by requiring the token to end alphanumeric or in one of
# the few suffix characters that carry meaning (`#` in "C#", `+` in "C++").
#
# The leading dot is safe because it must be followed by an alphanumeric: in
# "Kafka." the final period starts no token, and in "C#/.NET" the "/" is skipped
# and ".net" is matched whole.
#
# `&` is token-internal for the same reason: "AT&T" and "R&D" are single names.
# Without it "AT&T" tokenizes to ["at", "t"], and since "at" is a stopword the
# employer reduces to the single letter "t" -- effectively unsearchable.
_TOKEN_RE = re.compile(r"\.?[a-z0-9]+(?:[._/\-+#&][a-z0-9]+)*[+#]*")

# Characters that join the parts of a compound token. Splitting on these gives
# the sub-tokens; note `#`, `+` and `&` are absent because "c#", "c++" and "at&t"
# have no meaningful parts -- splitting them would reintroduce the collisions above.
_COMPOUND_SPLIT_RE = re.compile(r"[._/\-]")

# Words carrying no retrieval signal in resume prose. Deliberately tiny: an
# aggressive stopword list is a good way to break a query like "not null" or
# "up time", and at 200 documents BM25's IDF term already discounts common words.
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
        "into", "is", "it", "its", "of", "on", "or", "that", "the", "to", "with",
    }
)  # fmt: skip

# Sub-tokens shorter than this are dropped when splitting a compound, so that
# "node.js" does not also emit a bare "js"... it does, actually: 2 is the floor,
# which keeps "js", "ci", "cd", "go" and drops nothing real.
MIN_SUBTOKEN_LENGTH = 2


def tokenize(text: str, *, expand_compounds: bool = True) -> list[str]:
    """Lowercase `text` into BM25 terms, preserving technology names.

    Returns whole tokens first, each followed by its sub-parts when it is a
    compound. Duplicates are kept: BM25 is a bag-of-words model and term
    frequency is meaningful, so deduplicating here would distort scoring.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group()
        if token in STOPWORDS:
            continue
        tokens.append(token)

        if not expand_compounds:
            continue
        parts = [p for p in _COMPOUND_SPLIT_RE.split(token) if len(p) >= MIN_SUBTOKEN_LENGTH]
        # `len(parts) > 1` means it really was a compound; a plain word splits
        # into itself and must not be emitted twice.
        if len(parts) > 1:
            tokens.extend(part for part in parts if part not in STOPWORDS)

    return tokens
