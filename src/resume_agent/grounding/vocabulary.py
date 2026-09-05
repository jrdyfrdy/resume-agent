r"""Which technologies a rewritten bullet is allowed to name. Spec 5, `verify_grounding`.

    tech_generated = extract_tech_tokens(tailored.text)   # capitalized + known-tech regex
    assert tech_generated <= global_skill_vocabulary

Spec 3.2 on why `skills.yaml` doubles as this allow-list:

    "if the generated bullet says 'Kafka' and Kafka isn't in your vocabulary,
     that's a fabrication."

**The naive version does not work.** "Capitalised tokens" flags `Cut` at the
start of every bullet, `Redis` legitimately, and `AT&T` confusingly. Running
that against a skills vocabulary rejects almost everything.

The fix is the same one the number checker uses: compare against the source.

    tech_in_rewrite - tech_in_canonical  subset-of  skill_vocabulary

A technology the source already names needs no permission -- it is by definition
grounded. Only *newly introduced* technology has to appear in `skills.yaml`, and
that is exactly the fabrication signal, with no false positives from sentence
openers or from proper nouns the profile itself uses.
"""

from __future__ import annotations

import re

# Candidate technology mentions:
#   * an internally capitalised or all-caps token   PostgreSQL, AWS, DuckDB
#   * a capitalised word                            Redis, Kubernetes, Cut
#   * a token carrying technology punctuation       C#, .NET, Node.js, CI/CD
#
# Hyphens are NOT compound characters here, unlike in the BM25 tokenizer. In
# prose a hyphen usually joins a compound adjective -- "Cassandra-backed",
# "read-through" -- and keeping it whole hides the capitalised head behind a
# lowercase tail, so "Cassandra-backed" stops looking like a technology at all.
# Splitting lets "Cassandra" be judged on its own and drops "backed" harmlessly.
_TECH_CANDIDATE_RE = re.compile(r"\.?[A-Za-z][A-Za-z0-9]*(?:[.#+/][A-Za-z0-9#+]+)*")

# Words that appear capitalised in ordinary resume prose and are never
# technologies -- overwhelmingly the strong past-tense verbs the tailoring prompt
# asks for, which land at the start of a sentence and get capitalised there.
#
# Only needed for words a *rewrite* introduces that the source did not use: a
# verb shared with the canonical cancels out before this list is consulted. It
# matters when a rewrite opens with "Reduced" where the source said "Cut".
#
# Kept to verbs and function words on purpose. Adding plausible-sounding nouns
# here would start hiding real fabrications, and spec 12 is explicit that the
# check stays strict even when it is annoying.
_NON_TECH_WORDS = frozenset(
    {
        "a", "achieved", "added", "an", "and", "architected", "at", "authored",
        "automated", "built", "by", "consolidated", "created", "cut",
        "delivered", "designed", "developed", "drove", "eliminated", "enabled",
        "expanded", "for", "from", "grew", "halved", "held", "implemented",
        "improved", "in", "increased", "instrumented", "integrated", "into",
        "introduced", "launched", "led", "maintained", "migrated", "modernised",
        "modernized", "of", "on", "optimised", "optimized", "owned",
        "partnered", "prototyped", "rearchitected", "rebuilt", "reduced",
        "refactored", "removed", "repartitioned", "replaced", "scaled",
        "shipped", "simplified", "the", "to", "tripled", "with", "wrote",
    }
)  # fmt: skip


def _looks_like_technology(token: str) -> bool:
    """Whether a token is worth checking against the vocabulary at all.

    Plain capitalised words count. That is deliberate: "Cassandra" and
    "Kubernetes" carry no internal capitals and no punctuation, and excluding
    them would let the most ordinary fabrication -- naming a technology the
    candidate has never used -- pass unnoticed.

    The false positives that would otherwise create (every sentence-opening
    verb) are handled twice over: by the canonical-difference rule in
    `unsupported_technologies`, and by `_NON_TECH_WORDS` for the case where a
    rewrite opens with a different verb than the source.
    """
    if len(token) < 2 or token.lower() in _NON_TECH_WORDS:
        return False
    # Technology punctuation is a strong signal on its own: C#, .NET, Node.js.
    if any(char in token for char in ".#+/"):
        return True
    # ALLCAPS (AWS, SQL), internal capitals (PostgreSQL, DuckDB), or a plain
    # capitalised word (Redis, Kubernetes, Cassandra).
    return token[0].isupper()


# Where a sentence begins: the start of the text, after terminal punctuation, or
# at the start of a line -- optionally past a markdown marker, because the
# narratives are markdown and `# How I learn` capitalises "How" for exactly the
# same positional reason a sentence opener does.
_SENTENCE_START_RE = re.compile(
    r"(?:(?<=[.!?])\s+|^[ \t]*(?:[#>*\-]+[ \t]*)?)([A-Za-z][A-Za-z0-9]*)",
    re.MULTILINE,
)


def _sentence_initial_words(text: str) -> set[str]:
    return {match.group(1).lower() for match in _SENTENCE_START_RE.finditer(text)}


def extract_tech_tokens(text: str, *, prose: bool = False) -> set[str]:
    """Technology-looking tokens in `text`, lowercased for comparison.

    `prose=True` additionally ignores **plain** capitalised words that sit at the
    start of a sentence.

    That mode exists for the cover letter, and the reason is worth understanding.
    For a rewritten bullet the canonical-difference rule handles sentence
    openers: the source starts with a capitalised verb too, so it cancels. A
    letter is many sentences of ordinary prose, and its openers -- "Separately",
    "Your", "At" -- appear nowhere in the knowledge base, so every one of them
    would be reported as an invented technology. (Found exactly that way: the
    first letter run rejected itself over the word "Separately".)

    A token that looks technological by *shape* is still flagged wherever it
    appears: internal capitals (PostgreSQL), all caps (AWS), or technology
    punctuation (C#, .NET, Node.js). The narrow hole this leaves is a plain
    capitalised unknown technology as the very first word of a sentence --
    "Cassandra backs the product." Accepted knowingly: the alternative makes the
    check unusable on prose, and the judge still reads the whole letter.
    """
    tokens = {match.group().strip(".,;:") for match in _TECH_CANDIDATE_RE.finditer(text)}
    candidates = {token for token in tokens if _looks_like_technology(token)}

    if prose:
        openers = _sentence_initial_words(text)
        candidates = {
            token
            for token in candidates
            if token.lower() not in openers or not _is_plain_capitalised(token)
        }

    return {token.lower() for token in candidates}


def _is_plain_capitalised(token: str) -> bool:
    """A capitalised word with no other technology signal: `Separately`, `Cassandra`."""
    if any(char in token for char in ".#+/"):
        return False
    if token.isupper():
        return False
    return token[:1].isupper() and not any(char.isupper() for char in token[1:])


def unsupported_technologies(
    text: str, canonical: str, vocabulary: set[str], *, prose: bool = False
) -> set[str]:
    """Technologies the text introduces that are not in the skill vocabulary.

    `vocabulary` is `Profile.skill_vocabulary()` -- canonical names and every
    alias, lowercased. A non-empty result means the text named a technology the
    candidate has never recorded using.

    `prose=True` for multi-sentence text such as a cover letter; see
    `extract_tech_tokens`.
    """
    introduced = extract_tech_tokens(text, prose=prose) - extract_tech_tokens(
        canonical, prose=prose
    )
    return {token for token in introduced if token not in vocabulary}
