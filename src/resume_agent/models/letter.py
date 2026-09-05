"""The cover letter. Spec 5, `write_cover_letter`.

    "Structure: hook -> proof P1 -> proof P2 -> why-this-company -> close.
     Constraints: <= 320 words. No 'I am writing to apply for'. No adjective
     without evidence behind it."

The five paragraphs are separate fields rather than one blob of prose. That is
not tidiness: it lets the structure be checked without parsing, lets a critique
name the paragraph that is wrong, and stops the model from quietly collapsing
five moves into three.

`word_count` is computed here, never returned by the model -- CLAUDE.md rule 2,
the same reason `source_hash` and `estimated_lines` are computed elsewhere. A
model asked how many words it just wrote will guess, and the 320-word limit is
the one hard constraint in the milestone.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Spec 5. A hard number, enforced by counting rather than by asking.
MAX_LETTER_WORDS = 320

LetterCheck = Literal["length", "grounding", "consistency", "passed"]

# A "word" is a run of characters separated by whitespace, which is how a human
# and every word-count tool would read it. Deliberately not a token count: the
# limit exists because a hiring manager will not read more than a page.
_WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


class CoverLetterFields(BaseModel):
    """What the model returns. Spec 5's five-part structure, one field each."""

    model_config = ConfigDict(extra="forbid")

    hook: str = Field(
        description="Opening. Never 'I am writing to apply for'. Earn the next sentence."
    )
    proof_one: str = Field(description="First proof paragraph, grounded in a specific achievement.")
    proof_two: str = Field(description="Second proof paragraph, a different achievement.")
    why_this_company: str = Field(
        description="Why this employer specifically, drawn from the posting itself."
    )
    close: str = Field(description="Short close. No begging, no 'I look forward to hearing'.")
    bullet_ids_used: list[str] = Field(
        default_factory=list,
        description="The achievement ids this letter draws on. Must be ones provided.",
    )


class CoverLetter(CoverLetterFields):
    """A drafted letter with its measured length."""

    word_count: int

    @classmethod
    def from_fields(cls, fields: CoverLetterFields) -> CoverLetter:
        return cls(**fields.model_dump(), word_count=count_words(body_of(fields)))

    def body(self) -> str:
        return body_of(self)

    @property
    def paragraphs(self) -> list[str]:
        return [self.hook, self.proof_one, self.proof_two, self.why_this_company, self.close]


def body_of(fields: CoverLetterFields) -> str:
    """The five paragraphs as one document, in spec 5's order.

    A free function so it works on the model's raw output as well as on a
    finished `CoverLetter` -- the word count has to be measurable *before* the
    object that carries the word count exists.
    """
    parts = [
        fields.hook,
        fields.proof_one,
        fields.proof_two,
        fields.why_this_company,
        fields.close,
    ]
    return "\n\n".join(part.strip() for part in parts if part.strip())


class LetterVerification(BaseModel):
    """The outcome of checking a draft. Mirrors `VerificationResult` for bullets."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    check: LetterCheck
    critique: str | None = None

    @classmethod
    def ok(cls) -> LetterVerification:
        return cls(passed=True, check="passed")


class ConsistencyVerdict(BaseModel):
    """The judge's answer on whether a letter contradicts the resume."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["consistent", "inconsistent"]
    reason: str = Field(
        description=(
            "One sentence. If inconsistent, quote the letter's phrase and the resume "
            "bullet it conflicts with."
        )
    )
