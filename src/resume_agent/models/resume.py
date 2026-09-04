"""Rewritten bullets and the verdict on whether they are grounded. Spec 4.

`TailoredBullet.source_id` is the field the whole ethical premise hangs on:
every line on the finished resume points back at a `canonical` sentence in the
knowledge base that it may rephrase but must not contradict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Which check rejected a rewrite. Ordered cheapest-first, which is also the
# order they run in: the two free layers before the one that costs money.
VerificationLayer = Literal["numbers", "vocabulary", "judge", "passed"]


class TailoredBulletFields(BaseModel):
    """What the model returns for one rewritten bullet.

    `estimated_lines` is absent on purpose -- it is `ceil(len / 109)`, which is
    Python's job (CLAUDE.md rule 2). Asking a model to count lines is exactly
    the temptation the rule exists to remove.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="The bullet id this rewrites. Must be one that was given.")
    text: str = Field(description="The rewritten bullet.")
    keywords_used: list[str] = Field(
        default_factory=list,
        description="ATS keywords from the posting that were truthfully worked in.",
    )
    metrics_used: list[str] = Field(
        default_factory=list,
        description="Keys from the provided metrics dict whose values appear in the text.",
    )


class TailoredBullet(TailoredBulletFields):
    """A rewritten bullet with its computed line cost."""

    estimated_lines: int


class VerificationResult(BaseModel):
    """The outcome of checking one rewrite against its source."""

    model_config = ConfigDict(extra="forbid")

    bullet_id: str
    passed: bool
    layer: VerificationLayer
    # Fed straight back into the next tailoring attempt, so it has to be
    # actionable prose rather than an error code: the model reads it.
    critique: str | None = None

    @classmethod
    def ok(cls, bullet_id: str) -> VerificationResult:
        return cls(bullet_id=bullet_id, passed=True, layer="passed")


class JudgeVerdict(BaseModel):
    """The LLM judge's answer. Spec 5: "`supported` / `unsupported` + reason"."""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supported", "unsupported"]
    reason: str = Field(
        description=(
            "One sentence. If unsupported, name the specific phrase that overreaches "
            "and what the source actually says."
        )
    )
