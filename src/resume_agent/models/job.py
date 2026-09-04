"""What a job description becomes once it has been read. Spec 4.

The split between `JobSpecFields` and `JobSpec` is deliberate and is the whole
of CLAUDE.md rule 2 in miniature: **the LLM judges, Python counts.**

`JobSpecFields` is the schema handed to `with_structured_output`. It contains
only judgements -- what seniority is this, which requirements are load-bearing,
what is the register of the writing. `JobSpec` adds `source_hash`, which is a
sha256 and therefore something Python computes. A model asked to emit a hash
will produce a plausible-looking hex string that is not the hash of anything.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RequirementCategory = Literal["language", "framework", "tool", "domain", "practice", "soft"]
Seniority = Literal["intern", "junior", "mid", "senior", "staff", "lead", "unknown"]
Tone = Literal["formal", "conversational", "startup", "academic"]


def job_description_hash(raw_jd: str) -> str:
    """sha256 of the raw posting -- the identity of a job description.

    Whitespace is normalised first so that a JD copied twice from the same page
    with different trailing spaces is recognised as the same posting.
    """
    normalised = "\n".join(line.rstrip() for line in raw_jd.strip().splitlines())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class Requirement(BaseModel):
    """One thing the job is asking for."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        description="The requirement, in the posting's own words where possible."
    )
    category: RequirementCategory
    weight: int = Field(ge=1, le=5, description="How load-bearing this is in the posting, 1-5.")
    is_must_have: bool = Field(description="True for hard requirements, false for nice-to-haves.")

    # Added beyond spec 4. Spec 5 asks the prompt to distinguish stated
    # requirements from inferred priorities -- "a JD that mentions on-call three
    # times cares about ops maturity even if it never says so". That distinction
    # is worthless if it does not survive into the data structure: M3 should be
    # able to weight a stated must-have differently from something read between
    # the lines, and a fit report should never tell you that you failed to meet
    # a requirement nobody actually wrote down.
    is_inferred: bool = Field(
        default=False,
        description="False if stated in the posting; true if inferred from emphasis or repetition.",
    )


class JobSpecFields(BaseModel):
    """Everything the model is asked to judge about a posting.

    This exact schema is what `llm.with_structured_output()` receives, so every
    field description here is prompt surface -- the model reads them.
    """

    model_config = ConfigDict(extra="forbid")

    company: str
    title: str
    seniority: Seniority
    domain: str = Field(
        description="The problem space, e.g. 'fintech payments' or 'developer tooling'."
    )
    requirements: list[Requirement]
    responsibilities: list[str]
    ats_keywords: list[str] = Field(
        description="Verbatim terms from the posting worth mirroring exactly in a resume."
    )
    culture_signals: list[str]
    tone: Tone
    red_flags: list[str] = Field(
        description=(
            "Concerns a candidate should notice: unpaid overtime hints, fifteen must-haves "
            "for a junior role, vague compensation, churn signals. Empty if there are none."
        )
    )


class JobSpec(JobSpecFields):
    """A parsed posting, plus the hash that identifies it."""

    source_hash: str = Field(description="sha256 of the raw posting. Computed, never generated.")

    @classmethod
    def from_fields(cls, fields: JobSpecFields, raw_jd: str) -> JobSpec:
        return cls(**fields.model_dump(), source_hash=job_description_hash(raw_jd))

    # -- convenience views used by M3's scoring -----------------------------

    def must_haves(self) -> list[Requirement]:
        return [r for r in self.requirements if r.is_must_have]

    def stated_requirements(self) -> list[Requirement]:
        return [r for r in self.requirements if not r.is_inferred]

    def inferred_requirements(self) -> list[Requirement]:
        return [r for r in self.requirements if r.is_inferred]
