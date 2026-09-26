"""The three kinds of question Jev answers, and the answers it gives.

Shapes from TypeSafe's API reference (docs.typesafe.ai/api.md, checked
2026-09-26), which is the contract; the SDK is not used (see `client.py`):

    noul    "is this statement true?"   -> a probability, 0..1
    choice  "which of these options?"   -> the likeliest option, every option's
                                           probability, and a confidence
    score   "where on this ordered scale?" -> the probability-weighted level,
                                           every level's probability, confidence

Question wording is not written here. It lives in `prompts/decide_*.md`
(CLAUDE.md rule 5) and is loaded by `load_question`: YAML front matter for the
type and criteria, the markdown body as the instructions.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from resume_agent.llm import load_prompt

# From the API reference: a choice may have up to 255 options, a score 2 to 10
# levels. Checked here, where a mistake is a load error in a test, rather than
# as a 422 from the API in the middle of someone's run.
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class Noul(BaseModel):
    """A yes-or-no question. `criteria` may say what yes and no each mean."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["noul"] = "noul"
    instructions: str = Field(min_length=1)
    criteria: dict[Literal["true", "false"], str] | None = None


class Choice(BaseModel):
    """Pick one option. `criteria` maps each option's name to what it means."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["choice"] = "choice"
    instructions: str = Field(min_length=1)
    criteria: dict[str, str]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, criteria: dict[str, str]) -> dict[str, str]:
        if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"a choice needs 2 to {MAX_CHOICE_OPTIONS} options")
        return criteria


class Score(BaseModel):
    """Rate on an ordered scale. `criteria` runs from the low end to the high."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["score"] = "score"
    instructions: str = Field(min_length=1)
    criteria: list[str]

    @field_validator("criteria")
    @classmethod
    def _level_count(cls, criteria: list[str]) -> list[str]:
        if not MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        return criteria

    @property
    def top(self) -> int:
        """The highest level's number. Levels are numbered from 0."""
        return len(self.criteria) - 1


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


# -- answers -------------------------------------------------------------------
#
# `extra="ignore"` on everything that comes back: an API ten days old will add
# fields, and a new field is not a reason to fail someone's run.


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0, description="Probability the statement is true.")


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["choice"]
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["score"]
    score: float = Field(description="Probability-weighted level, 0 to the top level.")
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[int, float]


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int | None = None
    output_tokens: int | None = None


class Decision(BaseModel):
    """One response: an answer per question, keyed as the questions were."""

    model_config = ConfigDict(extra="ignore")

    model: str = ""
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)


# -- loading question wording ---------------------------------------------------


def load_question(name: str) -> Noul | Choice | Score:
    """`prompts/decide_<name>.md` as a question.

    Front matter between two `---` lines gives `type` and `criteria`; the body
    is the instructions. Loaded through `load_prompt`, so the question is
    versioned by `prompt_version` and can be overridden for an eval like any
    other prompt.
    """
    text = load_prompt(f"decide_{name}")
    header, body = _split_front_matter(text, name)
    criteria = header.get("criteria")
    if isinstance(criteria, dict):
        # YAML reads a bare `true:` key as the boolean True. The API wants the
        # string "true", and a choice's option names are strings too.
        header["criteria"] = {
            (str(key).lower() if isinstance(key, bool) else str(key)): value
            for key, value in criteria.items()
        }
    fields: dict[str, Any] = {**header, "instructions": body}
    kinds = {"noul": Noul, "choice": Choice, "score": Score}
    kind = kinds.get(str(fields.get("type")))
    if kind is None:
        raise ValueError(f"decide_{name}.md: type must be one of {', '.join(kinds)}")
    return kind.model_validate(fields)


def _split_front_matter(text: str, name: str) -> tuple[dict, str]:
    lines = text.strip().splitlines()
    closed = any(line.strip() == "---" for line in lines[1:])
    if not lines or lines[0].strip() != "---" or not closed:
        raise ValueError(f"decide_{name}.md needs front matter between '---' lines")
    end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    header = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(header, dict):
        raise ValueError(f"decide_{name}.md: front matter must be a mapping")
    return header, "\n".join(lines[end + 1:]).strip()
