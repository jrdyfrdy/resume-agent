"""The LLM-as-judge half of the eval. Spec 9.

    "LLM-as-judge rubric (1-5, one judge call per resume, with the JD in
     context): relevance to JD, specificity of claims, ATS keyword alignment,
     tone match, absence of filler"

One call per resume, on the cheaper model. Fifteen judge calls per eval run is
the whole cost of the scoring half, and the task -- read two things and give
five integers -- does not repay Opus.

Spec 9 also says the human spot-check is the real calibration set: *"Judges
drift; your eye is the calibration set."* Which is why `run_eval.py` prints the
three lowest-scoring resumes by name rather than only an average.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from resume_agent.llm import build_chat_model, load_prompt
from resume_agent.models.job import JobSpec
from resume_agent.models.resume import TailoredBullet

PROMPT_NAME = "judge_resume"

# The five dimensions from spec 9, in the order it lists them.
DIMENSIONS = ["relevance", "specificity", "ats_alignment", "tone_match", "absence_of_filler"]


class JudgeScores(BaseModel):
    """Spec 9's rubric. Integers 1-5, one per dimension."""

    model_config = ConfigDict(extra="forbid")

    relevance: int = Field(ge=1, le=5, description="Do these bullets answer what the posting asks?")
    specificity: int = Field(ge=1, le=5, description="Are the claims concrete enough to check?")
    ats_alignment: int = Field(ge=1, le=5, description="Posting's terminology, where honest.")
    tone_match: int = Field(ge=1, le=5, description="Does the register suit this employer?")
    absence_of_filler: int = Field(ge=1, le=5, description="How much text does no work?")
    reasoning: str = Field(description="One paragraph naming what moved a score.")

    @property
    def mean(self) -> float:
        """The single number the regression gate compares.

        Computed here rather than asked for, for the same reason every other
        aggregate in this project is (CLAUDE.md rule 2) -- and because a model
        asked for both the parts and the average will occasionally return an
        average that is not the average.
        """
        return round(sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS), 3)

    def as_dict(self) -> dict[str, float]:
        return {d: getattr(self, d) for d in DIMENSIONS} | {"mean": self.mean}


def judge_resume(
    job: JobSpec,
    tailored: list[TailoredBullet],
    *,
    llm: BaseChatModel | None = None,
) -> JudgeScores | None:
    """Score one resume. Returns None when there is nothing to score.

    A run that produced no bullets is a deterministic failure, already counted
    by `checks.py`. Sending it to the judge would spend a call to be told what
    is already known, and would drag the mean down twice for one fault.
    """
    if not tailored:
        return None

    llm = llm or build_chat_model("judge")
    structured = llm.with_structured_output(JudgeScores)

    requirements = "\n".join(
        f"- [{r.weight}/5] {'MUST' if r.is_must_have else 'nice'}: {r.text}"
        for r in job.requirements
    )
    bullets = "\n".join(f"- {b.text}" for b in tailored)

    result = structured.invoke(
        [
            SystemMessage(content=load_prompt(PROMPT_NAME)),
            HumanMessage(
                content=(
                    f"<posting>\n"
                    f"{job.title} at {job.company} ({job.seniority}, {job.domain})\n"
                    f"tone: {job.tone}\n\n"
                    f"requirements:\n{requirements}\n\n"
                    f"terms worth mirroring: {', '.join(job.ats_keywords)}\n"
                    f"</posting>\n\n"
                    f"<resume_bullets>\n{bullets}\n</resume_bullets>"
                )
            ),
        ]
    )
    if not isinstance(result, JudgeScores):
        result = JudgeScores.model_validate(result)
    return result
