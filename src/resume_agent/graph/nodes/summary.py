"""The professional summary, and the gate that keeps it honest.

Every other generated line on the resume rewrites exactly one `canonical`
sentence, so the grounding check has an obvious source to compare against. A
summary does not: it says what a *set* of achievements adds up to, and there is
no single sentence it is a rewrite of.

**So the source is the set.** `unsupported_numbers` and
`unsupported_technologies` already answer "what did this text assert that its
source did not"; here the source is the selected achievements joined together.
A figure or a technology in the summary that appears in none of them is exactly
the fabrication the per-bullet gate catches, found the same way with the same
tested code. This is the third place that pattern is reused -- the rewriter uses
one bullet, the chat uses your typed message, this uses the selection.

**Failure is silence, not a guess.** At the retry cap the summary is dropped and
the resume renders without one. A resume with no summary is complete; a resume
whose opening line claims something it does not evidence is worse than both.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.grounding.numbers import unsupported_numbers
from resume_agent.grounding.vocabulary import unsupported_technologies
from resume_agent.llm import build_chat_model, load_prompt
from resume_agent.models.profile import Profile

logger = logging.getLogger(__name__)

PROMPT_NAME = "write_summary"

# One retry. The first rejection tells the model exactly what it invented, which
# is the information it was missing; a second failure means it is not going to
# stop, and spending a third call to confirm that is money for nothing.
MAX_SUMMARY_ATTEMPTS = 2


def summary_source(profile: Profile, selected_ids: list[str], tailored: dict[str, str]) -> str:
    """The text the summary is allowed to draw on, joined.

    Tailored text where a bullet has it, canonical otherwise -- the same
    precedence the renderer uses, so the gate is checking against the words that
    will actually appear underneath the summary on the page.
    """
    parts: list[str] = []
    for bullet_id in selected_ids:
        if bullet_id in tailored:
            parts.append(tailored[bullet_id])
            continue
        try:
            parts.append(profile.bullet_by_id(bullet_id).canonical)
        except KeyError:  # pragma: no cover - selection only yields real ids
            continue
    return "\n".join(parts)


def check_summary(text: str, source: str) -> set[str]:
    """What the summary asserted that the selected achievements do not support.

    **An empty vocabulary, deliberately** -- the same call the chat's gate makes,
    and for the same reason. `skill_vocabulary()` is a legitimate allow-list when
    the claim being checked is one bullet's rewrite, because the skills list says
    what this person has used and the bullet says where. Here the claim is about
    the page as a whole, so a technology in skills.yaml that no *selected*
    achievement demonstrates is precisely the promise the reader will look for
    and not find. `write_summary.md` rule 2 says the same thing to the model;
    passing the vocabulary here would have let the gate disagree with the prompt.
    """
    invented = unsupported_numbers(text, {}, source)
    invented |= unsupported_technologies(text, source, set(), prose=True)
    return invented


def write_summary(
    profile: Profile,
    job_title: str,
    selected_ids: list[str],
    tailored: dict[str, str],
    *,
    llm: BaseChatModel | None = None,
) -> tuple[str, list[str]]:
    """Returns `(summary, complaints)`. An empty summary means it was dropped."""
    source = summary_source(profile, selected_ids, tailored)
    if not source.strip():
        return "", ["no selected achievements to summarise"]

    llm = llm or build_chat_model()
    complaints: list[str] = []

    for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
        message = (
            f"<target_role>\n{job_title}\n</target_role>\n\n"
            f"<selected_achievements>\n{source}\n</selected_achievements>"
        )
        if complaints:
            message += (
                "\n\n<rejected_last_attempt>\n"
                + "\n".join(complaints)
                + "\nRewrite without those.\n</rejected_last_attempt>"
            )

        response = llm.invoke([
            SystemMessage(content=load_prompt(PROMPT_NAME)),
            HumanMessage(content=message),
        ])
        text = _text_of(response).strip()

        invented = check_summary(text, source)
        if not invented:
            return text, []

        complaint = (
            f"attempt {attempt}: {', '.join(sorted(invented))} "
            "appears in none of the selected achievements"
        )
        logger.warning("summary: %s", complaint)
        complaints.append(complaint)

    logger.error("summary: dropped after %d attempts", MAX_SUMMARY_ATTEMPTS)
    return "", complaints


def _text_of(response: object) -> str:
    """Normalise a chat response to a string.

    Anthropic returns content as a list of typed blocks rather than a string,
    so the naive `response.content` is a list here and a str elsewhere.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
