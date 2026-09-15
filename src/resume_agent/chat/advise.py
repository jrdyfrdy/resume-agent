"""Talk about a profile, grounded in facts computed from it.

The model is handed `audit.py`'s counts and the profile's own sentences, and its
job is to explain and prioritise what is already there. That division is
CLAUDE.md rule 2 again -- Python counts, the model judges -- and it is what
separates this from generic resume advice: "add more metrics" is free everywhere,
"these four achievements record no numbers, and here is what that costs you" is
only possible because the list was computed first.

This is the one streaming call in the project. Advice is prose read as it
arrives; everything else here returns a structured object, which is why nothing
else streams.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from resume_agent.chat.audit import ProfileAudit, audit_profile
from resume_agent.llm import build_chat_model, load_prompt
from resume_agent.models.profile import Profile

logger = logging.getLogger(__name__)

PROMPT_NAME = "chat_advise"

# How much of the transcript goes back to the model. A conversation is the first
# thing in this project with an unbounded number of calls per user action, and
# resending the whole history each turn makes the cost quadratic in its length.
# Six turns is enough for "what about the second one?" to resolve.
TRANSCRIPT_TURNS = 6


def profile_briefing(profile: Profile, audit: ProfileAudit) -> str:
    """Everything the model may reason from, as one tagged block.

    The findings come first and carry their own consequence text, so the model
    does not have to know the pipeline to explain why something matters.
    """
    findings = "\n".join(
        f"- {finding.kind}: {', '.join(finding.subjects)}\n  why it matters: {finding.detail}"
        for finding in audit.findings()
    ) or "(nothing flagged)"

    themes = ", ".join(f"{name} x{count}" for name, count in audit.theme_counts.items())

    achievements = "\n".join(
        f"- {bullet.id}: {bullet.canonical.strip()}"
        + (f"\n  numbers: {bullet.metrics}" if bullet.metrics else "\n  numbers: none")
        for bullet in profile.all_bullets()
    )

    return (
        f"<shape>\n{audit.bullets} achievements across {audit.entries} roles, "
        f"{len(profile.skills)} skills, {audit.narratives} narratives.\n"
        f"themes: {themes or 'none'}\n</shape>\n\n"
        f"<findings>\n{findings}\n</findings>\n\n"
        f"<achievements>\n{achievements or '(none yet)'}\n</achievements>"
    )


def build_messages(
    message: str, profile: Profile, transcript: list[tuple[str, str]]
) -> list[BaseMessage]:
    """System prompt, briefing, recent turns, then the new question."""
    audit = audit_profile(profile)
    messages: list[BaseMessage] = [
        SystemMessage(content=load_prompt(PROMPT_NAME)),
        HumanMessage(content=profile_briefing(profile, audit)),
        AIMessage(content="I have read the profile. What would you like to know?"),
    ]
    for role, text in transcript[-TRANSCRIPT_TURNS:]:
        messages.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))
    messages.append(HumanMessage(content=message.strip()))
    return messages


async def advise(
    message: str,
    profile: Profile,
    transcript: list[tuple[str, str]],
    *,
    llm: BaseChatModel | None = None,
) -> AsyncIterator[str]:
    """Stream an answer token by token.

    `astream` yields message chunks whose `.content` is the delta. The chunk can
    be a list of content blocks rather than a string on some providers, so it is
    normalised here rather than in the transport.
    """
    llm = llm or build_chat_model(streaming=True)

    async for chunk in llm.astream(build_messages(message, profile, transcript)):
        text = _text_of(chunk)
        if text:
            yield text


def _text_of(chunk: object) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic-style content blocks: keep the text, drop everything else.
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""
