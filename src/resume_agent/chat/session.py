"""The transcript, and deciding which of the two things a message is.

Conversations live in memory, per profile, exactly like the run registry in
`api/app.py`: an append-only list read by index. That shape was chosen there so a
browser refresh mid-run replays rather than finding a drained queue, and a
transcript is even more obviously the same thing -- it *is* a replayable log.

Nothing is persisted to disk. A conversation is not career data; the career data
is whatever you accepted out of it, and that is in `profile/` with a backup.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from resume_agent.chat.extract import Proposal

# Turns kept per profile. Long enough to look back over a working session,
# bounded so an afternoon of chatting does not grow without limit.
MAX_TURNS = 200

# Messages that are clearly a question rather than dictation. Checked before the
# "does this look like dictation" heuristics, because "what did I do at Acme?"
# contains a role name and would otherwise read as something to file.
_ASKS = re.compile(
    r"^\s*(what|which|who|why|how|when|where|should|shall|can|could|would|do|does|"
    r"did|is|are|am|was|were|any|anything|tell me|show me|help|explain|summar|review|"
    r"critique|advi[cs]e|improve|suggest)\b",
    re.IGNORECASE,
)

# An instruction to take something *out* of the profile. Checked before `_ASKS`,
# because the polite form of a command is shaped exactly like a question --
# "Can you remove my narratives?" ends in a question mark and opens with a word
# `_ASKS` matches, but it is not asking anything.
#
# The subject is what separates the two, so the optional prefix requires "you":
# "can you delete X" is a command, "should I delete X" is a genuine question and
# falls through to `advise`, which is where someone weighing a decision wants to
# land. Anchored at the start so "the exporter deletes stale rows" -- a sentence
# describing work, not requesting one -- is never read as an instruction.
_MUTATES = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:go\s+ahead\s+and\s+)?"
    r"(?:remove|delete|drop|clear|erase|wipe|get\s+rid\s+of|take\s+out)\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """One exchange. `proposal` is present only when the message was dictation."""

    turn_id: str
    profile: str
    message: str
    intent: str = "advise"
    status: str = "running"
    reply: str = ""
    proposal: Proposal | None = None
    applied: list[str] = field(default_factory=list)
    error: str | None = None
    # Append-only, read by index -- see the module docstring.
    events: list[dict] = field(default_factory=list)

    def emit(self, kind: str, **payload: object) -> None:
        self.events.append({"kind": kind, **payload})


@dataclass
class Conversation:
    profile: str
    turns: list[Turn] = field(default_factory=list)

    def transcript(self) -> list[tuple[str, str]]:
        """Settled exchanges, oldest first, as (role, text) pairs."""
        pairs: list[tuple[str, str]] = []
        for turn in self.turns:
            if turn.status != "done":
                continue
            pairs.append(("user", turn.message))
            if turn.reply:
                pairs.append(("assistant", turn.reply))
        return pairs


class Registry:
    """Every conversation this process is holding. In-process, like `runs`."""

    def __init__(self) -> None:
        self._by_profile: dict[str, Conversation] = {}
        self._turns: dict[str, Turn] = {}

    def conversation(self, profile: str) -> Conversation:
        return self._by_profile.setdefault(profile, Conversation(profile=profile))

    def start(self, profile: str, message: str) -> Turn:
        conversation = self.conversation(profile)
        turn = Turn(
            turn_id=uuid.uuid4().hex[:12],
            profile=profile,
            message=message,
            intent=classify(message),
        )
        conversation.turns.append(turn)
        self._turns[turn.turn_id] = turn

        # Drop the oldest turns, and stop addressing them by id, so a long
        # session does not retain every proposal it ever made.
        while len(conversation.turns) > MAX_TURNS:
            self._turns.pop(conversation.turns.pop(0).turn_id, None)
        return turn

    def get(self, turn_id: str) -> Turn | None:
        return self._turns.get(turn_id)


def classify(message: str) -> str:
    """`advise` or `extract`.

    Deliberately a heuristic in Python rather than a routing call to the model:
    it is one decision per message, it would double the latency of every turn,
    and a wrong answer is cheap -- an extraction of a question returns nothing
    to accept, and advice about a dictation still reads as a sensible reply.

    The bias is towards `advise`, because the failure modes are asymmetric.
    Treating dictation as a question wastes a turn; treating a question as
    dictation puts a proposal in front of someone who did not ask for one.

    The one exception to that bias is a command to remove something. It has to
    reach `extract`, because `extract` is the only half that can produce a
    proposal -- and a removal request answered by `advise` produced the worst
    outcome this feature has had: prose agreeing to do it, and nothing done.
    """
    text = message.strip()
    if not text:
        return "advise"
    if _MUTATES.match(text):
        return "extract"
    if text.endswith("?") or _ASKS.match(text):
        return "advise"

    # Dictation is a person narrating: "I rewrote...", "We shipped...". Matched
    # on the pronoun rather than a past-tense suffix, because the verbs that
    # turn up here are mostly irregular -- rewrote, built, led, spent, shipped.
    narrates = re.match(r"\s*(i|we)\b", text, re.IGNORECASE) is not None
    return "extract" if narrates or len(text.split()) > 25 else "advise"
