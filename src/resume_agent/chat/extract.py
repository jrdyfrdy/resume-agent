"""Turn something you typed into profile entries -- and check it did not grow.

This is the one place in the project where a model's output becomes **ground
truth** rather than being measured against it. Everywhere else `canonical` is
your sentence and `metrics` are your numbers, and the grounding gate checks
generated resume text against them. Here the model writes those fields, so the
usual protection does not apply: `_validate_with_edit` proves only that the
profile still *loads*, never that a sentence in it is true.

**The gate, one level up.** The rule that makes this safe is the same rule the
rewriter obeys, with your message standing in for `canonical`: the model may
restructure what you said, and may not add to it. `grounding/numbers.py` and
`grounding/vocabulary.py` already answer "what did this text assert that its
source did not", so they are used unchanged --

    unsupported_numbers(proposed_text, {}, canonical=your_message)
    unsupported_technologies(proposed_text, your_message, vocabulary, prose=True)

-- and anything they return is shown to you in red before you accept it. A
number you never said cannot become a fact about your career by accident.

**Nothing here writes.** `apply` goes through `forms.create_entry` and
`forms.write_document`, so an accepted proposal inherits the staging validation,
the timestamped backup, the atomic replace and the comment preservation that
already exist. There is still exactly one code path that writes to a profile.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from resume_agent.grounding.numbers import extract_numbers, unsupported_numbers
from resume_agent.grounding.vocabulary import unsupported_technologies
from resume_agent.kb.forms import (
    create_entry,
    next_bullet_id,
    read_document,
    write_document,
)
from resume_agent.llm import build_chat_model, load_prompt, structured_output
from resume_agent.models.profile import Profile

logger = logging.getLogger(__name__)

PROMPT_NAME = "chat_extract"

# `YearMonth` in `models/profile.py`. Checked here so a bad date is a flag you
# can fix in the proposal, rather than a refused save after you accept.
_YEAR_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
# Tokens a technology name could be, for "did the user actually say this?".
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9.#+/-]*")


# ---------------------------------------------------------------------------
# What the model is asked for
# ---------------------------------------------------------------------------


class ProposedBullet(BaseModel):
    """One achievement, as the model heard it.

    No `id`: ids are generated from the entry they land in
    (`forms.next_bullet_id`), never authored. `metrics` values are strings
    because the model returns text; `forms._number_or_text` restores the type on
    the way to disk.
    """

    model_config = ConfigDict(extra="forbid")

    canonical: str = Field(description="One achievement, in the user's own terms.")
    metrics: dict[str, str] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    evidence: str | None = None


class ProposedEntry(BaseModel):
    """A job or a project the user described."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["experience", "project"]
    name: str = Field(description="Employer for a job, project name for a project.")
    title: str = ""
    location: str = ""
    start: str = Field(default="", description="YYYY-MM, empty if not stated.")
    end: str | None = None
    tech: list[str] = Field(default_factory=list)
    bullets: list[ProposedBullet] = Field(default_factory=list)


class ProposedBulletForEntry(BaseModel):
    """An achievement that belongs to a role already in the profile."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    bullet: ProposedBullet


class ProposedSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical: str
    aliases: list[str] = Field(default_factory=list)
    category: str = "other"
    level: Literal["expert", "working", "familiar"] = "working"


class ExtractionFields(BaseModel):
    """The structured-output schema. One call returns all of it."""

    model_config = ConfigDict(extra="forbid")

    reply: str = Field(
        default="",
        description="One or two sentences to show the user.",
    )
    entries: list[ProposedEntry] = Field(default_factory=list)
    bullets: list[ProposedBulletForEntry] = Field(default_factory=list)
    skills: list[ProposedSkill] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# What comes back out
# ---------------------------------------------------------------------------


class Flag(BaseModel):
    """Something in the proposal that your message does not support."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: Literal["invented_number", "invented_technology", "unknown_entry", "bad_date"]
    detail: str


class Item(BaseModel):
    """One accept-or-discard unit, with its own id so the UI can address it."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: Literal["entry", "bullet", "skill"]
    summary: str
    payload: dict
    flags: list[Flag] = Field(default_factory=list)


class Proposal(BaseModel):
    """Everything the model extracted, held in memory until you accept it."""

    model_config = ConfigDict(extra="forbid")

    reply: str = ""
    items: list[Item] = Field(default_factory=list)

    def flagged(self) -> list[Item]:
        return [item for item in self.items if item.flags]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract(
    message: str, profile: Profile, *, llm: BaseChatModel | None = None
) -> Proposal:
    """One structured call, then the gate. Never writes."""
    llm = llm or build_chat_model()
    structured = structured_output(llm, ExtractionFields)

    known = "\n".join(
        f"- {entry.id}: {getattr(entry, 'org', None) or getattr(entry, 'name', '')}"
        for entry in profile.entries()
    )
    vocabulary = "\n".join(sorted({skill.canonical for skill in profile.skills}))

    result = structured.invoke([
        SystemMessage(content=load_prompt(PROMPT_NAME)),
        HumanMessage(content=(
            f"<existing_roles>\n{known or '(none yet)'}\n</existing_roles>\n\n"
            f"<known_technologies>\n{vocabulary or '(none yet)'}\n</known_technologies>\n\n"
            f"<what_the_user_said>\n{message.strip()}\n</what_the_user_said>"
        )),
    ])
    # `with_structured_output` can hand back a dict depending on the provider.
    if not isinstance(result, ExtractionFields):
        result = ExtractionFields.model_validate(result)

    return build_proposal(result, message, profile)


def build_proposal(
    fields: ExtractionFields, message: str, profile: Profile
) -> Proposal:
    """Wrap the model's output as addressable items, each carrying its flags.

    **The only source is the message.** Note what is deliberately *not* used
    here: `profile.skill_vocabulary()`. The tailoring gate passes it because
    there `canonical` is the specific claim and the skills list legitimately
    says what the candidate has used. Here the specific claim is the sentence
    you just typed -- your skills list records that you have used Kafka
    somewhere, not that you used it on the thing you are describing now. Passing
    the vocabulary in would let "a queue" become "Kafka", which is exactly the
    upgrade `chat_extract.md` rule 3 forbids.
    """
    known_entries = {entry.id for entry in profile.entries()}
    items: list[Item] = []

    for index, entry in enumerate(fields.entries):
        item_id = f"entry-{index}"
        flags = list(_check_date(item_id, entry.start))
        for bullet in entry.bullets:
            flags.extend(_check_bullet(item_id, bullet, message))
        label = "Job" if entry.kind == "experience" else "Project"
        items.append(Item(
            item_id=item_id,
            kind="entry",
            summary=f"{label}: {entry.name}"
                    + (f" — {len(entry.bullets)} achievement(s)" if entry.bullets else ""),
            payload=entry.model_dump(),
            flags=flags,
        ))

    for index, addition in enumerate(fields.bullets):
        item_id = f"bullet-{index}"
        flags = list(_check_bullet(item_id, addition.bullet, message))
        if addition.entry_id not in known_entries:
            # The model named a role that does not exist. Left as a flag rather
            # than dropped, because the useful answer is usually "you meant this
            # other role" and only you can say which.
            flags.append(Flag(
                item_id=item_id,
                kind="unknown_entry",
                detail=f"No role called {addition.entry_id!r} exists in this profile.",
            ))
        items.append(Item(
            item_id=item_id,
            kind="bullet",
            summary=f"Achievement for {addition.entry_id}: "
                    f"{addition.bullet.canonical[:70]}",
            payload=addition.model_dump(),
            flags=flags,
        ))

    for index, skill in enumerate(fields.skills):
        item_id = f"skill-{index}"
        flags = []
        if skill.canonical.lower() not in _said(message):
            flags.append(Flag(
                item_id=item_id,
                kind="invented_technology",
                detail=f"You did not mention {skill.canonical!r}.",
            ))
        items.append(Item(
            item_id=item_id,
            kind="skill",
            summary=f"Skill: {skill.canonical}",
            payload=skill.model_dump(),
            flags=flags,
        ))

    return Proposal(reply=fields.reply.strip(), items=items)


def _check_bullet(item_id: str, bullet: ProposedBullet, message: str) -> list[Flag]:
    """The gate. Your message is the source; the proposal may not exceed it."""
    flags: list[Flag] = []

    # Numbers: in the sentence, and in every metric value.
    invented = unsupported_numbers(bullet.canonical, {}, message)
    for key, value in bullet.metrics.items():
        for number in extract_numbers(str(value)):
            if number not in extract_numbers(message):
                invented.add(f"{key}={value}")
    if invented:
        flags.append(Flag(
            item_id=item_id,
            kind="invented_number",
            detail=f"{_listed(invented)} appears nowhere in what you wrote.",
        ))

    # Technologies: anything named that you did not say. An empty vocabulary is
    # the point -- see `build_proposal`.
    unknown = unsupported_technologies(bullet.canonical, message, set(), prose=True)
    unknown |= {name for name in bullet.skills if name.lower() not in _said(message)}
    if unknown:
        flags.append(Flag(
            item_id=item_id,
            kind="invented_technology",
            detail=f"{_listed(unknown)} appears nowhere in what you wrote.",
        ))
    return flags


def _check_date(item_id: str, start: str) -> list[Flag]:
    """`YearMonth` is `YYYY-MM`; anything else refuses to load later."""
    if start and not _YEAR_MONTH.match(start):
        return [Flag(
            item_id=item_id,
            kind="bad_date",
            detail=f"{start!r} is not a YYYY-MM date; you will be asked for one.",
        )]
    return []


def _listed(values: set[str]) -> str:
    """Quote a set for a person to read, not a Python repr.

    These strings go straight into the proposal card, where `['80']` reads as a
    leaked implementation detail rather than as the number the model invented.
    """
    quoted = [f"“{value}”" for value in sorted(values)]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + f" and {quoted[-1]}"


def _said(message: str) -> set[str]:
    return set(message.lower().split()) | {
        token.lower() for token in _WORD.findall(message)
    }


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------


def apply(
    proposal: Proposal, profile_dir: Path, accepted: list[str]
) -> list[str]:
    """Write the accepted items. Returns a line per change, for the transcript.

    **Skills go first.** There is no transactional multi-file write here -- each
    call to `write_document` validates the whole directory on its own -- so a
    bullet naming a new technology must not land before the skill that makes it
    resolvable, or the intermediate state fails to load and the save is refused.
    """
    wanted = {item.item_id for item in proposal.items} & set(accepted)
    items = [item for item in proposal.items if item.item_id in wanted]
    done: list[str] = []

    for item in sorted(items, key=lambda i: {"skill": 0, "entry": 1, "bullet": 2}[i.kind]):
        if item.kind == "skill":
            done.append(_add_skill(profile_dir, item))
        elif item.kind == "entry":
            done.append(_add_entry(profile_dir, item))
        else:
            done.append(_add_bullet(profile_dir, item))
    return done


def _add_skill(profile_dir: Path, item: Item) -> str:
    document = read_document(profile_dir, "skills.yaml")
    rows = document["data"]["skills"]
    if any(r["canonical"].lower() == item.payload["canonical"].lower() for r in rows):
        return f"{item.payload['canonical']} was already in your skills"
    rows.append(item.payload)
    write_document(profile_dir, "skills.yaml", document["data"])
    return f"added the skill {item.payload['canonical']}"


def _add_entry(profile_dir: Path, item: Item) -> str:
    payload = item.payload
    relative = create_entry(profile_dir, payload["kind"], payload["name"])

    document = read_document(profile_dir, relative)
    data = document["data"]
    for key in ("title", "location", "start", "end", "tech"):
        if payload.get(key):
            data[key] = payload[key]

    entry_id = data["id"]
    for proposed in payload.get("bullets") or []:
        data["bullets"].append(_bullet_row(entry_id, data["bullets"], proposed))

    write_document(profile_dir, relative, data)
    return f"added {payload['name']} with {len(payload.get('bullets') or [])} achievement(s)"


def _add_bullet(profile_dir: Path, item: Item) -> str:
    entry_id = item.payload["entry_id"]
    relative = _file_for_entry(profile_dir, entry_id)

    document = read_document(profile_dir, relative)
    data = document["data"]
    data["bullets"].append(_bullet_row(entry_id, data["bullets"], item.payload["bullet"]))
    write_document(profile_dir, relative, data)
    return f"added an achievement to {entry_id}"


def _bullet_row(entry_id: str, existing: list[dict], proposed: dict) -> dict:
    """A bullet as the form layer expects it, with a generated id.

    `confidence` is left at the schema default rather than being set from the
    model: it is a statement about how well *you* can substantiate something,
    which nothing in this pipeline is in a position to decide.
    """
    return {
        "id": next_bullet_id(entry_id, [b["id"] for b in existing]),
        "canonical": proposed["canonical"],
        "metrics": proposed.get("metrics") or {},
        "skills": proposed.get("skills") or [],
        "themes": proposed.get("themes") or [],
        "evidence": proposed.get("evidence"),
        "confidence": "verified",
    }


def _file_for_entry(profile_dir: Path, entry_id: str) -> str:
    from resume_agent.kb.loader import load_profile  # noqa: PLC0415 - avoids a cycle
    from resume_agent.kb.writer import relative_profile_files

    profile = load_profile(Path(profile_dir))
    if entry_id not in {entry.id for entry in profile.entries()}:
        raise ValueError(f"no role called {entry_id!r} in this profile")

    for relative in relative_profile_files(profile_dir):
        if relative.endswith(".yaml") and "/" in relative:
            document = read_document(profile_dir, relative)
            if document["data"].get("id") == entry_id:
                return relative
    raise ValueError(f"could not find the file holding {entry_id!r}")
