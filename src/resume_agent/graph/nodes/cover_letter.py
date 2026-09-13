"""The cover letter, as a LangGraph subgraph. Spec 5, `write_cover_letter`.

    "Inputs: JobSpec, CompanyBrief, top-3 EvidenceMatch, narratives/.
     Structure: hook -> proof P1 -> proof P2 -> why-this-company -> close.
     Constraints: <= 320 words. No 'I am writing to apply for'. No adjective
     without evidence behind it. Consistency check: every claim must also be
     supported by the same KB, and must not contradict the resume's selected
     bullets."

**Why a subgraph and not two more nodes in the parent.** Spec 10 maps
"Subgraphs" to this milestone, and the shape fits: drafting and verifying a
letter is a self-contained loop with its own retry budget, and the parent graph
has no business knowing that the loop exists. It calls one node and either gets
a letter or does not.

    draft -> verify -> draft   (retry, cap 2)
                    -> END     (verified, or out of attempts)

**Three checks, cheapest first**, the same discipline as M4's bullet verifier:

    length       count the words                     free
    grounding    numbers and technologies vs the KB  free  (reuses M4's modules)
    consistency  contradicts the resume?             one cheap call

The grounding layer is M4's `unsupported_numbers` and `unsupported_technologies`
applied to the whole knowledge base rather than to a single bullet, because a
letter legitimately draws on several achievements at once.
"""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from resume_agent.graph.state import AgentState
from resume_agent.grounding.numbers import unsupported_numbers
from resume_agent.grounding.vocabulary import unsupported_technologies
from resume_agent.kb.loader import load_profile
from resume_agent.llm import build_chat_model, load_prompt
from resume_agent.models.fit import EvidenceMatch
from resume_agent.models.job import JobSpec
from resume_agent.models.letter import (
    MAX_LETTER_WORDS,
    ConsistencyVerdict,
    CoverLetter,
    CoverLetterFields,
    LetterVerification,
)
from resume_agent.models.profile import Profile

logger = logging.getLogger(__name__)

WRITE_PROMPT = "write_cover_letter"
VERIFY_PROMPT = "verify_letter"

# CLAUDE.md rule 7. Same budget as the bullet grounding cycle: two retries.
MAX_LETTER_ATTEMPTS = 2

# Spec 5: "top-3 EvidenceMatch".
TOP_EVIDENCE = 3


# ---------------------------------------------------------------------------
# The free checks
# ---------------------------------------------------------------------------


def verify_length(letter: CoverLetter) -> LetterVerification:
    """Spec 5's one hard number. Counted, never asked."""
    if letter.word_count <= MAX_LETTER_WORDS:
        return LetterVerification.ok()
    return LetterVerification(
        passed=False,
        check="length",
        critique=(
            f"The letter is {letter.word_count} words; the limit is {MAX_LETTER_WORDS}. "
            f"Cut roughly {letter.word_count - MAX_LETTER_WORDS} words. Tighten the "
            f"paragraph that is doing the least work rather than trimming evenly."
        ),
    )


def verify_letter_grounding(letter: CoverLetter, profile: Profile) -> LetterVerification:
    """Numbers and technologies, against the whole knowledge base.

    A letter draws on several achievements, so the allowed set is the union of
    every bullet's metrics and canonical text -- unlike a rewritten bullet,
    which is checked against its own source alone.
    """
    all_metrics: dict[str, object] = {}
    for index, bullet in enumerate(profile.all_bullets()):
        for key, value in bullet.metrics.items():
            # Namespaced so two bullets with a `latency_after_ms` key do not
            # collide and silently narrow the allowed set.
            all_metrics[f"{index}_{key}"] = value

    corpus = " ".join(b.canonical for b in profile.all_bullets())
    body = letter.body()

    bad_numbers = unsupported_numbers(body, all_metrics, corpus)  # type: ignore[arg-type]
    if bad_numbers:
        return LetterVerification(
            passed=False,
            check="grounding",
            critique=(
                f"The number(s) {sorted(bad_numbers)} appear in the letter but are not "
                f"in the candidate's recorded data. Remove them, or use a figure that "
                f"is actually recorded. Do not compute new numbers from existing ones."
            ),
        )

    # prose=True: a letter is many sentences, and its capitalised openers
    # ("Separately", "Your") appear nowhere in the knowledge base. See
    # grounding/vocabulary.py for why bullets do not need this and letters do.
    bad_tech = unsupported_technologies(body, corpus, profile.skill_vocabulary(), prose=True)
    if bad_tech:
        return LetterVerification(
            passed=False,
            check="grounding",
            critique=(
                f"The letter names {sorted(bad_tech)}, which the candidate has no "
                f"recorded experience with. Write only about technologies in their "
                f"skills list."
            ),
        )

    return LetterVerification.ok()


def verify_consistency(
    letter: CoverLetter,
    resume_bullets: list[str],
    *,
    llm: BaseChatModel | None = None,
) -> LetterVerification:
    """Does the letter contradict the resume it will be sent with?

    The judge sees the resume's *actual tailored bullets*, which is what makes
    spec 5's "must not contradict the resume's selected bullets" a check rather
    than an aspiration.
    """
    if not resume_bullets:
        # Nothing to contradict. Not a pass we can claim, but not a failure
        # either -- and inventing a verdict from an empty resume would be worse.
        return LetterVerification.ok()

    llm = llm or build_chat_model("judge")
    structured = llm.with_structured_output(ConsistencyVerdict)

    rendered = "\n".join(f"- {text}" for text in resume_bullets)
    verdict = structured.invoke(
        [
            SystemMessage(content=load_prompt(VERIFY_PROMPT)),
            HumanMessage(
                content=(
                    f"<resume_bullets>\n{rendered}\n</resume_bullets>\n\n"
                    f"<cover_letter>\n{letter.body()}\n</cover_letter>"
                )
            ),
        ]
    )
    if not isinstance(verdict, ConsistencyVerdict):
        verdict = ConsistencyVerdict.model_validate(verdict)

    if verdict.verdict == "consistent":
        return LetterVerification.ok()
    return LetterVerification(passed=False, check="consistency", critique=verdict.reason)


def verify_cover_letter(
    letter: CoverLetter,
    profile: Profile,
    resume_bullets: list[str],
    *,
    use_judge: bool = True,
    llm: BaseChatModel | None = None,
) -> LetterVerification:
    """Run the checks cheapest-first and return the first failure."""
    for check in (
        lambda: verify_length(letter),
        lambda: verify_letter_grounding(letter, profile),
    ):
        result = check()
        if not result.passed:
            return result

    if not use_judge:
        return LetterVerification.ok()
    return verify_consistency(letter, resume_bullets, llm=llm)


# ---------------------------------------------------------------------------
# The two nodes
# ---------------------------------------------------------------------------


def _render_evidence(profile: Profile, evidence: list[EvidenceMatch]) -> str:
    """The top-3 achievements, deduplicated, with their metrics spelled out."""
    seen: list[str] = []
    for match in sorted(evidence, key=lambda m: -m.relevance):
        if match.bullet_id not in seen:
            seen.append(match.bullet_id)
        if len(seen) >= TOP_EVIDENCE:
            break

    lines = []
    for bullet_id in seen:
        bullet = profile.bullet_by_id(bullet_id)
        line = f"- id: {bullet_id}\n  achievement: {bullet.canonical.strip()}"
        if bullet.metrics:
            rendered = ", ".join(f"{k}={v}" for k, v in bullet.metrics.items())
            line += f"\n  allowed numbers: {rendered}"
        lines.append(line)
    return "\n".join(lines)


def draft_cover_letter(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """One drafting pass. The retry is an edge, not a loop in here."""
    profile = load_profile(state["profile_path"])
    job: JobSpec = state["job_spec"]

    llm = llm or build_chat_model()
    structured = llm.with_structured_output(CoverLetterFields)

    narratives = "\n\n".join(f"## {n.name}\n{n.content.strip()}" for n in profile.narratives)
    critiques = state.get("letter_critiques", [])

    sections = [
        f"<posting>\nrole: {job.title} at {job.company}\n"
        f"seniority: {job.seniority}\ndomain: {job.domain}\ntone: {job.tone}\n"
        f"what they say they want: {'; '.join(r.text for r in job.requirements[:8])}\n"
        f"how they describe themselves: {'; '.join(job.culture_signals) or 'not stated'}\n"
        f"</posting>",
        f"<achievements>\n{_render_evidence(profile, state.get('evidence', []))}\n</achievements>",
        f"<narratives>\n{narratives}\n</narratives>",
        f"<limit>{MAX_LETTER_WORDS} words maximum, across all five paragraphs.</limit>",
    ]
    if critiques:
        joined = "\n".join(f"- {c}" for c in critiques)
        sections.append(f"<previous_attempt_rejected>\n{joined}\n</previous_attempt_rejected>")

    result = structured.invoke(
        [
            SystemMessage(content=load_prompt(WRITE_PROMPT)),
            HumanMessage(content="\n\n".join(sections)),
        ]
    )
    if not isinstance(result, CoverLetterFields):
        result = CoverLetterFields.model_validate(result)

    # Drop invented bullet ids for the same reason every other node does: a
    # claimed source that does not exist is a claim attached to nothing.
    known = {b.id for b in profile.all_bullets()}
    result.bullet_ids_used = [bid for bid in result.bullet_ids_used if bid in known]

    letter = CoverLetter.from_fields(result)
    logger.info("cover letter drafted: %d words", letter.word_count)
    return {"cover_letter": letter, "letter_attempts": state.get("letter_attempts", 0) + 1}


def verify_cover_letter_node(state: AgentState, llm: BaseChatModel | None = None) -> dict:
    """Check the draft; on failure record the critique for the next attempt."""
    letter = state.get("cover_letter")
    if letter is None:
        return {}

    profile = load_profile(state["profile_path"])
    resume_bullets = [t.text for t in state.get("tailored", [])]

    result = verify_cover_letter(
        letter,
        profile,
        resume_bullets,
        use_judge=state["options"].use_judge,
        llm=llm,
    )
    if result.passed:
        return {"letter_verified": True}

    logger.info("cover letter rejected at %s: %s", result.check, result.critique)
    return {
        "letter_verified": False,
        "letter_critiques": [result.critique or "rejected"],
    }


def route_after_letter_verify(state: AgentState) -> str:
    """Retry until verified or out of attempts.

    At the cap there is no letter. Unlike a bullet -- where dropping one leaves
    a slightly weaker resume -- a letter is a single artifact, so the cap
    behaviour is "no letter, loudly". The resume still ships. Shipping an
    unverified letter would defeat the verifier that M4 exists to provide.
    """
    if state.get("letter_verified"):
        return "done"
    if state.get("letter_attempts", 0) >= MAX_LETTER_ATTEMPTS + 1:
        return "give_up"
    return "draft"


def abandon_letter(state: AgentState) -> dict:
    """Cap reached. Record it loudly and continue without a cover letter."""
    critiques = state.get("letter_critiques", [])
    logger.error(
        "ABANDONING the cover letter after %d attempts; the resume will be sent without "
        "one. Critiques: %s",
        state.get("letter_attempts", 0),
        " || ".join(critiques),
    )
    return {
        "cover_letter": None,
        "errors": [
            f"cover letter abandoned after {state.get('letter_attempts', 0)} attempts: "
            f"{critiques[-1] if critiques else 'unknown'}"
        ],
    }


def build_cover_letter_subgraph(
    *,
    draft_llm: BaseChatModel | None = None,
    judge_llm: BaseChatModel | None = None,
):
    """Compile the subgraph. Added to the parent as a single node."""
    builder = StateGraph(AgentState)

    builder.add_node("draft", lambda state: draft_cover_letter(state, llm=draft_llm))
    builder.add_node("verify", lambda state: verify_cover_letter_node(state, llm=judge_llm))
    builder.add_node("give_up", abandon_letter)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_letter_verify,
        {"draft": "draft", "give_up": "give_up", "done": END},
    )
    builder.add_edge("give_up", END)

    return builder.compile()
