"""The chat: the audit, the extraction gate, and applying a proposal.

Everything runs against fake models and throwaway copies of `profile.example`.
Nothing here costs money and nothing here touches a real profile.

The centre of gravity is the gate. This is the only place in the project where a
model authors ground truth rather than being measured against it, so the tests
that matter are the ones that feed it a model which embellishes and assert the
embellishment is caught.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from resume_agent.chat.audit import audit_profile, has_weak_opener
from resume_agent.chat.extract import (
    ExtractionFields,
    ProposedBullet,
    ProposedBulletForEntry,
    ProposedEntry,
    ProposedSkill,
    apply,
    build_proposal,
    extract,
)
from resume_agent.chat.session import Registry, classify
from resume_agent.kb.loader import load_profile
from resume_agent.kb.writer import read_profile_file, scaffold_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_EXAMPLE = REPO_ROOT / "profile.example"

AN_ENTRY = "exp_halvorsen_bright"


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    target = tmp_path / "profile"
    scaffold_profile(PROFILE_EXAMPLE, target)
    return target


@pytest.fixture
def profile(profile_dir: Path):
    return load_profile(profile_dir)


class ScriptedExtractor(BaseChatModel):
    """Returns a fixed `ExtractionFields`, whatever it is asked.

    Same shape as `ScriptedWriter` in test_tailor.py: override
    `with_structured_output`, because that is what `llm.structured_output`
    delegates to.
    """

    fields: ExtractionFields = ExtractionFields(reply="ok")
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-extractor"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> ExtractionFields:
                fake.call_count += 1
                return fake.fields

        return _Runnable()


# ===========================================================================
# The gate -- the reason this feature is safe to have
# ===========================================================================


SAID = (
    "I rewrote the nightly export at Halvorsen so it streams instead of "
    "buffering in memory, and it stopped paging us overnight."
)


def test_a_number_the_user_never_said_is_flagged(profile) -> None:
    """The failure this whole design exists to prevent: a figure the model
    invented becoming a permanent fact about someone's career."""
    fields = ExtractionFields(
        reply="Added one achievement.",
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(
                canonical="Rewrote the nightly export to stream, cutting memory use by 80%.",
                metrics={"memory_saved_pct": "80"},
            ),
        )],
    )

    proposal = build_proposal(fields, SAID, profile)
    flags = proposal.items[0].flags

    assert [f.kind for f in flags] == ["invented_number"]
    assert "80" in flags[0].detail


def test_a_technology_the_user_never_named_is_flagged(profile) -> None:
    """"A queue" must not become Kafka -- not even when Kafka is in the skills
    list, because that records having used it somewhere, not here."""
    fields = ExtractionFields(
        reply="Added one achievement.",
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(
                canonical="Rewrote the nightly export to stream through Kafka.",
                skills=["Kafka"],
            ),
        )],
    )

    proposal = build_proposal(fields, SAID, profile)

    assert any(f.kind == "invented_technology" for f in proposal.items[0].flags)


def test_faithful_extraction_is_not_flagged(profile) -> None:
    """The gate must not cry wolf, or it trains you to click through it.

    `themes` are deliberately not checked: they are the user's own filing
    system, not a claim about what they did, and a theme cannot end up on a
    resume as a fact.
    """
    fields = ExtractionFields(
        reply="Added one achievement.",
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(
                canonical="Rewrote the nightly export to stream instead of buffering in "
                          "memory, ending overnight pages.",
                themes=["reliability", "backend"],
            ),
        )],
    )

    proposal = build_proposal(fields, SAID, profile)

    assert proposal.items[0].flags == []
    assert proposal.flagged() == []


def test_a_skill_inferred_from_context_is_still_flagged(profile) -> None:
    """The subtle case, and the one that justifies ignoring skills.yaml here.

    Halvorsen's recorded tech list contains Python, so tagging this achievement
    `python` is a plausible inference. It is still an inference: the user did
    not say the export was Python, and `skills` feeds retrieval and the printed
    skills section. Flagged, not refused -- if it is true they accept it.
    """
    fields = ExtractionFields(bullets=[ProposedBulletForEntry(
        entry_id=AN_ENTRY,
        bullet=ProposedBullet(
            canonical="Rewrote the nightly export to stream instead of buffering.",
            skills=["python"],
        ),
    )])

    flags = build_proposal(fields, SAID, profile).items[0].flags
    assert [f.kind for f in flags] == ["invented_technology"]


def test_a_number_the_user_did_say_is_allowed(profile) -> None:
    said = "I cut the p95 from 820ms to 50ms across 12 endpoints."
    fields = ExtractionFields(
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(
                canonical="Cut p95 latency from 820ms to 50ms across 12 endpoints.",
                metrics={"before_ms": "820", "after_ms": "50", "endpoints": "12"},
            ),
        )],
    )

    assert build_proposal(fields, said, profile).items[0].flags == []


def test_a_new_skill_makes_its_own_bullet_legal(profile) -> None:
    """Saying "I used Rust" should propose the skill *and* the achievement
    without the achievement being flagged for naming a skill that does not exist
    yet -- the skill in the same proposal is what makes it resolvable."""
    said = "At Halvorsen I built a Rust service for parsing the ledger files."
    fields = ExtractionFields(
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(
                canonical="Built a Rust service for parsing ledger files.",
                skills=["Rust"],
            ),
        )],
        skills=[ProposedSkill(canonical="Rust", category="language", level="working")],
    )

    proposal = build_proposal(fields, said, profile)

    assert proposal.flagged() == []


def test_an_entry_id_that_does_not_exist_is_flagged(profile) -> None:
    fields = ExtractionFields(
        bullets=[ProposedBulletForEntry(
            entry_id="exp_somewhere_else",
            bullet=ProposedBullet(canonical="Did a thing."),
        )],
    )

    flags = build_proposal(fields, "I did a thing.", profile).items[0].flags
    assert [f.kind for f in flags] == ["unknown_entry"]


def test_a_vague_date_is_flagged_rather_than_guessed(profile) -> None:
    """`YearMonth` is YYYY-MM. Catching it here makes it a fixable flag instead
    of a refused save after you have already accepted."""
    fields = ExtractionFields(
        entries=[ProposedEntry(kind="experience", name="Kestrel", start="a couple of years ago")],
    )

    said = "I was at Kestrel a couple of years ago."
    flags = build_proposal(fields, said, profile).items[0].flags
    assert [f.kind for f in flags] == ["bad_date"]


def test_extract_calls_the_model_once_and_gates_the_result(profile) -> None:
    model = ScriptedExtractor(fields=ExtractionFields(
        reply="ok",
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(canonical="Improved throughput by 300%."),
        )],
    ))

    proposal = extract(SAID, profile, llm=model)

    assert model.call_count == 1
    assert proposal.flagged(), "the gate did not run on the model's output"


# ===========================================================================
# Applying -- goes through the form layer or not at all
# ===========================================================================


def test_nothing_is_written_until_you_accept(profile_dir: Path, profile) -> None:
    before = read_profile_file(profile_dir, "experience/halvorsen_bright.yaml")

    fields = ExtractionFields(bullets=[ProposedBulletForEntry(
        entry_id=AN_ENTRY, bullet=ProposedBullet(canonical="Did something."),
    )])
    proposal = build_proposal(fields, "I did something.", profile)
    apply(proposal, profile_dir, accepted=[])

    assert read_profile_file(profile_dir, "experience/halvorsen_bright.yaml") == before


def test_accepting_a_bullet_writes_it_through_the_form_layer(
    profile_dir: Path, profile
) -> None:
    """Which means it inherits the backup, the whole-directory validation and
    the comment preservation -- asserted here, because a second write path is
    exactly what this feature must not become."""
    fields = ExtractionFields(bullets=[ProposedBulletForEntry(
        entry_id=AN_ENTRY,
        bullet=ProposedBullet(
            canonical="Rewrote the nightly export to stream instead of buffering.",
            skills=["python"], themes=["reliability"],
        ),
    )])
    proposal = build_proposal(fields, SAID, profile)

    apply(proposal, profile_dir, accepted=[proposal.items[0].item_id])

    after = read_profile_file(profile_dir, "experience/halvorsen_bright.yaml")
    assert "Rewrote the nightly export" in after
    assert "# Employer name carries both an ampersand" in after, "comments were lost"
    assert "canonical: >-" in after

    reloaded = load_profile(profile_dir)
    assert len(reloaded.all_bullets()) == 14
    assert reloaded.bullet_by_id("exp_halvorsen_bright.b5")


def test_accepting_a_new_role_creates_it(profile_dir: Path, profile) -> None:
    fields = ExtractionFields(entries=[ProposedEntry(
        kind="experience", name="Kestrel Health", title="Backend Engineer",
        location="Remote", start="2021-03",
        bullets=[ProposedBullet(canonical="Built the ingest pipeline.")],
    )])
    proposal = build_proposal(fields, "I was at Kestrel Health from 2021-03.", profile)

    apply(proposal, profile_dir, accepted=[proposal.items[0].item_id])

    reloaded = load_profile(profile_dir)
    entry = next(e for e in reloaded.experience if e.org == "Kestrel Health")
    assert entry.title == "Backend Engineer"
    assert len(entry.bullets) == 1
    assert entry.bullets[0].id.startswith(entry.id + ".")


def test_a_skill_is_written_before_the_bullet_that_needs_it(
    profile_dir: Path, profile
) -> None:
    """There is no transactional multi-file write: each save validates the whole
    directory on its own. A bullet naming a new technology that landed first
    would leave a state that does not load, and the save would be refused."""
    said = "At Halvorsen I built a Rust parser for the ledger files."
    fields = ExtractionFields(
        bullets=[ProposedBulletForEntry(
            entry_id=AN_ENTRY,
            bullet=ProposedBullet(canonical="Built a Rust parser for ledger files.",
                                  skills=["Rust"]),
        )],
        skills=[ProposedSkill(canonical="Rust", category="language", level="working")],
    )
    proposal = build_proposal(fields, said, profile)

    apply(proposal, profile_dir, accepted=[i.item_id for i in proposal.items])

    reloaded = load_profile(profile_dir)
    assert "rust" in reloaded.skill_vocabulary()
    assert any("Rust" in b.canonical for b in reloaded.all_bullets())


def test_accepting_only_the_unflagged_items_is_possible(
    profile_dir: Path, profile
) -> None:
    """The point of per-item ids: you keep the good half of a proposal."""
    fields = ExtractionFields(bullets=[
        ProposedBulletForEntry(entry_id=AN_ENTRY, bullet=ProposedBullet(
            canonical="Rewrote the nightly export to stream instead of buffering.")),
        ProposedBulletForEntry(entry_id=AN_ENTRY, bullet=ProposedBullet(
            canonical="Cut memory use by 80%.")),
    ])
    proposal = build_proposal(fields, SAID, profile)
    clean = [item.item_id for item in proposal.items if not item.flags]

    assert len(clean) == 1
    apply(proposal, profile_dir, accepted=clean)

    after = read_profile_file(profile_dir, "experience/halvorsen_bright.yaml")
    assert "Rewrote the nightly export" in after
    assert "80%" not in after


# ===========================================================================
# The audit
# ===========================================================================


def test_the_audit_finds_the_theme_that_cannot_all_be_selected(profile) -> None:
    """The example profile has nine `backend` achievements against a cap of
    three -- which is why an `analyze` run on it selects seven bullets for a
    thirty-two line budget and rejects six for the theme cap."""
    audit = audit_profile(profile)

    assert audit.theme_counts["backend"] == 9
    assert "backend" in audit.crowded_themes
    assert any(f.kind == "crowded_themes" for f in audit.findings())


def test_the_audit_finds_achievements_with_no_numbers(profile_dir: Path) -> None:
    """The highest-leverage finding, and one nothing else surfaces: a bullet
    with no metrics can never carry a figure onto a page."""
    from resume_agent.kb.forms import read_document, write_document

    document = read_document(profile_dir, "experience/halvorsen_bright.yaml")
    document["data"]["bullets"][0]["metrics"] = {}
    write_document(profile_dir, "experience/halvorsen_bright.yaml", document["data"])

    audit = audit_profile(load_profile(profile_dir))

    assert "exp_halvorsen_bright.b1" in audit.bullets_without_metrics
    assert audit.findings()[0].kind == "no_metrics", "should be reported first"


def test_the_audit_reports_declared_but_unevidenced_skills(profile) -> None:
    audit = audit_profile(profile)
    assert "Go" in audit.unused_skills


def test_findings_are_ordered_by_what_they_cost(profile) -> None:
    audit = audit_profile(profile)
    kinds = [f.kind for f in audit.findings()]
    assert kinds == sorted(kinds, key=lambda k: [
        "no_metrics", "crowded_themes", "thin_entry",
        "weak_opener", "unverified", "unused_skill",
    ].index(k))


@pytest.mark.parametrize(
    "sentence,weak",
    [
        ("Responsible for the billing service", True),
        ("Worked on the ingest pipeline", True),
        ("Building the ingest pipeline", True),
        ("Utilized Kafka for the queue", True),
        ("Cut p95 latency from 820ms to 50ms", False),
        ("Migrated 9 services to Kubernetes", False),
        ("Engineering the release process", False),
    ],
)
def test_weak_openers(sentence: str, weak: bool) -> None:
    assert has_weak_opener(sentence) is weak


# ===========================================================================
# Routing and the transcript
# ===========================================================================


@pytest.mark.parametrize(
    "message,intent",
    [
        ("What should I improve?", "advise"),
        ("how does the theme cap work", "advise"),
        ("Review my profile", "advise"),
        ("Which bullets have no numbers?", "advise"),
        ("I rewrote the nightly export so it streams.", "extract"),
        ("we shipped a new billing service last spring", "extract"),
        ("hello", "advise"),
        ("", "advise"),
    ],
)
def test_routing(message: str, intent: str) -> None:
    """Biased towards advice on purpose: treating dictation as a question wastes
    a turn, but treating a question as dictation puts a proposal in front of
    someone who never asked for one."""
    assert classify(message) == intent


def test_the_transcript_only_carries_settled_turns() -> None:
    registry = Registry()
    first = registry.start("profile", "What should I improve?")
    first.reply = "Two things."
    first.status = "done"
    registry.start("profile", "and the second?")   # still running

    assert registry.conversation("profile").transcript() == [
        ("user", "What should I improve?"),
        ("assistant", "Two things."),
    ]


def test_turns_are_addressable_by_id_and_bounded() -> None:
    registry = Registry()
    turn = registry.start("profile", "hello")

    assert registry.get(turn.turn_id) is turn
    assert registry.get("nope") is None
