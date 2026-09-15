"""Structured (form) editing of a profile.

The whole point of `forms.py` is that a user edits fields and the app owns the
file format. The risk that buys is a serializer: it now rewrites files the user
never asked it to touch, in a gitignored directory with no undo. Most of what is
here is about that.

Every test runs on a throwaway copy of `profile.example`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_agent.kb.forms import (
    FORMS,
    HIDDEN_KEYS,
    create_entry,
    delete_document,
    document_role,
    next_bullet_id,
    read_document,
    skill_usage,
    write_document,
)
from resume_agent.kb.loader import load_profile
from resume_agent.kb.writer import (
    ProfileWriteError,
    read_profile_file,
    relative_profile_files,
    scaffold_profile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_EXAMPLE = REPO_ROOT / "profile.example"

AN_ENTRY = "experience/halvorsen_bright.yaml"


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    target = tmp_path / "profile"
    scaffold_profile(PROFILE_EXAMPLE, target)
    return target


def editable(profile: Path) -> list[str]:
    """Everything a form can open. README.md is documentation, not content."""
    return [f for f in relative_profile_files(profile) if not f.endswith("README.md")]


# ===========================================================================
# The property the whole design rests on
# ===========================================================================


def test_opening_and_saving_without_editing_changes_nothing(profile: Path) -> None:
    """Byte-identical, every file.

    This is the test `test_a_save_preserves_comments_and_formatting` in
    test_profile_writer.py cannot do: that one exercises the writer, which still
    moves text verbatim, so it stays green no matter what a serializer above it
    destroys. This one goes through the serializer.
    """
    for relative in editable(profile):
        before = read_profile_file(profile, relative)
        write_document(profile, relative, read_document(profile, relative)["data"])
        assert read_profile_file(profile, relative) == before, relative


def test_an_edit_preserves_everything_it_did_not_touch(profile: Path) -> None:
    """A form edits one field; the file should differ in one field.

    `yaml.safe_dump` would have stripped the header comment, unfolded the `>-`
    blocks, expanded `tech: [...]` one-per-line and re-sorted every key -- on a
    file whose comments are the user's own notes.
    """
    document = read_document(profile, AN_ENTRY)
    document["data"]["title"] = "Staff Backend Engineer"
    write_document(profile, AN_ENTRY, document["data"])

    after = read_profile_file(profile, AN_ENTRY)
    assert "title: Staff Backend Engineer" in after
    assert "# Employer name carries both an ampersand" in after
    assert "canonical: >-" in after
    assert "tech: [python, fastapi," in after
    assert "end: null # still there" in after
    assert after.index("org:") < after.index("title:") < after.index("start:")


def test_a_comment_stays_with_its_bullet_when_one_is_inserted_above(profile: Path) -> None:
    """Rows are matched by id, not position -- otherwise inserting a bullet
    would slide every comment onto the wrong achievement."""
    document = read_document(profile, AN_ENTRY)
    bullets = document["data"]["bullets"]
    entry_id = document["data"]["id"]

    bullets.insert(0, {
        "id": next_bullet_id(entry_id, [b["id"] for b in bullets]),
        "canonical": "Did a brand new thing worth mentioning.",
        "metrics": {}, "skills": [], "themes": [], "confidence": "verified",
    })
    write_document(profile, AN_ENTRY, document["data"])

    after = read_profile_file(profile, AN_ENTRY)
    kubernetes_note = "# Kubernetes evidence exists so that"
    assert kubernetes_note in after
    assert after.index(kubernetes_note) < after.index("id: exp_halvorsen_bright.b4")


def test_changed_text_that_was_folded_stays_folded(profile: Path) -> None:
    """A rewritten `canonical` must not become one enormous quoted line."""
    document = read_document(profile, AN_ENTRY)
    document["data"]["bullets"][0]["canonical"] = "A completely different sentence " * 6
    write_document(profile, AN_ENTRY, document["data"])

    assert read_profile_file(profile, AN_ENTRY).count("canonical: >-") == 4


def test_changed_chips_stay_flow_style(profile: Path) -> None:
    """`skills: [redis, caching]`, not nine lines of `- redis`."""
    document = read_document(profile, AN_ENTRY)
    document["data"]["bullets"][0]["skills"] = ["redis", "python"]
    write_document(profile, AN_ENTRY, document["data"])

    assert "skills: [redis, python]" in read_profile_file(profile, AN_ENTRY)


def test_metrics_keep_their_types(profile: Path) -> None:
    """The browser sends every value as text. If `820` came back quoted, the
    grounding gate would be comparing a string against a number."""
    document = read_document(profile, AN_ENTRY)
    assert document["data"]["bullets"][0]["metrics"]["latency_before_ms"] == 820

    document["data"]["bullets"][0]["metrics"] = {
        "latency_before_ms": "820", "ratio": "0.5", "note": "three regions",
    }
    write_document(profile, AN_ENTRY, document["data"])

    metrics = load_profile(profile).bullet_by_id("exp_halvorsen_bright.b1").metrics
    assert metrics["latency_before_ms"] == 820
    assert metrics["ratio"] == 0.5
    assert metrics["note"] == "three regions"


def test_a_narrative_is_a_title_and_prose(profile: Path) -> None:
    """No Markdown editor: these files are one `# Title` and paragraphs, and the
    cover-letter node builds its own heading from the filename anyway."""
    relative = "narratives/how_i_learn.md"
    document = read_document(profile, relative)

    assert document["kind"] == "prose"
    assert document["data"]["title"] == "How I learn"
    assert "#" not in document["data"]["body"]
    assert "\n\n" in document["data"]["body"], "paragraph breaks were lost"

    write_document(profile, relative, document["data"])
    assert read_profile_file(profile, relative) == read_profile_file(profile, relative)
    assert load_profile(profile).narrative("how_i_learn") is not None


# ===========================================================================
# What the form will not let you do
# ===========================================================================


def test_ids_are_never_offered_as_fields(profile: Path) -> None:
    """Renaming an entry id has to cascade to every bullet id in the same save,
    and orphans the ids already written into the tracker. Ids are plumbing: the
    form carries them and generates them, but never offers them for typing."""
    for role, (_kind, _root, fields) in FORMS.items():
        names = {f.name for f in fields} | {
            sub.name for f in fields for sub in f.fields
        }
        assert not names & set(HIDDEN_KEYS), f"{role} exposes an id or type field"


def test_a_generated_bullet_id_is_namespaced_and_free(profile: Path) -> None:
    existing = ["e.b1", "e.b2", "e.b4"]
    assert next_bullet_id("e", existing) == "e.b3"
    assert next_bullet_id("e", []) == "e.b1"


def test_deleting_a_used_skill_is_refused_and_changes_nothing(profile: Path) -> None:
    """The cross-file failure a per-file form cannot see. The save is refused by
    the writer's staging validation, so the file is untouched."""
    document = read_document(profile, "skills.yaml")
    before = read_profile_file(profile, "skills.yaml")

    document["data"]["skills"] = [
        row for row in document["data"]["skills"] if row["canonical"] != "Kubernetes"
    ]

    with pytest.raises(ProfileWriteError, match="not in skills.yaml"):
        write_document(profile, "skills.yaml", document["data"])

    assert read_profile_file(profile, "skills.yaml") == before


def test_skill_usage_counts_aliases_against_the_canonical_row(profile: Path) -> None:
    """What the UI warns with before removing a skill. Aliases roll up, because
    the row being deleted takes its aliases with it."""
    usage = skill_usage(profile)

    assert usage["Kubernetes"] > 0
    assert set(usage) == {skill.canonical for skill in load_profile(profile).skills}


def test_a_file_no_form_knows_is_refused(profile: Path) -> None:
    with pytest.raises(ProfileWriteError, match="no form knows"):
        document_role("out/resume.pdf")


@pytest.mark.parametrize(
    "relative,expected",
    [
        ("identity.yaml", "identity"),
        ("skills.yaml", "skills"),
        ("education.yaml", "education"),
        ("certifications.yaml", "certifications"),
        ("experience/halvorsen_bright.yaml", "experience"),
        ("projects/ledgerlint.yaml", "project"),
        ("narratives/values.md", "narrative"),
    ],
)
def test_every_file_in_a_profile_has_a_form(relative: str, expected: str) -> None:
    assert document_role(relative) == expected


def test_the_spec_covers_every_file_a_profile_contains(profile: Path) -> None:
    """A file the editor lists but no form can open would be a dead end."""
    for relative in editable(profile):
        assert document_role(relative) in FORMS


# ===========================================================================
# Creating and removing entries
# ===========================================================================


def test_creating_a_job_produces_a_profile_that_still_loads(profile: Path) -> None:
    relative = create_entry(profile, "experience", "Northgate Robotics")

    assert relative == "experience/northgate_robotics.yaml"
    entry = next(e for e in load_profile(profile).experience if e.org == "Northgate Robotics")
    assert entry.id == "exp_northgate_robotics"
    assert entry.bullets == []


def test_a_new_entry_never_reuses_an_id(profile: Path) -> None:
    """Ids are unique across experience *and* projects -- they share a namespace."""
    first = create_entry(profile, "experience", "Repeat Co")
    second = create_entry(profile, "project", "Repeat Co")

    ids = {entry.id for entry in load_profile(profile).entries()}
    assert "exp_repeat_co" in ids and "prj_repeat_co" in ids
    assert first != second


def test_a_created_entry_can_be_opened_as_a_form(profile: Path) -> None:
    relative = create_entry(profile, "project", "Tiny Tool")
    document = read_document(profile, relative)

    assert document["role"] == "project"
    assert document["data"]["name"] == "Tiny Tool"


def test_deleting_an_entry_keeps_a_backup(profile: Path) -> None:
    backup = delete_document(profile, "projects/tilecache.yaml")

    assert backup.is_file()
    assert "prj_tilecache" in backup.read_text(encoding="utf-8")
    assert not (profile / "projects/tilecache.yaml").exists()
    load_profile(profile)


def test_a_deletion_that_would_break_the_profile_is_undone(profile: Path) -> None:
    """identity.yaml is required; removing it must not leave a directory that
    cannot load."""
    with pytest.raises(ProfileWriteError):
        delete_document(profile, "identity.yaml")

    assert (profile / "identity.yaml").is_file()
    load_profile(profile)


# ===========================================================================
# Suggestions -- what makes the skills rule unfailable
# ===========================================================================


def test_skills_are_offered_as_a_closed_list(profile: Path) -> None:
    """The strongest argument for forms: a combobox backed by the live
    vocabulary means `_skills_resolve_to_vocabulary` cannot be violated by
    picking from it."""
    document = read_document(profile, AN_ENTRY)

    skills_field = next(
        f for f in document["spec"] if f["name"] == "bullets"
    )["fields"]
    technologies = next(f for f in skills_field if f["name"] == "skills")

    assert technologies["closed"] is True and technologies["suggest"] == "skills"
    assert "Kubernetes" in document["suggestions"]["skills"]


def test_a_broken_profile_still_yields_a_usable_form(profile: Path) -> None:
    """Suggestions come from loading the profile, which is exactly what is
    failing when you most need the editor."""
    (profile / "skills.yaml").write_text("skills: [\n", encoding="utf-8")

    document = read_document(profile, AN_ENTRY)
    assert document["spec"]
    assert document["suggestions"]["skills"] == []
