"""The profile schema and its cross-file integrity checks.

Most of these assert that bad data is *rejected*. The knowledge base is the
foundation of everything downstream (spec 3), and a schema that silently accepts
a typo'd skill or a duplicated bullet id turns into a confusing retrieval miss in
M1 or a mis-attributed bullet in `run.json` at M5.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from resume_agent.kb.loader import ProfileLoadError, load_profile
from resume_agent.models.profile import Bullet, Profile

from .conftest import PROFILE_EXAMPLE

# --- the example profile ---------------------------------------------------


def test_example_profile_loads(example_profile: Profile) -> None:
    assert example_profile.identity.name == "John Doe"
    assert len(example_profile.experience) == 2
    assert len(example_profile.projects) == 3
    assert len(example_profile.all_bullets()) == 13


def test_example_profile_carries_the_escape_hazards(example_profile: Profile) -> None:
    """The example data must keep exercising the escaper end to end.

    If someone later "tidies" these strings out of profile.example, the
    end-to-end PDF test silently stops testing anything interesting.
    """
    orgs = {e.org for e in example_profile.experience}
    assert "Halvorsen & Bright R&D" in orgs

    canonicals = " ".join(b.canonical for b in example_profile.all_bullets())
    for hazard in ["AT&T", "100%", "user_id", "^18", "~50ms", "<100ms", ">2M", "| jq", "{z}"]:
        assert hazard in canonicals, f"profile.example no longer contains {hazard!r}"

    assert "C#" in {s.canonical for s in example_profile.skills}


def test_current_role_has_no_end_date(example_profile: Profile) -> None:
    current = [e for e in example_profile.experience if e.end is None]
    assert len(current) == 1
    assert current[0].org == "Halvorsen & Bright R&D"


# --- derived views ---------------------------------------------------------


def test_skill_vocabulary_includes_aliases_lowercased(example_profile: Profile) -> None:
    vocab = example_profile.skill_vocabulary()
    assert "postgresql" in vocab  # canonical
    assert "postgres" in vocab  # alias
    assert "k8s" in vocab
    assert "c#" in vocab
    assert "PostgreSQL" not in vocab  # everything is lowercased


def test_bullet_by_id(example_profile: Profile) -> None:
    bullet = example_profile.bullet_by_id("exp_halvorsen_bright.b1")
    assert "820ms" in bullet.canonical
    assert bullet.metrics["latency_before_ms"] == 820
    with pytest.raises(KeyError):
        example_profile.bullet_by_id("nope.b1")


def test_metrics_accept_int_float_and_str() -> None:
    """Plan D2: real metrics are not all integers."""
    bullet = Bullet(
        id="x.b1",
        canonical="text",
        metrics={"count": 12, "ratio": 0.94, "regions": "3 regions"},
    )
    assert bullet.metrics == {"count": 12, "ratio": 0.94, "regions": "3 regions"}


# --- rejection cases -------------------------------------------------------


def _example_payload() -> dict:
    """A valid Profile payload, as nested dicts, ready to be corrupted."""
    return copy.deepcopy(load_profile(PROFILE_EXAMPLE).model_dump())


def test_unknown_skill_is_rejected() -> None:
    """Plan D5: a technology not in skills.yaml is the M4 fabrication signal."""
    payload = _example_payload()
    payload["experience"][0]["bullets"][0]["skills"].append("kafkaesque")
    with pytest.raises(ValidationError, match="not in skills.yaml"):
        Profile.model_validate(payload)


def test_unknown_tech_is_rejected() -> None:
    payload = _example_payload()
    payload["experience"][0]["tech"].append("cobol")
    with pytest.raises(ValidationError, match="not in skills.yaml"):
        Profile.model_validate(payload)


def test_bullet_id_must_be_namespaced_to_its_entry() -> None:
    payload = _example_payload()
    payload["experience"][0]["bullets"][0]["id"] = "some_other_entry.b1"
    with pytest.raises(ValidationError, match="must start with"):
        Profile.model_validate(payload)


def test_duplicate_bullet_ids_are_rejected() -> None:
    payload = _example_payload()
    payload["experience"][0]["bullets"][1]["id"] = payload["experience"][0]["bullets"][0]["id"]
    with pytest.raises(ValidationError, match="duplicate bullet ids"):
        Profile.model_validate(payload)


def test_duplicate_entry_ids_are_rejected() -> None:
    payload = _example_payload()
    duplicate = copy.deepcopy(payload["experience"][0])
    payload["experience"].append(duplicate)
    with pytest.raises(ValidationError, match="duplicate entry ids"):
        Profile.model_validate(payload)


def test_unknown_yaml_key_is_rejected() -> None:
    """extra="forbid": `cannonical:` must be an error, not a dropped field."""
    payload = _example_payload()
    payload["experience"][0]["bullets"][0]["cannonical"] = "typo"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Profile.model_validate(payload)


@pytest.mark.parametrize("bad_date", ["2023", "2023-13", "2023-1", "june 2023", ""])
def test_malformed_year_month_is_rejected(bad_date: str) -> None:
    payload = _example_payload()
    payload["experience"][0]["start"] = bad_date
    with pytest.raises(ValidationError):
        Profile.model_validate(payload)


def test_invalid_confidence_value_is_rejected() -> None:
    payload = _example_payload()
    payload["experience"][0]["bullets"][0]["confidence"] = "pretty sure"
    with pytest.raises(ValidationError):
        Profile.model_validate(payload)


# --- the loader ------------------------------------------------------------


def test_missing_directory_is_a_profile_load_error(tmp_path: Path) -> None:
    with pytest.raises(ProfileLoadError, match="profile directory not found"):
        load_profile(tmp_path / "does_not_exist")


def test_missing_required_file_is_a_profile_load_error(tmp_path: Path) -> None:
    (tmp_path / "skills.yaml").write_text("skills: []", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="identity.yaml"):
        load_profile(tmp_path)


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    (tmp_path / "identity.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    (tmp_path / "skills.yaml").write_text("skills: []", encoding="utf-8")
    with pytest.raises(ProfileLoadError, match="identity.yaml"):
        load_profile(tmp_path)


def test_entries_are_loaded_in_filename_order(example_profile: Profile) -> None:
    """Deterministic load order -- the golden .tex snapshot compares bytes."""
    assert [e.id for e in example_profile.experience] == [
        "exp_halvorsen_bright",
        "exp_northwind_data",
    ]
    assert [p.id for p in example_profile.projects] == [
        "prj_ledgerlint",
        "prj_tilecache",
        "prj_vitals_dashboard",
    ]


# ===========================================================================
# Leadership and publications are entries, not a second-class shape
#
# Their whole justification is that carrying bullets makes them entries, and
# being entries earns them every check below for free. These tests exist to
# prove that claim rather than assume it -- a type left out of
# `Profile.entries()` loads fine and silently skips all of them.
# ===========================================================================


STUDENT_PROFILE = PROFILE_EXAMPLE.parent / "tests" / "fixtures" / "profile.student"


def test_the_student_fixture_has_all_four_entry_types() -> None:
    profile = load_profile(STUDENT_PROFILE)

    kinds = {entry.type for entry in profile.entries()}
    assert kinds == {"experience", "project", "leadership", "publication"}


def test_a_leadership_bullet_id_must_be_namespaced(tmp_path: Path) -> None:
    directory = _copy_student(tmp_path)
    path = directory / "leadership" / "student_radio.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("ldr_student_radio.b1", "elsewhere.b1"),
        encoding="utf-8",
    )

    with pytest.raises(ProfileLoadError, match="must start with"):
        load_profile(directory)


def test_a_publication_may_not_name_an_unknown_skill(tmp_path: Path) -> None:
    directory = _copy_student(tmp_path)
    path = directory / "publications" / "caption_study.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("skills: [Python]", "skills: [Kubernetes]"),
        encoding="utf-8",
    )

    with pytest.raises(ProfileLoadError, match="not in skills.yaml"):
        load_profile(directory)


def test_entry_ids_are_unique_across_every_type(tmp_path: Path) -> None:
    """One namespace, which is why `create_entry` prefixes ldr_ and pub_."""
    directory = _copy_student(tmp_path)
    path = directory / "leadership" / "student_radio.yaml"
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("id: ldr_student_radio", "id: exp_harlow_press")
        .replace("ldr_student_radio.b", "exp_harlow_press.bz"),
        encoding="utf-8",
    )

    with pytest.raises(ProfileLoadError, match="duplicate entry ids"):
        load_profile(directory)


def _copy_student(tmp_path: Path) -> Path:
    directory = tmp_path / "profile"
    shutil.copytree(STUDENT_PROFILE, directory)
    return directory
