"""The profile writer. The only code in this project that writes career data.

Every test works on a throwaway copy of `profile.example`. Nothing here may
touch a real `profile/` -- and the autouse fixture in `conftest.py` additionally
pins the backup directory into `tmp_path`, because a backup is itself a write.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from resume_agent.kb.loader import load_profile
from resume_agent.kb.writer import (
    ProfileWriteError,
    default_backup_dir,
    read_profile_file,
    relative_profile_files,
    resolve_editable_path,
    scaffold_profile,
    write_profile_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_EXAMPLE = REPO_ROOT / "profile.example"

AN_ENTRY = "experience/halvorsen_bright.yaml"


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    """A disposable copy of the example profile."""
    target = tmp_path / "profile"
    scaffold_profile(PROFILE_EXAMPLE, target)
    return target


# ===========================================================================
# Containment -- the security boundary of the whole feature
# ===========================================================================


@pytest.mark.parametrize(
    "bad",
    [
        "../secrets.yaml",
        "../../etc/passwd",
        "experience/../../escape.yaml",
        "/etc/passwd",
        "",
        "   ",
        " identity.yaml",
    ],
)
def test_paths_outside_the_profile_are_refused(profile: Path, bad: str) -> None:
    """The editor sends this path from the browser. Without the check, a request
    for `../../.ssh/id_rsa` would be honoured -- and on a PUT, written to."""
    with pytest.raises(ProfileWriteError):
        resolve_editable_path(profile, bad)


def test_yml_is_refused_with_the_reason(profile: Path) -> None:
    """`loader.py` globs `*.yaml`. Saving `.yml` would write a file that looks
    right in the editor, parses fine, and is read by nothing."""
    with pytest.raises(ProfileWriteError, match="never .yml"):
        resolve_editable_path(profile, "experience/new_job.yml")


def test_non_profile_file_types_are_refused(profile: Path) -> None:
    with pytest.raises(ProfileWriteError, match="editable"):
        resolve_editable_path(profile, "out/resume.pdf")


def test_the_file_list_is_in_load_order(profile: Path) -> None:
    """Root files, then experience, projects, narratives -- the order the
    profile is assembled in, not alphabetical across the whole tree."""
    files = relative_profile_files(profile)

    assert files[0].endswith(".yaml") and "/" not in files[0]
    assert "identity.yaml" in files
    assert AN_ENTRY in files
    assert [f for f in files if f.startswith("experience/")] == sorted(
        f for f in files if f.startswith("experience/")
    )
    assert files.index("identity.yaml") < files.index(AN_ENTRY)


# ===========================================================================
# The reason this writes text and not objects
# ===========================================================================


def test_a_save_preserves_comments_and_formatting(profile: Path) -> None:
    """The whole design rests on this.

    `yaml.safe_dump` would strip the header comment, reflow the `>-` folded
    scalars into quoted lines, expand `tech: [python, ...]` one-per-line and
    sort the keys. Those comments are the user's own notes on their own career
    data, and `profile/` is gitignored -- there is no undo. Round-tripping text
    is what makes a save non-destructive.
    """
    before = read_profile_file(profile, AN_ENTRY)
    assert "# Employer name carries both an ampersand" in before
    assert "canonical: >-" in before
    assert "tech: [python," in before

    edited = before.replace("Senior Backend Engineer", "Staff Backend Engineer")
    write_profile_file(profile, AN_ENTRY, edited)

    after = read_profile_file(profile, AN_ENTRY)
    assert after == edited
    assert "# Employer name carries both an ampersand" in after
    assert "canonical: >-" in after
    assert "tech: [python," in after
    assert "Staff Backend Engineer" in after


def test_writes_are_lf_even_when_the_browser_sends_crlf(profile: Path) -> None:
    """`profile_content_hash` hashes raw bytes, so CRLF would invalidate the
    search index on a save that changed nothing, and fight .gitattributes."""
    original = read_profile_file(profile, AN_ENTRY)
    write_profile_file(profile, AN_ENTRY, original.replace("\n", "\r\n"))

    raw = (profile / AN_ENTRY).read_bytes()
    assert b"\r\n" not in raw
    assert read_profile_file(profile, AN_ENTRY) == original


# ===========================================================================
# Validation is whole-directory, not per-file
# ===========================================================================


def test_a_break_in_another_file_refuses_the_save(profile: Path) -> None:
    """The cross-file case that a per-file check cannot see.

    Removing Kubernetes from `skills.yaml` invalidates a bullet in an experience
    file the user never opened, because `Profile._skills_resolve_to_vocabulary`
    runs across the whole directory.
    """
    skills = read_profile_file(profile, "skills.yaml")
    assert "canonical: Kubernetes" in skills
    without = "\n".join(
        line for line in skills.splitlines() if "Kubernetes" not in line and "k8s" not in line
    )

    with pytest.raises(ProfileWriteError, match="not in skills.yaml"):
        write_profile_file(profile, "skills.yaml", without)

    assert read_profile_file(profile, "skills.yaml") == skills, "the refused save still wrote"


def test_a_refused_save_leaves_the_file_byte_identical(profile: Path) -> None:
    """The guarantee that makes the feature safe to use on real data."""
    before = (profile / AN_ENTRY).read_bytes()

    with pytest.raises(ProfileWriteError):
        write_profile_file(profile, AN_ENTRY, "id: [this is not valid YAML\n")

    assert (profile / AN_ENTRY).read_bytes() == before


def test_a_refused_save_leaves_the_profile_loadable(profile: Path) -> None:
    with pytest.raises(ProfileWriteError):
        write_profile_file(profile, "identity.yaml", "name: only-a-name\n")

    load_profile(profile)  # raises if the directory was left broken


def test_the_error_names_the_file_and_the_problem(profile: Path) -> None:
    """These messages go straight to the editor, so they have to be readable."""
    with pytest.raises(ProfileWriteError) as caught:
        write_profile_file(profile, AN_ENTRY, "id: [broken\n")

    message = str(caught.value)
    assert "halvorsen_bright.yaml" in message
    assert "not valid YAML" in message


def test_the_validation_error_is_fit_to_show_someone_mid_edit(profile: Path) -> None:
    """This message goes straight into the editor, so it is the one the user
    reads most. Raw, it names an internal staging directory (making a correct
    refusal look like a tool bug) and trails a truncated dump of the entire
    profile plus a docs URL, which buries the one useful line."""
    skills = read_profile_file(profile, "skills.yaml")
    entry = read_profile_file(profile, AN_ENTRY)

    with pytest.raises(ProfileWriteError) as caught:
        write_profile_file(
            profile, AN_ENTRY, entry.replace("skills: [redis", "skills: [not-a-real-skill, redis")
        )

    message = str(caught.value)
    assert "not-a-real-skill" in message
    assert "skills.yaml" in message
    assert "staging" not in message
    assert "input_type" not in message
    assert "pydantic.dev" not in message
    assert read_profile_file(profile, "skills.yaml") == skills


def test_no_staging_directory_survives_a_failure(profile: Path) -> None:
    """The staging copy is an implementation detail and must not leak into the
    workspace -- least of all next to someone's career data."""
    with pytest.raises(ProfileWriteError):
        write_profile_file(profile, AN_ENTRY, "id: [broken\n")

    assert not list(profile.parent.glob(".*staging*"))


# ===========================================================================
# Backups -- the only undo a gitignored directory has
# ===========================================================================


def test_a_successful_save_backs_up_the_previous_text(profile: Path) -> None:
    before = read_profile_file(profile, AN_ENTRY)

    backup = write_profile_file(profile, AN_ENTRY, before + "\n# a new comment\n")

    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == before
    assert default_backup_dir() in backup.parents


def test_each_save_keeps_its_own_backup(profile: Path) -> None:
    """Backups accumulate rather than overwrite. One rescue file that is itself
    overwritten by the next mistake is not a rescue."""
    text = read_profile_file(profile, AN_ENTRY)
    first = write_profile_file(profile, AN_ENTRY, text + "\n# one\n")
    second = write_profile_file(profile, AN_ENTRY, text + "\n# two\n")

    assert first != second
    assert first.is_file() and second.is_file()


def test_a_refused_save_takes_no_backup(profile: Path) -> None:
    """Validation runs first, so a rejected edit does not fill the backup
    directory with copies of a file that never changed."""
    with pytest.raises(ProfileWriteError):
        write_profile_file(profile, AN_ENTRY, "id: [broken\n")

    assert not list(default_backup_dir().rglob("*.yaml"))


# ===========================================================================
# Creating a profile
# ===========================================================================


def test_scaffold_produces_a_profile_that_loads(tmp_path: Path) -> None:
    target = tmp_path / "profile"
    scaffold_profile(PROFILE_EXAMPLE, target)

    assert load_profile(target).identity.name == "John Doe"


def test_scaffold_refuses_to_overwrite(profile: Path) -> None:
    """Creating is cheap and repeatable. Clobbering a profile someone has
    already written is not."""
    with pytest.raises(ProfileWriteError, match="already exists"):
        scaffold_profile(PROFILE_EXAMPLE, profile)


def test_scaffold_refuses_a_source_that_is_not_a_profile(tmp_path: Path) -> None:
    empty = tmp_path / "not-a-profile"
    empty.mkdir()

    with pytest.raises(ProfileWriteError, match="identity.yaml"):
        scaffold_profile(empty, tmp_path / "profile")


def test_the_jd_cache_is_isolated_from_the_real_one(tmp_path: Path) -> None:
    """A guard on the conftest fixture, kept beside the other data-hygiene
    tests. `JobSpecCache` carried its own hardcoded default and so escaped the
    cache-directory override, and the tests that pre-seed a parse were writing
    their fixture JobSpec into the developer's real `.cache/jd/`.
    """
    from resume_agent.jd_cache import JobSpecCache

    resolved = JobSpecCache().cache_dir.resolve()

    assert Path(".cache/jd").resolve() != resolved, "tests would write to the real JD cache"
    assert resolved.is_relative_to(Path(os.environ["RESUME_AGENT_CACHE_DIR"]).resolve())


def test_the_example_profile_is_never_touched(profile: Path) -> None:
    """A guard on the fixture itself. `profile.example` is what the golden
    snapshot test renders, so a writer test that mutated it would break an
    unrelated suite in a way that looks like a template regression."""
    before = (PROFILE_EXAMPLE / AN_ENTRY).read_bytes()

    write_profile_file(profile, AN_ENTRY, read_profile_file(profile, AN_ENTRY) + "\n# edit\n")

    assert (PROFILE_EXAMPLE / AN_ENTRY).read_bytes() == before
