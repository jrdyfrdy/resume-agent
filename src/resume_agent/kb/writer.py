"""Write to a profile directory. The only module in this project that does.

`profile/` is real career data, it is gitignored, and nothing regenerates it --
so unlike `.cache/` or `.index/`, a bad write here is permanent. Every rule in
this module exists because of that one fact.

**Why text, not objects.** The obvious design is "parse YAML, edit the object,
dump it back". It is also wrong here: PyYAML's `safe_dump` discards every
comment, reflows `>-` folded scalars into quoted lines, expands flow-style lists
(`tech: [python, fastapi]`) one-per-line, and sorts keys alphabetically. The
profile YAML carries load-bearing commentary -- `skills.yaml` explains *why* a
skill vocabulary is an allow-list, and the entry files explain which escape
hazard each employer name is testing. A round trip would silently delete all of
it on the first save, with no git history to recover from.

So this module moves **file text**, never parsed objects. What the editor sends
is byte-for-byte what lands on disk. Structured editing through a
comment-preserving parser (`ruamel.yaml`) is a reasonable thing to add later;
it is a new dependency and a separate decision.

**Validation is whole-directory, not per-file.** `Profile` has three
cross-file validators: globally unique ids, bullet ids namespaced to their
entry, and every skill resolving through `skills.yaml`. Deleting one alias from
`skills.yaml` can invalidate an experience file nobody touched. So a save is
staged into a copy of the entire directory and `load_profile` runs against that
copy; the real directory is only touched once the whole thing is known to load.
A profile that does not load is never a state this module leaves behind.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from resume_agent.kb.loader import (
    EXPERIENCE_DIR,
    IDENTITY_FILE,
    NARRATIVES_DIR,
    PROJECTS_DIR,
    SKILLS_FILE,
    ProfileLoadError,
    load_profile,
)

PROFILE_BACKUP_DIR_ENV_VAR = "RESUME_AGENT_PROFILE_BACKUPS"
BACKUP_DIR = Path(".profile-backups")

# What the editor is allowed to open and save. `.yml` is deliberately absent:
# `loader.py` globs `*.yaml` only, so saving `foo.yml` would write a file that
# loads fine, looks right in the editor, and is read by nothing.
EDITABLE_SUFFIXES = {".yaml", ".md"}

# Directories a profile may contain files in, plus the root itself.
EDITABLE_DIRS = (EXPERIENCE_DIR, PROJECTS_DIR, NARRATIVES_DIR)


class ProfileWriteError(RuntimeError):
    """A write was refused, or could not be completed."""


def default_backup_dir() -> Path:
    """Where pre-write copies go.

    Resolved on every call rather than bound as a default argument, the same
    reason as `default_tracker_db` and `default_cache_dir`: the test suite has
    to be able to point it somewhere disposable, and a value frozen at import
    cannot be redirected. Here it matters most -- a test that backed up into the
    user's real backup directory would be writing beside their career data.
    """
    override = os.environ.get(PROFILE_BACKUP_DIR_ENV_VAR)
    return Path(override) if override else BACKUP_DIR


# ---------------------------------------------------------------------------
# Locating files safely
# ---------------------------------------------------------------------------


def resolve_editable_path(profile_dir: Path, relative: str) -> Path:
    """Turn a browser-supplied relative path into a real one, or refuse.

    The containment check is the security boundary of the whole feature. The
    editor sends a path from the client, and without this a request for
    `../../.ssh/id_rsa` or an absolute path would be honoured. Comparing
    *resolved* paths is what makes it sound -- a purely textual check for `..`
    is defeated by symlinks.
    """
    if not relative or relative != relative.strip():
        raise ProfileWriteError("path must not be empty or padded with whitespace")

    candidate = Path(relative)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise ProfileWriteError(f"path must be relative to the profile: {relative!r}")

    root = Path(profile_dir).resolve()
    target = (root / candidate).resolve()

    if target != root and root not in target.parents:
        raise ProfileWriteError(f"path escapes the profile directory: {relative!r}")

    if target.suffix not in EDITABLE_SUFFIXES:
        raise ProfileWriteError(
            f"only {', '.join(sorted(EDITABLE_SUFFIXES))} files are editable, got {target.suffix!r}"
            + (" -- the loader reads .yaml, never .yml" if target.suffix == ".yml" else "")
        )
    return target


def relative_profile_files(profile_dir: Path) -> list[str]:
    """Every file the editor may open, as forward-slash relative paths.

    Root files first, then the entry directories in load order, so the list
    reads the way the profile is assembled rather than alphabetically.
    """
    root = Path(profile_dir)
    if not root.is_dir():
        raise ProfileWriteError(f"profile directory not found: {root}")

    found: list[str] = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix in EDITABLE_SUFFIXES
    )
    for directory in EDITABLE_DIRS:
        sub = root / directory
        if not sub.is_dir():
            continue
        found.extend(
            f"{directory}/{path.name}"
            for path in sorted(sub.iterdir())
            if path.is_file() and path.suffix in EDITABLE_SUFFIXES
        )
    return found


def read_profile_file(profile_dir: Path, relative: str) -> str:
    """The current text of one file."""
    path = resolve_editable_path(profile_dir, relative)
    if not path.is_file():
        raise ProfileWriteError(f"no such file in this profile: {relative}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def write_profile_file(profile_dir: Path, relative: str, text: str) -> Path:
    """Save one file, but only if the whole profile still loads afterwards.

    Returns the path of the backup taken before the write.

    The order is deliberate and is the point of the function:

      1. resolve and contain the path
      2. stage the edit in a throwaway copy of the directory
      3. load the *copy* -- this is where a cross-file break surfaces
      4. back up the real file
      5. replace it atomically

    Steps 2 and 3 cost a directory copy on every save. For a profile of a few
    dozen small YAML files that is microseconds, and it buys the guarantee that
    a refused save leaves the file on disk exactly as it was.
    """
    root = Path(profile_dir)
    path = resolve_editable_path(root, relative)
    if not path.is_file():
        raise ProfileWriteError(f"no such file in this profile: {relative}")

    # Normalise to LF. `.gitattributes` declares `* text=auto eol=lf`, and
    # `profile_content_hash` hashes raw bytes -- a browser posting CRLF would
    # rewrite every line and invalidate the search index on a no-op save.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")

    _validate_with_edit(root, relative, normalised)

    backup = _back_up(root, path, relative)

    # Atomic replace, same shape as `jd_cache.put`. `with_name` rather than
    # `with_suffix`: entry files are user-named and `with_suffix` would eat a
    # dot that happens to be in the stem.
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(normalised, encoding="utf-8", newline="\n")
    temp_path.replace(path)
    return backup


def _validate_with_edit(profile_dir: Path, relative: str, text: str) -> None:
    """Load a copy of the profile with this edit applied. Raise if it breaks.

    Validating the single file in isolation would not be enough. `Profile`'s
    validators run across the whole directory: a skill removed from
    `skills.yaml` invalidates every bullet elsewhere that named it, and a
    renamed entry id orphans its own bullets. Only a whole-directory load can
    see that.
    """
    staging_parent = Path(profile_dir).parent / f".{Path(profile_dir).name}.staging"
    shutil.rmtree(staging_parent, ignore_errors=True)
    try:
        shutil.copytree(profile_dir, staging_parent)
        staged = resolve_editable_path(staging_parent, relative)
        staged.write_text(text, encoding="utf-8", newline="\n")
        try:
            load_profile(staging_parent)
        except ProfileLoadError as exc:
            raise ProfileWriteError(
                _readable(str(exc), staging_parent, Path(profile_dir))
            ) from exc
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _readable(message: str, staging: Path, real: Path) -> str:
    """Make a loader error fit to show someone mid-edit.

    Two things have to go. The staging directory is an implementation detail of
    this module, and naming it makes a correct refusal look like a bug in the
    tool rather than a typo in the file. And Pydantic appends
    `[type=value_error, input_value={...}, input_type=dict]` plus a docs URL --
    a truncated dump of the entire profile, which buries the one line that says
    which skill is missing.

    What is kept is the sentence the validator wrote, which is already good:
    the model errors in `models/profile.py` name the bullet and the value.
    """
    cleaned = message.replace(str(staging), str(real)).replace(staging.name, real.name)
    # To end of line, not to the first `]`: the dumped `input_value` is full of
    # brackets, so a bracket-matching pattern stops in the middle of it and
    # leaves `}]}, input_type=dict]` behind.
    cleaned = re.sub(r"\s*\[type=[^\n]*", "", cleaned)
    cleaned = re.sub(r"\n\s*For further information visit \S+", "", cleaned)
    cleaned = re.sub(r"\n\s*Value error, ", "\n", cleaned)
    return cleaned.strip()


def _back_up(profile_dir: Path, path: Path, relative: str) -> Path:
    """Copy the current file somewhere recoverable before overwriting it.

    `profile/` is gitignored, so there is no `git checkout` to fall back on.
    This is the only undo that exists.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = default_backup_dir() / Path(profile_dir).name
    destination = root / stamp / relative

    # Two saves inside the same second would otherwise land on the same path and
    # the second would overwrite the first -- destroying the copy of the text you
    # most likely want back, since fixing a mistake usually means saving twice
    # quickly. Found by a test; a seconds-resolution timestamp is not an
    # identity.
    attempt = 1
    while destination.exists():
        destination = root / f"{stamp}-{attempt}" / relative
        attempt += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


# ---------------------------------------------------------------------------
# Creating a profile
# ---------------------------------------------------------------------------


def scaffold_profile(source: Path, target: Path) -> None:
    """Copy an existing profile to a new directory.

    The answer to "where do I put my career knowledge base": this is what the
    UI's create button runs, so that the first step is a working profile rather
    than seven empty files and a schema to read.

    Refuses to overwrite. Creating is cheap and repeatable; clobbering a profile
    someone has already written is not.
    """
    source, target = Path(source), Path(target)
    if not (source / IDENTITY_FILE).is_file():
        raise ProfileWriteError(f"{source} is not a profile (no {IDENTITY_FILE})")
    if target.exists():
        raise ProfileWriteError(f"{target} already exists -- refusing to overwrite it")

    shutil.copytree(source, target)
    try:
        load_profile(target)
    except ProfileLoadError as exc:  # pragma: no cover - source is always valid
        shutil.rmtree(target, ignore_errors=True)
        raise ProfileWriteError(f"copied profile does not load: {exc}") from exc


# Only `identity.yaml` and `skills.yaml` are `required=True` in `load_profile`,
# and every list on `Profile` defaults to empty -- so this is the smallest thing
# that loads. The skills table keeps its explanatory header because that text
# documents the schema rather than any particular person's stack.
_EMPTY_IDENTITY = """\
name: {name}
email: ""
phone: ""
location: ""
links: []
work_authorization: ""
"""

_EMPTY_SKILLS = """\
# The canonical skill vocabulary (spec 3.2).
#
# This table does double duty:
#   1. query expansion at retrieval time -- "k8s" in a posting finds "Kubernetes"
#   2. the allow-list for the fabrication check -- a generated bullet naming a
#      technology that is not in here is, by definition, invented
#
# `canonical` is written in *display* casing ("PostgreSQL", not "postgresql")
# because it is what the Technical Skills section prints. Matching is
# case-insensitive throughout, so bullets can still write `skills: [postgresql]`.
skills: []
"""

_EMPTY_NARRATIVES_README = """\
# Narratives

Short pieces of prose about how you work -- why you chose this field, the
hardest thing you have debugged, what you are trying to get better at. One
markdown file each; the filename is the name.

They are never copied into a resume. They are raw material for the cover
letter, which is why they can be informal: write them the way you would explain
it to someone, not the way you would write a bullet point.

This README is skipped when the profile loads, so it is safe to keep here.
"""


def create_empty_profile(target: Path, *, name: str = "") -> None:
    """Start a profile with nothing in it but you.

    The counterpart to `scaffold_profile`, and the better default of the two.
    Copying the worked example gives someone a directory that loads immediately,
    which is genuinely useful for trying the tool out -- but it also means a
    profile you intend to *use* begins as a stranger's career, and every job,
    project and narrative in it has to be found and deleted before your own
    material is the only thing in there. Starting empty has no such step.
    """
    target = Path(target)
    if target.exists():
        raise ProfileWriteError(f"{target} already exists -- refusing to overwrite it")

    (target / EXPERIENCE_DIR).mkdir(parents=True)
    (target / PROJECTS_DIR).mkdir(parents=True)
    (target / NARRATIVES_DIR).mkdir(parents=True)

    written = {
        IDENTITY_FILE: _EMPTY_IDENTITY.format(name=name.strip() or '""'),
        SKILLS_FILE: _EMPTY_SKILLS,
        f"{NARRATIVES_DIR}/README.md": _EMPTY_NARRATIVES_README,
    }
    for relative, text in written.items():
        (target / relative).write_text(text, encoding="utf-8", newline="\n")

    try:
        load_profile(target)
    except ProfileLoadError as exc:  # pragma: no cover - the skeleton is fixed
        shutil.rmtree(target, ignore_errors=True)
        raise ProfileWriteError(f"new profile does not load: {exc}") from exc
