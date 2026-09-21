"""Read `profile/` (or `profile.example/`) off disk into a validated `Profile`.

Spec 3.3 describes this as "YAML -> Pydantic -> SQLite". M0 does the first arrow
only; the SQLite mirror and the vector index are M1's definition of done, so
there is no stub for them here.

The one job this module adds on top of Pydantic is *locating the failure*. A raw
`ValidationError` says `experience.0.bullets.2.canonical`, which is useless when
`experience` came from five different files. Every error raised from here names
the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from resume_agent.models.profile import Profile


class ProfileLoadError(Exception):
    """A profile directory could not be read or did not validate."""


# Directory layout from spec 3.2. Kept as module constants so tests and the CLI
# error messages agree on what a profile is supposed to contain.
IDENTITY_FILE = "identity.yaml"
EDUCATION_FILE = "education.yaml"
SKILLS_FILE = "skills.yaml"
CERTIFICATIONS_FILE = "certifications.yaml"
AWARDS_FILE = "awards.yaml"
EXPERIENCE_DIR = "experience"
PROJECTS_DIR = "projects"
LEADERSHIP_DIR = "leadership"
PUBLICATIONS_DIR = "publications"
NARRATIVES_DIR = "narratives"


def load_profile(profile_dir: Path) -> Profile:
    """Load and validate a whole profile directory.

    Raises `ProfileLoadError` for anything wrong: missing directory, malformed
    YAML, schema violation, or a cross-file integrity failure from `Profile`.
    """
    profile_dir = Path(profile_dir)
    if not profile_dir.is_dir():
        raise ProfileLoadError(f"profile directory not found: {profile_dir}")

    identity = _read_mapping(profile_dir / IDENTITY_FILE, required=True)

    # education.yaml and certifications.yaml hold a list under a named key, so a
    # human editing them sees what the file is about without opening the schema.
    education = _read_keyed_list(profile_dir / EDUCATION_FILE, key="education")
    skills = _read_keyed_list(profile_dir / SKILLS_FILE, key="skills", required=True)
    certifications = _read_keyed_list(profile_dir / CERTIFICATIONS_FILE, key="certifications")
    awards = _read_keyed_list(profile_dir / AWARDS_FILE, key="awards")

    narratives = _read_narratives(profile_dir / NARRATIVES_DIR)
    experience = _read_entry_dir(profile_dir / EXPERIENCE_DIR)
    projects = _read_entry_dir(profile_dir / PROJECTS_DIR)
    leadership = _read_entry_dir(profile_dir / LEADERSHIP_DIR)
    publications = _read_entry_dir(profile_dir / PUBLICATIONS_DIR)

    payload: dict[str, Any] = {
        "identity": identity,
        "education": education,
        "skills": skills,
        "certifications": certifications,
        "awards": awards,
        "narratives": narratives,
        "experience": experience,
        "projects": projects,
        "leadership": leadership,
        "publications": publications,
    }

    try:
        return Profile.model_validate(payload)
    except ValidationError as exc:
        # Cross-file checks (unique ids, skill vocabulary) can only fail here,
        # once every file is in one object -- so this message points at the
        # directory rather than at a single file.
        raise ProfileLoadError(f"profile at {profile_dir} is invalid:\n{exc}") from exc


# --- file-level helpers ----------------------------------------------------


def _read_yaml(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except OSError as exc:
        raise ProfileLoadError(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProfileLoadError(f"{path} is not valid YAML:\n{exc}") from exc


def _read_mapping(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ProfileLoadError(f"required file missing: {path}")
        return {}
    data = _read_yaml(path)
    if not isinstance(data, dict):
        raise ProfileLoadError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _read_keyed_list(path: Path, *, key: str, required: bool = False) -> list[Any]:
    """Read a file shaped like `{key}:` followed by a list."""
    if not path.exists():
        if required:
            raise ProfileLoadError(f"required file missing: {path}")
        return []
    data = _read_mapping(path)
    items = data.get(key, [])
    if not isinstance(items, list):
        raise ProfileLoadError(f"{path}: key {key!r} must hold a list, got {type(items).__name__}")
    return items


def _read_narratives(directory: Path) -> list[dict[str, str]]:
    """Read every `*.md` in `narratives/`, one entry per file.

    README.md is skipped: it documents the directory for a human rather than
    describing the profile's owner, and feeding it to a cover-letter prompt
    would be feeding the model instructions meant for you.

    Sorted by filename, like the entry directories, so load order is
    deterministic and prompts are byte-stable between runs.
    """
    if not directory.is_dir():
        return []
    narratives = []
    for path in sorted(directory.glob("*.md")):
        if path.stem.lower() == "readme":
            continue
        try:
            narratives.append({"name": path.stem, "content": path.read_text(encoding="utf-8")})
        except OSError as exc:
            raise ProfileLoadError(f"cannot read {path}: {exc}") from exc
    return narratives


def _read_entry_dir(directory: Path) -> list[dict[str, Any]]:
    """Read every `*.yaml` in a directory, one entry per file.

    Sorted by filename so the render order is deterministic. That matters more
    than it sounds: the golden .tex snapshot test compares bytes, and a
    filesystem that returns files in a different order would fail it spuriously.
    """
    if not directory.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        data = _read_yaml(path)
        if not isinstance(data, dict):
            raise ProfileLoadError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
        entries.append(data)
    return entries
