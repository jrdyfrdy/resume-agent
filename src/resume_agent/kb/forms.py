"""Structured editing of a profile: field specs in, file text out.

The raw-text editor asks the user to know YAML -- indentation, `>-` folded
scalars, flow lists, which keys exist. This module is what lets the web UI show
input fields instead and own the file format itself.

**It sits on top of `writer.py`, not beside it.** Everything here ends by handing
finished text to `write_profile_file`, so the staging validation, the pre-write
backup, the atomic replace, the LF normalisation and the path containment all
apply unchanged. There is exactly one code path that writes to a profile.

**Why ruamel and not PyYAML.** A form edits one field; the file it writes back
should differ in one field. `yaml.safe_dump` discards every comment, reflows the
folded scalars, expands `tech: [python, ...]` one-per-line and re-sorts the keys
-- so the first save through a form would silently destroy the user's own notes,
in a directory that is gitignored and has no undo. ruamel round-trips all of it.
Verified empirically: with the settings in `_yaml()` below, every file in
`profile.example/` load-dumps byte-identically.

**Why an explicit field table** rather than generating forms from the Pydantic
schema: seven file types, each needing hand-written labels and help text anyway.
A JSON-Schema-to-form renderer would be more machinery than the problem has
(CLAUDE.md: favour explicit, readable constructions).
"""

from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import FoldedScalarString, LiteralScalarString

from resume_agent.kb.loader import (
    CERTIFICATIONS_FILE,
    EDUCATION_FILE,
    EXPERIENCE_DIR,
    IDENTITY_FILE,
    NARRATIVES_DIR,
    PROJECTS_DIR,
    SKILLS_FILE,
    load_profile,
)
from resume_agent.kb.writer import (
    ProfileWriteError,
    read_profile_file,
    resolve_editable_path,
    write_profile_file,
)

DocumentKind = Literal["mapping", "keyed_list", "entry", "prose"]

# Straight from `models/profile.py`. Duplicated deliberately rather than
# introspected: these are UI copy as much as they are constraints, and a form
# that silently gained an option because a Literal changed would be worse than
# one that fails a test.
CONFIDENCE = ("verified", "approximate", "claim")
SENIORITY = ("intern", "junior", "mid", "senior", "staff", "lead")
SKILL_LEVEL = ("expert", "working", "familiar")

YEAR_MONTH = r"^\d{4}-(0[1-9]|1[0-2])$"


@dataclass(frozen=True)
class Field:
    """One control in a form."""

    name: str
    label: str
    kind: Literal["text", "textarea", "month", "select", "chips", "keyvalue", "objects"]
    required: bool = False
    choices: tuple[str, ...] = ()
    # Which suggestion list backs a `chips` control. "skills" is a closed list
    # (the profile refuses to load otherwise); the rest are open and merely
    # offer what is already in use.
    suggest: str | None = None
    closed: bool = False
    help: str = ""
    placeholder: str = ""
    fields: tuple[Field, ...] = ()
    # Which key identifies a row of an `objects` list across a save. Matching on
    # it is what keeps a comment attached to the bullet it describes when
    # another bullet is inserted above it.
    key: str | None = None


BULLET_FIELDS: tuple[Field, ...] = (
    Field("canonical", "What you did", "textarea", required=True,
          help="One achievement, written fully and truthfully. Every rewrite the agent "
               "produces is a rephrasing of this sentence and nothing more.",
          placeholder="Cut p95 checkout latency from 820ms to 50ms across 12 endpoints by "
                      "repartitioning the events table on user_id."),
    Field("metrics", "Numbers", "keyvalue",
          help="The ONLY numbers a rewrite of this bullet may use. A figure that is not "
               "here, and not derivable from here, is treated as invented and the bullet "
               "is dropped."),
    Field("skills", "Technologies", "chips", suggest="skills", closed=True,
          help="Must already exist in Skills. That is what stops the agent quietly "
               "upgrading “a queue” into “Kafka”."),
    Field("themes", "Themes", "chips", suggest="themes",
          help="Free grouping. At most three bullets per theme reach the page, so spread "
               "them or most of your bullets can never be picked together."),
    Field("seniority_signal", "Seniority signal", "select", choices=("",) + SENIORITY),
    Field("confidence", "Confidence", "select", choices=CONFIDENCE,
          help="“claim” is excluded when you run with strict."),
    Field("evidence", "Evidence", "textarea",
          help="Where you could prove it. Never printed — it is for you, at interview.",
          placeholder="PR #4412; Grafana p95 dashboard, Nov 2024"),
)

_ENTRY_TAIL: tuple[Field, ...] = (
    Field("start", "Started", "month", required=True),
    Field("end", "Ended", "month", help="Leave empty for “Present”."),
    Field("tech", "Technologies", "chips", suggest="skills", closed=True),
    Field("bullets", "Achievements", "objects", fields=BULLET_FIELDS, key="id"),
)

# role -> (document kind, YAML root key or None, fields)
FORMS: dict[str, tuple[DocumentKind, str | None, tuple[Field, ...]]] = {
    "identity": ("mapping", None, (
        Field("name", "Full name", "text", required=True),
        Field("email", "Email", "text", required=True),
        Field("phone", "Phone", "text", required=True),
        Field("location", "Location", "text", required=True, placeholder="Oakland, CA"),
        Field("links", "Links", "objects", key="label", fields=(
            Field("label", "Shown as", "text", required=True,
                  placeholder="github.com/you"),
            Field("url", "Address", "text", required=True,
                  placeholder="https://github.com/you"),
        )),
        Field("work_authorization", "Work authorisation", "text"),
    )),
    "education": ("keyed_list", "education", (
        Field("education", "Education", "objects", key="institution", fields=(
            Field("institution", "Institution", "text", required=True),
            Field("degree", "Degree", "text", required=True),
            Field("location", "Location", "text", required=True),
            Field("start", "Started", "month", required=True),
            Field("end", "Ended", "month"),
            Field("gpa", "GPA", "text", help="Written as text, e.g. 3.7."),
            Field("coursework", "Coursework", "chips", suggest="coursework"),
        )),
    )),
    "skills": ("keyed_list", "skills", (
        Field("skills", "Skills", "objects", key="canonical", fields=(
            Field("canonical", "Name", "text", required=True,
                  help="Display casing — this is what prints on the resume.",
                  placeholder="PostgreSQL"),
            Field("aliases", "Also written as", "chips", suggest="aliases",
                  help="How postings might spell it. Used to match a posting to this "
                       "skill, and accepted wherever you name it elsewhere."),
            Field("category", "Category", "text", required=True, placeholder="data"),
            Field("level", "Level", "select", required=True, choices=SKILL_LEVEL),
            Field("first_used", "First used", "month"),
        )),
    )),
    "certifications": ("keyed_list", "certifications", (
        Field("certifications", "Certifications", "objects", key="name", fields=(
            Field("name", "Name", "text", required=True),
            Field("issuer", "Issuer", "text", required=True),
            Field("issued", "Issued", "month", required=True),
            Field("credential_url", "Verification link", "text"),
        )),
    )),
    "experience": ("entry", None, (
        Field("org", "Employer", "text", required=True),
        Field("title", "Job title", "text", required=True),
        Field("location", "Location", "text", required=True),
    ) + _ENTRY_TAIL),
    "project": ("entry", None, (
        Field("name", "Project", "text", required=True),
        Field("role", "Your role", "text"),
        Field("repo_url", "Repository", "text"),
        Field("live_url", "Live link", "text"),
    ) + _ENTRY_TAIL),
    "narrative": ("prose", None, (
        Field("title", "Title", "text", required=True, placeholder="How I learn"),
        Field("body", "Prose", "textarea", required=True,
              help="Plain paragraphs — no formatting needed. Keep it qualitative: "
                   "do not introduce numbers or technologies that appear nowhere else in "
                   "your knowledge base, or the cover letter will fail verification."),
    )),
}

# Keys the form owns but never shows. `type` is fixed by which directory the
# file lives in, and ids are plumbing: renaming an entry id has to cascade to
# every bullet id in the same save, and orphans the ids already recorded in the
# tracker. The form generates them and preserves them; it does not offer them.
HIDDEN_KEYS = ("id", "type")


def _yaml() -> YAML:
    """A round-tripper tuned to reproduce this project's YAML exactly.

    Every setting here was derived by dumping `profile.example/` and diffing,
    not guessed:

    * `indent(sequence=4, offset=2)` -- the files write `bullets:` then `  - id:`.
      ruamel's default puts the dash at the parent's indentation.
    * `width` -- the default of 80 rewraps `tech: [python, fastapi, ...]` across
      two lines.
    * the None representer -- ruamel emits an empty value for None; these files
      write the literal `null`, and `end: null` carries a comment explaining it.

    With these, all nine files in `profile.example/` load-dump byte-identically.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.representer.add_representer(
        type(None),
        lambda representer, _data: representer.represent_scalar("tag:yaml.org,2002:null", "null"),
    )
    return yaml


# ---------------------------------------------------------------------------
# Which form does a file get?
# ---------------------------------------------------------------------------


def document_role(relative: str) -> str:
    """Map a path inside a profile to the form that edits it."""
    path = relative.replace("\\", "/")
    if path.startswith(f"{EXPERIENCE_DIR}/"):
        return "experience"
    if path.startswith(f"{PROJECTS_DIR}/"):
        return "project"
    if path.startswith(f"{NARRATIVES_DIR}/"):
        return "narrative"
    role = {
        IDENTITY_FILE: "identity",
        EDUCATION_FILE: "education",
        SKILLS_FILE: "skills",
        CERTIFICATIONS_FILE: "certifications",
    }.get(path)
    if role is None:
        raise ProfileWriteError(f"no form knows how to edit {relative!r}")
    return role


def _spec(role: str) -> list[dict[str, Any]]:
    _kind, _root, fields = FORMS[role]
    return [asdict(f) for f in fields]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_document(profile_dir: Path, relative: str) -> dict[str, Any]:
    """The form spec, the current values, and what to suggest in comboboxes."""
    role = document_role(relative)
    kind, root_key, _fields = FORMS[role]
    text = read_profile_file(profile_dir, relative)

    if kind == "prose":
        data: dict[str, Any] = _split_prose(text)
    else:
        loaded = _yaml().load(text) or {}
        data = _plain(loaded)
        if kind == "keyed_list":
            data = {root_key: data.get(root_key) or []}

    return {
        "path": relative,
        "role": role,
        "kind": kind,
        "spec": _spec(role),
        "data": data,
        "suggestions": suggestions(profile_dir),
    }


def suggestions(profile_dir: Path) -> dict[str, list[str]]:
    """What the comboboxes offer.

    `skills` is the one closed list -- naming anything outside it makes the
    profile refuse to load, so offering it as a constrained choice is what turns
    `Profile._skills_resolve_to_vocabulary` from a rejection you discover after
    saving into something the form cannot violate. The rest are free text and
    are merely conveniences.
    """
    try:
        profile = load_profile(Path(profile_dir))
    except Exception:  # noqa: BLE001 - a broken profile still deserves a usable form
        return {"skills": [], "themes": [], "aliases": [], "coursework": []}

    themes = {theme for bullet in profile.all_bullets() for theme in bullet.themes}
    return {
        "skills": sorted({skill.canonical for skill in profile.skills}),
        "themes": sorted(themes),
        "aliases": [],
        "coursework": sorted({c for e in profile.education for c in e.coursework}),
    }


def skill_usage(profile_dir: Path) -> dict[str, int]:
    """How many bullets and entries name each skill, case-insensitively.

    Deleting a skill row -- or even one of its aliases -- can invalidate files
    the user never opened. The save is already refused, but that costs them the
    edit, so the form warns first and this is the number it warns with.
    """
    try:
        profile = load_profile(Path(profile_dir))
    except Exception:  # noqa: BLE001
        return {}

    counts: dict[str, int] = {}
    for entry in profile.entries():
        for name in entry.tech:
            counts[name.lower()] = counts.get(name.lower(), 0) + 1
        for bullet in entry.bullets:
            for name in bullet.skills:
                counts[name.lower()] = counts.get(name.lower(), 0) + 1

    # Roll aliases up onto the canonical name, since that is the row being removed.
    rolled: dict[str, int] = {}
    for skill in profile.skills:
        total = counts.get(skill.canonical.lower(), 0)
        total += sum(counts.get(alias.lower(), 0) for alias in skill.aliases)
        rolled[skill.canonical] = total
    return rolled


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_document(profile_dir: Path, relative: str, data: dict[str, Any]) -> Path:
    """Apply form values to a file and save it. Returns the backup path.

    The merge happens **into the loaded document**, never into a fresh one: that
    is what keeps every comment, the folded scalars, the flow lists and the key
    order attached to the nodes that survive the edit.
    """
    role = document_role(relative)
    kind, root_key, fields = FORMS[role]

    if kind == "prose":
        return write_profile_file(profile_dir, relative, _join_prose(data))

    yaml = _yaml()
    document = yaml.load(read_profile_file(profile_dir, relative)) or {}

    if kind == "keyed_list":
        spec = next(f for f in fields if f.name == root_key)
        document[root_key] = _merge_list(
            document.get(root_key) or [], data.get(root_key) or [], spec
        )
    else:
        _merge_mapping(document, data, fields)

    buffer = io.StringIO()
    yaml.dump(document, buffer)
    return write_profile_file(profile_dir, relative, buffer.getvalue())


def _merge_mapping(node: Any, data: dict[str, Any], fields: tuple[Field, ...]) -> None:
    """Set each known field on an existing mapping node, in place.

    **A field whose value did not change is not reassigned.** That single rule is
    what preserves the file's style: ruamel carries the flow-vs-block choice and
    the folded-scalar wrapping on the *node*, and assigning a plain Python `str`
    or `list` back over it -- even an equal one -- throws that away. Reassigning
    only what the user actually edited means an untouched file comes back
    byte-identical, which the tests assert.
    """
    for spec in fields:
        if spec.name not in data:
            continue
        value = data[spec.name]

        if spec.kind == "objects":
            node[spec.name] = _merge_list(node.get(spec.name) or [], value or [], spec)
            continue

        value = _coerce(spec, value)
        existing = node.get(spec.name) if hasattr(node, "get") else None
        if spec.name in node and _plain(existing) == value:
            continue

        if value is None and not spec.required:
            # An optional field cleared in the form. Keep the key with an
            # explicit null rather than deleting it: `end: null` is meaningful
            # here ("Present") and carries a comment in the example files.
            node[spec.name] = None
        elif value is not None:
            node[spec.name] = _styled(spec, value, existing)


def _merge_list(existing: Any, rows: list[dict[str, Any]], spec: Field) -> Any:
    """Rebuild a list of mappings, reusing the nodes that survived.

    Matching on `spec.key` rather than position is what keeps a comment attached
    to the row it describes when another row is inserted above it. A row with no
    match is new and gets a plain mapping -- correctly, since there is no comment
    to carry.
    """
    by_key = {}
    if spec.key:
        for node in existing:
            if isinstance(node, dict) and spec.key in node:
                by_key[str(node[spec.key])] = node

    merged = []
    for row in rows:
        node = by_key.get(str(row.get(spec.key))) if spec.key else None
        if node is None:
            node = {}
            # Hidden keys are carried by the form, not shown in it, so they have
            # to be written explicitly for a new row.
            for hidden in HIDDEN_KEYS:
                if hidden in row:
                    node[hidden] = row[hidden]
        _merge_mapping(node, row, spec.fields)
        merged.append(node)

    if hasattr(existing, "copy_attributes"):
        # Preserve the sequence's own style (block vs flow) and any comment
        # attached to the list as a whole.
        replacement = type(existing)(merged)
        existing.copy_attributes(replacement)
        return replacement
    return merged


def _styled(spec: Field, value: Any, existing: Any) -> Any:
    """Give a genuinely changed value the style its neighbours use.

    Only reached when the user edited the field, so there is no node left to
    inherit from -- the style has to be chosen. Two cases matter:

    * a `chips` list is written flow-style (`skills: [redis, caching]`), which is
      how every list of short tokens in these files is written;
    * text that was a folded block stays a folded block, so a long `canonical`
      does not become one enormous quoted line in the middle of the file.

    A re-folded scalar is emitted on one line rather than hand-wrapped at 80
    columns, because `width` has to stay high to stop the flow lists wrapping.
    It is still folded, still valid, and only the sentence just edited is
    affected.
    """
    if spec.kind == "chips":
        sequence = CommentedSeq(value)
        sequence.fa.set_flow_style()
        return sequence
    if isinstance(existing, FoldedScalarString) and isinstance(value, str):
        return FoldedScalarString(value)
    if isinstance(existing, LiteralScalarString) and isinstance(value, str):
        return LiteralScalarString(value)
    return value


def _coerce(spec: Field, value: Any) -> Any:
    """Turn a browser value into what the schema expects."""
    if spec.kind == "keyvalue":
        return {str(k): _number_or_text(v) for k, v in (value or {}).items()}
    if spec.kind == "chips":
        return [str(v) for v in (value or [])]
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return value


def _number_or_text(value: Any) -> Any:
    """`metrics` values are `int | float | str`.

    The browser sends everything as text, so "820" has to come back as an int or
    a round trip would rewrite every metric in the file as a quoted string --
    and the grounding gate compares these against the numbers in a rewrite.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _plain(node: Any) -> Any:
    """ruamel's types are dict/list subclasses; JSON needs the plain ones."""
    if isinstance(node, dict):
        return {str(k): _plain(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_plain(v) for v in node]
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    return str(node)


# ---------------------------------------------------------------------------
# Narratives
# ---------------------------------------------------------------------------


def _split_prose(text: str) -> dict[str, str]:
    """A narrative is a `# Title` and paragraphs. Nothing else is used.

    Checked against every file in `profile.example/narratives/`: one H1, then
    plain paragraphs -- no lists, links or emphasis. The cover-letter node builds
    its own heading from the filename, so the H1 is redundant to the pipeline and
    exists for the human reading the file.
    """
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return {"title": lines[0][2:].strip(), "body": "\n".join(lines[1:]).strip()}
    return {"title": "", "body": text.strip()}


def _join_prose(data: dict[str, Any]) -> str:
    title = str(data.get("title") or "").strip()
    body = str(data.get("body") or "").strip()
    heading = f"# {title}\n\n" if title else ""
    return f"{heading}{body}\n"


# ---------------------------------------------------------------------------
# Creating and removing whole entries
# ---------------------------------------------------------------------------


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "untitled"


def create_entry(profile_dir: Path, role: str, name: str) -> str:
    """Add a job or a project. Returns the new file's relative path.

    Both the filename and the id are generated. The id has to be unique across
    experience *and* projects -- they share one namespace -- and every bullet
    added later is namespaced under it.
    """
    if role not in ("experience", "project"):
        raise ProfileWriteError(f"cannot create a {role!r}")

    directory = EXPERIENCE_DIR if role == "experience" else PROJECTS_DIR
    prefix = "exp" if role == "experience" else "prj"
    slug = slugify(name)

    taken = _existing_ids(profile_dir)
    entry_id = f"{prefix}_{slug}"
    suffix = 2
    while entry_id in taken:
        entry_id = f"{prefix}_{slug}_{suffix}"
        suffix += 1

    target = Path(profile_dir) / directory / f"{slug}.yaml"
    count = 2
    while target.exists():
        target = Path(profile_dir) / directory / f"{slug}_{count}.yaml"
        count += 1
    relative = f"{directory}/{target.name}"

    # A minimal entry that already loads: `bullets` has no default, `start` is
    # required, and `type` is fixed by the directory. Written through the same
    # writer as everything else, so it is validated and backed up like any save.
    scaffold: dict[str, Any] = {"id": entry_id, "type": role}
    if role == "experience":
        scaffold.update({"org": name, "title": "", "location": ""})
    else:
        scaffold.update({"name": name})
    scaffold.update({"start": _this_month(), "end": None, "tech": [], "bullets": []})

    yaml = _yaml()
    buffer = io.StringIO()
    yaml.dump(scaffold, buffer)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    try:
        write_profile_file(profile_dir, relative, buffer.getvalue())
    except ProfileWriteError:
        target.unlink(missing_ok=True)
        raise
    return relative


def next_bullet_id(entry_id: str, existing: list[str]) -> str:
    """`<entry id>.b<next free number>`.

    Generated, never typed: `Profile._bullet_ids_are_namespaced` requires the
    prefix, and a hand-edited id is a way to break a file that has no upside.
    """
    used = set(existing)
    index = 1
    while f"{entry_id}.b{index}" in used:
        index += 1
    return f"{entry_id}.b{index}"


def delete_document(profile_dir: Path, relative: str) -> Path:
    """Remove a file, keeping a copy. Returns the backup path."""
    from resume_agent.kb.writer import _back_up  # noqa: PLC0415 - avoids a cycle

    path = resolve_editable_path(profile_dir, relative)
    if not path.is_file():
        raise ProfileWriteError(f"no such file in this profile: {relative}")

    backup = _back_up(Path(profile_dir), path, relative)
    path.unlink()
    try:
        load_profile(Path(profile_dir))
    except Exception as exc:  # noqa: BLE001 - put it back rather than leave a broken profile
        path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        raise ProfileWriteError(f"removing {relative} would break the profile: {exc}") from exc
    return backup


def _existing_ids(profile_dir: Path) -> set[str]:
    try:
        profile = load_profile(Path(profile_dir))
    except Exception:  # noqa: BLE001
        return set()
    return {entry.id for entry in profile.entries()}


def _this_month() -> str:
    from datetime import UTC, datetime  # noqa: PLC0415 - only needed here

    return datetime.now(UTC).strftime("%Y-%m")


