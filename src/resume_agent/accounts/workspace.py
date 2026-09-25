"""Where a user's career file becomes a folder the rest of the code can use.

Everything in `kb/` -- the loader, the validated writer, the comment-preserving
form editor, the search index, the chat's accept path -- operates on a
**directory of YAML files**. In multi-user mode the truth lives in Postgres.
This module is the only bridge between the two, and it is deliberately the only
new mechanism: none of that code changes.

    async with workspaces.profile(user) as profile_dir:
        ...existing kb / forms / chat code, unchanged...
    # on a clean exit: whatever changed in the folder is written to Postgres

**The folder is named after the user's id.** Every store that would otherwise
leak between users is keyed on the profile directory's *name*: the search index
(`.index/<name>.db`), the backups, the chat transcripts, and the profile the API
resolves. Two users each with a folder called `profile` would share one search
index, so one person's achievements could be retrieved into another's resume.
Naming the folder after the user makes all of those per-user by construction.

**Rebuilt from Postgres at the start of every operation.** A copy left on disk
by the last request is never trusted: if a crash landed between a file being
written and it being persisted, the disk would be ahead of the database, and the
next request must not build on that. Rebuilding costs a few milliseconds.

**Persisted by diffing, on a clean exit only.** The folder is read before and
after, and the difference is written. That captures every mutation whichever
function made it -- there is no list of write functions to keep in step with
`kb/`. If the block raises, nothing is persisted, so a half-finished operation
leaves the database exactly as it was.

**A brand-new user has no folder at all, not an empty one.** Profile creation
refuses to overwrite an existing directory, so an empty one would make "Create
my file" fail for exactly the person it is for.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from resume_agent.accounts.db import AccountsDB, User

# What a profile is made of. Anything else under the folder -- there should be
# nothing -- is not career data and is not persisted.
PROFILE_SUFFIXES = {".yaml", ".md"}


class Workspaces:
    """Per-user folders under `root`, kept in step with Postgres."""

    def __init__(self, db: AccountsDB, root: Path) -> None:
        self.db = db
        self.root = Path(root)
        self._locks: dict[str, asyncio.Lock] = {}

    def parent_for(self, user: User) -> Path:
        """The directory `discover_profiles` scans for this user. It holds
        exactly one profile, so the picker can only ever offer their own."""
        return self.root / "users" / user.id

    def folder_for(self, user: User) -> Path:
        return self.parent_for(user) / user.id

    def _lock(self, user_id: str) -> asyncio.Lock:
        # One lock per person. Free Render is a single process, so an
        # in-process lock is a correct way to stop two of one user's saves
        # interleaving; different users never wait on each other.
        return self._locks.setdefault(user_id, asyncio.Lock())

    @asynccontextmanager
    async def profile(self, user: User) -> AsyncIterator[Path]:
        """The user's profile as a folder, for the duration of one operation."""
        folder = self.folder_for(user)
        async with self._lock(user.id):
            files = await asyncio.to_thread(self.db.profile_files, user.id)
            materialize(folder, files)
            before = read_tree(folder)

            yield folder  # an exception here skips everything below: nothing persisted

            written, deleted = diff(before, read_tree(folder))
            if written or deleted:
                await asyncio.to_thread(
                    self.db.save_profile_changes, user.id, written=written, deleted=deleted
                )

    async def snapshot_for_run(self, user: User, run_id: str) -> Path:
        """A private copy for one run to read from.

        A run reads the profile for about a minute. Saving a change in the
        meantime must not alter what it is reading, so it gets its own copy --
        still named after the user, so the search index is shared and only
        rebuilt when the content actually differs.
        """
        folder = self.root / "runs" / run_id / user.id
        async with self._lock(user.id):
            files = await asyncio.to_thread(self.db.profile_files, user.id)
        materialize(folder, files)
        return folder

    def discard_run(self, run_id: str) -> None:
        shutil.rmtree(self.root / "runs" / run_id, ignore_errors=True)


def materialize(folder: Path, files: dict[str, str]) -> None:
    """Make `folder` contain exactly `files` -- and not exist, if there are none."""
    shutil.rmtree(folder, ignore_errors=True)
    if not files:
        folder.parent.mkdir(parents=True, exist_ok=True)
        return
    for relative, text in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # LF, always: `profile_content_hash` hashes raw bytes, so CRLF would make
        # the search index look stale on every request.
        path.write_text(text, encoding="utf-8", newline="\n")


def read_tree(folder: Path) -> dict[str, str]:
    """Every profile file under `folder`, as `{posix relative path: text}`."""
    if not folder.is_dir():
        return {}
    return {
        path.relative_to(folder).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in PROFILE_SUFFIXES
    }


def diff(before: dict[str, str], after: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    """What changed: files created or modified, and files removed."""
    written = {path: text for path, text in after.items() if before.get(path) != text}
    deleted = set(before) - set(after)
    return written, deleted
