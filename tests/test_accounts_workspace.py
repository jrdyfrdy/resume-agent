"""The workspace: Postgres on one side, the unchanged file-based code on the other.

The claim this module makes is that `kb/` needs no changes to become
multi-user. These tests hold it to that by driving the real loader, writer,
form editor and search index through a workspace, and checking what lands in the
database -- and that one person's data never reaches another's.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from resume_agent.accounts.db import AccountsDB
from resume_agent.accounts.workspace import Workspaces, diff, materialize, read_tree
from resume_agent.kb.forms import create_entry, delete_document, read_document, write_document
from resume_agent.kb.index import ProfileIndex, index_path_for
from resume_agent.kb.loader import load_profile
from resume_agent.kb.retriever import HybridRetriever
from resume_agent.kb.writer import ProfileWriteError, create_empty_profile, write_profile_file

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = REPO_ROOT / "profile.example"
STUDENT = REPO_ROOT / "tests" / "fixtures" / "profile.student"


@pytest.fixture
def db(tmp_path: Path) -> AccountsDB:
    store = AccountsDB(f"sqlite:///{tmp_path / 'accounts.db'}")
    store.create_schema()
    return store


@pytest.fixture
def workspaces(db: AccountsDB, tmp_path: Path) -> Workspaces:
    return Workspaces(db, tmp_path / "workspaces")


def _approved(db: AccountsDB, n: int):
    user = db.sign_in(
        google_sub=f"sub-{n}", email=f"u{n}@example.com", name=f"U{n}", admin_email=None
    )
    return db.decide(user.id, "approved", by="test")


def _seed(db: AccountsDB, user, source: Path) -> None:
    """Load a whole on-disk profile into the database as this user's."""
    db.save_profile_changes(user.id, written=read_tree(source), deleted=set())


def _run(coroutine):
    return asyncio.run(coroutine)


# ===========================================================================
# The small pieces
# ===========================================================================


def test_diff_finds_created_modified_and_deleted() -> None:
    before = {"a.yaml": "1", "b.yaml": "2", "c.yaml": "3"}
    after = {"a.yaml": "1", "b.yaml": "CHANGED", "d.yaml": "new"}

    written, deleted = diff(before, after)

    assert written == {"b.yaml": "CHANGED", "d.yaml": "new"}
    assert deleted == {"c.yaml"}


def test_a_user_with_no_files_gets_no_folder(tmp_path: Path) -> None:
    """Profile creation refuses an existing target, so an *empty* folder would
    make "Create my file" fail for exactly the person it is for."""
    folder = tmp_path / "users" / "u1" / "u1"

    materialize(folder, {})

    assert not folder.exists()
    assert folder.parent.is_dir()


def test_read_tree_ignores_anything_that_is_not_a_profile_file(tmp_path: Path) -> None:
    materialize(tmp_path / "p", {"identity.yaml": "x", "narratives/a.md": "y"})
    (tmp_path / "p" / "stray.tmp").write_text("junk")

    assert set(read_tree(tmp_path / "p")) == {"identity.yaml", "narratives/a.md"}


# ===========================================================================
# The existing code, unchanged, running through a workspace
# ===========================================================================


def test_the_folder_is_named_after_the_user(db, workspaces) -> None:
    """The one idea the design rests on. See the module docstring."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)

    async def go():
        async with workspaces.profile(user) as folder:
            return folder

    folder = _run(go())
    assert folder.name == user.id
    assert folder.parent.name == user.id


def test_the_unchanged_loader_reads_a_workspace(db, workspaces) -> None:
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)

    async def go():
        async with workspaces.profile(user) as folder:
            return load_profile(folder)

    profile = _run(go())
    assert profile.identity.name == "John Doe"
    assert len(profile.all_bullets()) == 13


def test_a_form_save_lands_in_the_database(db, workspaces) -> None:
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)

    async def go():
        async with workspaces.profile(user) as folder:
            document = read_document(folder, "identity.yaml")
            data = dict(document["data"])
            data["location"] = "Lisbon, PT"
            write_document(folder, "identity.yaml", data)

    _run(go())

    assert "Lisbon, PT" in db.profile_files(user.id)["identity.yaml"]
    assert len(db.profile_history(user.id, "identity.yaml")) == 2


def test_creating_and_deleting_an_entry_both_land(db, workspaces) -> None:
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)

    async def add():
        async with workspaces.profile(user) as folder:
            return create_entry(folder, "project", "Weather Station")

    relative = _run(add())
    assert relative in db.profile_files(user.id)

    async def remove():
        async with workspaces.profile(user) as folder:
            delete_document(folder, relative)

    _run(remove())
    assert relative not in db.profile_files(user.id)
    assert db.profile_history(user.id, relative)[0]["deleted"] is True


def test_a_new_user_can_create_their_file(db, workspaces) -> None:
    """The path a friend takes on day one."""
    user = _approved(db, 1)

    async def go():
        async with workspaces.profile(user) as folder:
            create_empty_profile(folder, name="Jane Doe")

    _run(go())

    files = db.profile_files(user.id)
    assert "identity.yaml" in files
    assert "Jane Doe" in files["identity.yaml"]


# ===========================================================================
# The database, not the disk, is the truth
# ===========================================================================


def test_a_failed_operation_persists_nothing(db, workspaces) -> None:
    """If the block raises, the database is left exactly as it was -- even if
    the folder was changed before the failure."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)
    before = db.profile_files(user.id)

    async def go():
        async with workspaces.profile(user) as folder:
            (folder / "identity.yaml").write_text("name: half-finished\n")
            raise RuntimeError("the operation died here")

    with pytest.raises(RuntimeError):
        _run(go())

    assert db.profile_files(user.id) == before


def test_a_refused_save_changes_nothing(db, workspaces) -> None:
    """The writer's own validation still runs, and a refusal is not a change."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)
    before = db.profile_files(user.id)

    async def go():
        async with workspaces.profile(user) as folder:
            with pytest.raises(ProfileWriteError):
                write_profile_file(folder, "identity.yaml", "id: [not valid\n")

    _run(go())
    assert db.profile_files(user.id) == before


def test_a_stale_folder_on_disk_is_never_trusted(db, workspaces) -> None:
    """Something left on disk by an earlier crash must not leak into the next
    operation, let alone into the database."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)
    folder = workspaces.folder_for(user)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "identity.yaml").write_text("name: LEFT OVER FROM A CRASH\n")

    async def go():
        async with workspaces.profile(user) as live:
            return (live / "identity.yaml").read_text()

    assert "LEFT OVER" not in _run(go())
    assert "LEFT OVER" not in db.profile_files(user.id)["identity.yaml"]


# ===========================================================================
# One person, another person
# ===========================================================================


def test_two_people_get_two_folders(db, workspaces) -> None:
    a, b = _approved(db, 1), _approved(db, 2)
    _seed(db, a, EXAMPLE)
    _seed(db, b, STUDENT)

    async def name_of(user):
        async with workspaces.profile(user) as folder:
            return load_profile(folder).identity.name

    assert _run(name_of(a)) == "John Doe"
    assert _run(name_of(b)) == "Jane Doe"


def test_a_shared_folder_name_would_have_shared_the_index() -> None:
    """Why the folders are named after users rather than called `profile`.

    The search index is keyed on the folder's *name*. This is the hazard, stated
    as a fact about the unchanged code; the next test is the fix.
    """
    assert index_path_for(Path("alice/profile")) == index_path_for(Path("bob/profile"))


def test_the_index_cannot_cross_between_users(
    db, workspaces, embeddings, tmp_path: Path, monkeypatch
) -> None:
    """Two people with different careers; each search returns only their own.

    Run through the real index at its real default location, because that
    default is exactly what would have been shared.
    """
    monkeypatch.chdir(tmp_path)  # `.index/` is relative to the working directory
    a, b = _approved(db, 1), _approved(db, 2)
    _seed(db, a, EXAMPLE)
    _seed(db, b, STUDENT)

    async def search(user, query):
        async with workspaces.profile(user) as folder:
            profile = load_profile(folder)
            index, _ = ProfileIndex.open(profile, folder, embeddings=embeddings)
            try:
                hits = HybridRetriever(index, profile, embeddings).search(query, k=8)
                own = {bullet.id for bullet in profile.all_bullets()}
                return {hit.bullet_id for hit in hits}, own, index_path_for(folder)
            finally:
                index.close()

    # The same query for both, broad enough to hit either career.
    a_hits, a_own, a_index = _run(search(a, "built a service and wrote tests"))
    b_hits, b_own, b_index = _run(search(b, "built a service and wrote tests"))

    assert a_index != b_index
    assert a_hits and a_hits <= a_own, "A's search returned bullets that are not A's"
    assert b_hits and b_hits <= b_own, "B's search returned bullets that are not B's"
    assert not (a_hits & b_own)
    assert not (b_hits & a_own)


def test_one_persons_save_never_reaches_another(db, workspaces) -> None:
    a, b = _approved(db, 1), _approved(db, 2)
    _seed(db, a, EXAMPLE)
    _seed(db, b, EXAMPLE)  # identical content, so only the id tells them apart

    async def edit(user):
        async with workspaces.profile(user) as folder:
            text = (folder / "identity.yaml").read_text()
            write_profile_file(folder, "identity.yaml", text.replace("John Doe", "Alice"))

    _run(edit(a))

    assert "Alice" in db.profile_files(a.id)["identity.yaml"]
    assert "Alice" not in db.profile_files(b.id)["identity.yaml"]


# ===========================================================================
# Runs read from their own copy
# ===========================================================================


def test_a_run_is_not_affected_by_a_save_made_during_it(db, workspaces) -> None:
    """A run reads the profile for about a minute. What it reads must be what
    was there when it started."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)

    snapshot = _run(workspaces.snapshot_for_run(user, "run-1"))

    async def edit():
        async with workspaces.profile(user) as folder:
            text = (folder / "identity.yaml").read_text()
            write_profile_file(folder, "identity.yaml", text.replace("John Doe", "Renamed"))

    _run(edit())

    assert load_profile(snapshot).identity.name == "John Doe"
    assert snapshot.name == user.id, "named after the user, so the index is shared"

    workspaces.discard_run("run-1")
    assert not snapshot.exists()


def test_one_persons_operations_run_one_at_a_time(db, workspaces) -> None:
    """Two of one person's saves must not interleave: each materializes,
    edits and persists as a unit."""
    user = _approved(db, 1)
    _seed(db, user, EXAMPLE)
    order: list[str] = []

    async def op(tag: str):
        async with workspaces.profile(user):
            order.append(f"{tag}-in")
            await asyncio.sleep(0.02)
            order.append(f"{tag}-out")

    async def both():
        await asyncio.gather(op("a"), op("b"))

    _run(both())
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])
