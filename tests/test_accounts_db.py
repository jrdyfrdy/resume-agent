"""The multi-user store, against SQLite always and Postgres when one is provided.

Every test runs against both backends through the `db` fixture, so the two
dialects cannot quietly drift apart. Postgres runs only when
`RESUME_AGENT_TEST_DATABASE_URL` is set -- the same pattern as the live model
tests. **Point it at a throwaway database: these tests drop its tables.**
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from resume_agent.accounts.db import AccountsDB, ago

POSTGRES_URL_ENV_VAR = "RESUME_AGENT_TEST_DATABASE_URL"
OWNER = "owner@example.com"


def _backends() -> list:
    backends = [pytest.param("sqlite", id="sqlite")]
    backends.append(pytest.param(
        "postgres",
        id="postgres",
        marks=pytest.mark.skipif(
            not os.environ.get(POSTGRES_URL_ENV_VAR),
            reason=f"set {POSTGRES_URL_ENV_VAR} to a throwaway Postgres to run these",
        ),
    ))
    return backends


@pytest.fixture(params=_backends())
def db(request, tmp_path: Path) -> Iterator[AccountsDB]:
    if request.param == "sqlite":
        store = AccountsDB(f"sqlite:///{tmp_path / 'accounts.db'}")
    else:
        store = AccountsDB(os.environ[POSTGRES_URL_ENV_VAR])
        # A clean slate every test. Children first, for the foreign keys.
        for table in ("runs", "profile_versions", "profile_files", "users"):
            store._run(f"DROP TABLE IF EXISTS {table}")  # noqa: SLF001 - test setup
    store.create_schema()
    yield store
    store.close()


def _friend(db: AccountsDB, n: int = 1):
    return db.sign_in(
        google_sub=f"sub-friend-{n}", email=f"friend{n}@example.com",
        name=f"Friend {n}", admin_email=OWNER,
    )


def _owner(db: AccountsDB):
    return db.sign_in(google_sub="sub-owner", email=OWNER, name="Owner", admin_email=OWNER)


# ===========================================================================
# Signing in, and the approval state it leaves you in
# ===========================================================================


def test_a_new_account_waits_for_approval(db: AccountsDB) -> None:
    user = _friend(db)

    assert user.status == "pending"
    assert not user.approved
    assert not user.is_admin


def test_the_owner_is_approved_on_first_sign_in(db: AccountsDB) -> None:
    """The bootstrap: the very first account has nobody to approve it."""
    user = _owner(db)

    assert user.approved
    assert user.is_admin


def test_the_owner_email_is_matched_case_insensitively(db: AccountsDB) -> None:
    user = db.sign_in(
        google_sub="sub-owner", email="Owner@Example.COM", name="Owner", admin_email=OWNER
    )

    assert user.is_admin
    assert user.email == "owner@example.com"


def test_setting_the_owner_email_later_still_works(db: AccountsDB) -> None:
    """Sign up first, set RESUME_AGENT_ADMIN_EMAIL second: the next sign-in
    promotes you. Otherwise the order of two deploy steps would lock you out."""
    early = db.sign_in(google_sub="sub-owner", email=OWNER, name="Owner", admin_email=None)
    assert early.status == "pending"

    later = db.sign_in(google_sub="sub-owner", email=OWNER, name="Owner", admin_email=OWNER)

    assert later.id == early.id
    assert later.approved and later.is_admin


def test_the_google_sub_is_the_identity_not_the_email(db: AccountsDB) -> None:
    """A friend who changes their address keeps their account and profile."""
    first = _friend(db)
    db.save_profile_changes(first.id, written={"identity.yaml": "name: F"}, deleted=set())

    renamed = db.sign_in(
        google_sub="sub-friend-1", email="new.address@example.com",
        name="Friend 1", admin_email=OWNER,
    )

    assert renamed.id == first.id
    assert renamed.email == "new.address@example.com"
    assert db.profile_files(renamed.id) == {"identity.yaml": "name: F"}


def test_a_returning_friend_stays_pending_until_approved(db: AccountsDB) -> None:
    """Signing in again is not a way around the queue."""
    _friend(db)
    again = _friend(db)

    assert again.status == "pending"


# ===========================================================================
# Deciding
# ===========================================================================


def test_approve_reject_and_revoke(db: AccountsDB) -> None:
    owner, friend = _owner(db), _friend(db)

    approved = db.decide(friend.id, "approved", by=owner.id)
    assert approved and approved.approved and approved.decided_by == owner.id

    revoked = db.decide(friend.id, "rejected", by=owner.id)
    assert revoked and revoked.status == "rejected"


def test_the_list_puts_pending_people_first(db: AccountsDB) -> None:
    owner = _owner(db)
    a, b = _friend(db, 1), _friend(db, 2)
    db.decide(a.id, "approved", by=owner.id)

    order = [user.id for user in db.users()]

    assert order.index(b.id) < order.index(a.id)
    assert order.index(b.id) < order.index(owner.id)


# ===========================================================================
# Profile files -- and that one person cannot see another's
# ===========================================================================


def test_files_are_saved_and_read_back(db: AccountsDB) -> None:
    user = _friend(db)

    db.save_profile_changes(
        user.id,
        written={"identity.yaml": "name: F", "experience/acme.yaml": "id: exp_acme"},
        deleted=set(),
    )

    assert db.profile_files(user.id) == {
        "experience/acme.yaml": "id: exp_acme",
        "identity.yaml": "name: F",
    }


def test_every_save_appends_a_version(db: AccountsDB) -> None:
    """What replaces the backups folder: nothing is overwritten."""
    user = _friend(db)
    for text in ("name: A", "name: B", "name: C"):
        db.save_profile_changes(user.id, written={"identity.yaml": text}, deleted=set())

    history = db.profile_history(user.id, "identity.yaml")

    assert [v["text"] for v in history] == ["name: C", "name: B", "name: A"]
    assert db.profile_files(user.id) == {"identity.yaml": "name: C"}


def test_a_deletion_is_recorded_not_forgotten(db: AccountsDB) -> None:
    user = _friend(db)
    db.save_profile_changes(user.id, written={"projects/x.yaml": "id: prj_x"}, deleted=set())

    db.save_profile_changes(user.id, written={}, deleted={"projects/x.yaml"})

    assert db.profile_files(user.id) == {}
    history = db.profile_history(user.id, "projects/x.yaml")
    assert history[0]["deleted"] is True
    assert history[1]["text"] == "id: prj_x", "the deleted file must stay recoverable"


def test_one_person_cannot_see_anothers_files(db: AccountsDB) -> None:
    a, b = _friend(db, 1), _friend(db, 2)
    db.save_profile_changes(a.id, written={"identity.yaml": "name: A"}, deleted=set())

    assert db.profile_files(b.id) == {}
    assert db.profile_history(b.id, "identity.yaml") == []


# ===========================================================================
# Runs -- the history, the PDF, and the quota counter
# ===========================================================================


def test_a_run_counts_against_the_quota_while_it_is_still_running(db: AccountsDB) -> None:
    """Counting only finished runs would let someone start ten at once."""
    user = _friend(db)
    db.start_run("run-1", user.id)

    assert db.runs_since(ago(24), user_id=user.id) == 1
    assert db.runs_since(ago(24)) == 1


def test_the_quota_counts_one_person_and_everyone_separately(db: AccountsDB) -> None:
    a, b = _friend(db, 1), _friend(db, 2)
    for n in range(3):
        db.start_run(f"a-{n}", a.id)
    db.start_run("b-0", b.id)

    assert db.runs_since(ago(24), user_id=a.id) == 3
    assert db.runs_since(ago(24), user_id=b.id) == 1
    assert db.runs_since(ago(24)) == 4


def test_a_pdf_round_trips_as_bytes(db: AccountsDB) -> None:
    user = _friend(db)
    db.start_run("run-1", user.id)
    pdf = b"%PDF-1.7\n\x00\xff binary payload"

    db.finish_run("run-1", status="done", company="Acme", title="Engineer",
                  overall_fit=0.7, pdf=pdf)

    assert db.run_pdf(user.id, "run-1") == pdf
    assert db.runs(user.id)[0]["has_pdf"] is True
    assert db.runs(user.id)[0]["company"] == "Acme"


def test_anothers_run_is_indistinguishable_from_no_run(db: AccountsDB) -> None:
    """Ownership lives in the WHERE clause. A run id belonging to someone else
    returns exactly what a made-up id does, so ids cannot even be confirmed."""
    a, b = _friend(db, 1), _friend(db, 2)
    db.start_run("run-a", a.id)
    db.finish_run("run-a", status="done", pdf=b"%PDF secret")

    assert db.run_pdf(b.id, "run-a") is None
    assert db.run_pdf(b.id, "never-existed") is None
    assert db.owns_run(b.id, "run-a") is False
    assert db.runs(b.id) == []


# ===========================================================================
# Deleting an account
# ===========================================================================


def test_deleting_an_account_removes_every_trace(db: AccountsDB) -> None:
    """It is their career history. `PRAGMA foreign_keys` is off by default in
    SQLite, and without it this cascade silently does nothing."""
    user = _friend(db)
    db.save_profile_changes(user.id, written={"identity.yaml": "name: F"}, deleted=set())
    db.start_run("run-1", user.id)

    db.delete_user(user.id)

    assert db.user(user.id) is None
    assert db.profile_files(user.id) == {}
    assert db.profile_history(user.id, "identity.yaml") == []
    assert db.runs_since(ago(24)) == 0


def test_an_unknown_database_url_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported DATABASE_URL"):
        AccountsDB("mysql://nope")
