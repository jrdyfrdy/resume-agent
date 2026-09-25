"""Everything the multi-user mode stores, and every query that touches it.

**Two backends, one set of queries.** SQLite for tests and local development,
Postgres (Neon) when deployed. The SQL is written once, with `?` placeholders,
and translated to psycopg's `%s` on the way out. There are four tables and about
fifteen queries; an ORM would be more code than the thing it abstracts.

**Why every query lives in this file.** Isolation between users is a property of
the `WHERE user_id = ?` clauses, and it is much easier to check that property
when every one of them is on one screen. Nothing outside this module writes
SQL.

**One connection per process, behind a lock.** Free Render is a single process,
and Neon's free tier limits connections, so a pool would be managing contention
that does not exist. The lock makes the connection safe to use from the worker
threads that `asyncio.to_thread` hands these calls to. Neon closes idle
connections after a few minutes -- it scales to zero -- so a broken connection
is reopened once and the query retried, rather than surfacing to a user as a
500 on their first click of the morning.

**Timestamps are ISO-8601 text in UTC.** They sort lexicographically in both
databases, compare with plain `>=`, and avoid the two dialects' different ideas
of what a timestamp type is.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

# How long a connection may sit unused before it is pinged before use. Well
# under Neon's five-minute suspend, so a suspended compute is always noticed.
_IDLE_CHECK_S = 30

Dialect = Literal["sqlite", "postgres"]
Status = Literal["pending", "approved", "rejected"]

# The schema, written once. `{serial_pk}` and `{bytes}` are the only two places
# the dialects disagree about a type.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    google_sub  TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending',
    is_admin    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TEXT NOT NULL,
    decided_at  TEXT,
    decided_by  TEXT
);

CREATE TABLE IF NOT EXISTS profile_files (
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, path)
);

CREATE TABLE IF NOT EXISTS profile_versions (
    id          {serial_pk},
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    deleted     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS profile_versions_by_path
    ON profile_versions (user_id, path, created_at);

CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    company      TEXT,
    title        TEXT,
    overall_fit  REAL,
    pdf          {bytes}
);
CREATE INDEX IF NOT EXISTS runs_by_user ON runs (user_id, created_at);
"""

_TYPES: dict[Dialect, dict[str, str]] = {
    "sqlite": {"serial_pk": "INTEGER PRIMARY KEY AUTOINCREMENT", "bytes": "BLOB"},
    "postgres": {"serial_pk": "BIGSERIAL PRIMARY KEY", "bytes": "BYTEA"},
}


def now() -> str:
    """UTC, ISO-8601, microsecond precision. Sortable as text."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def ago(hours: float) -> str:
    """The timestamp `hours` before now, in the same sortable form."""
    return (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class User:
    id: str
    google_sub: str
    email: str
    name: str
    status: Status
    is_admin: bool
    created_at: str
    decided_at: str | None
    decided_by: str | None

    @property
    def approved(self) -> bool:
        return self.status == "approved"


class AccountsDB:
    """The whole multi-user store. See the module docstring."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.dialect: Dialect = _dialect_of(url)
        self._lock = threading.Lock()
        self._conn: Any = None
        self._last_used = 0.0

    # -- connection ----------------------------------------------------------

    def _open(self) -> Any:
        if self.dialect == "sqlite":
            path = self.url.removeprefix("sqlite:///")
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            # One connection used from several worker threads, always under
            # `self._lock`, so the same-thread check would only get in the way.
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Off by default in SQLite, per connection. Without it
            # `ON DELETE CASCADE` is silently ignored and deleting an account
            # would leave its career data behind.
            conn.execute("PRAGMA foreign_keys = ON")
            return conn

        import psycopg  # noqa: PLC0415 - only the deployed mode needs it
        from psycopg.rows import dict_row  # noqa: PLC0415

        # `prepare_threshold=None`: psycopg otherwise turns a query it has run
        # five times into a server-side prepared statement, and behind a
        # transaction-mode pooler -- Neon's `-pooler` connection string is
        # PgBouncer -- the next query can land on a connection that never saw
        # it. At a dozen queries a request, preparing saves nothing worth that.
        return psycopg.connect(self.url, row_factory=dict_row, prepare_threshold=None)

    def _live(self) -> Any:
        """A connection that is known to work.

        Neon suspends idle compute, and when it does the connection dies
        without psycopg noticing -- `closed` stays False until a query fails.
        So a connection idle for longer than `_IDLE_CHECK_S` is pinged first,
        and reopened if the ping fails. The cost is one round trip after a
        quiet spell; the alternative is a 500 on a friend's first click of the
        morning.
        """
        stale = time.monotonic() - self._last_used > _IDLE_CHECK_S
        if self._conn is not None and self.dialect == "postgres" and stale:
            try:
                self._conn.execute("SELECT 1")
            except Exception:  # noqa: BLE001 - any failure means reconnect
                self._conn = None
        if self._conn is None or getattr(self._conn, "closed", False):
            self._conn = self._open()
        self._last_used = time.monotonic()
        return self._conn

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """A cursor inside one transaction: committed on success, rolled back
        on any exception. Every write in this module goes through here, so a
        save that fails halfway leaves nothing half-written."""
        with self._lock:
            conn = self._live()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.dialect == "postgres" else statement

    def _rows(self, statement: str, params: tuple = ()) -> list[dict]:
        with self._transaction() as cursor:
            cursor.execute(self._sql(statement), params)
            return [dict(row) for row in cursor.fetchall()]

    def _one(self, statement: str, params: tuple = ()) -> dict | None:
        rows = self._rows(statement, params)
        return rows[0] if rows else None

    def _run(self, statement: str, params: tuple = ()) -> None:
        with self._transaction() as cursor:
            cursor.execute(self._sql(statement), params)

    def create_schema(self) -> None:
        ddl = _SCHEMA.format(**_TYPES[self.dialect])
        with self._transaction() as cursor:
            # One statement at a time: psycopg will not run several in one call.
            for statement in (s.strip() for s in ddl.split(";")):
                if statement:
                    cursor.execute(statement)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- users ---------------------------------------------------------------

    def sign_in(
        self, *, google_sub: str, email: str, name: str, admin_email: str | None
    ) -> User:
        """Create or refresh the account behind a verified Google identity.

        Keyed on `google_sub`, Google's stable id for the person. An email can
        change; the sub cannot, so a friend who renames their address keeps
        their profile.

        New accounts are `pending`. The one exception is `admin_email`, which is
        approved and made admin on every sign-in: it is the bootstrap for the
        first account, which has nobody to approve it, and re-checking it each
        time means setting the variable after signing up still works.
        """
        email = email.strip().lower()
        is_owner = bool(admin_email) and email == admin_email.strip().lower()

        existing = self._one("SELECT * FROM users WHERE google_sub = ?", (google_sub,))
        if existing is None:
            user_id = uuid.uuid4().hex
            self._run(
                "INSERT INTO users (id, google_sub, email, name, status, is_admin, "
                "created_at, decided_at, decided_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id, google_sub, email, name,
                    "approved" if is_owner else "pending",
                    is_owner, now(),
                    now() if is_owner else None,
                    "bootstrap" if is_owner else None,
                ),
            )
        else:
            user_id = existing["id"]
            self._run(
                "UPDATE users SET email = ?, name = ? WHERE id = ?", (email, name, user_id)
            )
            if is_owner and not (existing["is_admin"] and existing["status"] == "approved"):
                self._run(
                    "UPDATE users SET status = 'approved', is_admin = ?, decided_at = ?, "
                    "decided_by = 'bootstrap' WHERE id = ?",
                    (True, now(), user_id),
                )

        user = self.user(user_id)
        assert user is not None  # just written
        return user

    def user(self, user_id: str) -> User | None:
        row = self._one("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(row) if row else None

    def user_by_email(self, email: str) -> User | None:
        row = self._one("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
        return _user(row) if row else None

    def users(self) -> list[User]:
        """Everyone, pending first -- they are the ones waiting on a decision."""
        rows = self._rows(
            "SELECT * FROM users ORDER BY CASE status WHEN 'pending' THEN 0 "
            "WHEN 'approved' THEN 1 ELSE 2 END, created_at"
        )
        return [_user(row) for row in rows]

    def decide(self, user_id: str, status: Status, *, by: str) -> User | None:
        """Approve, reject, or revoke (which is rejecting someone approved)."""
        self._run(
            "UPDATE users SET status = ?, decided_at = ?, decided_by = ? WHERE id = ?",
            (status, now(), by, user_id),
        )
        return self.user(user_id)

    def delete_user(self, user_id: str) -> None:
        """Remove an account and, by cascade, every file, version and run."""
        self._run("DELETE FROM users WHERE id = ?", (user_id,))

    # -- profile files -------------------------------------------------------

    def profile_files(self, user_id: str) -> dict[str, str]:
        """The user's career file, as `{relative path: text}`."""
        rows = self._rows(
            "SELECT path, text FROM profile_files WHERE user_id = ? ORDER BY path",
            (user_id,),
        )
        return {row["path"]: row["text"] for row in rows}

    def save_profile_changes(
        self, user_id: str, *, written: dict[str, str], deleted: set[str]
    ) -> None:
        """Apply one operation's changes, in one transaction.

        Every change also appends to `profile_versions`, which is what replaces
        the backups folder: nothing is overwritten, so any earlier state of any
        file can be recovered. And it survives a restart, where a backups folder
        on free Render's disk would not.
        """
        stamp = now()
        with self._transaction() as cursor:
            for path, text in sorted(written.items()):
                if self.dialect == "postgres":
                    upsert = (
                        "INSERT INTO profile_files (user_id, path, text, updated_at) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT (user_id, path) "
                        "DO UPDATE SET text = EXCLUDED.text, updated_at = EXCLUDED.updated_at"
                    )
                else:
                    upsert = (
                        "INSERT INTO profile_files (user_id, path, text, updated_at) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT (user_id, path) "
                        "DO UPDATE SET text = excluded.text, updated_at = excluded.updated_at"
                    )
                cursor.execute(self._sql(upsert), (user_id, path, text, stamp))
                cursor.execute(
                    self._sql(
                        "INSERT INTO profile_versions (user_id, path, text, deleted, "
                        "created_at) VALUES (?, ?, ?, ?, ?)"
                    ),
                    (user_id, path, text, False, stamp),
                )
            for path in sorted(deleted):
                cursor.execute(
                    self._sql("DELETE FROM profile_files WHERE user_id = ? AND path = ?"),
                    (user_id, path),
                )
                cursor.execute(
                    self._sql(
                        "INSERT INTO profile_versions (user_id, path, text, deleted, "
                        "created_at) VALUES (?, ?, ?, ?, ?)"
                    ),
                    (user_id, path, "", True, stamp),
                )

    def profile_history(self, user_id: str, path: str, limit: int = 20) -> list[dict]:
        """Earlier versions of one file, newest first."""
        rows = self._rows(
            "SELECT text, deleted, created_at FROM profile_versions "
            "WHERE user_id = ? AND path = ? ORDER BY id DESC LIMIT ?",
            (user_id, path, limit),
        )
        return [{**row, "deleted": bool(row["deleted"])} for row in rows]

    # -- runs ----------------------------------------------------------------

    def start_run(self, run_id: str, user_id: str) -> None:
        """Recorded when the run *starts*, so a run still in flight counts
        against the quota. Counting only finished runs would let someone start
        ten at once."""
        self._run(
            "INSERT INTO runs (id, user_id, created_at, status) VALUES (?, ?, ?, ?)",
            (run_id, user_id, now(), "running"),
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        company: str | None = None,
        title: str | None = None,
        overall_fit: float | None = None,
        pdf: bytes | None = None,
    ) -> None:
        self._run(
            "UPDATE runs SET status = ?, company = ?, title = ?, overall_fit = ?, pdf = ? "
            "WHERE id = ?",
            (status, company, title, overall_fit, pdf, run_id),
        )

    def runs(self, user_id: str, limit: int = 50) -> list[dict]:
        """One user's runs, newest first -- the History tab. No PDF bytes."""
        rows = self._rows(
            "SELECT id, created_at, status, company, title, overall_fit, "
            "CASE WHEN pdf IS NULL THEN 0 ELSE 1 END AS has_pdf "
            "FROM runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [{**row, "has_pdf": bool(row["has_pdf"])} for row in rows]

    def run_pdf(self, user_id: str, run_id: str) -> bytes | None:
        """Ownership is in the WHERE clause: another user's run id returns
        nothing at all, exactly as an id that never existed would."""
        row = self._one(
            "SELECT pdf FROM runs WHERE id = ? AND user_id = ?", (run_id, user_id)
        )
        if row is None or row["pdf"] is None:
            return None
        return bytes(row["pdf"])  # psycopg returns a memoryview for BYTEA

    def owns_run(self, user_id: str, run_id: str) -> bool:
        return self._one(
            "SELECT 1 AS found FROM runs WHERE id = ? AND user_id = ?", (run_id, user_id)
        ) is not None

    def runs_since(self, since: str, *, user_id: str | None = None) -> int:
        """How many runs started since `since` -- one user's, or everyone's."""
        if user_id is None:
            row = self._one("SELECT COUNT(*) AS n FROM runs WHERE created_at >= ?", (since,))
        else:
            row = self._one(
                "SELECT COUNT(*) AS n FROM runs WHERE user_id = ? AND created_at >= ?",
                (user_id, since),
            )
        return int(row["n"]) if row else 0


def _dialect_of(url: str) -> Dialect:
    if url.startswith("sqlite:///"):
        return "sqlite"
    if url.startswith(("postgresql://", "postgres://")):
        return "postgres"
    raise ValueError(
        f"unsupported DATABASE_URL {url.split('://')[0]!r}: "
        "expected sqlite:///path or postgresql://..."
    )


def _user(row: dict) -> User:
    return User(
        id=row["id"],
        google_sub=row["google_sub"],
        email=row["email"],
        name=row["name"],
        status=row["status"],
        is_admin=bool(row["is_admin"]),
        created_at=row["created_at"],
        decided_at=row["decided_at"],
        decided_by=row["decided_by"],
    )
