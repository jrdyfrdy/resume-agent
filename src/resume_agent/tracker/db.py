"""The application tracker. Spec 3.3 ("SQLite for ... the application tracker").

Plain `sqlite3` and hand-written SQL rather than an ORM. The table has one
shape, the queries are three, and an ORM would add a dependency and a layer of
indirection to save perhaps twenty lines.

Unlike `.index/` and `.cache/`, this database is **not** derived and **not**
disposable -- it is the only record of what was actually sent where, and the
`outcome` column is data you cannot regenerate. It therefore lives in `out/` by
default rather than in a cache directory, and nothing in this project ever
deletes it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from resume_agent.tracker.models import ApplicationRow

TRACKER_DB = Path("out") / "applications.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS applications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    applied_on          TEXT NOT NULL,
    company             TEXT NOT NULL,
    title               TEXT NOT NULL,
    jd_hash             TEXT NOT NULL,
    run_dir             TEXT NOT NULL,
    recommendation      TEXT,
    overall_fit         REAL,
    page_count          INTEGER,
    cover_letter_words  INTEGER,
    bullet_ids          TEXT NOT NULL DEFAULT '[]',
    dropped_bullets     TEXT NOT NULL DEFAULT '[]',
    outcome             TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS applications_jd_hash ON applications (jd_hash);
CREATE INDEX IF NOT EXISTS applications_outcome ON applications (outcome);
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else TRACKER_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def insert_application(row: ApplicationRow, db_path: Path | None = None) -> int:
    """Record one application. Returns its id."""
    connection = _connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO applications (
                    applied_on, company, title, jd_hash, run_dir, recommendation,
                    overall_fit, page_count, cover_letter_words, bullet_ids,
                    dropped_bullets, outcome, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.applied_on,
                    row.company,
                    row.title,
                    row.jd_hash,
                    row.run_dir,
                    row.recommendation,
                    row.overall_fit,
                    row.page_count,
                    row.cover_letter_words,
                    # The two lists are JSON in a TEXT column. At this scale the
                    # alternative -- a join table -- would be more machinery than
                    # the question "which bullets did I send?" is worth.
                    json.dumps(row.bullet_ids),
                    json.dumps(row.dropped_bullets),
                    row.outcome,
                    row.notes,
                ),
            )
        return int(cursor.lastrowid or 0)
    finally:
        connection.close()


def _to_row(record: sqlite3.Row) -> ApplicationRow:
    data = dict(record)
    data["bullet_ids"] = json.loads(data["bullet_ids"])
    data["dropped_bullets"] = json.loads(data["dropped_bullets"])
    return ApplicationRow.model_validate(data)


def list_applications(
    db_path: Path | None = None, limit: int = 50, outcome: str | None = None
) -> list[ApplicationRow]:
    """Recent applications, newest first."""
    connection = _connect(db_path)
    try:
        if outcome is None:
            records = connection.execute(
                "SELECT * FROM applications ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            records = connection.execute(
                "SELECT * FROM applications WHERE outcome = ? ORDER BY id DESC LIMIT ?",
                (outcome, limit),
            ).fetchall()
        return [_to_row(record) for record in records]
    finally:
        connection.close()


def set_outcome(
    application_id: int, outcome: str, db_path: Path | None = None, notes: str | None = None
) -> bool:
    """Record what came of an application. Returns False if the id is unknown."""
    connection = _connect(db_path)
    try:
        with connection:
            cursor = connection.execute(
                "UPDATE applications SET outcome = ?, notes = COALESCE(?, notes) WHERE id = ?",
                (outcome, notes, application_id),
            )
        return cursor.rowcount > 0
    finally:
        connection.close()


def bullets_by_outcome(db_path: Path | None = None) -> dict[str, dict[str, int]]:
    """How often each bullet appears in applications with each outcome.

    Spec 11's stretch idea, made answerable: "which bullets appear in
    applications that got callbacks?" The answer is only as good as the
    `outcome` column you fill in, which is why the column exists from day one
    rather than being added once there is data worth analysing -- by then the
    data would already be missing.
    """
    counts: dict[str, dict[str, int]] = {}
    for row in list_applications(db_path, limit=10_000):
        if not row.outcome:
            continue
        for bullet_id in row.bullet_ids:
            counts.setdefault(bullet_id, {})
            counts[bullet_id][row.outcome] = counts[bullet_id].get(row.outcome, 0) + 1
    return counts
