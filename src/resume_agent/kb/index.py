"""The searchable mirror of the knowledge base. Spec 3.3.

    "Load YAML -> validate with Pydantic -> write to SQLite for structured
     queries and the application tracker, and a vector index over
     `canonical + skills + themes` for semantic recall."

Both live in one SQLite file: ordinary tables for the bullets, and a `sqlite-vec`
virtual table for their embeddings. One file, one engine, no server.

**The YAML is the source of truth; this file is a cache.** Everything here is
derived and `.index/` is gitignored, so the index can always be thrown away and
rebuilt. To make that safe automatically, the database records a hash of the
YAML it was built from, and `open()` rebuilds when the profile has moved on.

**On distance metrics.** fastembed returns L2-normalised vectors, and for unit
vectors squared euclidean distance and cosine similarity are the same ordering:

    ||a - b||^2 = 2 - 2*(a . b)

So sqlite-vec's default L2 metric ranks identically to cosine, and the exact
similarity is recoverable as `1 - distance^2 / 2`. That identity is asserted in
the tests, along with a cross-check against a plain numpy dot product -- because
"the vector store silently used the wrong metric" is a failure that looks like
mediocre retrieval rather than an error.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from resume_agent.kb.embeddings import DEFAULT_DIMENSIONS, DEFAULT_MODEL, FastEmbedEmbeddings
from resume_agent.kb.tokenize import tokenize
from resume_agent.models.profile import Bullet, Entry, Profile

INDEX_DIR = Path(".index")

# Bumped whenever the schema or the embedded-text recipe changes, so that an
# index built by an older version is rebuilt rather than silently reused.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexedBullet:
    """A bullet plus everything retrieval and display need, denormalised."""

    bullet_id: str
    entry_id: str
    entry_label: str  # "Halvorsen & Bright R&D" or "LedgerLint"
    entry_type: str  # "experience" | "project"
    embed_text: str
    tokens: list[str]
    bullet: Bullet


def index_path_for(profile_dir: Path, index_dir: Path = INDEX_DIR) -> Path:
    """`profile.example` -> `.index/profile.example.db`.

    Keyed on the directory name so a real `profile/` and the example profile do
    not fight over one database.
    """
    return Path(index_dir) / f"{Path(profile_dir).name}.db"


def profile_content_hash(profile_dir: Path) -> str:
    """A hash of every YAML file in the profile, for staleness detection.

    Filenames are included alongside contents and the list is sorted, so that
    renaming or adding a file changes the hash even when the text does not.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(profile_dir).rglob("*.yaml")):
        digest.update(path.relative_to(profile_dir).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_embed_text(bullet: Bullet) -> str:
    """The text that gets embedded, per spec 3.3: canonical + skills + themes.

    Note what is *not* in here: the parent entry's `tech` list and the employer
    name. Those are contextual rather than claims of the bullet itself, and
    including them would let a bullet match a technology it never mentions --
    which is the retrieval-side version of the fabrication problem M4 guards
    against. First thing to revisit if recall turns out to be too tight.
    """
    parts = [bullet.canonical]
    if bullet.skills:
        parts.append(" ".join(bullet.skills))
    if bullet.themes:
        parts.append(" ".join(bullet.themes))
    return " ".join(parts)


def _entry_label(entry: Entry) -> str:
    return entry.label


def collect_indexed_bullets(profile: Profile) -> list[IndexedBullet]:
    """Flatten the profile into the unit of retrieval: one row per bullet.

    Spec explicitly warns against chunking here -- the bullets are already
    atomic, which is the entire point of the schema in section 3.1.
    """
    rows: list[IndexedBullet] = []
    for entry in profile.entries():
        for bullet in entry.bullets:
            embed_text = build_embed_text(bullet)
            rows.append(
                IndexedBullet(
                    bullet_id=bullet.id,
                    entry_id=entry.id,
                    entry_label=_entry_label(entry),
                    entry_type=entry.type,
                    embed_text=embed_text,
                    tokens=tokenize(embed_text),
                    bullet=bullet,
                )
            )
    return rows


def _serialize_vector(vector: list[float]) -> bytes:
    """sqlite-vec stores float32 vectors as raw little-endian bytes."""
    return struct.pack(f"{len(vector)}f", *vector)


def _connect(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    # sqlite-vec ships as a loadable extension; without this the vec0 virtual
    # table simply does not exist.
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


class ProfileIndex:
    """A built index, open for querying."""

    def __init__(self, connection: sqlite3.Connection, bullets: list[IndexedBullet]) -> None:
        self.connection = connection
        self._bullets = bullets
        self._by_id = {b.bullet_id: b for b in bullets}

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        profile: Profile,
        profile_dir: Path,
        db_path: Path | None = None,
        embeddings: FastEmbedEmbeddings | None = None,
    ) -> ProfileIndex:
        """Embed every bullet and write a fresh database, replacing any existing one."""
        db_path = db_path or index_path_for(profile_dir)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.unlink(missing_ok=True)

        embeddings = embeddings or FastEmbedEmbeddings()
        rows = collect_indexed_bullets(profile)
        vectors = embeddings.embed_documents([row.embed_text for row in rows])

        connection = _connect(db_path)
        with connection:
            connection.executescript(
                """
                CREATE TABLE meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE bullets (
                    rowid       INTEGER PRIMARY KEY,
                    bullet_id   TEXT NOT NULL UNIQUE,
                    entry_id    TEXT NOT NULL,
                    entry_label TEXT NOT NULL,
                    entry_type  TEXT NOT NULL,
                    embed_text  TEXT NOT NULL,
                    tokens      TEXT NOT NULL,
                    canonical   TEXT NOT NULL,
                    confidence  TEXT NOT NULL
                );
                CREATE INDEX bullets_entry_id ON bullets (entry_id);
                """
            )
            # `rowid` is shared between the two tables: vec0 keys on rowid, so
            # this is what joins an embedding back to its bullet.
            connection.execute(
                f"CREATE VIRTUAL TABLE bullet_vectors USING vec0("
                f"embedding float[{DEFAULT_DIMENSIONS}])"
            )

            for rowid, (row, vector) in enumerate(zip(rows, vectors, strict=True), start=1):
                connection.execute(
                    "INSERT INTO bullets (rowid, bullet_id, entry_id, entry_label, entry_type,"
                    " embed_text, tokens, canonical, confidence)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        rowid,
                        row.bullet_id,
                        row.entry_id,
                        row.entry_label,
                        row.entry_type,
                        row.embed_text,
                        json.dumps(row.tokens),
                        row.bullet.canonical,
                        row.bullet.confidence,
                    ),
                )
                connection.execute(
                    "INSERT INTO bullet_vectors (rowid, embedding) VALUES (?, ?)",
                    (rowid, _serialize_vector(vector)),
                )

            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("profile_hash", profile_content_hash(profile_dir)),
                    ("model_name", embeddings.model_name),
                    ("dimensions", str(DEFAULT_DIMENSIONS)),
                ],
            )

        return cls(connection, rows)

    @classmethod
    def open(
        cls,
        profile: Profile,
        profile_dir: Path,
        db_path: Path | None = None,
        *,
        rebuild: bool = False,
        embeddings: FastEmbedEmbeddings | None = None,
    ) -> tuple[ProfileIndex, bool]:
        """Open the index, rebuilding it if missing, stale or from an older schema.

        Returns `(index, was_rebuilt)` so callers can tell the user what happened
        rather than appearing to hang for a second on a silent re-embed.
        """
        db_path = db_path or index_path_for(profile_dir)

        if rebuild or not db_path.is_file():
            return cls.build(profile, profile_dir, db_path, embeddings), True

        connection: sqlite3.Connection | None = None
        try:
            connection = _connect(db_path)
            meta = {row["key"]: row["value"] for row in connection.execute("SELECT * FROM meta")}
        except sqlite3.DatabaseError:
            # A truncated or corrupt file is not worth diagnosing; it is a cache.
            # Closing first is not optional: `build()` unlinks the file, and on
            # Windows an open handle makes that a PermissionError. (On POSIX the
            # unlink would succeed and quietly leak the connection instead.)
            if connection is not None:
                connection.close()
            return cls.build(profile, profile_dir, db_path, embeddings), True

        stale = (
            meta.get("schema_version") != str(SCHEMA_VERSION)
            or meta.get("profile_hash") != profile_content_hash(profile_dir)
            or meta.get("model_name") != (embeddings.model_name if embeddings else DEFAULT_MODEL)
        )
        if stale:
            connection.close()
            return cls.build(profile, profile_dir, db_path, embeddings), True

        return cls(connection, collect_indexed_bullets(profile)), False

    # -- querying -----------------------------------------------------------

    def bullets(self) -> list[IndexedBullet]:
        return list(self._bullets)

    def get(self, bullet_id: str) -> IndexedBullet:
        return self._by_id[bullet_id]

    def dense_search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        """k nearest bullets to `vector`, as `(bullet_id, cosine_similarity)`.

        The stored distance is euclidean; the conversion below is exact because
        both sides are unit vectors (see the module docstring).
        """
        rows = self.connection.execute(
            """
            SELECT b.bullet_id AS bullet_id, v.distance AS distance
            FROM bullet_vectors v
            JOIN bullets b ON b.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(vector), k),
        ).fetchall()
        return [(row["bullet_id"], 1.0 - (row["distance"] ** 2) / 2.0) for row in rows]

    def close(self) -> None:
        self.connection.close()
