"""Durable graph state, so a paused run survives the process that started it.

Spec 5, `human_review`: "Requires a checkpointer and a stable `thread_id`."
Spec 8's M7: "`--interactive` pauses, **survives process restart** via
checkpointer, resumes correctly."

Two decisions carry that requirement.

**SqliteSaver, not InMemorySaver.** The in-memory saver ships with langgraph and
is what every tutorial uses, and it loses everything when the process exits --
which is the exact scenario this milestone exists to handle. A paused run has to
be readable by a *different* invocation of the CLI.

**Constructed directly from a connection, not via `from_conn_string`.** That
helper is a context manager: it closes the saver on exit, which is right for a
script that pauses and resumes inside one process and wrong for a CLI that
pauses, exits, and is run again later. Each invocation opens its own connection
to the same file, which is how restart survival actually works.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from resume_agent.models.job import job_description_hash

# Alongside the retrieval index: both are derived state, both are gitignored,
# and neither is worth backing up.
CHECKPOINT_DB = Path(".index") / "checkpoints.db"

# Every Pydantic model that crosses the checkpointer, named explicitly.
#
# Without this, langgraph warns on every resume that it is "deserializing an
# unregistered type ... this will be blocked in a future version" -- so a future
# upgrade would turn a working `resume-agent resume-run` into a hard failure,
# and only for people who happened to have a run paused at the time.
#
# Listed as (module, class) pairs rather than allowing the whole package,
# because the allowlist is a deserialisation boundary: anything on it can be
# reconstructed from a checkpoint file, and that list should be one somebody
# chose rather than a wildcard.
ALLOWED_CHECKPOINT_TYPES = [
    ("resume_agent.graph.state", "RunOptions"),
    ("resume_agent.models.job", "JobSpec"),
    ("resume_agent.models.job", "JobSpecFields"),
    ("resume_agent.models.job", "Requirement"),
    ("resume_agent.models.fit", "FitReport"),
    ("resume_agent.models.fit", "EvidenceMatch"),
    ("resume_agent.models.fit", "SelectionResult"),
    ("resume_agent.models.resume", "TailoredBullet"),
    ("resume_agent.models.resume", "TailoredBulletFields"),
    ("resume_agent.models.letter", "CoverLetter"),
    ("resume_agent.models.letter", "CoverLetterFields"),
]


def thread_id_for(raw_jd: str, profile_path: str | Path) -> str:
    """The identity of "this posting, against this profile".

    Must be **stable across processes**: `resume-agent resume --jd job.txt` has
    to compute the same id the earlier `run` did, with nothing carried over but
    the arguments. So it is derived from the inputs rather than generated -- a
    uuid would be correct once and useless the moment the process exits.

    Derived from the JD's content hash rather than its path, so moving or
    renaming the file does not orphan a paused run.
    """
    digest = hashlib.sha256()
    digest.update(job_description_hash(raw_jd).encode("utf-8"))
    digest.update(str(Path(profile_path).name).encode("utf-8"))
    return digest.hexdigest()[:24]


def open_checkpointer(path: Path | None = None) -> tuple[SqliteSaver, sqlite3.Connection]:
    """Open the checkpoint store. Returns the saver and the connection to close.

    `check_same_thread=False` because LangGraph may touch the connection from a
    worker thread; the CLI uses one connection for one run, so there is no
    concurrent-write hazard to guard against.

    The connection is returned rather than hidden because somebody has to close
    it, and a saver that silently leaked a file handle on every invocation would
    be a bad neighbour to the tracker database sitting next to it.
    """
    db_path = Path(path) if path else CHECKPOINT_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(
        connection,
        serde=JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES),
    )
    # Idempotent, and required on a fresh database: without it the first write
    # fails on a missing `checkpoints` table.
    saver.setup()
    return saver, connection


def thread_config(thread_id: str) -> dict:
    """The `config` LangGraph needs to associate a run with its checkpoint."""
    return {"configurable": {"thread_id": thread_id}}
