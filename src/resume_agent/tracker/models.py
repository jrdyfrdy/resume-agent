"""One row per application sent. Spec 5, `finalize`: "insert a tracker row".

    "That `run.json` is your future dataset. Six months from now you'll want to
     ask 'which bullets appear in applications that got callbacks?' -- you can
     only answer that if you logged it from the start."

`run.json` records one run in its own directory. This table is the index across
all of them, which is what makes that question answerable with a query instead
of a filesystem walk.

`outcome` starts empty and is filled in by hand, later, when you hear back. That
is the whole point: the correlation between what you sent and what came of it
cannot be computed, only recorded.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

# Free text on purpose rather than an enum. Real outcomes are messier than any
# list drawn up in advance -- "recruiter screen", "ghosted", "rejected after
# take-home" -- and a schema that forces them into four buckets loses the
# detail that makes the record worth keeping.
SUGGESTED_OUTCOMES = ["callback", "rejected", "ghosted", "offer", "withdrawn"]


class ApplicationRow(BaseModel):
    """One application, as stored."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None  # assigned by SQLite
    applied_on: str = Field(default_factory=lambda: date.today().isoformat())

    company: str
    title: str
    # Ties the row back to the posting, so two applications to the same job are
    # recognisable as such even if the file was renamed.
    jd_hash: str
    run_dir: str

    recommendation: str | None = None
    overall_fit: float | None = None
    page_count: int | None = None
    cover_letter_words: int | None = None

    # The two lists spec 11's stretch idea needs: what was sent, and what the
    # verifier refused to let through.
    bullet_ids: list[str] = Field(default_factory=list)
    dropped_bullets: list[str] = Field(default_factory=list)

    outcome: str | None = None
    notes: str | None = None
