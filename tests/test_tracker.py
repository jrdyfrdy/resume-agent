"""The application tracker.

Spec 5 asks `finalize` to "insert a tracker row"; spec 11's stretch idea asks
"which bullets appear in applications that got callbacks?". The second question
is the reason the first exists, and it is only answerable if the row is written
from the very first application rather than once there is data worth analysing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_agent.tracker.db import (
    bullets_by_outcome,
    insert_application,
    list_applications,
    set_outcome,
)
from resume_agent.tracker.models import ApplicationRow


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "applications.db"


def row(**overrides) -> ApplicationRow:
    base = {
        "company": "Acme Corp",
        "title": "Backend Engineer",
        "jd_hash": "abc123",
        "run_dir": "out/acme_backend_2026-09-05",
        "recommendation": "apply",
        "overall_fit": 0.71,
        "page_count": 1,
        "bullet_ids": ["exp_a.b1", "exp_a.b2"],
    }
    base.update(overrides)
    return ApplicationRow(**base)


def test_insert_and_read_back(db: Path) -> None:
    application_id = insert_application(row(), db)
    assert application_id > 0

    stored = list_applications(db)
    assert len(stored) == 1
    assert stored[0].company == "Acme Corp"
    assert stored[0].bullet_ids == ["exp_a.b1", "exp_a.b2"]
    assert stored[0].outcome is None


def test_the_schema_is_created_on_demand(db: Path) -> None:
    """No migration step, no setup command -- the first write makes the table."""
    assert not db.exists()
    insert_application(row(), db)
    assert db.is_file()


def test_newest_first(db: Path) -> None:
    insert_application(row(company="First"), db)
    insert_application(row(company="Second"), db)
    assert [r.company for r in list_applications(db)] == ["Second", "First"]


def test_outcome_starts_empty_and_is_filled_in_later(db: Path) -> None:
    """The correlation this table exists for cannot be computed, only recorded."""
    application_id = insert_application(row(), db)
    assert list_applications(db)[0].outcome is None

    assert set_outcome(application_id, "callback", db) is True
    assert list_applications(db)[0].outcome == "callback"


def test_setting_an_outcome_on_an_unknown_id_is_reported(db: Path) -> None:
    assert set_outcome(999, "callback", db) is False


def test_filter_by_outcome(db: Path) -> None:
    first = insert_application(row(company="Callback Co"), db)
    insert_application(row(company="Silent Co"), db)
    set_outcome(first, "callback", db)

    assert [r.company for r in list_applications(db, outcome="callback")] == ["Callback Co"]


def test_bullets_by_outcome_answers_spec_11s_question(db: Path) -> None:
    """"Which bullets appear in applications that got callbacks?"."""
    good = insert_application(row(bullet_ids=["exp_a.b1", "exp_a.b2"]), db)
    bad = insert_application(row(bullet_ids=["exp_a.b2", "exp_a.b3"]), db)
    set_outcome(good, "callback", db)
    set_outcome(bad, "rejected", db)

    counts = bullets_by_outcome(db)

    assert counts["exp_a.b1"] == {"callback": 1}
    assert counts["exp_a.b2"] == {"callback": 1, "rejected": 1}
    assert counts["exp_a.b3"] == {"rejected": 1}


def test_applications_without_an_outcome_are_excluded_from_the_analysis(db: Path) -> None:
    """An unrecorded outcome is not evidence of anything, in either direction."""
    insert_application(row(bullet_ids=["exp_a.b1"]), db)
    assert bullets_by_outcome(db) == {}


def test_dropped_bullets_are_recorded(db: Path) -> None:
    """What the verifier refused is as interesting as what was sent."""
    insert_application(row(dropped_bullets=["exp_a.b9"]), db)
    assert list_applications(db)[0].dropped_bullets == ["exp_a.b9"]


def test_finalize_writes_a_row(tmp_path: Path, monkeypatch) -> None:
    """Spec 5: "insert a tracker row"."""
    from resume_agent.graph.nodes import finalize as finalize_module
    from resume_agent.graph.state import RunOptions
    from resume_agent.models.job import JobSpec, JobSpecFields

    db = tmp_path / "applications.db"
    monkeypatch.setattr(finalize_module, "insert_application", lambda r: insert_application(r, db))

    job = JobSpec.from_fields(
        JobSpecFields(
            company="Acme Corp",
            title="Backend Engineer",
            seniority="mid",
            domain="payments",
            requirements=[],
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )
    finalize_module.finalize(
        {
            "profile_path": "profile.example",
            "options": RunOptions(out_dir=str(tmp_path)),
            "job_spec": job,
            "tailored": [],
            "selected": [],
            "page_count": 1,
        }
    )

    stored = list_applications(db)
    assert len(stored) == 1
    assert stored[0].company == "Acme Corp"
    assert stored[0].jd_hash == job.source_hash


def test_a_tracker_failure_does_not_fail_the_run(tmp_path: Path, monkeypatch, caplog) -> None:
    """The artifacts are already on disk; losing a bookkeeping row is not fatal.

    It is still logged as an error, because the tracker is the only record that
    survives the run directory being tidied away.
    """
    import logging

    from resume_agent.graph.nodes import finalize as finalize_module
    from resume_agent.graph.state import RunOptions
    from resume_agent.models.job import JobSpec, JobSpecFields

    def explode(_row):
        raise OSError("disk full")

    monkeypatch.setattr(finalize_module, "insert_application", explode)

    job = JobSpec.from_fields(
        JobSpecFields(
            company="Acme",
            title="Engineer",
            seniority="mid",
            domain="d",
            requirements=[],
            responsibilities=[],
            ats_keywords=[],
            culture_signals=[],
            tone="formal",
            red_flags=[],
        ),
        "raw",
    )

    with caplog.at_level(logging.ERROR):
        result = finalize_module.finalize(
            {
                "profile_path": "profile.example",
                "options": RunOptions(out_dir=str(tmp_path)),
                "job_spec": job,
                "tailored": [],
                "selected": [],
            }
        )

    assert result["out_dir"], "a tracker failure took the whole run down"
    assert any("tracker row" in record.getMessage() for record in caplog.records)
