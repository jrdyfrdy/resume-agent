"""`resume-agent users`: approving people when the web page is the broken part."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from resume_agent.accounts.db import AccountsDB
from resume_agent.cli import ExitCode, app

runner = CliRunner()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AccountsDB:
    url = f"sqlite:///{tmp_path / 'accounts.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    db = AccountsDB(url)
    db.create_schema()
    db.sign_in(google_sub="1", email="owner@example.com", name="Owner",
               admin_email="owner@example.com")
    db.sign_in(google_sub="2", email="alex@example.com", name="Alex", admin_email=None)
    return db


def test_list_shows_who_is_waiting(db) -> None:
    result = runner.invoke(app, ["users", "list"])

    assert result.exit_code == 0, result.output
    lines = [line for line in result.output.splitlines() if "@" in line]
    assert "alex@example.com" in lines[0], "the people waiting come first"
    assert "pending" in lines[0]
    assert "1 waiting" in result.output


def test_approve_and_reject_by_email(db) -> None:
    assert runner.invoke(app, ["users", "approve", "Alex@Example.com"]).exit_code == 0
    alex = db.user_by_email("alex@example.com")
    assert alex.status == "approved"
    assert alex.decided_by == "cli"

    assert runner.invoke(app, ["users", "reject", "alex@example.com"]).exit_code == 0
    assert db.user_by_email("alex@example.com").status == "rejected"


def test_an_unknown_email_says_they_must_sign_in_first(db) -> None:
    result = runner.invoke(app, ["users", "approve", "nobody@example.com"])

    assert result.exit_code == ExitCode.NO_SUCH_USER
    assert "sign in once" in result.output


def test_it_needs_only_the_database_url(monkeypatch) -> None:
    """No Google or session settings: this works when the web app cannot start."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = runner.invoke(app, ["users", "list"])

    assert result.exit_code == ExitCode.USAGE
    assert "DATABASE_URL" in result.output
