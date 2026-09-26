"""Shadow mode (M11 J5): Jev answers beside a checker, and nothing changes.

The two model checkers -- the grounding judge in `verify` and the cover
letter's consistency judge -- decide, always. With shadow mode on, Jev is asked
the same question and both answers are written to a log, so the eval harness
can say how often they agree and list every line Jev would have passed that the
checker refused. That list is the only evidence that could ever justify letting
Jev share the check, and CLAUDE.md rule 1 forbids loosening it without evidence.

**Never on the hosted site.** The log holds career text -- a source achievement
and its rewrite -- so shadow mode is off in multi-user mode whatever the
environment says. The eval harness switches it on, and runs on the made-up
example profile.

On when `RESUME_AGENT_JEV_SHADOW_LOG` names a file, and Jev is on.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from resume_agent.decisions import DecisionsUnavailable, ask, jev_enabled, load_question

logger = logging.getLogger(__name__)

SHADOW_LOG_ENV_VAR = "RESUME_AGENT_JEV_SHADOW_LOG"
# accounts.auth.MULTIUSER_ENV_VAR, read directly: a graph node has no business
# importing the web layer, and the test suite checks the two names agree.
MULTIUSER_ENV_VAR = "RESUME_AGENT_MULTIUSER"

_write = threading.Lock()


def shadow_log() -> Path | None:
    """Where shadow rows go, or None when shadow mode is off."""
    path = os.environ.get(SHADOW_LOG_ENV_VAR, "").strip()
    hosted = os.environ.get(MULTIUSER_ENV_VAR, "").strip().lower() in ("1", "true", "yes")
    if not path or hosted or not jev_enabled():
        return None
    return Path(path)


def shadow(
    kind: str,
    question: str,
    state: dict[str, Any],
    *,
    checker_passed: bool,
    record: dict[str, Any],
) -> None:
    """Ask Jev `decide_<question>` about `state`, beside a checker that said
    `checker_passed`, and log both. Never raises; never changes a verdict."""
    log = shadow_log()
    if log is None:
        return
    try:
        answer = ask(state, {"q": load_question(question)}).answers["q"]
    except DecisionsUnavailable as exc:
        logger.info("shadow: Jev did not answer (%s); nothing recorded", exc)
        return

    row = {"kind": kind, "jev": round(answer.noul, 4), "checker_passed": checker_passed, **record}
    with _write:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")


# -- the report the eval harness prints -----------------------------------------

# "Jev is sure it passes" at each of these. 0.9 is the one the list is made for.
THRESHOLDS = (0.8, 0.9, 0.95)
LIST_AT = 0.9


def summarise(rows: list[dict]) -> dict:
    """What the log says, per checker: agreement, and the disagreements that matter.

    For each threshold T: how many lines Jev was at least T sure pass, how many
    of those the checker *refused* (the number that must be zero before Jev
    could ever skip the checker), and how many checker calls it could have
    skipped. At `LIST_AT`, every refused line is listed in full.
    """
    summary: dict[str, dict] = {}
    for kind in sorted({row["kind"] for row in rows}):
        mine = [row for row in rows if row["kind"] == kind]
        agree = sum((row["jev"] >= 0.5) == row["checker_passed"] for row in mine)
        by_threshold = {}
        for threshold in THRESHOLDS:
            sure = [row for row in mine if row["jev"] >= threshold]
            refused = [row for row in sure if not row["checker_passed"]]
            by_threshold[str(threshold)] = {
                "jev_sure_it_passes": len(sure),
                "checker_refused_those": len(refused),
                "checks_that_could_be_skipped": len(sure) - len(refused),
            }
        summary[kind] = {
            "lines": len(mine),
            "checker_passed": sum(row["checker_passed"] for row in mine),
            "agreement_at_0.5": round(agree / len(mine), 3) if mine else None,
            "by_threshold": by_threshold,
            "refused_but_jev_sure": [
                row for row in mine if row["jev"] >= LIST_AT and not row["checker_passed"]
            ],
        }
    return summary


def render(summary: dict) -> str:
    lines = ["", "shadow: Jev beside the checkers (the checkers decided every line)", ""]
    if not summary:
        return "\n".join([*lines, "  no rows -- was anything checked?", ""])
    for kind, part in summary.items():
        lines.append(
            f"{kind}: {part['lines']} lines, checker passed {part['checker_passed']}, "
            f"agreement at 0.5: {part['agreement_at_0.5']:.0%}"
        )
        for threshold, counts in part["by_threshold"].items():
            lines.append(
                f"  Jev >= {threshold}: sure of {counts['jev_sure_it_passes']}, "
                f"checker refused {counts['checker_refused_those']} of those, "
                f"{counts['checks_that_could_be_skipped']} checks skippable"
            )
        for row in part["refused_but_jev_sure"]:
            text = row.get("rewrite") or row.get("letter") or ""
            lines.append(f"  ! Jev {row['jev']:.2f} but refused: {text[:100]!r}")
            if row.get("reason"):
                lines.append(f"      checker: {row['reason']}")
        lines.append("")
    lines.append(
        "Jev could only ever share a check if 'checker refused' is 0 at the chosen "
        "threshold, across enough lines to mean something -- and with your sign-off."
    )
    return "\n".join(lines) + "\n"
