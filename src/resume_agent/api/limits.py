"""Rate limits for the two public modes. Neither applies when you run this locally.

`RunLimiter` is the public demo's, keyed on address because a demo visitor has
no other identity. `AccountQuota` is multi-user mode's, keyed on the account;
see its docstring for why it counts from the database instead.

A run costs real money against whichever API key the server was started with, so
on a public URL the interesting question is not "can someone break in" -- there
is nothing to break into once `_strip_mutating_routes` has run -- but "can
someone empty the owner's account by clicking a button four hundred times".

**Two limits, because they defend different things.**

`per_ip` keeps one person from hammering it, and is the one a visitor actually
notices. It is not a security control: addresses are cheap, and the header it
reads can be spoofed by anyone willing to try.

`per_day` is the one that protects the wallet, and it is deliberately blunt --
a single global counter with no notion of who. If a hundred people arrive at
once, the hundred-and-first is turned away, and that is the correct outcome for
a demo. A limit that cannot be evaded by changing address is worth more here
than a fair one.

**In-process, like the run registry.** This app is a single process serving a
single demo; a shared store would be inventing a deployment story that does not
exist. It resets when the dyno restarts, which is a real gap and an acceptable
one for something whose worst case is a few extra dollars.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Enough to paste two or three postings and see what the tool does; not enough
# to sit there running it. Overridable so the owner can tune without a deploy.
DEFAULT_PER_IP = 3
DEFAULT_PER_IP_WINDOW_S = 60 * 60
DEFAULT_PER_DAY = 40

PER_IP_ENV_VAR = "RESUME_AGENT_DEMO_RUNS_PER_IP"
PER_DAY_ENV_VAR = "RESUME_AGENT_DEMO_RUNS_PER_DAY"


def _positive_int(name: str, fallback: int) -> int:
    """An unparseable or absurd override falls back rather than disabling the
    limit -- the failure mode of `int(os.environ[...])` here is an unbounded
    bill."""
    raw = os.environ.get(name, "").strip()
    if not raw.isdigit() or int(raw) <= 0:
        return fallback
    return int(raw)


@dataclass
class RunLimiter:
    """How many runs a visitor, and the demo as a whole, may start."""

    per_ip: int = field(default_factory=lambda: _positive_int(PER_IP_ENV_VAR, DEFAULT_PER_IP))
    per_day: int = field(default_factory=lambda: _positive_int(PER_DAY_ENV_VAR, DEFAULT_PER_DAY))
    window_s: int = DEFAULT_PER_IP_WINDOW_S

    _by_ip: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _today: deque[float] = field(default_factory=deque)

    def check(self, ip: str, now: float | None = None) -> str | None:
        """`None` to allow. Otherwise the sentence to show the visitor.

        Records the run as taken when it allows, so the caller must only call
        this once per attempt, immediately before starting the run.
        """
        now = time.time() if now is None else now

        self._prune(self._today, now, 24 * 60 * 60)
        if len(self._today) >= self.per_day:
            return (
                "This demo has hit its limit for today. It runs on the owner's "
                "own API key, so the cap is what keeps it free to try. "
                "Clone the repo to run it without limits."
            )

        seen = self._by_ip[ip]
        self._prune(seen, now, self.window_s)
        if len(seen) >= self.per_ip:
            minutes = max(1, int((self.window_s - (now - seen[0])) // 60))
            return (
                f"You have used your {self.per_ip} runs for the hour. "
                f"Try again in about {minutes} minute{'s' if minutes != 1 else ''}, "
                "or clone the repo to run it without limits."
            )

        seen.append(now)
        self._today.append(now)
        return None

    @staticmethod
    def _prune(stamps: deque[float], now: float, window_s: int) -> None:
        while stamps and now - stamps[0] >= window_s:
            stamps.popleft()

    def remaining_today(self) -> int:
        self._prune(self._today, time.time(), 24 * 60 * 60)
        return max(0, self.per_day - len(self._today))


# ---------------------------------------------------------------------------
# Multi-user mode
# ---------------------------------------------------------------------------

DEFAULT_PER_USER = 5
DEFAULT_CONCURRENT = 1

PER_USER_ENV_VAR = "RESUME_AGENT_RUNS_PER_USER_PER_DAY"
ACCOUNTS_PER_DAY_ENV_VAR = "RESUME_AGENT_RUNS_PER_DAY"
CONCURRENT_ENV_VAR = "RESUME_AGENT_CONCURRENT_RUNS"


@dataclass
class AccountQuota:
    """How many resumes each approved person, and everyone together, may make.

    **The counts come from the accounts database, not from memory.** The demo's
    in-process counter resets whenever the free instance goes to sleep, which on
    a site people visit occasionally is most of the time -- so a limit held in
    memory would be no limit at all. Each run is recorded when it starts, and
    "used today" is a count of rows from the last 24 hours. This class only
    decides, given those counts; it is pure arithmetic so it can be tested as such.

    `concurrent` is not a quota but a memory ceiling: the free instance has
    512 MB, and two runs at once -- each with the embedding model and a LaTeX
    compile -- do not fit. A run over the ceiling waits its turn rather than
    being refused.
    """

    per_user: int = field(
        default_factory=lambda: _positive_int(PER_USER_ENV_VAR, DEFAULT_PER_USER)
    )
    per_day: int = field(
        default_factory=lambda: _positive_int(ACCOUNTS_PER_DAY_ENV_VAR, DEFAULT_PER_DAY)
    )
    concurrent: int = field(
        default_factory=lambda: _positive_int(CONCURRENT_ENV_VAR, DEFAULT_CONCURRENT)
    )

    def refusal(self, used_by_user: int, used_by_everyone: int) -> str | None:
        """`None` to allow; otherwise the sentence to show the person."""
        if used_by_user >= self.per_user:
            return (
                f"You have made your {self.per_user} resumes for today. "
                "You can make more tomorrow."
            )
        if used_by_everyone >= self.per_day:
            return (
                "The site has reached its limit for today -- it runs on the owner's "
                "own API key. Try again tomorrow."
            )
        return None

    def remaining(self, used_by_user: int, used_by_everyone: int) -> int:
        return max(0, min(self.per_user - used_by_user, self.per_day - used_by_everyone))
