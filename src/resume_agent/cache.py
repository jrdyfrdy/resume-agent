"""Disk cache for the expensive nodes. Spec 12.

    "Cache JD parses by hash. `tailor_bullets` is the expensive node; batch it
     by section. A full run should land in the low tens of cents. If it doesn't,
     you're re-parsing something you already parsed."

M2 cached JD parses. This generalises the same pattern -- atomic write, corrupt
entry treated as a miss, key includes the prompt version -- to the two nodes
that dominate the bill: `score_fit` and `tailor_bullets`.

The reason this matters more than the money: M5's layout loop re-runs selection
and rendering several times per posting, and without caching, every iteration of
a loop you are *debugging* pays full price for scoring that has not changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

CACHE_DIR = Path(".cache")
CACHE_DIR_ENV_VAR = "RESUME_AGENT_CACHE_DIR"

T = TypeVar("T", bound=BaseModel)


def default_cache_dir() -> Path:
    """Where cached model output lives.

    Resolved on every call rather than bound as a default argument, so that
    setting `RESUME_AGENT_CACHE_DIR` takes effect -- a default argument would be
    captured once at import and ignore the environment. The test suite relies on
    this to keep each run's cache in a temp directory: without it, one test's
    cached result silently satisfies the next test's "did it call the model?"
    assertion.
    """
    override = os.environ.get(CACHE_DIR_ENV_VAR)
    return Path(override) if override else CACHE_DIR


def content_key(*parts: str) -> str:
    """A stable short key from any number of identifying strings.

    Every caller passes the prompt version and the model id among the parts, so
    that editing a prompt or switching models invalidates rather than silently
    serving output the current configuration would not produce.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")  # separator, so ("ab","c") != ("a","bc")
    return digest.hexdigest()[:24]


class ModelListCache:
    """Caches a list of Pydantic models as one JSON file per key."""

    def __init__(self, model_type: type[T], namespace: str, cache_dir: Path | None = None) -> None:
        self.model_type = model_type
        self.cache_dir = (Path(cache_dir) if cache_dir else default_cache_dir()) / namespace

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> list[T] | None:
        """The cached list, or None if absent or unreadable.

        A corrupt entry is a miss, not an error -- this is a cache, and the only
        useful response to a bad file is to recompute.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [self.model_type.model_validate(item) for item in raw]
        except (OSError, ValidationError, json.JSONDecodeError):
            return None

    def put(self, key: str, items: list[T]) -> None:
        """Write a list to the cache, atomically.

        Temp file plus replace: an interrupted write would otherwise leave
        truncated JSON that `get` reads as a miss, quietly costing a full call
        on every subsequent run.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps([item.model_dump() for item in items], indent=2), encoding="utf-8"
        )
        temp_path.replace(path)
