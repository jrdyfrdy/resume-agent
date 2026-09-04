"""On-disk cache for parsed job descriptions. Spec 5.

    "Cache on sha256(raw_jd) -- you will re-run this graph many times on the same
     JD while debugging, and you shouldn't pay for it twice."

**Deviation from the spec, deliberately.** The key here is
`sha256(raw_jd + model + prompt_version)`, not the JD alone. Keying on the JD
alone means that editing `prompts/parse_jd.md` serves the old parse forever, and
prompt iteration is the thing this project is built to practise -- a caching
scheme that hides prompt changes would quietly defeat the exercise. Including
the model id has the same motivation: comparing Opus against Sonnet on the same
posting has to produce two cache entries, not one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from resume_agent.models.job import JobSpec, job_description_hash

CACHE_DIR = Path(".cache") / "jd"


def jd_cache_key(raw_jd: str, model: str, prompt_version: str) -> str:
    """The identity of "this posting, parsed by this model, with this prompt"."""
    digest = hashlib.sha256()
    digest.update(job_description_hash(raw_jd).encode("utf-8"))
    digest.update(model.encode("utf-8"))
    digest.update(prompt_version.encode("utf-8"))
    return digest.hexdigest()[:24]


class JobSpecCache:
    """A directory of parsed postings, one JSON file per cache key."""

    def __init__(self, cache_dir: Path = CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> JobSpec | None:
        """Return the cached parse, or None if absent or unreadable.

        A cache miss and a corrupt entry are the same event to the caller: parse
        it again. This is a cache, so a bad file is never worth an exception.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None
        try:
            return JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError):
            return None

    def put(self, key: str, spec: JobSpec) -> None:
        """Write a parse to the cache.

        Written via a temporary file and an atomic replace: an interrupted write
        would otherwise leave a truncated JSON file that `get` has to treat as a
        miss, silently costing another API call every run.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(path)
