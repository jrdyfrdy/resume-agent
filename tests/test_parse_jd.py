"""JD parsing: the node, the cache, and the snapshots.

**Nothing here spends money by default.** The happy path runs against a fake
chat model (CLAUDE.md working style: "every node gets at least a happy-path test
with a fake LLM"), and the snapshot tests validate committed JSON. The one test
that would call the API is marked `llm` and skips without credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from resume_agent.graph.nodes.parse_jd import (
    PROMPT_NAME,
    JobDescriptionParseError,
    parse_job_description,
)
from resume_agent.jd_cache import JobSpecCache, jd_cache_key
from resume_agent.llm import has_credentials, load_prompt, model_for, prompt_version
from resume_agent.models.job import JobSpec, JobSpecFields, job_description_hash

REPO_ROOT = Path(__file__).resolve().parent.parent
JD_DIR = REPO_ROOT / "evals" / "datasets" / "jds"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots" / "jd"

JD_LEVELS = ["junior", "mid", "senior"]

requires_llm = pytest.mark.skipif(
    not has_credentials(), reason="no model credentials set -- skipping live model call"
)


# --- a fake model that returns a fixed JobSpecFields ------------------------


SAMPLE_FIELDS = JobSpecFields(
    company="Brightloom",
    title="Junior Software Engineer (Backend)",
    seniority="junior",
    domain="small-business accounting software",
    requirements=[
        {
            "text": "Strong Python",
            "category": "language",
            "weight": 5,
            "is_must_have": True,
            "is_inferred": False,
        },
        {
            "text": "Comfort carrying an on-call pager",
            "category": "practice",
            "weight": 3,
            "is_must_have": False,
            "is_inferred": True,
        },
    ],
    responsibilities=["Build backend services in Python and FastAPI"],
    ats_keywords=["Python", "FastAPI", "PostgreSQL", "CI/CD"],
    culture_signals=["small team", "ship on Fridays"],
    tone="startup",
    red_flags=["16 requirements for a role asking 2+ years of experience"],
)


class FakeStructuredModel(BaseChatModel):
    """A chat model whose `with_structured_output` returns a canned object.

    Written by hand rather than using a LangChain fake because the fakes return
    messages, and the code under test goes through `with_structured_output` --
    which is the part whose contract matters here.
    """

    fields: JobSpecFields = SAMPLE_FIELDS
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-structured"

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError("this fake is only used via with_structured_output")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        fake = self

        class _Runnable:
            def invoke(self, _messages: Any, **_kw: Any) -> JobSpecFields:
                fake.call_count += 1
                return fake.fields

        return _Runnable()


@pytest.fixture
def fake_llm() -> FakeStructuredModel:
    return FakeStructuredModel()


@pytest.fixture
def cache(tmp_path: Path) -> JobSpecCache:
    return JobSpecCache(tmp_path / "jd")


# --- the node ---------------------------------------------------------------


def test_parses_into_a_jobspec(fake_llm: FakeStructuredModel, cache: JobSpecCache) -> None:
    spec, hit = parse_job_description("Some posting text", llm=fake_llm, cache=cache)
    assert isinstance(spec, JobSpec)
    assert hit is False
    assert spec.company == "Brightloom"
    assert spec.seniority == "junior"


def test_source_hash_is_computed_not_generated(
    fake_llm: FakeStructuredModel, cache: JobSpecCache
) -> None:
    """CLAUDE.md rule 2. The model never sees this field.

    `SAMPLE_FIELDS` has no `source_hash` at all -- if it appeared in the LLM's
    schema, this test could not exist.
    """
    raw = "Some posting text"
    spec, _ = parse_job_description(raw, llm=fake_llm, cache=cache)
    assert spec.source_hash == job_description_hash(raw)
    assert "source_hash" not in JobSpecFields.model_fields


def test_empty_jd_is_rejected(fake_llm: FakeStructuredModel, cache: JobSpecCache) -> None:
    with pytest.raises(JobDescriptionParseError, match="empty"):
        parse_job_description("   \n  ", llm=fake_llm, cache=cache)
    assert fake_llm.call_count == 0, "spent a call on an empty posting"


def test_stated_and_inferred_are_separable(
    fake_llm: FakeStructuredModel, cache: JobSpecCache
) -> None:
    """The distinction spec 5 asks for has to survive into the data."""
    spec, _ = parse_job_description("Some posting", llm=fake_llm, cache=cache)
    assert [r.text for r in spec.stated_requirements()] == ["Strong Python"]
    assert [r.text for r in spec.inferred_requirements()] == ["Comfort carrying an on-call pager"]


# --- caching: spec 5 and the second half of M2's DoD ------------------------


def test_second_run_is_a_cache_hit(fake_llm: FakeStructuredModel, cache: JobSpecCache) -> None:
    """M2 DoD: "cache hit on second run"."""
    raw = "Some posting text"

    first, hit_first = parse_job_description(raw, llm=fake_llm, cache=cache)
    second, hit_second = parse_job_description(raw, llm=fake_llm, cache=cache)

    assert hit_first is False
    assert hit_second is True
    assert first == second
    assert fake_llm.call_count == 1, "the cached run still called the model"


def test_no_cache_forces_a_reparse(fake_llm: FakeStructuredModel, cache: JobSpecCache) -> None:
    raw = "Some posting text"
    parse_job_description(raw, llm=fake_llm, cache=cache)
    _, hit = parse_job_description(raw, llm=fake_llm, cache=cache, use_cache=False)
    assert hit is False
    assert fake_llm.call_count == 2


def test_whitespace_only_changes_still_hit_the_cache(
    fake_llm: FakeStructuredModel, cache: JobSpecCache
) -> None:
    """A posting re-copied with different trailing spaces is the same posting."""
    parse_job_description("line one\nline two", llm=fake_llm, cache=cache)
    _, hit = parse_job_description("line one   \nline two  \n", llm=fake_llm, cache=cache)
    assert hit is True


def test_cache_key_changes_with_the_prompt() -> None:
    """The deviation from spec 5, asserted.

    Keying on the JD alone would freeze every parsed posting at whatever the
    prompt said when it was first run -- and iterating on prompts is the point.
    """
    raw = "Some posting"
    assert jd_cache_key(raw, "m", "v1") != jd_cache_key(raw, "m", "v2")


def test_cache_key_changes_with_the_model() -> None:
    raw = "Some posting"
    assert jd_cache_key(raw, "claude-opus-5", "v1") != jd_cache_key(raw, "claude-sonnet-5", "v1")


def test_corrupt_cache_entry_is_treated_as_a_miss(
    fake_llm: FakeStructuredModel, cache: JobSpecCache
) -> None:
    raw = "Some posting text"
    parse_job_description(raw, llm=fake_llm, cache=cache)

    # `model_for()`, not a hardcoded id: the cache key follows the active
    # provider, so pinning Anthropic here made this test pass only on a machine
    # with no other key set. It broke the moment a DeepSeek key appeared -- by
    # corrupting a file the parse was no longer looking at, which made the
    # second call a cache *hit* and the assertion below fail for a reason that
    # had nothing to do with corrupt entries.
    key = jd_cache_key(raw, model_for(), prompt_version(PROMPT_NAME))
    cache.path_for(key).write_text("{ not json", encoding="utf-8")

    _, hit = parse_job_description(raw, llm=fake_llm, cache=cache)
    assert hit is False, "a corrupt cache entry should re-parse, not raise"


def test_cache_writes_are_atomic(cache: JobSpecCache) -> None:
    """No `.tmp` file should survive a completed write."""
    spec = JobSpec.from_fields(SAMPLE_FIELDS, "raw")
    cache.put("abc123", spec)
    assert cache.get("abc123") == spec
    assert not list(cache.cache_dir.glob("*.tmp"))


# --- the prompt -------------------------------------------------------------


def test_prompt_exists_and_covers_inferred_priorities() -> None:
    """Spec 5 requires the prompt to make this distinction; assert it says so."""
    prompt = load_prompt(PROMPT_NAME)
    assert "is_inferred" in prompt
    assert "on-call" in prompt.lower()


def test_prompt_version_is_stable_and_short() -> None:
    version = prompt_version(PROMPT_NAME)
    assert version == prompt_version(PROMPT_NAME)
    assert len(version) == 12


# --- the dataset and its snapshots -----------------------------------------


@pytest.mark.parametrize("level", JD_LEVELS)
def test_jd_file_exists_and_is_substantial(level: str) -> None:
    path = JD_DIR / f"{level}.txt"
    assert path.is_file(), f"missing {path}"
    assert len(path.read_text(encoding="utf-8")) > 500


@pytest.mark.parametrize("level", JD_LEVELS)
def test_snapshot_is_a_valid_jobspec(level: str) -> None:
    """M2 DoD: "3 saved JDs (junior, mid, senior) parse to valid JobSpec".

    Validates the committed parse rather than re-running the model, so this
    holds in CI and on a machine with no API key.
    """
    path = SNAPSHOT_DIR / f"{level}.json"
    if not path.is_file():
        pytest.skip("no snapshot yet: regenerate with REGEN_SNAPSHOTS=1 (needs an API key)")

    try:
        spec = JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        pytest.fail(f"{path} is not a valid JobSpec:\n{exc}")

    assert spec.requirements, "a posting with no requirements is a parse failure"
    assert spec.source_hash == job_description_hash(
        (JD_DIR / f"{level}.txt").read_text(encoding="utf-8")
    ), "snapshot is stale -- the JD text changed since it was generated"


@pytest.mark.parametrize(
    ("level", "expected_seniority"),
    [("junior", "junior"), ("mid", "mid"), ("senior", "senior")],
)
def test_snapshot_seniority(level: str, expected_seniority: str) -> None:
    path = SNAPSHOT_DIR / f"{level}.json"
    if not path.is_file():
        pytest.skip("no snapshot yet")
    spec = JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.seniority == expected_seniority


def test_junior_snapshot_flags_the_overlong_requirement_list() -> None:
    """junior.txt asks for 16 things from someone with 2 years of experience.

    Spec 5 names this exact shape as a red flag ("15 must haves for a junior
    role"), and it is the reason that JD is in the dataset.
    """
    path = SNAPSHOT_DIR / "junior.json"
    if not path.is_file():
        pytest.skip("no snapshot yet")
    spec = JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.red_flags, "the junior posting should not come back clean"


def test_senior_snapshot_does_not_invent_red_flags() -> None:
    """senior.txt is a clean posting. Inventing concerns is its own failure."""
    path = SNAPSHOT_DIR / "senior.json"
    if not path.is_file():
        pytest.skip("no snapshot yet")
    spec = JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert len(spec.red_flags) <= 1, f"invented concerns about a clean posting: {spec.red_flags}"


def test_mid_snapshot_infers_operational_maturity() -> None:
    """mid.txt mentions on-call, incidents, postmortems, SLOs and toil in five
    places without ever listing "ops experience" as a requirement.

    This is spec 5's own example of an inferred priority, and the single most
    load-bearing assertion about prompt quality in this milestone.
    """
    path = SNAPSHOT_DIR / "mid.json"
    if not path.is_file():
        pytest.skip("no snapshot yet")
    spec = JobSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.inferred_requirements(), "no inferred priorities found in a posting full of them"


# --- the one test that spends money -----------------------------------------


@pytest.mark.llm
@requires_llm
@pytest.mark.parametrize("level", JD_LEVELS)
def test_live_parse_matches_snapshot_shape(level: str) -> None:
    """Parse for real and check the result is structurally sound.

    Deliberately does not assert equality with the snapshot: the model is not
    deterministic, and a test that demanded byte-identical output would fail for
    reasons that say nothing about correctness. Set REGEN_SNAPSHOTS=1 to write
    the result to disk instead of only checking it.
    """
    raw = (JD_DIR / f"{level}.txt").read_text(encoding="utf-8")
    spec, _ = parse_job_description(raw)

    assert spec.requirements
    assert spec.ats_keywords
    assert spec.source_hash == job_description_hash(raw)

    if os.environ.get("REGEN_SNAPSHOTS"):
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        (SNAPSHOT_DIR / f"{level}.json").write_text(
            json.dumps(spec.model_dump(), indent=2) + "\n", encoding="utf-8"
        )
