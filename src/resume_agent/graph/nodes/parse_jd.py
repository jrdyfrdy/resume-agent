"""Turn a raw job posting into a validated `JobSpec`. Spec 5, `parse_jd`.

    "Structured output via `llm.with_structured_output(JobSpec)`. Cache on
     sha256(raw_jd). Prompt should push the model to distinguish stated
     requirements from *inferred* priorities."

**On the shape of this module.** Spec 5 describes nodes as `(state) -> dict`,
and this file sits at the path spec 7 gives that node. What it exports today is
a plain function rather than a state adapter, because `AgentState` has around
twenty fields belonging to milestones that do not exist yet, and defining it now
to satisfy one caller would be scaffolding the graph before the graph. The
state adapter is three lines and arrives in M5 alongside `graph/build.py`.

The function is the part worth testing anyway: everything interesting here is
prompt, schema and cache behaviour, none of which cares about a TypedDict.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from resume_agent.jd_cache import JobSpecCache, jd_cache_key
from resume_agent.llm import build_chat_model, load_prompt, model_for, prompt_version
from resume_agent.models.job import JobSpec, JobSpecFields

PROMPT_NAME = "parse_jd"


class JobDescriptionParseError(RuntimeError):
    """The model returned something that is not a valid JobSpec."""


def parse_job_description(
    raw_jd: str,
    *,
    llm: BaseChatModel | None = None,
    cache: JobSpecCache | None = None,
    use_cache: bool = True,
    model: str | None = None,
) -> tuple[JobSpec, bool]:
    """Parse a posting. Returns `(spec, was_cache_hit)`.

    `was_cache_hit` is returned rather than logged because the caller is what
    decides whether that matters -- the CLI prints it, and M2's definition of
    done asserts it.
    """
    if not raw_jd.strip():
        raise JobDescriptionParseError("job description is empty")

    # Resolved here rather than as a default argument, because the id goes into
    # the cache key below: a value frozen at import time would let a run on one
    # provider serve a JobSpec parsed by another straight off disk.
    model = model or model_for()

    cache = cache or JobSpecCache()
    key = jd_cache_key(raw_jd, model, prompt_version(PROMPT_NAME))

    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached, True

    # Built here rather than at import time so that everything in this module
    # except an actual live call works without credentials.
    llm = llm or build_chat_model(model=model)

    # The structured-output schema is JobSpecFields, not JobSpec: `source_hash`
    # is a sha256 and belongs to Python (CLAUDE.md rule 2). A model asked for a
    # hash returns a plausible hex string that is the hash of nothing.
    structured = llm.with_structured_output(JobSpecFields)

    messages = [
        SystemMessage(content=load_prompt(PROMPT_NAME)),
        HumanMessage(content=f"<job_posting>\n{raw_jd.strip()}\n</job_posting>"),
    ]

    result = structured.invoke(messages)
    if not isinstance(result, JobSpecFields):
        # with_structured_output can hand back a dict depending on the provider
        # and settings; normalise so the error surfaces here rather than as an
        # AttributeError three frames away.
        try:
            result = JobSpecFields.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - re-raised as our own error type
            raise JobDescriptionParseError(
                f"model did not return a valid JobSpecFields: {exc}"
            ) from exc

    spec = JobSpec.from_fields(result, raw_jd)

    if use_cache:
        cache.put(key, spec)

    return spec, False
