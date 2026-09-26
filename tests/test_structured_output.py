"""Settling a model's structured answer: what `llm.structured_output` returns.

Found on the first hosted run. DeepSeek answered the posting parse with its tool
arguments inside a wrapper it invented -- `{"job_json": {...}}` -- so every field
of `JobSpecFields` looked missing, and the page showed eleven Pydantic errors.

`WrapsItsAnswer` reproduces that through LangChain's own `include_raw`
machinery: its tool call is parsed by the real `PydanticToolsParser`, which
fails exactly as it did on the server, before `settle_structured` sees it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from resume_agent.graph.nodes.parse_jd import parse_job_description
from resume_agent.llm import StructuredOutputError, settle_structured, structured_output
from resume_agent.models.job import JobSpecFields

FIELDS = {
    "company": "Acme Corp",
    "title": "RPA Developer",
    "seniority": "mid",
    "domain": "automation",
    "requirements": [
        {"text": "UiPath", "category": "tool", "weight": 5, "is_must_have": True},
    ],
    "responsibilities": ["Build bots"],
    "ats_keywords": ["UiPath"],
    "culture_signals": [],
    "tone": "formal",
    "red_flags": [],
}


class WrapsItsAnswer(BaseChatModel):
    """Answers each structured call with one tool call carrying `answer` --
    or, from the second call on, `later`, when that is set."""

    answer: Any
    later: Any = None
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "wraps-its-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls += 1
        args = self.answer if self.calls == 1 or self.later is None else self.later
        if isinstance(args, str):  # arguments the client could not parse as JSON
            message = AIMessage(content="", invalid_tool_calls=[
                {"name": "JobSpecFields", "args": args, "id": "call_1", "error": "bad json"},
            ])
        else:
            message = AIMessage(content="", tool_calls=[
                {"name": "JobSpecFields", "args": args, "id": "call_1", "type": "tool_call"},
            ])
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, *, method=None, include_raw=False, **kwargs):
        # The provider's `method` is the one argument the base class does not
        # take; everything else is LangChain's own tool-call parsing.
        return BaseChatModel.with_structured_output(self, schema, include_raw=include_raw)


def ask(answer: Any) -> Any:
    return structured_output(WrapsItsAnswer(answer=answer), JobSpecFields).invoke(
        [HumanMessage(content="<job_posting>...</job_posting>")]
    )


def test_a_well_formed_answer_passes_straight_through() -> None:
    assert ask(FIELDS) == JobSpecFields.model_validate(FIELDS)


def test_the_answer_deepseek_actually_gave_is_unwrapped(caplog) -> None:
    """The failure from the first hosted run, reproduced end to end."""
    with caplog.at_level(logging.WARNING, logger="resume_agent.llm"):
        result = ask({"job_json": FIELDS})

    assert result == JobSpecFields.model_validate(FIELDS)
    assert "wrapped its JobSpecFields answer in 'job_json'" in caplog.text


def test_a_wrapper_holding_json_text_is_unwrapped_too() -> None:
    assert ask({"job": json.dumps(FIELDS)}).company == "Acme Corp"


def test_arguments_the_client_could_not_parse_are_given_one_more_look() -> None:
    assert ask(json.dumps({"job_json": FIELDS})).title == "RPA Developer"


def test_a_real_field_is_never_mistaken_for_a_wrapper() -> None:
    """`{"company": {...}}` is an answer with ten fields missing, not a wrapper:
    unwrapping it would validate the wrong thing, or worse, succeed."""
    with pytest.raises(StructuredOutputError):
        ask({"company": FIELDS})


def test_an_unusable_answer_fails_with_a_sentence_not_eleven_errors() -> None:
    model = WrapsItsAnswer(answer={"job_json": {"company": "Acme Corp"}})

    with pytest.raises(StructuredOutputError) as caught:
        structured_output(model, JobSpecFields).invoke([HumanMessage(content="x")])

    assert model.calls == 2, "asked once more, and no more than that"
    message = str(caught.value)
    assert "Trying again usually works" in message
    assert "validation error" not in message
    assert caught.value.__cause__ is not None, "the detail is kept for the log"


def test_one_bad_answer_is_followed_by_one_more_ask() -> None:
    """A malformed answer is usually a one-off. Failing on it would end the run
    -- and on the hosted site, spend one of the person's runs for the day."""
    model = WrapsItsAnswer(answer={"company": "only this"}, later=FIELDS)

    result = structured_output(model, JobSpecFields).invoke([HumanMessage(content="x")])

    assert result.company == "Acme Corp"
    assert model.calls == 2


def test_the_posting_parse_survives_the_wrapper() -> None:
    """The node that failed on the server, with the answer that broke it."""
    spec, from_cache = parse_job_description(
        "RPA Developer at Acme Corp. UiPath required.",
        llm=WrapsItsAnswer(answer={"job_json": FIELDS}),
        use_cache=False,
    )

    assert from_cache is False
    assert spec.company == "Acme Corp"
    assert spec.requirements[0].text == "UiPath"


def test_anything_already_parsed_is_left_alone() -> None:
    """A model integration or a test stand-in that parses for itself."""
    parsed = JobSpecFields.model_validate(FIELDS)

    assert settle_structured(JobSpecFields, parsed) is parsed
    assert settle_structured(JobSpecFields, {"company": "x"}) == {"company": "x"}


def test_the_model_call_still_reports_itself_to_the_page() -> None:
    """The run page's "calling the model" line comes from `on_chat_model_start`,
    raised inside the call. The call now happens inside the settling lambda, so
    the events have to find their way out of it -- as they do from a graph node.
    """
    import asyncio  # noqa: PLC0415

    from langchain_core.runnables import RunnableLambda  # noqa: PLC0415

    def node(messages):
        return structured_output(WrapsItsAnswer(answer=FIELDS), JobSpecFields).invoke(messages)

    async def events() -> list[str]:
        stream = RunnableLambda(node).astream_events(
            [HumanMessage(content="a posting")], version="v2"
        )
        return [event["event"] async for event in stream]

    seen = asyncio.run(events())

    assert "on_chat_model_start" in seen
    assert "on_chat_model_end" in seen
