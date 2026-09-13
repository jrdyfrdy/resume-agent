"""The provider seam in `llm.py`.

The project was built against Anthropic; this is what lets it run on anything
that speaks the OpenAI protocol instead. Every test here is offline -- the keys
are obvious fakes and no client is ever invoked, only constructed.

Isolation is per-test rather than in `conftest.py` on purpose. `requires_llm` in
`test_parse_jd.py` is evaluated at collection time, so pinning a provider in an
autouse fixture would select the live tests against the real environment and
then run them against a pinned fake one.
"""

from __future__ import annotations

import pytest

from resume_agent.graph.nodes.score import ScoringResult
from resume_agent.graph.nodes.tailor import TailoringResult
from resume_agent.graph.nodes.verify import JudgeVerdict
from resume_agent.jd_cache import jd_cache_key
from resume_agent.llm import (
    BASE_URL_ENV_VAR,
    GENERATION_MODEL_ENV_VAR,
    JUDGE_MODEL_ENV_VAR,
    PROVIDER_ENV_VAR,
    PROVIDERS,
    MissingCredentialsError,
    ProviderConfigError,
    build_chat_model,
    credentials_message,
    has_credentials,
    model_for,
    resolve_provider,
)
from resume_agent.models.job import JobSpecFields
from resume_agent.models.letter import ConsistencyVerdict, CoverLetterFields

FAKE_KEY = "not-a-real-key"

PROVIDER_ENV_VARS = [
    PROVIDER_ENV_VAR,
    BASE_URL_ENV_VAR,
    GENERATION_MODEL_ENV_VAR,
    JUDGE_MODEL_ENV_VAR,
    *(provider.key_env_var for provider in PROVIDERS.values()),
]


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test from an empty environment.

    Without this the suite would behave differently depending on which keys
    happen to be in the developer's shell.
    """
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ===========================================================================
# Which provider is active
# ===========================================================================


def test_anthropic_is_the_default_with_nothing_set() -> None:
    """So a new user's first error is "set ANTHROPIC_API_KEY", which is
    actionable, rather than "no provider configured", which is not."""
    assert resolve_provider().name == "anthropic"
    assert has_credentials() is False


def test_a_key_in_the_environment_selects_its_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting one key is the whole configuration story for the common case."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)

    provider = resolve_provider()
    assert provider.name == "deepseek"
    assert provider.client == "openai"
    assert has_credentials() is True


def test_anthropic_wins_when_several_keys_are_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection order is defined, not incidental: someone who adds a second key
    should not silently change which model writes their resume."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)

    assert resolve_provider().name == "anthropic"


def test_an_explicit_provider_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepseek")

    assert resolve_provider().name == "deepseek"


def test_provider_names_are_case_and_space_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVIDER_ENV_VAR, "  DeepSeek ")
    assert resolve_provider().name == "deepseek"


def test_an_unknown_provider_name_lists_the_valid_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo should say what to type instead, not raise KeyError."""
    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepsek")

    with pytest.raises(ProviderConfigError) as caught:
        resolve_provider()

    message = str(caught.value)
    assert "deepsek" in message
    assert "deepseek" in message and "anthropic" in message


# ===========================================================================
# Overrides
# ===========================================================================


def test_model_overrides_apply_on_top_of_a_known_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point a known provider at a different model without editing llm.py."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, "deepseek-flash")

    assert model_for("generation") == "deepseek-flash"


def test_the_judge_falls_back_to_the_generation_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuring a judge model is optional. Falling back to the generation
    model is always correct, just more expensive than it needs to be."""
    monkeypatch.setenv(PROVIDER_ENV_VAR, "custom")
    monkeypatch.setenv("RESUME_AGENT_API_KEY", FAKE_KEY)
    monkeypatch.setenv(BASE_URL_ENV_VAR, "http://localhost:11434/v1")
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, "llama3.3")

    assert model_for("generation") == "llama3.3"
    assert model_for("judge") == "llama3.3"


def test_a_custom_provider_without_a_model_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVIDER_ENV_VAR, "custom")
    monkeypatch.setenv(BASE_URL_ENV_VAR, "http://localhost:11434/v1")

    with pytest.raises(ProviderConfigError, match=GENERATION_MODEL_ENV_VAR):
        resolve_provider()


def test_a_custom_provider_without_an_endpoint_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Better than sending the request to OpenAI's default endpoint with a key
    that is not an OpenAI key, which fails with an authentication error that
    points at entirely the wrong thing."""
    monkeypatch.setenv(PROVIDER_ENV_VAR, "custom")
    monkeypatch.setenv(GENERATION_MODEL_ENV_VAR, "llama3.3")

    with pytest.raises(ProviderConfigError, match=BASE_URL_ENV_VAR):
        resolve_provider()


# ===========================================================================
# The thing most likely to bite silently
# ===========================================================================


def test_switching_provider_changes_the_jd_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parse is cached under (posting, model, prompt version).

    If the model id did not move with the provider, switching to DeepSeek would
    serve Claude's parse of the same posting off disk -- and you would be
    measuring the wrong model while believing you had switched.
    """
    posting = "We are hiring a backend engineer."

    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    anthropic_key = jd_cache_key(posting, model_for(), "v1")

    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    deepseek_key = jd_cache_key(posting, model_for(), "v1")

    assert anthropic_key != deepseek_key


def test_the_model_id_is_not_frozen_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """`model_for` is called, not captured. This project has been bitten twice
    by import-time defaults (the model cache directory, then the tracker
    database); here it would be hardest to notice."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    first = model_for()

    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)

    assert model_for() != first


# ===========================================================================
# Building clients
# ===========================================================================


def test_the_client_class_follows_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert type(build_chat_model()).__name__ == "ChatAnthropic"

    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    assert type(build_chat_model()).__name__ == "ChatOpenAI"


def test_the_deepseek_client_points_at_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)

    generation = build_chat_model()
    judge = build_chat_model("judge")

    assert generation.model_name == "deepseek-v4-pro"
    assert judge.model_name == "deepseek-flash"
    assert "api.deepseek.com" in str(generation.openai_api_base)


def test_no_key_raises_missing_credentials_with_instructions() -> None:
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"):
        build_chat_model()


@pytest.mark.parametrize(
    "schema",
    [JobSpecFields, ScoringResult, TailoringResult, JudgeVerdict, CoverLetterFields,
     ConsistencyVerdict],
    ids=lambda s: s.__name__,
)
def test_every_schema_converts_for_an_openai_client(
    monkeypatch: pytest.MonkeyPatch, schema: type
) -> None:
    """Every model call in this project goes through `with_structured_output`,
    and these schemas were only ever exercised against Anthropic's tool format.

    Nested models are where OpenAI-compatible schema conversion tends to break,
    and `JobSpecFields` carries a list of nested `Requirement`s. This says the
    schemas convert; whether a given model *fills them well* is what `make eval`
    is for.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    assert build_chat_model().with_structured_output(schema) is not None


# ===========================================================================
# What the user is told
# ===========================================================================


def test_credentials_message_names_the_active_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telling a DeepSeek user to set ANTHROPIC_API_KEY is worse than saying
    nothing: they would set it, and it still would not work."""
    monkeypatch.setenv(PROVIDER_ENV_VAR, "deepseek")
    message = credentials_message()

    assert "DEEPSEEK_API_KEY" in message
    assert "ANTHROPIC_API_KEY" not in message


def test_credentials_message_never_contains_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The message is printed to a terminal and, via the API, rendered into a
    web page. It names variables, never their values."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-value-do-not-print")
    assert "sk-secret-value-do-not-print" not in credentials_message()


def test_a_misconfigured_provider_is_reported_rather_than_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`has_credentials` is called by the web UI's readiness check. A bad
    provider name should light up the blocked state that explains itself, not
    return a 500."""
    monkeypatch.setenv(PROVIDER_ENV_VAR, "nonsense")

    assert has_credentials() is False
    assert "nonsense" in credentials_message()
