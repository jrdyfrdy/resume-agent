"""Model access and prompt loading.

Everything that talks to a language model goes through here, so that swapping
providers, changing a model id, or checking what a prompt currently says is a
single-file operation rather than a grep.

CLAUDE.md rule 5: prompts are versioned artifacts on disk, never inline string
literals. `load_prompt` reads them and `prompt_version` hashes them -- that hash
is part of the JD cache key, so editing a prompt invalidates the parses it
produced instead of silently serving stale ones.

PROVIDERS
---------
The project was built against Anthropic and Anthropic is still the default, but
nothing above this file knows that. A `Provider` is just "which client class,
which key, which two model ids, which endpoint", and `build_chat_model` is the
only place that turns one into a live client.

Anything that speaks the OpenAI protocol -- DeepSeek, OpenRouter, Together,
Groq, a local Ollama -- is reachable by adding a row to `PROVIDERS`, or with no
code at all through the `RESUME_AGENT_*` overrides at the bottom of that table.

One caveat worth knowing before switching: every prompt in `prompts/` was
written and tuned against Claude, and every call in this project goes through
`with_structured_output`. A different model will parse and rewrite differently.
That is a measurable question, not a guess -- `make eval` (M8) scores a run
against the baseline, so change the provider and then go and look.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal

from langchain_core.language_models import BaseChatModel

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# The two jobs a model does here. Generation writes (parse, score, tailor, the
# letter, LaTeX repair); judge checks (grounding entailment, letter consistency,
# the eval rubric). They are separate because the checking task is narrower and
# does not need the expensive model.
Role = Literal["generation", "judge"]


@dataclass(frozen=True)
class Provider:
    """One way to reach a model.

    `client` selects the class, not the vendor: everything OpenAI-compatible
    uses `ChatOpenAI` with a different `base_url`, which is the whole reason one
    dependency covers most of the market.
    """

    name: str
    client: Literal["anthropic", "openai"]
    key_env_var: str
    generation_model: str
    judge_model: str
    # None means "the SDK's own default endpoint" -- correct for Anthropic,
    # which has no base_url to override.
    base_url: str | None = None
    # None means "send no effort parameter at all". Effort is spelled
    # differently by every vendor, so it lives on the provider rather than
    # being a global constant.
    effort: str | None = None


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        client="anthropic",
        key_env_var="ANTHROPIC_API_KEY",
        # Claude Opus 5: $5 / $25 per MTok. A JD parse is roughly 2-3k tokens in
        # and 1-2k out, so about $0.05-0.07 -- paid once per posting, because
        # the result is cached on a hash of the JD.
        generation_model="claude-opus-5",
        # The grounding judge (spec 5: "Use a cheap model"). Sonnet 5 is
        # $2 / $10 against Opus's $5 / $25, and the task is a narrow entailment
        # call with the source sentence right there in the prompt.
        #
        # Not dropped to Haiku despite "cheap", because this is the check that
        # makes the tool ethical to use (spec 12) and semantic inflation is a
        # subtle judgement.
        judge_model="claude-sonnet-5",
        # Extraction is not reasoning-heavy, and effort is the first quality
        # lever worth tuning per route rather than globally. Medium keeps the
        # inference quality `is_inferred` depends on without paying for depth
        # this workload does not repay.
        effort="medium",
    ),
    "deepseek": Provider(
        name="deepseek",
        client="openai",
        key_env_var="DEEPSEEK_API_KEY",
        # Current DeepSeek lineup as of 2026-09. The `deepseek-chat` and
        # `deepseek-reasoner` names that LangChain's integration page still
        # documents were retired on 2026-07-24; do not put them back.
        # V4-Pro for writing, Flash for checking, mirroring the Opus/Sonnet
        # split above rather than inventing a different shape.
        generation_model="deepseek-v4-pro",
        judge_model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",
        # DeepSeek exposes low/high/max thinking effort. Deliberately not wired
        # up: picking one without measuring would be guessing, and `make eval`
        # is cheap enough here to answer it properly.
        effort=None,
    ),
    "openrouter": Provider(
        name="openrouter",
        client="openai",
        key_env_var="OPENROUTER_API_KEY",
        # Reaches Claude without an Anthropic account, which keeps the prompts
        # working if a swap turns out to cost more quality than it saves money.
        generation_model="anthropic/claude-opus-5",
        judge_model="anthropic/claude-sonnet-5",
        base_url="https://openrouter.ai/api/v1",
    ),
    "custom": Provider(
        name="custom",
        client="openai",
        key_env_var="RESUME_AGENT_API_KEY",
        # Nothing sensible to default to. Both must come from the environment,
        # and `resolve_provider` says so plainly rather than sending "" to an
        # endpoint and letting it 404.
        generation_model="",
        judge_model="",
    ),
}

# Checked in this order when no provider is named explicitly, so that having a
# key in your shell is enough to run.
DETECTION_ORDER = ("anthropic", "deepseek", "openrouter", "custom")

PROVIDER_ENV_VAR = "RESUME_AGENT_PROVIDER"
BASE_URL_ENV_VAR = "RESUME_AGENT_BASE_URL"
GENERATION_MODEL_ENV_VAR = "RESUME_AGENT_GENERATION_MODEL"
JUDGE_MODEL_ENV_VAR = "RESUME_AGENT_JUDGE_MODEL"

# Lets one prompt be swapped for another without touching the file on disk.
# Format: "name=path,name=path". Used by the eval harness to run a variant
# (M8 needs to prove a deliberately worsened prompt scores measurably worse),
# and by nothing else -- ordinary runs read prompts/ unchanged.
PROMPT_OVERRIDE_ENV_VAR = "RESUME_AGENT_PROMPT_OVERRIDES"


class MissingCredentialsError(RuntimeError):
    """No API key is configured for the active provider."""


class ProviderConfigError(RuntimeError):
    """The provider itself is misconfigured -- bad name, or missing settings."""


# ---------------------------------------------------------------------------
# Which provider are we talking to?
# ---------------------------------------------------------------------------


def resolve_provider() -> Provider:
    """The active provider, with environment overrides applied.

    Resolved on every call rather than captured at import time. That is not
    fussiness: the model id is part of the JD cache key, so a value frozen at
    import could let a run on one provider serve a `JobSpec` parsed by another
    off disk. This project has twice been bitten by import-time defaults (the
    model cache directory, then the tracker database); this is the case where
    it would be hardest to notice.
    """
    named = os.environ.get(PROVIDER_ENV_VAR, "").strip().lower()

    if named:
        if named not in PROVIDERS:
            raise ProviderConfigError(
                f"{PROVIDER_ENV_VAR}={named!r} is not a known provider. "
                f"Valid names: {', '.join(sorted(PROVIDERS))}."
            )
        provider = PROVIDERS[named]
    else:
        provider = _detect_provider()

    # Overrides apply on top of whichever provider was chosen, so you can point
    # a known provider at a different model without editing this file.
    base_url = os.environ.get(BASE_URL_ENV_VAR, "").strip()
    generation = os.environ.get(GENERATION_MODEL_ENV_VAR, "").strip()
    judge = os.environ.get(JUDGE_MODEL_ENV_VAR, "").strip()

    provider = replace(
        provider,
        base_url=base_url or provider.base_url,
        generation_model=generation or provider.generation_model,
        # A judge model is optional to configure: falling back to the generation
        # model is always correct, just more expensive than it needs to be.
        judge_model=judge or provider.judge_model or generation or provider.generation_model,
    )

    if not provider.generation_model:
        raise ProviderConfigError(
            f"provider {provider.name!r} has no model configured. "
            f"Set {GENERATION_MODEL_ENV_VAR} (and optionally {JUDGE_MODEL_ENV_VAR})."
        )
    if provider.client == "openai" and not provider.base_url:
        raise ProviderConfigError(
            f"provider {provider.name!r} has no endpoint configured. "
            f"Set {BASE_URL_ENV_VAR}."
        )
    return provider


def _detect_provider() -> Provider:
    """The first provider whose key is present in the environment.

    Falls back to Anthropic when nothing is set, so that the error a new user
    sees is "set ANTHROPIC_API_KEY" rather than "no provider configured" --
    the first is actionable, the second sends you to the docs.
    """
    for name in DETECTION_ORDER:
        candidate = PROVIDERS[name]
        if os.environ.get(candidate.key_env_var):
            return candidate
    return PROVIDERS["anthropic"]


def model_for(role: Role = "generation") -> str:
    """The model id this provider uses for `role`.

    Exposed separately from `build_chat_model` because callers need the id
    itself, not just a client: it goes into the cache key, so that switching
    models invalidates cached work rather than silently serving another
    model's answers.
    """
    provider = resolve_provider()
    return provider.judge_model if role == "judge" else provider.generation_model


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def has_credentials() -> bool:
    """Whether a live model call is possible right now.

    Used to skip the tests that would spend money, and to fail the CLI early
    with instructions instead of a traceback from inside the SDK.

    A misconfigured provider counts as "no" rather than raising, because one of
    the callers is the web UI's readiness check: a bad provider name should
    light up the blocked state that explains itself, not return a 500.
    """
    try:
        return bool(os.environ.get(resolve_provider().key_env_var))
    except ProviderConfigError:
        return False


def credentials_message() -> str:
    """What to tell someone who has no working credentials.

    A function rather than a constant because the variable to set depends on
    which provider is active, and naming the wrong one is worse than saying
    nothing.
    """
    try:
        provider = resolve_provider()
    except ProviderConfigError as exc:
        return f"{exc}\n"

    var = provider.key_env_var
    return f"""\
No credentials found for provider "{provider.name}".

resume-agent reads {var} from the environment. Set it in your shell:

  Windows      setx {var} "..."      (then open a new terminal)
  macOS/Linux  export {var}="..."

Models in use: {provider.generation_model} (generation), {provider.judge_model} (judge)

To use a different provider, set {PROVIDER_ENV_VAR} to one of:
  {', '.join(sorted(PROVIDERS))}

Everything that does not call a model -- `build`, `search`, `index` -- works
without credentials.
"""


# ---------------------------------------------------------------------------
# Building a client
# ---------------------------------------------------------------------------


def build_chat_model(
    role: Role = "generation",
    model: str | None = None,
    **kwargs: object,
) -> BaseChatModel:
    """A live client for `role` on the active provider.

    `max_tokens` is generous on purpose: a JobSpec for a long posting runs to a
    few thousand tokens of JSON, and hitting the cap truncates mid-structure,
    which surfaces as a confusing validation error rather than as "too short".

    Imports are inside the function so that an unused provider's package is
    never imported -- and so that this module keeps loading if one of them is
    not installed.
    """
    provider = resolve_provider()
    if not os.environ.get(provider.key_env_var):
        raise MissingCredentialsError(credentials_message())

    model_id = model or model_for(role)

    if provider.client == "anthropic":
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs: dict[str, object] = {}
        if provider.effort:
            # `effort` lives inside output_config, not at the top level. Passed
            # through model_kwargs because it is newer than this
            # langchain-anthropic release's typed surface.
            anthropic_kwargs["model_kwargs"] = {"output_config": {"effort": provider.effort}}

        return ChatAnthropic(
            model_name=model_id,
            max_tokens_to_sample=8000,
            **anthropic_kwargs,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_id,
        base_url=provider.base_url,
        api_key=os.environ[provider.key_env_var],
        max_tokens=8000,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def prompt_overrides() -> dict[str, Path]:
    """Parse the override map from the environment. Empty in normal runs."""
    raw = os.environ.get(PROMPT_OVERRIDE_ENV_VAR, "")
    overrides: dict[str, Path] = {}
    for pair in raw.split(","):
        name, _, path = pair.partition("=")
        if name.strip() and path.strip():
            overrides[name.strip()] = Path(path.strip())
    return overrides


def prompt_path(name: str) -> Path:
    """Where `name` is read from, honouring any override."""
    return prompt_overrides().get(name, PROMPT_DIR / f"{name}.md")


@cache
def _read_prompt(path: str) -> str:
    """Cached on the resolved path, not on the prompt name.

    Caching on the name would make an override invisible for the rest of the
    process -- the first read would win and the variant would silently score
    identically to the baseline, which is the one result an eval must never
    produce by accident.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"no prompt at {file}")
    return file.read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    """Read `prompts/<name>.md`, or its override."""
    return _read_prompt(str(prompt_path(name)))


def prompt_version(name: str) -> str:
    """A short hash of a prompt's current text.

    Goes into the JD cache key. Without it, editing `parse_jd.md` would leave
    every previously parsed posting frozen at the old prompt's output -- and the
    entire point of keeping prompts as versioned files is being able to iterate
    on them and see the difference.
    """
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()[:12]
