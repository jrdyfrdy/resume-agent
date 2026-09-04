"""Model access and prompt loading.

Everything that talks to a language model goes through here, so that swapping
providers, changing a model id, or checking what a prompt currently says is a
single-file operation rather than a grep.

CLAUDE.md rule 5: prompts are versioned artifacts on disk, never inline string
literals. `load_prompt` reads them and `prompt_version` hashes them -- that hash
is part of the JD cache key, so editing a prompt invalidates the parses it
produced instead of silently serving stale ones.
"""

from __future__ import annotations

import hashlib
import os
from functools import cache
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Claude Opus 5: $5 / $25 per MTok. A JD parse is roughly 2-3k tokens in and
# 1-2k out, so about $0.05-0.07 -- paid once per posting, because the result is
# cached on a hash of the JD. Change this one constant to move the whole project
# to another model; `claude-sonnet-5` is $2 / $10 if this ever runs hot.
PARSE_MODEL = "claude-opus-5"

# Extraction is not a reasoning-heavy task, and effort is the first quality
# lever worth tuning per route rather than globally. Medium keeps the inference
# quality that `is_inferred` depends on without paying for depth this workload
# does not repay.
PARSE_EFFORT = "medium"

API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class MissingCredentialsError(RuntimeError):
    """No API key is configured."""


CREDENTIALS_MESSAGE = f"""\
No Anthropic credentials found.

resume-agent reads {API_KEY_ENV_VAR} from the environment. Set it in your shell:

  Windows   setx {API_KEY_ENV_VAR} "sk-ant-..."     (then open a new terminal)
  macOS/Linux  export {API_KEY_ENV_VAR}="sk-ant-..."

Get a key at https://console.anthropic.com/settings/keys

Everything that does not call a model -- `build`, `search`, `index` -- works
without one.
"""


def has_credentials() -> bool:
    """Whether a live model call is possible right now.

    Used to skip the tests that would spend money, and to fail the CLI early
    with instructions instead of a traceback from inside the SDK.
    """
    return bool(os.environ.get(API_KEY_ENV_VAR))


def build_chat_model(
    model: str = PARSE_MODEL,
    effort: str = PARSE_EFFORT,
    **kwargs: object,
) -> BaseChatModel:
    """The configured Claude client.

    `max_tokens` is generous on purpose: a JobSpec for a long posting runs to a
    few thousand tokens of JSON, and hitting the cap truncates mid-structure,
    which surfaces as a confusing validation error rather than as "too short".
    """
    if not has_credentials():
        raise MissingCredentialsError(CREDENTIALS_MESSAGE)

    return ChatAnthropic(
        model_name=model,
        max_tokens_to_sample=8000,
        # `effort` lives inside output_config, not at the top level. Passed
        # through model_kwargs because it is newer than this langchain-anthropic
        # release's typed surface.
        model_kwargs={"output_config": {"effort": effort}},
        **kwargs,
    )


@cache
def load_prompt(name: str) -> str:
    """Read `prompts/<name>.md`."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt named {name!r} at {path}")
    return path.read_text(encoding="utf-8")


@cache
def prompt_version(name: str) -> str:
    """A short hash of a prompt's current text.

    Goes into the JD cache key. Without it, editing `parse_jd.md` would leave
    every previously parsed posting frozen at the old prompt's output -- and the
    entire point of keeping prompts as versioned files is being able to iterate
    on them and see the difference.
    """
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()[:12]
