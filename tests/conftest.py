"""Shared fixtures.

The compile fixtures are session-scoped on purpose: shelling out to tectonic is
by far the slowest thing in the suite, and every end-to-end assertion is about
the same one PDF.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_agent.cache import CACHE_DIR_ENV_VAR
from resume_agent.kb.embeddings import FastEmbedEmbeddings
from resume_agent.kb.index import ProfileIndex
from resume_agent.kb.loader import load_profile
from resume_agent.kb.retriever import HybridRetriever
from resume_agent.kb.writer import PROFILE_BACKUP_DIR_ENV_VAR
from resume_agent.latex.compile import CompileResult, compile_tex, find_compiler
from resume_agent.latex.context import build_resume_context
from resume_agent.latex.env import render_template
from resume_agent.models.profile import Profile
from resume_agent.tracker.db import TRACKER_DB_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_EXAMPLE = REPO_ROOT / "profile.example"
TEMPLATE_PATH = REPO_ROOT / "src" / "resume_agent" / "templates" / "jake_resume.tex.j2"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "jake_resume.tex"

RESUME_TEMPLATE = "jake_resume.tex.j2"

# Every test that shells out to a real compiler carries this marker, so a machine
# without one can still run the rest with `-m "not latex"`.
requires_latex = pytest.mark.skipif(
    find_compiler() is None,
    reason="no LaTeX compiler found (tectonic/latexmk/docker) -- see latex/compile.py",
)


@pytest.fixture(scope="session")
def example_profile() -> Profile:
    return load_profile(PROFILE_EXAMPLE)


@pytest.fixture(scope="session")
def rendered_tex(example_profile: Profile) -> str:
    return render_template(RESUME_TEMPLATE, build_resume_context(example_profile))


@pytest.fixture(scope="session")
def template_source() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def compiled_resume(rendered_tex: str, tmp_path_factory: pytest.TempPathFactory) -> CompileResult:
    """Compile the example profile once and share the result."""
    out_dir = tmp_path_factory.mktemp("compiled_resume")
    return compile_tex(rendered_tex, out_dir, job_name="resume")


# --- M1: retrieval -----------------------------------------------------------
#
# Session-scoped because loading the ONNX model and embedding the corpus is the
# slowest thing in the suite by an order of magnitude, and every retrieval test
# wants the same index.


@pytest.fixture(scope="session")
def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()


@pytest.fixture(scope="session")
def profile_index(
    example_profile: Profile,
    embeddings: FastEmbedEmbeddings,
    tmp_path_factory: pytest.TempPathFactory,
) -> ProfileIndex:
    db_path = tmp_path_factory.mktemp("index") / "profile.example.db"
    return ProfileIndex.build(example_profile, PROFILE_EXAMPLE, db_path, embeddings)


@pytest.fixture(scope="session")
def retriever(
    profile_index: ProfileIndex,
    example_profile: Profile,
    embeddings: FastEmbedEmbeddings,
) -> HybridRetriever:
    return HybridRetriever(profile_index, example_profile, embeddings)


@pytest.fixture(autouse=True)
def isolated_model_cache(tmp_path_factory: pytest.TempPathFactory, monkeypatch) -> None:
    """Give every test its own empty model-output cache.

    Autouse and non-negotiable. Without it, the first test to call a fake LLM
    writes a cache entry that satisfies the next test's "was the model called?"
    assertion -- so the suite would pass while the code under test did nothing,
    which is the worst kind of green.
    """
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path_factory.mktemp("model_cache")))

    # Same reasoning, higher stakes: `finalize` writes a tracker row, so any
    # test that runs the graph end to end would otherwise put a fake
    # application into the user's real, non-regenerable tracker.
    monkeypatch.setenv(
        TRACKER_DB_ENV_VAR, str(tmp_path_factory.mktemp("tracker") / "applications.db")
    )

    # Highest stakes of the three. `kb/writer.py` copies a file before
    # overwriting it, and that copy is the only undo a gitignored `profile/`
    # has. Without this, a writer test would drop backups into the user's real
    # backup directory, beside their career data.
    monkeypatch.setenv(
        PROFILE_BACKUP_DIR_ENV_VAR, str(tmp_path_factory.mktemp("profile_backups"))
    )
