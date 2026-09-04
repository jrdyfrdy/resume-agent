"""Dense embeddings, behind LangChain's `Embeddings` interface.

Two decisions are worth understanding here.

**Why a local model.** `BAAI/bge-small-en-v1.5` runs through ONNX Runtime (no
torch), is 67MB downloaded once, and then works offline. That keeps the test
suite free, fast and deterministic -- which matters a great deal by M8, where the
eval harness re-runs a whole dataset of job descriptions on every prompt change.
A hosted embedding API would put a network call and a bill in that loop.

**Why the LangChain interface anyway.** Subclassing
`langchain_core.embeddings.Embeddings` costs two methods and buys two things:
the rest of the codebase depends on an interface rather than on fastembed, so
swapping in OpenAI or Voyage later is a one-class change; and it is the same
abstraction the LangChain ecosystem uses everywhere, which is worth being
fluent in.

**Asymmetric retrieval.** BGE models are trained so that a *query* and a
*passage* are embedded differently -- the query gets an instruction prefix.
fastembed exposes this as `query_embed` vs `embed`, and using the right one on
each side is free accuracy. Getting this backwards is a common and silent
quality regression, which is why the two paths are separate methods here rather
than one shared helper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:  # pragma: no cover
    from fastembed import TextEmbedding

# 384 dimensions, 0.067GB. The smallest BGE that is still good at technical
# prose; `bge-base-en-v1.5` (768 dims, 0.21GB) is the next step up if retrieval
# quality ever becomes the bottleneck.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSIONS = 384

MODEL_CACHE_ENV_VAR = "RESUME_AGENT_MODEL_CACHE"


def default_model_cache_dir() -> Path:
    """Where to keep the downloaded ONNX model.

    fastembed's own default is a temp directory, which on Windows means the OS
    is free to delete a 67MB download between runs and silently re-fetch it. A
    stable per-user location is worth the six lines.
    """
    override = os.environ.get(MODEL_CACHE_ENV_VAR)
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "resume-agent" / "fastembed"


class FastEmbedEmbeddings(Embeddings):
    """A local, offline `Embeddings` implementation backed by fastembed."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir or default_model_cache_dir()
        # Loading the ONNX model takes a second or two and may download it, so
        # it is deferred until something actually embeds. Constructing an index
        # object should not pay that cost.
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            from fastembed import TextEmbedding

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(self.cache_dir))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus passages. Order is preserved."""
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, with the model's query-side instruction prefix."""
        return next(iter(self.model.query_embed(text))).tolist()
