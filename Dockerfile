# The hosted image. Not how you should run this locally -- `uv run
# resume-agent serve` is, and it needs none of this.
#
# Which site it serves is set on the host, not here: RESUME_AGENT_MULTIUSER=1
# for accounts (friends sign in, you approve them) or RESUME_AGENT_DEMO=1 for
# the read-only tour. With neither, `serve` refuses to start on a public
# address, so a forgotten variable stops the deploy instead of exposing the
# no-sign-in local tool.
#
# Two things make this bigger than a plain Python image, and both are load
# bearing: tectonic, because a resume that never compiles is not a demo, and
# the ONNX embedding model, because retrieval runs before any of the LLM calls
# do. Baking the model in at build time rather than fetching it on first
# request is what keeps the first visitor from waiting 40 seconds for a 67MB
# download on a cold start.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    RESUME_AGENT_MODEL_CACHE=/opt/models

# tectonic pulls fonts and packages on first compile; letting it do that during
# the build means the first visitor does not pay for it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates fontconfig \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
      | tar -xz -C /usr/local/bin tectonic \
    && tectonic --version

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# The only profile in the image. The demo serves it; with accounts it is what
# "start from the worked example" copies -- and the only thing that may be
# copied, whatever else is in the working directory. `.dockerignore` keeps your
# real `profile/` out of the build.
COPY profile.example ./profile.example

# Warm both caches at build time, so a cold start is a cold start and not a
# download. Failure is non-fatal: a slower first request beats an unbuildable
# image.
RUN uv run resume-agent index --profile profile.example || true
RUN uv run resume-agent build --profile profile.example --out /tmp/warm || true

EXPOSE 8000
# Render supplies $PORT. Binding 0.0.0.0 is safe only in the two hosted modes,
# and `serve` checks: accounts put every route behind sign-in and approval, the
# demo removes every route that writes. Anything else is refused at startup.
CMD ["sh", "-c", "uv run resume-agent serve --host 0.0.0.0 --port ${PORT:-8000}"]
