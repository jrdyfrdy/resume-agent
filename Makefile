# Convenience targets. Everything here is a thin wrapper around `uv run`, so if
# you do not have `make` (Windows, typically) the commands underneath work
# directly -- see README.md.

.PHONY: setup test test-fast lint fmt build index search parse-jd analyze run review applications graph snapshots calibrate calibrate-budget golden clean

setup:
	uv sync

test:
	uv run pytest -v

# Skip the tests that shell out to a real LaTeX compiler.
test-fast:
	uv run pytest -v -m "not latex"

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

build:
	uv run resume-agent build --profile profile.example --out out/

index:
	uv run resume-agent index --profile profile.example

# Inspect retrieval by hand. Override the query: make search Q="redis caching"
Q ?= kubernetes
search:
	uv run resume-agent search "$(Q)" --explain

JD ?= evals/datasets/jds/mid.txt
parse-jd:
	uv run resume-agent parse-jd --jd $(JD)

analyze:
	uv run resume-agent analyze --jd $(JD)

run:
	uv run resume-agent run --jd $(JD)

review:
	uv run resume-agent run --jd $(JD) --interactive

applications:
	uv run resume-agent applications

# Regenerate the graph diagram in the README from the compiled graph.
graph:
	uv run python scripts/render_graph.py

# Regenerate the committed JD parse snapshots. Needs ANTHROPIC_API_KEY;
# costs roughly $$0.20 on Claude Opus 5.
snapshots:
	REGEN_SNAPSHOTS=1 uv run pytest tests/test_parse_jd.py -m llm

# Re-derive CHARS_PER_LINE. Run after any change to the template's geometry,
# font or list nesting, then update the constant in latex/metrics.py.
calibrate:
	uv run python scripts/calibrate_chars_per_line.py

# Re-derive the one-page line budget. Run after any template geometry change.
calibrate-budget:
	uv run python scripts/calibrate_line_budget.py

# Regenerate the golden .tex snapshot. Deliberate two-step: run this, then read
# the diff before committing it.
golden:
	REGEN_GOLDEN=1 uv run pytest tests/test_render_compile.py::test_golden_tex_snapshot

clean:
	rm -rf out/
