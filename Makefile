# Convenience targets. Everything here is a thin wrapper around `uv run`, so if
# you do not have `make` (Windows, typically) the commands underneath work
# directly -- see README.md.

.PHONY: setup test lint fmt build calibrate clean

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

# Re-derive CHARS_PER_LINE. Run after any change to the template's geometry,
# font or list nesting, then update the constant in latex/metrics.py.
calibrate:
	uv run python scripts/calibrate_chars_per_line.py

# Regenerate the golden .tex snapshot. Deliberate two-step: run this, then read
# the diff before committing it.
golden:
	REGEN_GOLDEN=1 uv run pytest tests/test_render_compile.py::test_golden_tex_snapshot

clean:
	rm -rf out/
