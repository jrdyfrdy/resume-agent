"""Write the graph's mermaid diagram into the README.

Run with:  uv run python scripts/render_graph.py

The diagram is generated, never hand-drawn. A hand-drawn one is a claim about
the topology; this one is the topology, read straight out of the compiled graph,
so it cannot quietly stop being true when an edge moves.

Uses `draw_mermaid()` rather than `draw_mermaid_png()`: the text form needs no
graphviz and no network call, renders natively on GitHub, and diffs as text when
the graph changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resume_agent.graph.build import build_graph  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

START_MARKER = "<!-- graph:start -->"
END_MARKER = "<!-- graph:end -->"


def render() -> str:
    """The mermaid source for the compiled graph."""
    return build_graph().get_graph().draw_mermaid().strip()


def main() -> None:
    diagram = render()
    block = f"{START_MARKER}\n\n```mermaid\n{diagram}\n```\n\n{END_MARKER}"

    text = README.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        print(
            f"README is missing the {START_MARKER} / {END_MARKER} markers; "
            f"add them where the diagram should live.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    before = text.split(START_MARKER)[0]
    after = text.split(END_MARKER)[1]
    README.write_text(before + block + after, encoding="utf-8")

    print(f"Wrote {diagram.count(chr(10)) + 1} lines of mermaid into {README.name}")


if __name__ == "__main__":
    main()
