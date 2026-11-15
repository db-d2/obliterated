"""Render the lab notes as Markdown.

    uv run python scripts/lab_notes.py

docs/lab-notes.json is the source: one dated entry per fix, finding, method change, decision,
milestone or open question, each with the commit or file behind it. The results dashboard
reads the same file. Entries carry aggregate numbers and public repo names only, never a
model's answers. Numbers stand as reported on the day; a later entry corrects an earlier one
instead of rewriting it, and an open question a later entry settles names that entry in
``closed_by``.
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
HEAD = """# Lab notes

Generated from `docs/lab-notes.json` by `scripts/lab_notes.py`; edit that file, not this one.
Numbers stand as reported on the day, and a later entry corrects an earlier one. Evidence
under `runs/` is local to the machine the experiments ran on: scores, records and model
answers are not committed.
"""


def line(n: dict, dated: bool = False) -> str:
    text = f"- **{n['kind']}** ({n['id']}{', ' + n['date'] if dated else ''}): {n['note']}"
    if n.get("closed_by"):
        text += f" (closed by {n['closed_by']})"
    if n.get("evidence"):
        text += f" Evidence: {n['evidence']}."
    return text


def render(notes: list[dict]) -> str:
    """Markdown for ``notes``: the questions nothing has closed yet, then every entry by day,
    newest day first and in the order written within a day."""
    parts = [HEAD]
    still_open = [n for n in notes if n["kind"] == "Open" and not n.get("closed_by")]
    if still_open:
        parts.append("## Still open\n\n" + "\n".join(line(n, dated=True) for n in still_open) + "\n")
    for day in sorted({n["date"] for n in notes}, reverse=True):
        parts.append(f"## {day}\n\n" + "\n".join(line(n) for n in notes if n["date"] == day) + "\n")
    return "\n".join(parts)


def main() -> None:
    notes = json.loads((DOCS / "lab-notes.json").read_text())["rows"]
    (DOCS / "lab-notes.md").write_text(render(notes))
    print(f"{len(notes)} notes written to docs/lab-notes.md")


if __name__ == "__main__":
    main()
