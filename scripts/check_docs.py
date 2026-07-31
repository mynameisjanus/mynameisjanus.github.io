#!/usr/bin/env python3
"""Structural checks on course content, run as a CI gate before the site builds.

These are the checks that do not need the market-data caches, which are
gitignored and therefore absent in CI. Verifying that every pinned `# =>`
output still matches its code requires `data/*.parquet` and runs locally;
see Part IX lesson two for why that split exists and what it costs.

Checks:
  1. A written lesson carries no "Under development" scaffold banner.
  2. Every fenced python block's pinned output uses the house `# =>` / `#    `
     prefixes, so the output is machine-separable from the code.
  3. A written lesson ends with "## Where this goes next" and carries exactly
     one "!!! abstract \"Key takeaways\"".
  4. Relative markdown links between docs resolve to files that exist.

Exit code 0 if clean, 1 otherwise. Usage: python scripts/check_docs.py [docs_dir]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STUB = "Under development"
FENCE = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:|#)([^)#]+\.md)(?:#[^)]*)?\)")


def is_written(text: str) -> bool:
    """A lesson is written once the scaffold banner is gone."""
    return STUB not in text


def check_file(path: Path, docs: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    bad: list[str] = []

    for m in LINK.finditer(text):
        target = (path.parent / m.group(1)).resolve()
        if not target.exists():
            line = text[: m.start()].count("\n") + 1
            bad.append(f"{path}:{line}: broken link -> {m.group(1)}")

    for m in FENCE.finditer(text):
        lines = m.group(1).splitlines()
        start = text[: m.start()].count("\n") + 1
        # Only the trailing run of comment lines is a pinned-output block; a
        # `# =>` anywhere else is the inline form and is left alone.
        i = len(lines)
        while i and lines[i - 1].startswith("#"):
            i -= 1
        tail = lines[i:]
        marker = next((j for j, ln in enumerate(tail) if ln.startswith("# =>")), None)
        if marker is None:
            continue
        # Lines after the '# =>' marker are continuations and must keep the
        # 5-character prefix so the pinned text stays aligned with the first line.
        for j, ln in enumerate(tail[marker + 1:], start=marker + 1):
            if not (ln.startswith("#    ") or ln.rstrip() == "#"):
                bad.append(f"{path}:{start + i + j}: pinned output continuation must "
                           f"start with '#    ' or be '#', got {ln[:40]!r}")

    # Course lessons live at docs/part-NN-*/NN-*.md. The appendix is a migrated
    # mathematics reference with a different shape and is deliberately exempt.
    rel = path.relative_to(docs)
    is_lesson = (len(rel.parts) == 2 and rel.parts[0].startswith("part-")
                 and path.name != "index.md")
    if is_written(text) and is_lesson:
        if "## Where this goes next" not in text:
            bad.append(f"{path}: written lesson has no "
                       f"'## Where this goes next' section")
        n = text.count('!!! abstract "Key takeaways"')
        if n != 1:
            bad.append(f"{path}: expected 1 'Key takeaways' block, found {n}")
    return bad


def main(argv: list[str]) -> int:
    docs = Path(argv[1] if len(argv) > 1 else "docs")
    if not docs.is_dir():
        print(f"no such directory: {docs}", file=sys.stderr)
        return 1
    files = sorted(docs.rglob("*.md"))
    problems: list[str] = []
    for f in files:
        problems.extend(check_file(f, docs))
    for p in problems:
        print(p)
    print(f"checked {len(files)} markdown files, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
