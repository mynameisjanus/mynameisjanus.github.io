#!/usr/bin/env python3
"""One-off migration of the old Jekyll math pages to MkDocs Material.

Reads pages/probability/*.md and pages/statistics/*.md, converts Jekyll/kramdown
constructs to Material-flavored markdown, and writes ordered files into
docs/appendix/{probability,statistics}/.

Conversions:
  - front matter stripped; `# {title}` prepended (old pages have no H1)
  - {% include custom/series_*_next.html %} removed (Material footer nav replaces it)
  - {% include note|warning|tip|important.html content="..." %} -> admonitions
  - {{site.data.alerts.proof}}..{{site.data.alerts.end}} -> collapsible "Proof" block
  - {{site.data.alerts.note}}..{{site.data.alerts.end}} -> note admonition
  - <p>/</p> wrappers stripped (math inside raw HTML is invisible to MathJax
    under Material's processHtmlClass config)
  - kramdown `$$..$$` used inline in prose -> `$..$`; display blocks kept
  - <img src="images/prob/..."> -> markdown image pointing at docs/assets/
  - standalone <br> lines dropped; runs of blank lines collapsed

Anything ambiguous is left unchanged and listed in the review report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pages whose front matter has an empty `weight:`, in their intended order.
# They are appended after the highest explicit weight in each section.
UNWEIGHTED_ORDER = {
    "probability": [
        "bernoulli_process",
        "splitting_bernoulli_processes",
        "merging_bernoulli_processes",
        "poisson_process",
        "markov_processes_1",
        "markov_processes_2",
    ],
    "statistics": [
        "intro_to_hypothesis_testing",
        "levels_and_pvalues",
    ],
}

SERIES_INCLUDE_RE = re.compile(r"^\s*\{%\s*include\s+custom/series_\w+_next\.html\s*%\}\s*$")
ALERT_INCLUDE_RE = re.compile(
    r"^(\s*)\{%\s*include\s+(note|tip|warning|important)\.html\s+content=\"(.+?)\"\s*%\}\s*$"
)
ALERT_OPEN_RE = re.compile(r"^\s*\{\{site\.data\.alerts\.(proof|note|tip|warning|important)\}\}\s*$")
ALERT_END_RE = re.compile(r"^\s*\{\{site\.data\.alerts\.end\}\}\s*$")
IMG_RE = re.compile(r'<img\s+src="images/prob/([^"]+)"(?:\s+style="width:(\d+)px[^"]*")?\s*/?>')
P_TAG_RE = re.compile(r"</?p[^>]*>")
BR_LINE_RE = re.compile(r"^\s*(<br\s*/?>\s*)+$")
INLINE_MATH_RE = re.compile(r"\$\$(.+?)\$\$")

report: list[str] = []
redirects: list[tuple[str, str]] = []


def parse_front_matter(text: str, src: str) -> tuple[dict, str]:
    lines = text.split("\n")
    if lines[0].strip() != "---":
        report.append(f"{src}: no front matter found")
        return {}, text
    meta = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1 :])
            return meta, body
        m = re.match(r"^(\w+):\s*(.*)$", line)
        if m:
            meta[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    report.append(f"{src}: unterminated front matter")
    return meta, text


def fix_inline_math(line: str, src: str, lineno: int) -> str:
    if "$$" not in line:
        return line
    if line.count("$$") % 2 != 0:
        report.append(f"{src}:{lineno}: odd number of $$ on a prose line - left unchanged")
        return line
    return INLINE_MATH_RE.sub(r"$\1$", line)


def convert_math(lines: list[str], src: str) -> list[str]:
    """Keep paragraph-level $$ display blocks; convert prose-embedded $$..$$ to $..$."""
    out = []
    in_fence = False
    in_display = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        if in_display:
            out.append(line)
            if stripped.endswith("$$"):
                in_display = False
            continue
        if stripped.startswith("$$"):
            count = stripped.count("$$")
            if stripped == "$$" or (count % 2 != 0):
                # bare opener, or `$$\begin{align}`-style opener: display block
                in_display = True
                out.append(line)
                continue
            if count == 2 and stripped.endswith("$$") and len(stripped) > 4:
                # single-line display equation
                out.append(line)
                continue
            # e.g. `$$a$$ and $$b$$ ...` - inline math that happens to start the line
            out.append(fix_inline_math(line, src, idx))
            continue
        out.append(fix_inline_math(line, src, idx))
    if in_display:
        report.append(f"{src}: unclosed $$ display block at EOF")
    return out


def convert_alert_blocks(lines: list[str], src: str) -> list[str]:
    out = []
    i = 0
    while i < len(lines):
        m = ALERT_OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        kind = m.group(1)
        body = []
        i += 1
        closed = False
        while i < len(lines):
            if ALERT_END_RE.match(lines[i]):
                closed = True
                i += 1
                break
            body.append(lines[i])
            i += 1
        if not closed:
            report.append(f"{src}: unterminated alerts.{kind} block - left unchanged")
            out.append("{{site.data.alerts." + kind + "}}")
            out.extend(body)
            continue
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        header = '??? note "Proof"' if kind == "proof" else f"!!! {kind}"
        out.append(header)
        for b in body:
            out.append(f"    {b}" if b.strip() else "")
        out.append("")
    return out


def transform(text: str, src: str, section: str) -> tuple[dict, str]:
    meta, body = parse_front_matter(text, src)
    lines = body.split("\n")

    # Line-level passes (fence-aware where it matters).
    staged = []
    in_fence = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            staged.append(line)
            continue
        if in_fence:
            staged.append(line)
            continue
        if SERIES_INCLUDE_RE.match(line):
            while staged and (BR_LINE_RE.match(staged[-1]) or not staged[-1].strip()):
                staged.pop()
            continue
        am = ALERT_INCLUDE_RE.match(line)
        if am:
            indent, kind, content = am.groups()
            content = fix_inline_math(content, src, idx)
            staged.append(f"!!! {kind}")
            staged.append(f"    {content}")
            staged.append("")
            continue
        line = P_TAG_RE.sub("", line)
        line = IMG_RE.sub(
            lambda m: f"![](../../assets/images/prob/{m.group(1)})"
            + (f'{{ width="{m.group(2)}" .center }}' if m.group(2) else "{ .center }"),
            line,
        )
        if BR_LINE_RE.match(line):
            continue
        if not line.strip():
            staged.append("")
            continue
        staged.append(line)

    staged = convert_math(staged, src)
    staged = convert_alert_blocks(staged, src)

    # Collapse blank runs, assemble with title.
    final = [f"# {meta.get('title', '(untitled)')}", ""]
    prev_blank = True
    for line in staged:
        if not line.strip():
            if prev_blank:
                continue
            final.append("")
            prev_blank = True
        else:
            final.append(line.rstrip())
            prev_blank = False
    while final and not final[-1].strip():
        final.pop()

    leftovers = [
        f"{src}:{n}: leftover Liquid/HTML: {l.strip()[:60]}"
        for n, l in enumerate(final, start=1)
        if ("{%" in l or "{{site." in l or "<p" in l.lower())
    ]
    report.extend(leftovers)
    return meta, "\n".join(final) + "\n"


def migrate_section(section: str) -> list[tuple[int, str, str]]:
    src_dir = ROOT / "pages" / section
    dst_dir = ROOT / "docs" / "appendix" / section
    dst_dir.mkdir(parents=True, exist_ok=True)

    weighted, unweighted = [], {}
    for path in sorted(src_dir.glob("*.md")):
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"), path.name)
        w = meta.get("weight", "")
        if w.isdigit():
            weighted.append((int(w), path))
        else:
            unweighted[path.stem] = path

    order = sorted(weighted)
    next_w = (max(w for w, _ in order) if order else 0) + 1
    for stem in UNWEIGHTED_ORDER[section]:
        if stem in unweighted:
            order.append((next_w, unweighted.pop(stem)))
            next_w += 1
    for stem, path in unweighted.items():
        report.append(f"{path.name}: unweighted page not in UNWEIGHTED_ORDER - appended last")
        order.append((next_w, path))
        next_w += 1

    entries = []
    for num, path in order:
        meta, converted = transform(
            path.read_text(encoding="utf-8"), f"{section}/{path.name}", section
        )
        slug = path.stem.replace("_", "-")
        dst_name = f"{num:02d}-{slug}.md"
        (dst_dir / dst_name).write_text(converted, encoding="utf-8")
        old = meta.get("permalink", f"{path.stem}.html")
        redirects.append((old, f"appendix/{section}/{num:02d}-{slug}/"))
        entries.append((num, meta.get("title", path.stem), f"appendix/{section}/{dst_name}"))
    return entries


def main() -> int:
    nav = {s: migrate_section(s) for s in ("probability", "statistics")}

    print("=== nav fragment (paste into mkdocs.yml) ===")
    print("      - Probability:")
    for _, title, rel in nav["probability"]:
        print(f"          - \"{title}\": {rel}")
    print("      - Statistics:")
    for _, title, rel in nav["statistics"]:
        print(f"          - \"{title}\": {rel}")

    print("\n=== redirect mapping (old -> new, for the PR description) ===")
    for old, new in redirects:
        print(f"  {old}: {new}")

    print(f"\n=== review report ({len(report)} items) ===")
    for item in report:
        print(f"  {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
