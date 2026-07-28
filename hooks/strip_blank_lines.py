"""Collapse the blank lines Jinja leaves behind in Material's templates.

MkDocs renders pages with `trim_blocks`/`lstrip_blocks` disabled, so every
`{% if %}`, `{% set %}` and `{% include %}` that produces no output still emits
its indentation and newline. `partials/nav-item.html` is a ~170-line recursive
macro of almost pure control flow, so a site with a large nav ends up ~70%
blank lines, in runs of 100+.

Whitespace outside `<pre>` is insignificant in HTML, so dropping those lines
cannot change rendering. Inside `<pre>` it is significant -- code samples do
contain blank lines -- so those regions are passed through untouched.
"""

import re

PRE = re.compile(r"<(pre|textarea)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
BLANK_LINE = re.compile(r"^[ \t]*\n", re.MULTILINE)


def _strip(output: str) -> str:
    out = []
    pos = 0
    for match in PRE.finditer(output):
        out.append(BLANK_LINE.sub("", output[pos : match.start()]))
        out.append(match.group(0))
        pos = match.end()
    out.append(BLANK_LINE.sub("", output[pos:]))
    return "".join(out)


def on_post_page(output: str, page=None, config=None) -> str:
    return _strip(output)


def on_post_template(output_content: str, template_name: str = "", config=None) -> str:
    # Theme templates (404.html, sitemap.xml) bypass on_post_page. Only touch
    # HTML -- leave sitemap.xml and friends exactly as the theme emits them.
    if not template_name.endswith(".html"):
        return output_content
    return _strip(output_content)
