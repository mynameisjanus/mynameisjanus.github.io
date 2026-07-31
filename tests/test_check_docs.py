"""Tests for the course-structure gate in scripts/check_docs.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_docs

LESSON = "docs/part-09-software-engineering/01-example.md"


def write(tmp_path, relpath, text):
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def check(tmp_path, relpath, text):
    p = write(tmp_path, relpath, text)
    return check_docs.check_file(p, tmp_path / "docs")


GOOD = """# Example

Prose.

```python
print("a")
# => a
```

!!! abstract "Key takeaways"
    - Something.

## Where this goes next

Onward.
"""


def test_a_well_formed_lesson_passes(tmp_path):
    assert check(tmp_path, LESSON, GOOD) == []


def test_scaffold_stub_is_exempt_from_structure_rules(tmp_path):
    stub = "# Example\n\n!!! warning \"Under development\"\n    Not written yet.\n"
    assert check(tmp_path, LESSON, stub) == []


def test_missing_where_this_goes_next_is_reported(tmp_path):
    out = check(tmp_path, LESSON, GOOD.replace("## Where this goes next", "## Other"))
    assert any("Where this goes next" in m for m in out)


def test_duplicate_key_takeaways_is_reported(tmp_path):
    out = check(tmp_path, LESSON, GOOD + '\n!!! abstract "Key takeaways"\n    - Dup.\n')
    assert any("found 2" in m for m in out)


def test_misaligned_pinned_continuation_is_reported(tmp_path):
    bad = GOOD.replace('print("a")\n# => a', 'print("a")\n# => a\n#  b')
    out = check(tmp_path, LESSON, bad)
    assert any("continuation" in m for m in out)


def test_inline_output_comment_is_not_treated_as_a_pinned_block(tmp_path):
    """`x = 1  # => 1` mid-block is the inline form and must not be flagged."""
    inline = GOOD.replace('print("a")\n# => a',
                          'x = 1  # => 1\nprint("a")\n# => a')
    assert check(tmp_path, LESSON, inline) == []


def test_log_lines_before_the_marker_are_allowed(tmp_path):
    """Some blocks show log output, then the result on the '# =>' line."""
    logged = GOOD.replace('print("a")\n# => a',
                          'print("a")\n# retrying...\n# => a')
    assert check(tmp_path, LESSON, logged) == []


def test_broken_relative_link_is_reported(tmp_path):
    out = check(tmp_path, LESSON, GOOD + "\n[gone](99-nope.md)\n")
    assert any("broken link" in m for m in out)


def test_resolving_relative_link_is_accepted(tmp_path):
    write(tmp_path, "docs/part-09-software-engineering/02-next.md", "# Next\n")
    assert check(tmp_path, LESSON, GOOD + "\n[next](02-next.md)\n") == []


def test_appendix_pages_are_exempt_from_lesson_structure(tmp_path):
    body = "# Appendix page\n\nNo takeaways here.\n"
    out = check(tmp_path, "docs/appendix/part-01-foundations/01-sets.md", body)
    assert out == []


@pytest.mark.parametrize("name", ["index.md"])
def test_part_index_pages_are_exempt(tmp_path, name):
    body = "# Part IX\n\nA landing page.\n"
    assert check(tmp_path, f"docs/part-09-software-engineering/{name}", body) == []
