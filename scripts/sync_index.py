#!/usr/bin/env python3
"""Generate docs/index.md from README.md.

README.md is the GitHub landing page and links to pages as `docs/<part>/<page>.md`.
The MkDocs site uses `docs/` as its root, so index.md must link to `<part>/<page>.md`.
This keeps a single source of truth: edit README.md, then run this script.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent
readme = (root / "README.md").read_text(encoding="utf-8")

# strip the leading docs/ from markdown link targets only (not from prose)
out = re.sub(r"\]\(docs/", "](", readme)
# the mkdocs.yml pointer in README refers to the repo root, which has no site equivalent
# front matter: the home page is an index, so give it the full width
out = ("---\nhide:\n  - navigation\n---\n\n"
       "<!-- AUTO-GENERATED from README.md by scripts/sync_index.py. Do not edit. -->\n\n"
       + out)

dest = root / "docs" / "index.md"
dest.write_text(out, encoding="utf-8")
print(f"wrote {dest.relative_to(root)} ({len(out.splitlines())} lines)")
