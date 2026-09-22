#!/usr/bin/env python3
"""Validate every internal markdown link and anchor in the wiki.

Anchors are checked against BOTH slug conventions, because the wiki is read in two
renderers that disagree:

  * GitHub          - each space becomes one hyphen, runs are NOT collapsed
  * python-markdown - whitespace runs ARE collapsed to a single hyphen
    (MkDocs/Material)

A heading like "Part V - Diffusion & Vision" therefore slugs differently in each.
A link is only accepted if it matches a heading under BOTH conventions, which forces
headings that are link targets to avoid em-dashes and ampersands.

Exit status 1 if anything is broken.
"""
import re
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"_site", "venv", ".venv", "node_modules", ".git"}

LINK = re.compile(r"(?<!!)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^()\s]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)")


def _clean(text: str) -> str:
    text = re.sub(r"`", "", text)
    text = re.sub(r"\*\*|__|\*|_", "", text)
    text = re.sub(r"\$[^$]*\$", "", text)          # inline math
    return re.sub(r"[^\w\s-]", "", text.lower())


def slug_github(text: str) -> str:
    return _clean(text).strip().replace(" ", "-")


def slug_pymd(text: str) -> str:
    return re.sub(r"\s+", "-", _clean(text)).strip("-")


def main() -> int:
    files = sorted(
        p for p in ROOT.rglob("*.md")
        if not any(part in SKIP_DIRS for part in p.parts)
    )

    anchors: dict[Path, set[str]] = {}
    for f in files:
        found: set[str] = set()
        in_code = False
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            m = HEADING.match(line)
            if m:
                found.add(slug_github(m.group(2)))
                found.add(slug_pymd(m.group(2)))
        anchors[f.resolve()] = found

    bad_file: list[tuple[str, str]] = []
    bad_anchor: list[tuple[str, str]] = []
    checked = 0

    for f in files:
        for _label, target in LINK.findall(f.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue

            if target.startswith("#"):                 # same-page anchor
                checked += 1
                if target[1:] not in anchors[f.resolve()]:
                    bad_anchor.append((str(f.relative_to(ROOT)), target))
                continue

            if not target.endswith(".md") and ".md#" not in target:
                continue                               # not a doc link (code fragment etc.)

            checked += 1
            path_part, _, anchor = target.partition("#")
            resolved = (f.parent / urllib.parse.unquote(path_part)).resolve()
            if not resolved.exists():
                bad_file.append((str(f.relative_to(ROOT)), target))
            elif anchor and anchor not in anchors.get(resolved, set()):
                bad_anchor.append((str(f.relative_to(ROOT)), target))

    print(f"{len(files)} files, {checked} internal links checked")
    for where, target in bad_file:
        print(f"  MISSING FILE   {where} -> {target}")
    for where, target in bad_anchor:
        print(f"  BROKEN ANCHOR  {where} -> {target}")

    if bad_file or bad_anchor:
        print(f"\nFAILED: {len(bad_file)} missing files, {len(bad_anchor)} broken anchors")
        return 1
    print("all links OK (valid under both GitHub and MkDocs slug rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
