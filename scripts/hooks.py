"""MkDocs hooks.

1. docs/index.md is generated from README.md by scripts/sync_index.py, so its
   "edit this page" button must point at README.md.
2. Pages use GitHub alert syntax for callouts so they render natively on GitHub:

       > [!TIP]
       > **Why three projections?** Because ...

   On the site these become Material admonitions, with a short bold lead-in
   lifted into the box title:

       !!! tip "Why three projections?"
           Because ...
"""
import re

TYPES = {"NOTE": "note", "TIP": "tip", "IMPORTANT": "info", "WARNING": "warning", "CAUTION": "danger"}
DEFAULT_TITLE = {"TIP": "Intuition", "WARNING": "Pitfall", "NOTE": "Note",
                 "IMPORTANT": "Important", "CAUTION": "Caution"}
ALERT = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")
LEAD = re.compile(r"^\*\*(?P<t>[^*]+?)\*\*\s*(?:—|–|:|\.)?\s*(?P<rest>.*)$")


def _convert(markdown: str) -> str:
    lines = markdown.split("\n")
    out, i, in_code = [], 0, False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else ALERT.match(line)
        if not m:
            out.append(line); i += 1; continue
        kind = m.group(1)
        body, i = [], i + 1
        while i < len(lines) and lines[i].startswith(">"):
            body.append(re.sub(r"^> ?", "", lines[i])); i += 1
        title = DEFAULT_TITLE[kind]
        if body:
            lead = LEAD.match(body[0])
            if lead:
                t = lead.group("t").strip().rstrip(":.")
                if len(t) <= 70 and not re.search(r"[$`\[\]]", t):
                    title = t
                    rest = lead.group("rest")
                    body[0] = rest[:1].upper() + rest[1:] if rest else ""
                    if not body[0]:
                        body.pop(0)
        out.append(f'!!! {TYPES[kind]} "{title.replace(chr(34), chr(39))}"')
        out.append("")
        out.extend(("    " + b) if b.strip() else "" for b in body)
        out.append("")
    return "\n".join(out)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == "index.md" and config.get("repo_url"):
        page.edit_url = config["repo_url"].rstrip("/") + "/edit/main/README.md"
    return _convert(markdown)
