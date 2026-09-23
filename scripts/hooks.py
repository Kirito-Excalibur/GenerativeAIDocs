"""MkDocs hooks.

docs/index.md is generated from README.md by scripts/sync_index.py, so its
"edit this page" button must point at README.md, not at the generated file.
"""

def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == "index.md" and config.get("repo_url"):
        page.edit_url = config["repo_url"].rstrip("/") + "/edit/main/README.md"
    return markdown
