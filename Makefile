.PHONY: serve build deploy check clean

# Port 8000 is often taken by other dev servers; override with `make serve PORT=9000`
PORT ?= 8001
PYTHON ?= python3

serve:            ## live-reload preview at http://127.0.0.1:$(PORT)
	$(PYTHON) scripts/sync_index.py && $(PYTHON) -m mkdocs serve -a 127.0.0.1:$(PORT)

build: check      ## build the static site into _site/
	$(PYTHON) -m mkdocs build --strict

check:            ## regenerate index.md and validate links
	$(PYTHON) scripts/sync_index.py
	$(PYTHON) scripts/check_links.py

deploy: check     ## push the built site to the gh-pages branch
	$(PYTHON) -m mkdocs gh-deploy --force

clean:
	rm -rf _site .cache
