format:
	black src/

docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

docs-deploy:
	mkdocs gh-deploy

dev:
	pip install -e ".[dev]"
	pre-commit install

help:
	@echo '----'
	@echo 'format:               autoformat local code changes'
	@echo 'docs-build:           build the documentation'
	@echo 'docs-serve:           preview docs locally'
	@echo 'docs-deploy:          deploy a new version of docs to Github Pages'
	@echo 'dev: install          autolabel from source with dev dependencies'
	@echo '----'