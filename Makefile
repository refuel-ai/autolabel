setup:
	pip install --upgrade pip
	pip install '.[all]'

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

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*

clean: clean-pyc clean-test

test: clean
	OPENAI_API_KEY=test_key ANTHROPIC_API_KEY=test_key REFUEL_API_KEY=test_key pytest

check: test

help:
	@echo '----'
	@echo 'format:               autoformat local code changes'
	@echo 'docs-build:           build the documentation'
	@echo 'docs-serve:           preview docs locally'
	@echo 'docs-deploy:          deploy a new version of docs to Github Pages'
	@echo 'dev: install          autolabel from source with dev dependencies'
	@echo 'clean-pyc:            remove Python file artifacts'
	@echo 'clean-test:           remove test and coverage artifacts'
	@echo 'clean:                remove all build, test and coverage artifacts'
	@echo 'test:                 clean previous build and test artifacts, and run all tests'
	@echo '----'