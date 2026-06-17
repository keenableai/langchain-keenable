.PHONY: all format lint test integration_test typing help

all: help

test:
	uv run --group test python -m pytest tests/unit_tests -q

integration_test:
	uv run --group test python -m pytest tests/integration_tests -q

lint:
	uv run --group lint python -m ruff check .
	uv run --group lint python -m ruff format --check .

format:
	uv run --group lint python -m ruff format .
	uv run --group lint python -m ruff check --fix .

typing:
	uv run --group typing python -m mypy langchain_keenable

help:
	@echo 'test             - run unit + standard unit tests'
	@echo 'integration_test - run live integration/standard tests'
	@echo 'lint             - run ruff check + format --check'
	@echo 'format           - apply ruff format + fixes'
	@echo 'typing           - run mypy'
