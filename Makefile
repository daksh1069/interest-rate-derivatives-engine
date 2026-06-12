.PHONY: help venv install dev fetch test lint format typecheck check clean

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

help:
	@echo "Targets:"
	@echo "  make install    Create venv and install runtime deps"
	@echo "  make dev        Install dev deps + editable package"
	@echo "  make fetch      Run the Phase 1 data pipeline (offline fallback if no FRED key)"
	@echo "  make test       Run pytest"
	@echo "  make lint       Run ruff linter"
	@echo "  make format     Auto-format with ruff"
	@echo "  make typecheck  Run mypy"
	@echo "  make check      lint + typecheck + test"
	@echo "  make clean      Remove caches and build artifacts"

$(VENV):
	$(PY) -m venv $(VENV)

install: $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e .

dev: $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

fetch:
	$(BIN)/python -m ird.data.fetch_sofr

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests

format:
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

typecheck:
	$(BIN)/mypy

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
