# Makefile for NanoAccel development

.PHONY: help install install-dev test test-cov lint format type-check clean build docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install NanoAccel in development mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

install-test:  ## Install test dependencies
	pip install -e ".[test]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=nanoaccel --cov-report=html --cov-report=term-missing

test-fast:  ## Run fast tests only (skip slow tests)
	pytest -m "not slow"

lint:  ## Run linting
	flake8 nanoaccel/ tests/ examples/
	mypy nanoaccel/

format:  ## Format code
	black nanoaccel/ tests/ examples/
	isort nanoaccel/ tests/ examples/

format-check:  ## Check code formatting
	black --check nanoaccel/ tests/ examples/
	isort --check-only nanoaccel/ tests/ examples/

type-check:  ## Run type checking
	mypy nanoaccel/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean  ## Build package
	python -m build

install-pre-commit:  ## Install pre-commit hooks
	pre-commit install

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

check-requirements:  ## Check system requirements
	nanoaccel --check-requirements --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

cpu-info:  ## Display CPU information
	nanoaccel --cpu-info

example-basic:  ## Run basic usage example
	python examples/basic_usage.py

example-speculative:  ## Run speculative decoding example
	python examples/speculative_decoding.py

docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

all-checks: format-check lint type-check test  ## Run all checks

ci: install-dev install-pre-commit all-checks  ## Run CI pipeline locally
