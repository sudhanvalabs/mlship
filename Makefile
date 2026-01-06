# Makefile for mlship development

.PHONY: help install test lint format clean build publish

help:
	@echo "mlship Development Commands"
	@echo ""
	@echo "  make install    Install package in development mode"
	@echo "  make test       Run tests"
	@echo "  make lint       Check code style"
	@echo "  make format     Format code"
	@echo "  make clean      Remove build artifacts"
	@echo "  make build      Build distribution packages"
	@echo "  make publish    Publish to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)"

install:
	uv pip install -e ".[dev,all]"

test:
	pytest -v --cov=mlship --cov-report=term-missing

test-fast:
	pytest -v

lint:
	black --check mlship/ tests/
	ruff check mlship/ tests/
	mypy mlship/

format:
	black mlship/ tests/
	ruff check --fix mlship/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*
