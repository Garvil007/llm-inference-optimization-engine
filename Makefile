.PHONY: setup lint format test hardware-info

setup:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest --cov=src --cov-report=term-missing

hardware-info:
	python -m inference_optimizer hardware-info
