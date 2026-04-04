.PHONY: install test lint format coverage clean

install:
	pip install -e ".[jax]" pytest coverage ruff black pre-commit

test:
	pytest -q

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

coverage:
	coverage run -m pytest -q
	coverage report -m

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
