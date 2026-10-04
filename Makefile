.PHONY: lint test smoke figures

LIMIT ?= 8

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

smoke:
	uv run python -m tollgate.collect.build_dataset --limit $(LIMIT)
	uv run python -m tollgate.train.train --limit $(LIMIT) --epochs 1
	uv run python -m tollgate.eval.metrics --limit $(LIMIT)

figures:
	uv run python -m tollgate.eval.plots
	uv run python -m tollgate.eval.cost_curve
