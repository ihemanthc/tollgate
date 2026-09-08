.PHONY: lint test smoke figures

LIMIT ?= 8

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

# CPU plumbing check on the existing dataset (collect-all builds it). Writes run `smoke`, never
# the real run, and evaluates it uncalibrated: a handful of rows cannot fit temperatures.
smoke:
	uv run tollgate train --limit $(LIMIT) --epochs 1 --run-name smoke
	uv run tollgate export-logits --checkpoint checkpoints/smoke/best --limit $(LIMIT)
	uv run tollgate evaluate --run smoke --limit $(LIMIT) --latency-samples $(LIMIT) --allow-uncalibrated
	uv run python -m tollgate.eval.plots

figures:
	uv run python -m tollgate.eval.plots
