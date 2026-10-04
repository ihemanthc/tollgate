# Tollgate

A calibrated LLM cost router. A fine-tuned [Laya](https://huggingface.co/convaiinnovations/laya)
encoder predicts the cheapest model tier that will answer a query acceptably, then routes to it.

Tiers: `local_small` | `mid_tier` | `frontier`. The training label is the **minimum sufficient
tier** — the cheapest tier whose answer an LLM judge rated equivalent to the frontier answer.

Calibration is the deliverable: ECE and Brier matter as much as accuracy. Temperature scaling is
fitted on the calibration split, never on train.

## Quickstart

```bash
uv sync
uv run python -m tollgate.collect.prompts --limit 32   # seed prompts -> data/seed.parquet
make smoke                                             # end-to-end on a tiny slice
make figures                                           # plots -> docs/assets/
```

## Results

Metrics are machine-written to `reports/results.json` by `make smoke`; see that file (and the
figures under `docs/assets/`) for current numbers.

## Development

```bash
make lint | make test
```
