"""Join seed + tier runs + judge verdicts into LabeledExample rows; split 70/15/15.

Splits are stratified on (label tier, source). Within each stratum, queries are ordered by a
seeded hash and assigned by systematic sampling from a hashed random start, so every stratum
splits 70/15/15 to within one row, and singleton strata spread across splits instead of all
landing in train. Assignment is deterministic for a given --split-seed and query set, but it is
recomputed on every build: fine-tune and fit temperature from the SAME build.

The calibration split exists only for temperature fitting. Consumers read rows through
read_split(), which returns exactly one split; training must only ever ask for Split.TRAIN.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import accumulate
from pathlib import Path
from typing import Annotated, TypeVar

import pandas as pd
import typer

from tollgate import paths
from tollgate.collect.judge import read_verdicts
from tollgate.collect.label import NeedsFn, QueryLabels, heuristic_needs, label_query
from tollgate.collect.runner import cache_key, load_records, read_cache, read_runs
from tollgate.schema import (
    TIER_ORDER,
    JudgeVerdict,
    LabeledExample,
    QueryRecord,
    Split,
    Tier,
    TierRun,
)

T = TypeVar("T")

DEFAULT_LIMIT = 8
SPLIT_FRACTIONS: tuple[tuple[Split, float], ...] = (
    (Split.TRAIN, 0.70),
    (Split.CALIBRATION, 0.15),
    (Split.TEST, 0.15),
)
SKIP_NO_FRONTIER = "no successful frontier run"
SKIP_UNDETERMINED = "a tier cheaper than the label has no verdict"


def _unit(*parts: str) -> float:
    """Deterministic uniform [0, 1) from string parts."""
    digest = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return int(digest[:16], 16) / 16**16


def _bucket_at(position: float, fractions: Sequence[tuple[T, float]]) -> T:
    bounds = accumulate(fraction for _, fraction in fractions)
    for (bucket, _), bound in zip(fractions, bounds, strict=True):
        if position < bound:
            return bucket
    return fractions[-1][0]  # float rounding at the top edge


def stratified_assign(
    strata: dict[str, tuple[str, ...]], fractions: Sequence[tuple[T, float]], seed: int = 0
) -> dict[str, T]:
    """Map query_id -> bucket, given query_id -> stratum key such as (tier, source)."""
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for qid, key in strata.items():
        groups[key].append(qid)
    buckets: dict[str, T] = {}
    for key, qids in groups.items():
        qids.sort(key=lambda q: _unit(str(seed), "query", q))
        start = _unit(str(seed), "stratum", *key)
        for k, qid in enumerate(qids):
            buckets[qid] = _bucket_at((k + start) / len(qids), fractions)
    return buckets


def assign_splits(strata: dict[str, tuple[str, ...]], seed: int = 0) -> dict[str, Split]:
    """Map query_id -> train / calibration / test, 70/15/15 within each stratum."""
    return stratified_assign(strata, SPLIT_FRACTIONS, seed)


def _true_cost(run: TierRun | None, prompt: str, cache_dir: Path) -> float | None:
    """TierRun.cost_usd is 0 on cache hits; the cache entry keeps what the call really cost."""
    if run is None:
        return None
    entry = read_cache(cache_dir, cache_key(prompt, run.model))
    return entry.cost_usd if entry is not None else run.cost_usd


def build_examples(
    records: Sequence[QueryRecord],
    runs: Sequence[TierRun],
    verdicts: Sequence[JudgeVerdict],
    *,
    cache_dir: Path | None = None,
    seed: int = 0,
    needs_fn: NeedsFn = heuristic_needs,
) -> tuple[list[LabeledExample], Counter[str]]:
    """Label and split every query whose label is fully determined; count the rest by reason."""
    cache_dir = cache_dir or paths.cache_dir()
    ids = [r.query_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query_id in seed records")
    runs_by_query: dict[str, dict[Tier, TierRun]] = defaultdict(dict)
    for run in runs:
        runs_by_query[run.query_id][run.tier] = run
    verdicts_by_query: dict[str, list[JudgeVerdict]] = defaultdict(list)
    for verdict in verdicts:
        verdicts_by_query[verdict.query_id].append(verdict)

    skipped: Counter[str] = Counter()
    labeled: list[tuple[QueryRecord, QueryLabels, dict[Tier, TierRun]]] = []
    for record in records:
        tier_runs = runs_by_query.get(record.query_id, {})
        frontier = tier_runs.get(Tier.FRONTIER)
        if frontier is None or frontier.error is not None:
            skipped[SKIP_NO_FRONTIER] += 1
            continue
        labels = label_query(frontier, verdicts_by_query.get(record.query_id, []), needs_fn)
        if not labels.determined:
            skipped[SKIP_UNDETERMINED] += 1
            continue
        labeled.append((record, labels, tier_runs))

    splits = assign_splits(
        {record.query_id: (labels.tier.value, record.source) for record, labels, _ in labeled},
        seed,
    )
    examples = [
        LabeledExample(
            query_id=record.query_id,
            prompt=record.prompt,
            source=record.source,
            split=splits[record.query_id],
            tier=labels.tier,
            needs_tools=labels.needs_tools,
            needs_rag=labels.needs_rag,
            judged_tiers=labels.judged_tiers,
            frontier_cost_usd=_true_cost(tier_runs[Tier.FRONTIER], record.prompt, cache_dir),
            label_cost_usd=_true_cost(tier_runs.get(labels.tier), record.prompt, cache_dir),
        )
        for record, labels, tier_runs in labeled
    ]
    return examples, skipped


def write_dataset(examples: Sequence[LabeledExample], path: Path | None = None) -> Path:
    path = path or paths.dataset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([e.model_dump(mode="json") for e in examples]).to_parquet(path, index=False)
    return path


def read_split(split: Split, path: Path | None = None) -> list[LabeledExample]:
    """Exactly one split. There is deliberately no way to read train and calibration together."""
    frame = pd.read_parquet(path or paths.dataset_path())
    frame = frame[frame["split"] == Split(split).value]
    frame = frame.astype(object).where(frame.notna(), None)
    return [
        LabeledExample.model_validate({**row, "judged_tiers": tuple(row["judged_tiers"])})
        for row in frame.to_dict(orient="records")
    ]


def report(examples: Sequence[LabeledExample], skipped: Counter[str]) -> str:
    frame = pd.DataFrame([e.model_dump(mode="json") for e in examples])
    tiers = [t.value for t in TIER_ORDER]
    counts = frame["tier"].value_counts().reindex(tiers, fill_value=0)
    dist = pd.DataFrame({"n": counts, "pct": (100 * counts / counts.sum()).round(1)})

    def crosstab(rows: str, order: list[str] | None = None) -> pd.DataFrame:
        table = pd.crosstab(frame[rows], frame["tier"], margins=True, margins_name="total")
        table = table.reindex(columns=[*tiers, "total"], fill_value=0)
        return table if order is None else table.reindex([*order, "total"], fill_value=0)

    return "\n".join(
        [
            f"{len(examples)} labeled examples; skipped: {dict(skipped) or 'none'}",
            "",
            "label distribution",
            dist.to_string(),
            "",
            "label x source",
            crosstab("source").to_string(),
            "",
            "label x split",
            crosstab("split", [s.value for s, _ in SPLIT_FRACTIONS]).to_string(),
        ]
    )


def build_dataset(
    limit: Annotated[int, typer.Option(help="Seed prompts to consider (first N).")] = DEFAULT_LIMIT,
    seed: Annotated[Path | None, typer.Option(help="[default: <data dir>/seed.parquet]")] = None,
    runs: Annotated[Path | None, typer.Option(help="[default: <data dir>/tier_runs.jsonl]")] = None,
    verdicts: Annotated[
        Path | None, typer.Option(help="[default: <data dir>/verdicts.jsonl]")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="[default: <data dir>/dataset.parquet]")] = None,
    cache_dir: Annotated[
        Path | None, typer.Option(help="Runner cache, for true costs. [default: <data dir>/cache]")
    ] = None,
    split_seed: Annotated[int, typer.Option(help="Seed for the stratified split.")] = 0,
) -> None:
    """Join seed + tier runs + verdicts into a labeled, 70/15/15-split dataset."""
    seed, runs = seed or paths.seed_path(), runs or paths.tier_runs_path()
    verdicts, out = verdicts or paths.verdicts_path(), out or paths.dataset_path()
    cache_dir = cache_dir or paths.cache_dir()
    for path, flag in ((seed, "--seed"), (runs, "--runs"), (verdicts, "--verdicts")):
        if not path.exists():
            raise typer.BadParameter(f"{path} not found.", param_hint=flag)
    records = load_records(seed, limit)
    examples, skipped = build_examples(
        records, read_runs(runs), read_verdicts(verdicts), cache_dir=cache_dir, seed=split_seed
    )
    if not examples:
        typer.echo(f"0 of {len(records)} queries labelable; skipped: {dict(skipped)}", err=True)
        raise typer.Exit(code=1)
    write_dataset(examples, out)
    typer.echo(report(examples, skipped))
    typer.echo(f"-> {out}")


if __name__ == "__main__":
    typer.run(build_dataset)
