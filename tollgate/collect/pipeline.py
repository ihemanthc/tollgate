"""`tollgate collect-all` and `tollgate judge`: seed -> tier runs -> judge -> dataset.

Resumable by construction: every finished model call is cached (runner.py) the moment it
returns and ledgered once, so a run killed at any point restarts by replaying the cache and
pays only for calls that never finished. Runs and verdicts files are upserted, never appended.

No paid call happens before a dry-run cost estimate has been printed, and none happens at all
without --yes. Free work (seed download, local Ollama calls) needs no confirmation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from pydantic import BaseModel

from tollgate import paths
from tollgate.collect.build_dataset import build_examples, report, write_dataset
from tollgate.collect.judge import (
    JUDGE_TAG,
    JUDGE_TEMPLATE,
    build_prompt,
    judge_all,
    read_verdicts,
    write_verdicts,
)
from tollgate.collect.prompts import SOURCES, load_seed_prompts, write_seed
from tollgate.collect.runner import (
    LedgerEntry,
    Runner,
    cache_key,
    load_records,
    read_cache,
    read_runs,
    write_runs,
)
from tollgate.config import (
    ModelConfig,
    RunnerConfig,
    TierConfig,
    check_judge_independent,
    judge_config,
    tier_config,
)
from tollgate.schema import TIER_ORDER, JudgeVerdict, QueryRecord, Tier, TierRun

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 8
# Expected completion lengths when the ledger has no history for a model yet.
DEFAULT_ANSWER_TOKENS = 400
DEFAULT_VERDICT_TOKENS = 60
MIN_LEDGER_ROWS = 3
CHAT_OVERHEAD_TOKENS = 8

Echo = Callable[[str], None]


def _tokens(text: str) -> int:
    """Rough token count (4 chars/token): an estimate, not a tokenizer."""
    return len(text) // 4 + CHAT_OVERHEAD_TOKENS


def _price(model: ModelConfig) -> tuple[float, float] | None:
    """USD per (prompt, completion) token from litellm's price table; None if it has none."""
    if model.local:
        return 0.0, 0.0
    try:
        import litellm

        info = litellm.get_model_info(model.model)
        return float(info["input_cost_per_token"]), float(info["output_cost_per_token"])
    except Exception:
        return None


class StageEstimate(BaseModel):
    stage: str
    model: str
    local: bool
    calls: int
    cached: int
    prompt_tokens: int  # uncached calls only, here and below
    completion_tokens: int
    completion_ceiling: int  # every uncached call hitting max_tokens
    usd_per_token: tuple[float, float] | None
    seconds: float | None = None  # local stages: from ledger latency

    @property
    def new_calls(self) -> int:
        return self.calls - self.cached

    def usd(self, ceiling: bool = False) -> float | None:
        if self.usd_per_token is None:
            return None
        completion = self.completion_ceiling if ceiling else self.completion_tokens
        return self.prompt_tokens * self.usd_per_token[0] + completion * self.usd_per_token[1]


class CostEstimate(BaseModel):
    queries: int
    stages: list[StageEstimate]

    @property
    def needs_confirmation(self) -> bool:
        """True when any uncached call would go to a hosted (possibly paid) model."""
        return any(s.new_calls and not s.local for s in self.stages)

    @property
    def unpriced(self) -> list[str]:
        return [s.model for s in self.stages if s.new_calls and s.usd() is None]

    def total(self, ceiling: bool = False) -> float:
        return sum(s.usd(ceiling) or 0.0 for s in self.stages)


class _History(BaseModel):
    completion_tokens: float
    latency_s: float


def _ledger_history(path: Path) -> dict[str, _History]:
    """Mean completion tokens and latency per model, once it has a few ledgered calls."""
    if not path.exists():
        return {}
    rows: dict[str, list[LedgerEntry]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = LedgerEntry.model_validate_json(line)
            rows[entry.model].append(entry)
    return {
        model: _History(
            completion_tokens=sum(e.completion_tokens for e in es) / len(es),
            latency_s=sum(e.latency_ms for e in es) / len(es) / 1000,
        )
        for model, es in rows.items()
        if len(es) >= MIN_LEDGER_ROWS
    }


def estimate(
    records: Sequence[QueryRecord],
    tiers: Sequence[TierConfig],
    judge_model: ModelConfig | None,
    cfg: RunnerConfig,
    runs: Sequence[TierRun] | None = None,
) -> CostEstimate:
    """What the pipeline would call, what the cache already holds, and what the rest costs.

    `runs` (judge-only estimates) says which candidate answers exist; without it every cheaper
    tier in `tiers` is assumed to succeed. Judge prompts are exact when both answers are cached,
    approximated from expected answer lengths otherwise. Judge parse retries are not included.
    """
    history = _ledger_history(cfg.ledger_path)

    def answer_tokens(model: str) -> int:
        h = history.get(model)
        return round(h.completion_tokens) if h else DEFAULT_ANSWER_TOKENS

    stages: list[StageEstimate] = []
    for tier in tiers:
        new = [
            r for r in records if read_cache(cfg.cache_dir, cache_key(r.prompt, tier.model)) is None
        ]
        h = history.get(tier.model)
        stages.append(
            StageEstimate(
                stage=tier.tier.value,
                model=tier.model,
                local=tier.local,
                calls=len(records),
                cached=len(records) - len(new),
                prompt_tokens=sum(_tokens(r.prompt) for r in new),
                completion_tokens=len(new) * answer_tokens(tier.model),
                completion_ceiling=len(new) * cfg.max_tokens,
                usd_per_token=_price(tier),
                seconds=(len(new) * h.latency_s / cfg.local_concurrency)
                if tier.local and h
                else None,
            )
        )
    if judge_model is not None:
        stages.append(_judge_estimate(records, tiers, judge_model, cfg, runs, answer_tokens))
    return CostEstimate(queries=len(records), stages=stages)


def _judge_estimate(
    records: Sequence[QueryRecord],
    tiers: Sequence[TierConfig],
    judge_model: ModelConfig,
    cfg: RunnerConfig,
    runs: Sequence[TierRun] | None,
    answer_tokens: Callable[[str], int],
) -> StageEstimate:
    by_query: dict[str, dict[Tier, TierRun]] = defaultdict(dict)
    for run in runs or []:
        by_query[run.query_id][run.tier] = run
    models = {t.tier: t.model for t in tiers}

    def answer(record: QueryRecord, tier: Tier) -> tuple[str | None, str]:
        """(known answer text or None, model) for one tier of one query."""
        if runs is not None:
            run = by_query[record.query_id].get(tier)
            return (run.completion if run and run.error is None else None), (
                run.model if run else ""
            )
        model = models.get(tier, "")
        hit = read_cache(cfg.cache_dir, cache_key(record.prompt, model)) if model else None
        return (hit.completion if hit else None), model

    calls = cached = prompt_tokens = 0
    template = _tokens(JUDGE_TEMPLATE)
    for record in records:
        reference, ref_model = answer(record, Tier.FRONTIER)
        if runs is not None and reference is None:
            continue  # judge_query skips a query without a frontier answer
        candidates = (
            [
                t
                for t, r in by_query[record.query_id].items()
                if t is not Tier.FRONTIER and r.error is None
            ]
            if runs is not None
            else [t.tier for t in tiers if t.tier is not Tier.FRONTIER]
        )
        for tier in candidates:
            calls += 1
            text, model = answer(record, tier)
            if reference is not None and text is not None:
                prompt = build_prompt(record.prompt, reference, text)
                if read_cache(cfg.cache_dir, cache_key(prompt, judge_model.model)) is not None:
                    cached += 1
                    continue
                prompt_tokens += _tokens(prompt)
            else:
                prompt_tokens += (
                    template
                    + _tokens(record.prompt)
                    + answer_tokens(ref_model)
                    + answer_tokens(model)
                )
    new = calls - cached
    history = _ledger_history(cfg.ledger_path).get(judge_model.model)
    return StageEstimate(
        stage=JUDGE_TAG,
        model=judge_model.model,
        local=judge_model.local,
        calls=calls,
        cached=cached,
        prompt_tokens=prompt_tokens,
        completion_tokens=new
        * (round(history.completion_tokens) if history else DEFAULT_VERDICT_TOKENS),
        completion_ceiling=new * cfg.max_tokens,
        usd_per_token=_price(judge_model),
    )


def format_estimate(est: CostEstimate) -> str:
    lines = [
        f"cost estimate for {est.queries} queries (dry run: nothing called yet)",
        f"{'stage':<12} {'model':<32} {'calls':>6} {'cached':>6} {'~in tok':>9} {'~out tok':>9}"
        f" {'~USD':>9} {'max USD':>9}",
    ]
    for s in est.stages:
        usd, cap = s.usd(), s.usd(ceiling=True)
        lines.append(
            f"{s.stage:<12} {s.model[:32]:<32} {s.calls:>6} {s.cached:>6} {s.prompt_tokens:>9}"
            f" {s.completion_tokens:>9} {'?' if usd is None else f'${usd:.4f}':>9}"
            f" {'?' if cap is None else f'${cap:.4f}':>9}"
        )
    lines.append(
        f"{'total':<12} {'':<32} {'':>6} {'':>6} {'':>9} {'':>9}"
        f" {f'${est.total():.4f}':>9} {f'${est.total(ceiling=True):.4f}':>9}"
    )
    lines.append("max USD assumes every uncached answer runs to max_tokens.")
    if est.unpriced:
        lines.append(
            f"WARNING: no litellm price for {sorted(set(est.unpriced))}: the total above leaves"
            " them out, and the cost ledger will record $0 for their calls."
        )
    for s in est.stages:
        if s.local and s.new_calls:
            eta = (
                "unknown (no ledger history yet)"
                if s.seconds is None
                else f"~{s.seconds / 3600:.1f} h"
            )
            lines.append(f"{s.stage}: {s.new_calls} local calls, {eta} on this machine")
    return "\n".join(lines)


def confirm_or_exit(est: CostEstimate, yes: bool, echo: Echo = typer.echo) -> None:
    """Print the estimate; stop here unless the run is free or --yes was given."""
    echo(format_estimate(est))
    if est.needs_confirmation and not yes:
        echo("dry run only: no model was called. Re-run with --yes to make these calls.")
        raise typer.Exit(code=0)


def ledger_totals(path: Path) -> dict[str, float]:
    """USD per ledger tag (tier name or 'judge') plus 'total', over the whole ledger."""
    totals: dict[str, float] = defaultdict(float)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = LedgerEntry.model_validate_json(line)
                totals[entry.tier] += entry.usd
                totals["total"] += entry.usd
    return dict(totals)


def ensure_seed(limit: int, sources: Sequence[str]) -> Path:
    """Reuse the seed if it has `limit` rows; otherwise regenerate it (free, deterministic)."""
    path = paths.seed_path()
    have = len(pd.read_parquet(path, columns=["query_id"])) if path.exists() else 0
    if have < limit:
        records = load_seed_prompts(limit, sources)
        write_seed(records, path)
        log.info("seed: %d prompts -> %s", len(records), path)
    return path


def check_local_models(models: Sequence[ModelConfig]) -> None:
    """Fail before any paid call if a local Ollama endpoint is down."""
    for model in models:
        if not model.local or not model.api_base:
            continue
        try:
            urllib.request.urlopen(f"{model.api_base.rstrip('/')}/api/tags", timeout=5).close()
        except (urllib.error.URLError, OSError) as exc:
            raise typer.BadParameter(
                f"{model.model}: Ollama is not reachable at {model.api_base} ({exc}). "
                "Start it (`ollama serve`) before collecting."
            ) from None


class PipelineResult(BaseModel):
    runs: int
    run_errors: int
    verdicts: int
    judge_failures: int
    examples: int
    skipped: dict[str, int]
    api_calls: int


def run_pipeline(
    records: Sequence[QueryRecord],
    tiers: Sequence[TierConfig],
    judge_model: ModelConfig,
    runner: Runner,
    *,
    split_seed: int = 0,
    echo: Echo = typer.echo,
) -> PipelineResult:
    """Tier runs, judging and dataset assembly for `records`; every file written is an upsert."""

    async def collect() -> tuple[list[TierRun], list[JudgeVerdict], int]:
        # One event loop for both stages: the runner's semaphores and locks belong to it.
        runs = await runner.run_many(records, tiers)
        write_runs(runs, paths.tier_runs_path())
        echo(f"tier runs: {len(runs)} ({sum(r.error is not None for r in runs)} errors)")
        verdicts, failed = await judge_all(runner, judge_model, records, runs)
        write_verdicts(verdicts, paths.verdicts_path())
        echo(f"verdicts: {len(verdicts)} ({failed} judge calls failed)")
        return runs, verdicts, failed

    runs, verdicts, failed = asyncio.run(collect())
    examples, skipped = build_examples(
        records,
        read_runs(paths.tier_runs_path()),
        read_verdicts(paths.verdicts_path()),
        cache_dir=runner.config.cache_dir,
        seed=split_seed,
    )
    if examples:
        write_dataset(examples, paths.dataset_path())
        echo(report(examples, skipped))
        echo(f"-> {paths.dataset_path()}")
    else:
        echo(f"0 of {len(records)} queries labelable; skipped: {dict(skipped)}")
    return PipelineResult(
        runs=len(runs),
        run_errors=sum(r.error is not None for r in runs),
        verdicts=len(verdicts),
        judge_failures=failed,
        examples=len(examples),
        skipped=dict(skipped),
        api_calls=runner.api_calls,
    )


def _configs() -> tuple[list[TierConfig], ModelConfig]:
    try:
        tiers, judge_model = [tier_config(t) for t in TIER_ORDER], judge_config()
        frontier = [t.model for t in tiers if t.tier is Tier.FRONTIER]
        check_judge_independent(judge_model.model, frontier)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    return tiers, judge_model


def _echo_ledger(before: dict[str, float]) -> None:
    after = ledger_totals(paths.ledger_path())
    spent = after.get("total", 0.0) - before.get("total", 0.0)
    by_tag = {k: round(v, 6) for k, v in sorted(after.items()) if k != "total"}
    typer.echo(
        f"cost ledger {paths.ledger_path()}: total ${after.get('total', 0.0):.4f} "
        f"(this run ${spent:.4f}); by stage {json.dumps(by_tag)}"
    )


def collect_all(
    limit: Annotated[int, typer.Option(help="First N seed prompts.")] = DEFAULT_LIMIT,
    yes: Annotated[bool, typer.Option("--yes", help="Make the paid calls estimated.")] = False,
    concurrency: Annotated[int, typer.Option(min=1, help="In-flight hosted calls.")] = 4,
    sources: Annotated[str, typer.Option(help="Seed sources, comma list.")] = ",".join(SOURCES),
    split_seed: Annotated[int, typer.Option(help="Seed for the stratified split.")] = 0,
) -> None:
    """Seed -> all tiers -> judge -> dataset; a killed run restarts without re-spending."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tiers, judge_model = _configs()
    seed = ensure_seed(limit, [s.strip() for s in sources.split(",") if s.strip()])
    records = load_records(seed, limit)
    runner = Runner(RunnerConfig(concurrency=concurrency))
    confirm_or_exit(estimate(records, tiers, judge_model, runner.config), yes)
    check_local_models([*tiers, judge_model])
    before = ledger_totals(paths.ledger_path())
    result = run_pipeline(records, tiers, judge_model, runner)
    typer.echo(f"{result.api_calls} new model calls")
    _echo_ledger(before)


def judge_cmd(
    limit: Annotated[int, typer.Option(help="First N seed prompts.")] = DEFAULT_LIMIT,
    yes: Annotated[bool, typer.Option("--yes", help="Make the paid calls estimated.")] = False,
    concurrency: Annotated[int, typer.Option(min=1, help="In-flight judge calls.")] = 4,
) -> None:
    """Judge existing tier runs of the first --limit seed prompts into verdicts.jsonl."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        judge_model = judge_config()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    for path in (paths.seed_path(), paths.tier_runs_path()):
        if not path.exists():
            raise typer.BadParameter(f"{path} not found; run run-tiers (or collect-all) first.")
    records = load_records(paths.seed_path(), limit)
    runs = read_runs(paths.tier_runs_path())
    try:
        check_judge_independent(
            judge_model.model, {r.model for r in runs if r.tier is Tier.FRONTIER}
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    runner = Runner(RunnerConfig(concurrency=concurrency))
    confirm_or_exit(estimate(records, [], judge_model, runner.config, runs=runs), yes)
    check_local_models([judge_model])
    before = ledger_totals(paths.ledger_path())
    verdicts, failed = asyncio.run(judge_all(runner, judge_model, records, runs))
    write_verdicts(verdicts, paths.verdicts_path())
    typer.echo(f"{len(verdicts)} verdicts -> {paths.verdicts_path()} ({failed} failed)")
    _echo_ledger(before)
