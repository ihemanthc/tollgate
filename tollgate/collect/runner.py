"""Run QueryRecords against model tiers. The only place Tollgate calls a model.

Every call is cached to data/cache/ by sha256(prompt, model) and never repeated; every real
(uncached) call appends one line to data/cost_ledger.jsonl. The one exception: a reply that hit
its output cap is asked again once the cap is raised, since the same prompt under a higher cap
can say more. A reply that finished on its own is never re-asked.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from pydantic import Field

from tollgate import paths
from tollgate.config import (
    ModelConfig,
    RunnerConfig,
    TierConfig,
    parse_tiers,
    tier_config,
)
from tollgate.schema import QueryRecord, TierRun, TollgateModel

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 8
# The output cap every call ran with before cache entries recorded their own.
LEGACY_MAX_TOKENS = 1024

CompletionFn = Callable[..., Awaitable[Any]]
CostFn = Callable[[Any], float]

# litellm exception class names worth retrying; anything else (auth, bad request) fails fast.
_RETRYABLE_LITELLM = (
    "RateLimitError",
    "APIConnectionError",
    "Timeout",
    "ServiceUnavailableError",
    "InternalServerError",
    "BadGatewayError",
)


class CachedCompletion(TollgateModel):
    """On-disk cache entry. cost_usd is what the original call cost."""

    model: str
    completion: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    latency_s: float = Field(ge=0.0)
    created_at: datetime
    max_tokens: int = Field(default=LEGACY_MAX_TOKENS, ge=1, description="Output cap of the call.")

    def cut_off_below(self, max_tokens: int) -> bool:
        """True if this reply ran into its cap and a call capped at `max_tokens` could say more.

        Reasoning models spend most of the cap thinking, so a capped reply can be a half
        sentence, half a JSON object, or empty.
        """
        return self.completion_tokens >= self.max_tokens and max_tokens > self.max_tokens


class LedgerEntry(TollgateModel):
    ts: datetime
    tier: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)


def cache_key(prompt: str, model: str) -> str:
    # NUL separator so (prompt, model) pairs can't collide by concatenation.
    return hashlib.sha256(f"{prompt}\x00{model}".encode()).hexdigest()


def quiet_http_logs() -> None:
    """Keep per-request INFO lines from litellm, openai and httpx out of the console."""
    for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "openai", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    return any(cls.__name__ in _RETRYABLE_LITELLM for cls in type(exc).__mro__)


def _litellm_completion(**kwargs: Any) -> Awaitable[Any]:
    import litellm

    litellm.suppress_debug_info = True
    return litellm.acompletion(**kwargs)


def _litellm_cost(response: Any) -> float:
    import litellm

    return float(litellm.completion_cost(completion_response=response))


class Runner:
    """Holds the concurrency semaphore and per-key locks for one batch of calls."""

    def __init__(
        self,
        config: RunnerConfig | None = None,
        completion_fn: CompletionFn = _litellm_completion,
        cost_fn: CostFn = _litellm_cost,
    ) -> None:
        self.config = config or RunnerConfig()
        self._completion_fn = completion_fn
        self._cost_fn = cost_fn
        self._semaphore = asyncio.Semaphore(self.config.concurrency)
        # Local models get their own, smaller pool: queueing inside Ollama would count against
        # the timeout, and hosted tiers should not wait behind a slow local model.
        self._local_semaphore = asyncio.Semaphore(self.config.local_concurrency)
        self._locks: dict[str, asyncio.Lock] = {}
        self.api_calls = 0

    async def complete(
        self, prompt: str, model: ModelConfig, *, tag: str, json_mode: bool = False
    ) -> tuple[CachedCompletion, bool]:
        """One single-user-message completion, cached. Returns (entry, cached); raises on failure.

        `tag` fills the ledger's tier column ('local_small', 'judge', ...).
        """
        key = cache_key(prompt, model.model)
        # Per-key lock: a duplicate in the same batch waits and then hits the cache.
        async with self._locks.setdefault(key, asyncio.Lock()):
            hit = self._read_cache(key)
            if hit is not None and not hit.cut_off_below(self.config.max_tokens):
                return hit, True
            if hit is not None:
                log.info(
                    "%s: cached reply was cut off at %d tokens; asking again with max_tokens=%d",
                    model.model,
                    hit.max_tokens,
                    self.config.max_tokens,
                )
            async with self._local_semaphore if model.local else self._semaphore:
                entry = await self._call_with_retry(prompt, model, json_mode)
            # Ledger first: if the cache write fails we still have a record of the spend.
            self._append_ledger(tag, entry)
            self._write_cache(key, entry)
            return entry, False

    async def run(self, record: QueryRecord, tier: TierConfig) -> TierRun:
        """Answer one query at one tier: cache hit, fresh call, or a TierRun with error set."""
        try:
            entry, cached = await self.complete(record.prompt, tier, tag=tier.tier.value)
        except Exception as exc:
            log.error("%s %s failed: %r", tier.tier, record.query_id[:12], exc)
            return TierRun(
                query_id=record.query_id,
                tier=tier.tier,
                model=tier.model,
                completion="",
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                latency_s=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _to_run(record, tier, entry, cached=cached)

    async def run_many(
        self, records: Sequence[QueryRecord], tiers: Sequence[TierConfig]
    ) -> list[TierRun]:
        """All (record, tier) pairs, concurrently; results in record-major, tier-minor order."""
        return list(await asyncio.gather(*(self.run(r, t) for r in records for t in tiers)))

    async def _call_with_retry(
        self, prompt: str, model: ModelConfig, json_mode: bool
    ) -> CachedCompletion:
        cfg = self.config
        for attempt in range(cfg.max_retries + 1):
            try:
                return await self._call_once(prompt, model, json_mode)
            except Exception as exc:
                if attempt >= cfg.max_retries or not is_retryable(exc):
                    raise
                delay = min(cfg.backoff_max_s, cfg.backoff_base_s * 2**attempt)
                delay *= random.uniform(0.5, 1.0)
                log.warning(
                    "%s attempt %d: %r; retrying in %.1fs", model.model, attempt + 1, exc, delay
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _call_once(
        self, prompt: str, model: ModelConfig, json_mode: bool
    ) -> CachedCompletion:
        cfg = self.config
        timeout = cfg.local_timeout_s if model.local else cfg.timeout_s
        kwargs: dict[str, Any] = {
            "model": model.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "timeout": timeout,
            # One retry layer: ours (_call_with_retry), not the OpenAI client's hidden one too.
            "max_retries": 0,
        }
        if model.api_base:
            kwargs["api_base"] = model.api_base
        if model.api_key:
            kwargs["api_key"] = model.api_key
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["drop_params"] = True  # providers without JSON mode still get the prompt
        self.api_calls += 1
        start = time.perf_counter()
        response = await asyncio.wait_for(self._completion_fn(**kwargs), timeout=timeout)
        latency_s = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return CachedCompletion(
            model=model.model,
            completion=response.choices[0].message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(response, model, prompt_tokens, completion_tokens),
            latency_s=latency_s,
            created_at=datetime.now(UTC),
            max_tokens=cfg.max_tokens,
        )

    def _cost(
        self, response: Any, model: ModelConfig, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Configured prices first (local is $0), then litellm's price table, else $0 + warning."""
        configured = model.price(prompt_tokens, completion_tokens)
        if configured is not None:
            return configured
        try:
            return self._cost_fn(response)
        except Exception as exc:
            log.warning(
                "no price for %s (%r); ledgering $0. Set TOLLGATE_<ROLE>_USD_PER_MTOK_IN/_OUT.",
                model.model,
                exc,
            )
            return 0.0

    def _cache_path(self, key: str) -> Path:
        return self.config.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> CachedCompletion | None:
        return read_cache(self.config.cache_dir, key)

    def _write_cache(self, key: str, entry: CachedCompletion) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(entry.model_dump_json(), encoding="utf-8")
        os.replace(tmp, path)

    def _append_ledger(self, tag: str, entry: CachedCompletion) -> None:
        line = LedgerEntry(
            ts=entry.created_at,
            tier=tag,
            model=entry.model,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            usd=entry.cost_usd,
            latency_ms=round(entry.latency_s * 1000, 1),
        )
        path = self.config.ledger_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line.model_dump_json() + "\n")


def read_cache(cache_dir: Path, key: str) -> CachedCompletion | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    return CachedCompletion.model_validate_json(path.read_text(encoding="utf-8"))


def read_usable_cache(cache_dir: Path, key: str, max_tokens: int) -> CachedCompletion | None:
    """The cache entry a Runner capped at `max_tokens` would reuse; None if it would call."""
    hit = read_cache(cache_dir, key)
    return None if hit is None or hit.cut_off_below(max_tokens) else hit


def _to_run(
    record: QueryRecord, tier: TierConfig, entry: CachedCompletion, *, cached: bool
) -> TierRun:
    return TierRun(
        query_id=record.query_id,
        tier=tier.tier,
        model=entry.model,
        completion=entry.completion,
        prompt_tokens=entry.prompt_tokens,
        completion_tokens=entry.completion_tokens,
        cost_usd=0.0 if cached else entry.cost_usd,
        latency_s=entry.latency_s,
        cached=cached,
    )


def load_records(path: Path, limit: int) -> list[QueryRecord]:
    frame = pd.read_parquet(path).head(limit)
    frame = frame.astype(object).where(frame.notna(), None)
    return [QueryRecord.model_validate(row) for row in frame.to_dict(orient="records")]


def read_runs(path: Path) -> list[TierRun]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [TierRun.model_validate_json(line) for line in lines if line.strip()]


def write_runs(runs: Sequence[TierRun], path: Path) -> Path:
    """Upsert by (query_id, tier), so tiers run in separate invocations accumulate."""
    merged = {(r.query_id, r.tier): r for r in read_runs(path)}
    merged.update({(r.query_id, r.tier): r for r in runs})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.writelines(run.model_dump_json() + "\n" for run in merged.values())
    os.replace(tmp, path)
    return path


def run_tiers(
    limit: Annotated[int, typer.Option(help="Number of seed prompts to run.")] = DEFAULT_LIMIT,
    tiers: Annotated[str, typer.Option(help="'all' or comma list of tiers.")] = "all",
    concurrency: Annotated[int, typer.Option(min=1, help="Max in-flight calls.")] = 4,
    yes: Annotated[bool, typer.Option("--yes", help="Make the paid calls estimated.")] = False,
    seed: Annotated[
        Path | None, typer.Option(help="Seed prompts. [default: <data dir>/seed.parquet]")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="TierRun JSONL. [default: <data dir>/tier_runs.jsonl]")
    ] = None,
) -> None:
    """Run the first --limit seed prompts against each requested tier."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    quiet_http_logs()
    seed, out = seed or paths.seed_path(), out or paths.tier_runs_path()
    tier_cfgs = [tier_config(t) for t in parse_tiers(tiers)]
    if not seed.exists():
        raise typer.BadParameter(
            f"{seed} not found; run `python -m tollgate.collect.prompts` first.",
            param_hint="--seed",
        )
    records = load_records(seed, limit)
    runner = Runner(RunnerConfig(concurrency=concurrency))
    from tollgate.collect.pipeline import check_local_models, confirm_or_exit, estimate

    confirm_or_exit(estimate(records, tier_cfgs, None, runner.config), yes)
    check_local_models(tier_cfgs)
    runs = asyncio.run(runner.run_many(records, tier_cfgs))
    write_runs(runs, out)
    errors = sum(r.error is not None for r in runs)
    typer.echo(
        f"{len(runs)} runs ({len(records)} prompts x {len(tier_cfgs)} tiers) -> {out}: "
        f"{runner.api_calls} api calls, {sum(r.cached for r in runs)} cached, {errors} errors, "
        f"${sum(r.cost_usd for r in runs):.4f}"
    )
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    from tollgate.config import load_env

    load_env()
    typer.run(run_tiers)
