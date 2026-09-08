"""Runner: cache skips calls, ledger on real calls only, retry/backoff, tier config."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tollgate import config
from tollgate.collect import runner as runner_mod
from tollgate.collect.runner import Runner, cache_key, read_runs, write_runs
from tollgate.config import RunnerConfig, TierConfig, parse_tiers, tier_config
from tollgate.schema import TIER_ORDER, QueryRecord, Tier

LOCAL = TierConfig(tier=Tier.LOCAL_SMALL, model="ollama/fake", local=True)
FRONTIER = TierConfig(tier=Tier.FRONTIER, model="openai/fake")


def _record(prompt: str = "What is 2+2?") -> QueryRecord:
    return QueryRecord(query_id=f"id-{prompt}", prompt=prompt, source="test")


class FakeLLM:
    """Stands in for litellm.acompletion; `failures` are raised before the first success."""

    def __init__(self, *failures: Exception) -> None:
        self.failures = list(failures)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        if self.failures:
            raise self.failures.pop(0)
        content = f"answer from {kwargs['model']}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


def _runner(tmp_path: Path, llm: FakeLLM, **overrides: Any) -> Runner:
    cfg = RunnerConfig(
        cache_dir=tmp_path / "cache",
        ledger_path=tmp_path / "ledger.jsonl",
        backoff_base_s=0.001,
        **overrides,
    )
    return Runner(cfg, completion_fn=llm, cost_fn=lambda _: 0.0125)


def _ledger(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "ledger.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_second_run_hits_cache_and_skips_call(tmp_path: Path) -> None:
    llm = FakeLLM()
    first = asyncio.run(_runner(tmp_path, llm).run(_record(), FRONTIER))
    second = asyncio.run(_runner(tmp_path, llm).run(_record(), FRONTIER))

    assert len(llm.calls) == 1
    assert (first.cached, first.cost_usd) == (False, 0.0125)
    assert (second.cached, second.cost_usd) == (True, 0.0)
    assert second.completion == first.completion == "answer from openai/fake"
    assert (tmp_path / "cache" / f"{cache_key(_record().prompt, FRONTIER.model)}.json").exists()


def test_ledger_line_per_real_call_only(tmp_path: Path) -> None:
    llm = FakeLLM()
    runner = _runner(tmp_path, llm)
    asyncio.run(runner.run_many([_record("a"), _record("b")], [LOCAL, FRONTIER]))
    asyncio.run(runner.run_many([_record("a")], [LOCAL, FRONTIER]))  # all cached

    lines = _ledger(tmp_path)
    assert len(lines) == len(llm.calls) == 4
    assert set(lines[0]) == {
        "ts",
        "tier",
        "model",
        "prompt_tokens",
        "completion_tokens",
        "usd",
        "latency_ms",
    }
    by_tier = {(x["tier"], x["usd"]) for x in lines}
    assert by_tier == {("local_small", 0.0), ("frontier", 0.0125)}


class CappedLLM(FakeLLM):
    """Every reply uses the whole output cap, like a reasoning model that never stops thinking."""

    async def __call__(self, **kwargs: Any) -> Any:
        response = await super().__call__(**kwargs)
        response.usage.completion_tokens = kwargs["max_tokens"]
        return response


def test_cut_off_reply_is_asked_again_only_under_a_higher_cap(tmp_path: Path) -> None:
    llm = CappedLLM()
    asyncio.run(_runner(tmp_path, llm, max_tokens=1024).run(_record(), FRONTIER))
    asyncio.run(_runner(tmp_path, llm, max_tokens=1024).run(_record(), FRONTIER))
    assert len(llm.calls) == 1  # same cap: the cut-off reply is what that call returns

    rerun = asyncio.run(_runner(tmp_path, llm, max_tokens=4096).run(_record(), FRONTIER))
    assert (len(llm.calls), llm.calls[-1]["max_tokens"], rerun.cached) == (2, 4096, False)
    assert len(_ledger(tmp_path)) == 2  # the re-ask is a real, ledgered call


def test_complete_reply_is_kept_when_the_cap_is_raised(tmp_path: Path) -> None:
    llm = FakeLLM()  # 3 completion tokens: finished well under any cap
    asyncio.run(_runner(tmp_path, llm, max_tokens=1024).run(_record(), FRONTIER))
    again = asyncio.run(_runner(tmp_path, llm, max_tokens=4096).run(_record(), FRONTIER))
    assert (len(llm.calls), again.cached) == (1, True)


def test_entries_without_a_recorded_cap_count_as_legacy(tmp_path: Path) -> None:
    llm = CappedLLM()
    asyncio.run(
        _runner(tmp_path, llm, max_tokens=runner_mod.LEGACY_MAX_TOKENS).run(_record(), FRONTIER)
    )
    path = tmp_path / "cache" / f"{cache_key(_record().prompt, FRONTIER.model)}.json"
    entry = json.loads(path.read_text())
    del entry["max_tokens"]  # written before entries recorded their cap
    path.write_text(json.dumps(entry))

    key = cache_key(_record().prompt, FRONTIER.model)
    assert runner_mod.read_usable_cache(tmp_path / "cache", key, 1024) is not None
    assert runner_mod.read_usable_cache(tmp_path / "cache", key, 4096) is None


def test_duplicate_in_one_batch_calls_once(tmp_path: Path) -> None:
    llm = FakeLLM()
    runs = asyncio.run(_runner(tmp_path, llm).run_many([_record(), _record()], [FRONTIER]))
    assert len(llm.calls) == 1
    assert sorted(r.cached for r in runs) == [False, True]


def _peak_in_flight(tmp_path: Path, tiers: list[TierConfig], **cfg: Any) -> dict[str, int]:
    in_flight: dict[str, int] = {}
    peak: dict[str, int] = {}

    async def slow(**kwargs: Any) -> Any:
        model = kwargs["model"]
        in_flight[model] = in_flight.get(model, 0) + 1
        peak[model] = max(peak.get(model, 0), in_flight[model])
        await asyncio.sleep(0.01)
        in_flight[model] -= 1
        return await FakeLLM()(**kwargs)

    runner = Runner(
        RunnerConfig(cache_dir=tmp_path / "c", ledger_path=tmp_path / "l", **cfg),
        completion_fn=slow,
    )
    asyncio.run(runner.run_many([_record(str(i)) for i in range(6)], tiers))
    return peak


def test_hosted_concurrency_is_bounded(tmp_path: Path) -> None:
    assert _peak_in_flight(tmp_path, [FRONTIER], concurrency=2) == {FRONTIER.model: 2}


def test_local_models_get_their_own_smaller_pool(tmp_path: Path) -> None:
    # Ollama on CPU is serial: queueing more calls there only burns their timeout.
    peak = _peak_in_flight(tmp_path, [LOCAL, FRONTIER], concurrency=4, local_concurrency=1)
    assert peak == {LOCAL.model: 1, FRONTIER.model: 4}


def test_local_calls_use_the_local_timeout(tmp_path: Path) -> None:
    llm = FakeLLM()
    cfg = RunnerConfig(
        cache_dir=tmp_path / "c", ledger_path=tmp_path / "l", timeout_s=5, local_timeout_s=600
    )
    asyncio.run(Runner(cfg, completion_fn=llm).run_many([_record()], [LOCAL, FRONTIER]))
    assert sorted(c["timeout"] for c in llm.calls) == [5, 600]


def test_transient_errors_retry_then_succeed(tmp_path: Path) -> None:
    llm = FakeLLM(ConnectionError("reset"), TimeoutError())
    run = asyncio.run(_runner(tmp_path, llm).run(_record(), LOCAL))
    assert run.error is None
    assert len(llm.calls) == 3
    assert len(_ledger(tmp_path)) == 1


def test_non_retryable_error_fails_fast_and_is_not_cached(tmp_path: Path) -> None:
    llm = FakeLLM(ValueError("bad request"))
    run = asyncio.run(_runner(tmp_path, llm).run(_record(), LOCAL))
    assert run.error == "ValueError: bad request"
    assert len(llm.calls) == 1
    assert _ledger(tmp_path) == []

    retry = asyncio.run(_runner(tmp_path, llm).run(_record(), LOCAL))
    assert retry.error is None and not retry.cached


def test_retries_exhausted(tmp_path: Path) -> None:
    llm = FakeLLM(*[ConnectionError("down")] * 5)
    run = asyncio.run(_runner(tmp_path, llm, max_retries=2).run(_record(), LOCAL))
    assert run.error is not None and "down" in run.error
    assert len(llm.calls) == 3


def test_parse_tiers() -> None:
    assert parse_tiers("all") == TIER_ORDER
    assert parse_tiers("frontier, local_small") == (Tier.LOCAL_SMALL, Tier.FRONTIER)
    with pytest.raises(ValueError):
        parse_tiers("gigantic")


def test_tier_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.MODEL_ENV_VARS[Tier.MID_TIER], raising=False)
    assert tier_config(Tier.LOCAL_SMALL).model == "ollama/qwen2.5:7b"
    with pytest.raises(ValueError, match="TOLLGATE_MID_TIER_MODEL"):
        tier_config(Tier.MID_TIER)
    monkeypatch.setenv(config.MODEL_ENV_VARS[Tier.FRONTIER], "anthropic/some-model")
    assert tier_config(Tier.FRONTIER).model == "anthropic/some-model"


def test_write_runs_upserts_across_invocations(tmp_path: Path) -> None:
    llm = FakeLLM()
    runner = _runner(tmp_path, llm)
    path = tmp_path / "runs.jsonl"
    write_runs(asyncio.run(runner.run_many([_record("a"), _record("b")], [LOCAL])), path)
    write_runs(asyncio.run(runner.run_many([_record("a")], [FRONTIER, LOCAL])), path)

    runs = read_runs(path)
    assert sorted((r.query_id, r.tier.value) for r in runs) == [
        ("id-a", "frontier"),
        ("id-a", "local_small"),
        ("id-b", "local_small"),
    ]
    assert next(r for r in runs if r.query_id == "id-a" and r.tier is Tier.LOCAL_SMALL).cached


def test_configured_prices_and_key_are_used(tmp_path: Path) -> None:
    llm = FakeLLM()
    priced = TierConfig(
        tier=Tier.FRONTIER,
        model="openai/claude-sonnet",
        api_base="https://llm.example.com/v1",
        api_key="sk-test",
        usd_per_mtok_in=3.0,
        usd_per_mtok_out=15.0,
    )

    def no_table_lookup(_: Any) -> float:
        raise AssertionError("litellm price table consulted despite configured prices")

    runner = Runner(
        RunnerConfig(cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl"),
        completion_fn=llm,
        cost_fn=no_table_lookup,
    )
    run = asyncio.run(runner.run(_record(), priced))
    # FakeLLM reports 7 prompt + 3 completion tokens: (7 * $3 + 3 * $15) / 1M
    assert run.cost_usd == pytest.approx(66e-6)
    assert (llm.calls[0]["api_key"], llm.calls[0]["api_base"]) == (
        "sk-test",
        "https://llm.example.com/v1",
    )
    assert _ledger(tmp_path)[0]["usd"] == pytest.approx(66e-6)


def test_only_our_retry_layer_retries(tmp_path: Path) -> None:
    llm = FakeLLM()
    asyncio.run(_runner(tmp_path, llm).run(_record(), FRONTIER))
    assert llm.calls[0]["max_retries"] == 0  # the OpenAI client must not retry underneath us


def test_quiet_http_logs() -> None:
    import logging

    from tollgate.collect.runner import quiet_http_logs

    quiet_http_logs()
    assert logging.getLogger("LiteLLM").level == logging.WARNING
    assert logging.getLogger("openai").level == logging.WARNING
