"""collect-all / judge: dry-run estimate, --yes gate, and resuming without re-spending."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from tollgate import paths
from tollgate.cli import app
from tollgate.collect import pipeline
from tollgate.collect.judge import build_prompt, read_verdicts, write_verdicts
from tollgate.collect.prompts import write_seed
from tollgate.collect.runner import Runner, read_runs, write_runs
from tollgate.config import ModelConfig, RunnerConfig, TierConfig
from tollgate.schema import JudgeVerdict, QueryRecord, Tier, TierRun

L = TierConfig(tier=Tier.LOCAL_SMALL, model="ollama/fake", local=True)
M = TierConfig(tier=Tier.MID_TIER, model="openai/mid")
F = TierConfig(tier=Tier.FRONTIER, model="openai/front")
TIERS = [L, M, F]
J = ModelConfig(model="openai/judge")
PRICES = {M.model: (1e-6, 4e-6), F.model: (1e-5, 5e-5), J.model: (2e-6, 8e-6)}
N = 6
REAL_PRICE = pipeline._price  # captured before the fixture swaps in fake prices


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(
        pipeline, "_price", lambda m: (0.0, 0.0) if m.local else PRICES.get(m.model)
    )
    return tmp_path / "data"


def _records(n: int = N) -> list[QueryRecord]:
    return [
        QueryRecord(query_id=f"q{i}", prompt=f"question {i}: " + "x" * 40, source=("a", "b")[i % 2])
        for i in range(n)
    ]


class FakeModels:
    """Tier answers name their model. The judge rates mid 'acceptable', local 'worse'."""

    def __init__(self, *failing: str) -> None:
        self.calls: Counter[str] = Counter()
        self.failing = set(failing)

    async def __call__(self, **kwargs: Any) -> Any:
        model, prompt = kwargs["model"], kwargs["messages"][0]["content"]
        self.calls[model] += 1
        if model in self.failing:
            raise ValueError("simulated outage")  # not retryable, so never cached
        if model == J.model:
            mid = f"<candidate>\nanswer from {M.model}" in prompt
            content = json.dumps({"verdict": "acceptable" if mid else "worse", "reason": "r"})
        else:
            content = f"answer from {model}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
        )


def _run(
    llm: FakeModels, records: list[QueryRecord] | None = None, *, canary: bool = True
) -> pipeline.PipelineResult:
    runner = Runner(RunnerConfig(), completion_fn=llm, cost_fn=lambda _: 0.001)
    return pipeline.run_pipeline(
        records or _records(), TIERS, J, runner, echo=lambda _: None, canary=canary
    )


def _ledger_lines() -> int:
    path = paths.ledger_path()
    return len(path.read_text().splitlines()) if path.exists() else 0


# --- estimate -----------------------------------------------------------------------------------


def test_estimate_counts_calls_and_prices_them() -> None:
    records = _records()
    est = pipeline.estimate(records, TIERS, J, RunnerConfig())
    by_stage = {s.stage: s for s in est.stages}
    assert [s.stage for s in est.stages] == ["local_small", "mid_tier", "frontier", "judge"]
    assert {s.stage: (s.calls, s.cached) for s in est.stages} == {
        "local_small": (N, 0),
        "mid_tier": (N, 0),
        "frontier": (N, 0),
        "judge": (2 * N, 0),  # local and mid, each judged against frontier
    }
    prompt = sum(pipeline._tokens(r.prompt) for r in records)
    mid = by_stage["mid_tier"]
    assert mid.usd() == pytest.approx(prompt * 1e-6 + N * pipeline.DEFAULT_ANSWER_TOKENS * 4e-6)
    cap = RunnerConfig().max_tokens
    assert mid.usd(ceiling=True) == pytest.approx(prompt * 1e-6 + N * cap * 4e-6)
    assert by_stage["local_small"].usd() == 0.0
    assert est.needs_confirmation and est.unpriced == []
    assert est.total() == pytest.approx(sum(s.usd() for s in est.stages))


def test_unpriced_model_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_price", lambda m: None if m is J else (0.0, 0.0))
    est = pipeline.estimate(_records(), TIERS, J, RunnerConfig())
    assert est.unpriced == [J.model]
    assert "WARNING: no price" in pipeline.format_estimate(est)


def test_without_yes_nothing_is_called(capsys: pytest.CaptureFixture[str]) -> None:
    est = pipeline.estimate(_records(), TIERS, J, RunnerConfig())
    with pytest.raises(typer.Exit) as stopped:
        pipeline.confirm_or_exit(est, yes=False)
    assert stopped.value.exit_code == 0
    out = capsys.readouterr().out
    assert "cost estimate for 6 queries" in out and "dry run only" in out
    pipeline.confirm_or_exit(est, yes=True)  # proceeds


def test_free_local_only_work_needs_no_confirmation() -> None:
    est = pipeline.estimate(_records(), [L], None, RunnerConfig())
    assert not est.needs_confirmation
    pipeline.confirm_or_exit(est, yes=False)  # does not exit


# --- full run, re-run, and resume ---------------------------------------------------------------


def test_pipeline_labels_everything_then_reruns_for_free() -> None:
    llm = FakeModels()
    result = _run(llm)
    assert (result.runs, result.run_errors, result.verdicts, result.examples) == (
        3 * N,
        0,
        2 * N,
        N,
    )
    assert _ledger_lines() == 5 * N  # 3 tiers + 2 judge calls per query
    labels = {v.candidate_tier: v.verdict for v in read_verdicts(paths.verdicts_path())}
    assert labels == {Tier.LOCAL_SMALL: "worse", Tier.MID_TIER: "acceptable"}

    calls_before = dict(llm.calls)
    again = _run(llm)
    assert dict(llm.calls) == calls_before and again.api_calls == 0
    assert _ledger_lines() == 5 * N
    after = pipeline.estimate(_records(), TIERS, J, RunnerConfig())
    assert not after.needs_confirmation and after.total() == 0.0


def test_run_killed_before_frontier_resumes_paying_only_for_what_is_missing() -> None:
    first = FakeModels(F.model)  # frontier never answers: as if killed before those calls
    result = _run(first, canary=False)
    assert result.run_errors == N and result.verdicts == 0 and result.examples == 0

    est = pipeline.estimate(_records(), TIERS, J, RunnerConfig())
    assert {s.stage: s.cached for s in est.stages} == {
        "local_small": N,
        "mid_tier": N,
        "frontier": 0,
        "judge": 0,
    }

    second = FakeModels()
    resumed = _run(second)
    assert second.calls == Counter({F.model: N, J.model: 2 * N})  # no local or mid re-spend
    assert resumed.examples == N
    assert _ledger_lines() == 5 * N  # every call ledgered exactly once across both runs


def test_run_killed_between_stages_rejudges_only() -> None:
    _run(FakeModels(J.model), canary=False)  # tier runs done, every judge call fails
    second = FakeModels()
    _run(second)
    assert second.calls == Counter({J.model: 2 * N})


def test_judge_estimate_is_exact_once_answers_exist() -> None:
    records = _records()
    _run(FakeModels(J.model), canary=False)
    runs = read_runs(paths.tier_runs_path())
    est = pipeline.estimate(records, [], J, RunnerConfig(), runs=runs)
    (judge,) = est.stages
    reference = f"answer from {F.model}"
    expected = sum(
        pipeline._tokens(build_prompt(r.prompt, reference, f"answer from {model}"))
        for r in records
        for model in (L.model, M.model)
    )
    assert (judge.calls, judge.cached, judge.prompt_tokens) == (2 * N, 0, expected)


def test_judge_estimate_skips_queries_without_frontier_answers() -> None:
    _run(FakeModels(F.model), canary=False)
    est = pipeline.estimate(
        _records(), [], J, RunnerConfig(), runs=read_runs(paths.tier_runs_path())
    )
    assert est.stages[0].calls == 0


# --- files and helpers --------------------------------------------------------------------------


def test_write_verdicts_upserts(tmp_path: Path) -> None:
    def verdict(tier: Tier, value: str) -> JudgeVerdict:
        return JudgeVerdict(
            query_id="q", candidate_tier=tier, verdict=value, reason="r", judge_model="j"
        )

    path = tmp_path / "v.jsonl"
    write_verdicts([verdict(Tier.LOCAL_SMALL, "worse"), verdict(Tier.MID_TIER, "worse")], path)
    write_verdicts([verdict(Tier.MID_TIER, "acceptable")], path)
    assert {v.candidate_tier: v.verdict for v in read_verdicts(path)} == {
        Tier.LOCAL_SMALL: "worse",
        Tier.MID_TIER: "acceptable",
    }


def test_ledger_totals_by_stage() -> None:
    _run(FakeModels())
    totals = pipeline.ledger_totals(paths.ledger_path())
    # the fake cost_fn charges $0.001 per hosted call; local calls are always $0
    assert totals["local_small"] == 0.0
    assert totals["mid_tier"] == pytest.approx(N * 0.001)
    assert totals["judge"] == pytest.approx(2 * N * 0.001)
    assert totals["total"] == pytest.approx(4 * N * 0.001)


def test_ensure_seed_reuses_a_long_enough_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    write_seed(_records(5))
    monkeypatch.setattr(pipeline, "load_seed_prompts", lambda *_: pytest.fail("re-downloaded"))
    assert pipeline.ensure_seed(3, ["gsm8k"]) == paths.seed_path()


# --- CLI ----------------------------------------------------------------------------------------


def test_collect_all_without_models_explains_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("TOLLGATE_MID_TIER_MODEL", "TOLLGATE_FRONTIER_MODEL", "TOLLGATE_JUDGE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    result = CliRunner().invoke(app, ["collect-all", "--limit", "2"])
    assert result.exit_code != 0
    assert "TOLLGATE_MID_TIER_MODEL is not set" in result.output


def test_judge_cli_dry_run_makes_no_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_JUDGE_MODEL", J.model)
    records = _records(2)
    write_seed(records)
    write_runs(
        [
            TierRun(
                query_id=r.query_id, tier=t.tier, model=t.model, completion="a",
                prompt_tokens=1, completion_tokens=1, cost_usd=0.0, latency_s=0.0,
            )
            for r in records
            for t in TIERS
        ],
        paths.tier_runs_path(),
    )  # fmt: skip
    result = CliRunner().invoke(app, ["judge", "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "dry run only" in result.output
    assert not paths.ledger_path().exists() and not paths.verdicts_path().exists()


def test_configured_prices_beat_the_price_table() -> None:
    priced = M.model_copy(update={"usd_per_mtok_in": 1.0, "usd_per_mtok_out": 4.0})
    assert REAL_PRICE(priced) == (1e-6, 4e-6)
    assert REAL_PRICE(L) == (0.0, 0.0)


# --- canary and unpriced refusal ----------------------------------------------------------------


@pytest.mark.parametrize("failing", [F.model, M.model])
def test_canary_stops_before_any_long_work(failing: str) -> None:
    llm = FakeModels(failing)
    with pytest.raises(pipeline.CanaryFailed, match=failing):
        _run(llm)
    # only the first query was tried, and the local tier was never started
    assert llm.calls[L.model] == 0
    assert sum(llm.calls.values()) == 2  # mid + frontier for query 0
    assert not paths.tier_runs_path().exists()


def test_canary_checks_the_judge_too() -> None:
    llm = FakeModels(J.model)
    with pytest.raises(pipeline.CanaryFailed, match="judge"):
        _run(llm)
    assert llm.calls[L.model] == 0


def test_canary_calls_are_reused_by_the_full_run() -> None:
    llm = FakeModels()
    _run(llm)
    # 3 tiers + 2 judge calls per query; the canary's mid/frontier/judge calls were not repeated
    assert sum(llm.calls.values()) == 5 * N


def test_yes_is_refused_while_a_paid_model_has_no_price(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipeline, "_price", lambda m: None if m is F else (0.0, 0.0))
    est = pipeline.estimate(_records(), TIERS, J, RunnerConfig())
    with pytest.raises(typer.Exit) as stopped:
        pipeline.confirm_or_exit(est, yes=True)
    assert stopped.value.exit_code == 2
    assert "refusing to run: no price for ['openai/front']" in capsys.readouterr().out


def test_an_explicit_zero_price_is_not_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    free = F.model_copy(update={"usd_per_mtok_in": 0.0, "usd_per_mtok_out": 0.0})
    monkeypatch.setattr(pipeline, "_price", REAL_PRICE)
    assert REAL_PRICE(free) == (0.0, 0.0)
    est = pipeline.estimate(_records(), [free], None, RunnerConfig())
    assert est.unpriced == []
    pipeline.confirm_or_exit(est, yes=True)  # proceeds
