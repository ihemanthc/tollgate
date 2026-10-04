"""build_dataset: joins and skip rules, true costs, stratified split, split isolation, CLI."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from tollgate.cli import app
from tollgate.collect import build_dataset as bd
from tollgate.collect.prompts import write_seed
from tollgate.collect.runner import CachedCompletion, cache_key, write_runs
from tollgate.schema import JudgeVerdict, LabeledExample, QueryRecord, Split, Tier, TierRun, Verdict

L, M, F = Tier.LOCAL_SMALL, Tier.MID_TIER, Tier.FRONTIER


def _rec(qid: str, source: str = "gsm8k") -> QueryRecord:
    return QueryRecord(query_id=qid, prompt=f"prompt {qid}", source=source)


def _run(qid: str, tier: Tier, cost: float = 0.0, error: str | None = None) -> TierRun:
    return TierRun(
        query_id=qid,
        tier=tier,
        model=f"{tier}-model",
        completion="" if error else "ok",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=cost,
        latency_s=0.0,
        cached=False,
        error=error,
    )


def _v(qid: str, tier: Tier, verdict: Verdict) -> JudgeVerdict:
    return JudgeVerdict(
        query_id=qid, candidate_tier=tier, verdict=verdict, reason="r", judge_model="j"
    )


def _fixture() -> tuple[list[QueryRecord], list[TierRun], list[JudgeVerdict]]:
    records = [_rec(q) for q in ("local", "mid", "front", "gap_ok", "gap", "no_front", "front_err")]
    runs = [
        _run(q, t, cost=0.5 if t is F else 0.1)
        for q in ("local", "mid", "front", "gap_ok", "gap")
        for t in (L, M, F)
    ]
    runs += [_run("no_front", L), _run("front_err", F, error="Timeout")]
    verdicts = [
        _v("local", L, "equivalent"),
        _v("local", M, "worse"),
        _v("mid", L, "worse"),
        _v("mid", M, "acceptable"),
        _v("front", L, "worse"),
        _v("front", M, "worse"),
        _v("gap_ok", L, "acceptable"),  # mid never judged: irrelevant, local already passes
        _v("gap", M, "acceptable"),  # local never judged: label could be too expensive
    ]
    return records, runs, verdicts


def test_labels_and_skip_reasons(tmp_path: Path) -> None:
    examples, skipped = bd.build_examples(*_fixture(), cache_dir=tmp_path)
    assert {e.query_id: e.tier for e in examples} == {
        "local": L,
        "mid": M,
        "front": F,
        "gap_ok": L,
    }
    assert skipped == Counter({bd.SKIP_NO_FRONTIER: 2, bd.SKIP_UNDETERMINED: 1})
    by_id = {e.query_id: e for e in examples}
    assert (by_id["mid"].frontier_cost_usd, by_id["mid"].label_cost_usd) == (0.5, 0.1)
    assert by_id["front"].label_cost_usd == 0.5
    assert by_id["gap_ok"].judged_tiers == (L,)


def test_costs_come_from_cache_when_run_was_a_cache_hit(tmp_path: Path) -> None:
    records, runs, verdicts = _fixture()
    run = next(r for r in runs if r.query_id == "front" and r.tier is F)
    runs = [r for r in runs if r is not run] + [
        run.model_copy(update={"cost_usd": 0.0, "cached": True})
    ]
    entry = CachedCompletion(
        model=run.model,
        completion="ok",
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.042,
        latency_s=0.0,
        created_at=datetime.now(UTC),
    )
    (tmp_path / f"{cache_key('prompt front', run.model)}.json").write_text(entry.model_dump_json())
    examples, _ = bd.build_examples(records, runs, verdicts, cache_dir=tmp_path)
    front = next(e for e in examples if e.query_id == "front")
    assert front.frontier_cost_usd == front.label_cost_usd == 0.042


def test_split_is_stratified_within_one_row() -> None:
    sizes = {("local_small", "lmsys"): 400, ("mid_tier", "gsm8k"): 203, ("frontier", "mmlu"): 37}
    strata = {f"{k[0]}-{k[1]}-{i}": k for k, n in sizes.items() for i in range(n)}
    splits = bd.assign_splits(strata)
    assert set(splits) == set(strata)
    for key, n in sizes.items():
        counts = Counter(s for q, s in splits.items() if strata[q] == key)
        for split, fraction in bd.SPLIT_FRACTIONS:
            assert abs(counts[split] - fraction * n) <= 1, (key, split, counts)


def test_singleton_strata_are_spread_across_splits() -> None:
    strata = {f"q{i}": (f"stratum{i}",) for i in range(3000)}
    counts = Counter(bd.assign_splits(strata).values())
    for split, fraction in bd.SPLIT_FRACTIONS:
        assert abs(counts[split] / 3000 - fraction) < 0.03


def test_split_is_deterministic_and_seeded() -> None:
    strata = {f"q{i}": ("t", "s") for i in range(200)}
    reversed_strata = dict(reversed(list(strata.items())))
    assert bd.assign_splits(strata, seed=1) == bd.assign_splits(reversed_strata, seed=1)
    assert bd.assign_splits(strata, seed=1) != bd.assign_splits(strata, seed=2)


def test_read_split_roundtrip_and_isolation(tmp_path: Path) -> None:
    examples = [
        LabeledExample(
            query_id=f"q{i}",
            prompt=f"p{i}",
            source="gsm8k",
            split=split,
            tier=L,
            needs_tools=False,
            needs_rag=i % 2 == 0,
            judged_tiers=(L, M) if i % 2 else (),
            frontier_cost_usd=None if i % 3 == 0 else 0.25,
            label_cost_usd=0.0,
        )
        for i, split in enumerate([Split.TRAIN] * 7 + [Split.CALIBRATION] * 2 + [Split.TEST] * 2)
    ]
    path = bd.write_dataset(examples, tmp_path / "dataset.parquet")
    per_split = {s: bd.read_split(s, path) for s in Split}
    assert all(e.split is s for s, rows in per_split.items() for e in rows)
    assert sorted(e.query_id for rows in per_split.values() for e in rows) == sorted(
        e.query_id for e in examples
    )
    assert {e.query_id for e in per_split[Split.TRAIN]}.isdisjoint(
        e.query_id for e in per_split[Split.CALIBRATION]
    )
    assert {e for rows in per_split.values() for e in rows} == set(examples)


def _write_inputs(tmp_path: Path) -> list[str]:
    records, runs, verdicts = _fixture()
    write_seed(records, tmp_path / "seed.parquet")
    write_runs(runs, tmp_path / "runs.jsonl")
    (tmp_path / "verdicts.jsonl").write_text("".join(v.model_dump_json() + "\n" for v in verdicts))
    return [
        "build-dataset",
        "--limit", "50",
        "--seed", str(tmp_path / "seed.parquet"),
        "--runs", str(tmp_path / "runs.jsonl"),
        "--verdicts", str(tmp_path / "verdicts.jsonl"),
        "--out", str(tmp_path / "dataset.parquet"),
        "--cache-dir", str(tmp_path / "cache"),
    ]  # fmt: skip


def test_cli_writes_dataset_and_prints_tables(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, _write_inputs(tmp_path))
    assert result.exit_code == 0, result.output
    for heading in ("4 labeled examples", "label distribution", "label x source", "label x split"):
        assert heading in result.output
    assert sum(len(bd.read_split(s, tmp_path / "dataset.parquet")) for s in Split) == 4


def test_cli_missing_verdicts_fails_cleanly(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    (tmp_path / "verdicts.jsonl").unlink()
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert not (tmp_path / "dataset.parquet").exists()


def test_cli_nothing_labelable_exits_nonzero(tmp_path: Path) -> None:
    args = _write_inputs(tmp_path)
    (tmp_path / "verdicts.jsonl").write_text("")  # frontier runs exist, but nothing judged
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert not (tmp_path / "dataset.parquet").exists()
