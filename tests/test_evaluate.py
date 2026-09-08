"""tollgate.eval.evaluate: results.json pieces, by hand-checked values; no model is loaded."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from tollgate.collect.runner import LedgerEntry
from tollgate.eval import evaluate as ev
from tollgate.eval.metrics import expected_calibration_error
from tollgate.schema import TIER_ORDER, JudgeVerdict, LabeledExample, Split, Tier, TierRun

L, M, F = TIER_ORDER
PRICE = {L: 0.0, M: 1e-6, F: 4e-6}  # USD per token, prompt and completion alike


def _logits(rows: list[tuple[int, float]]) -> np.ndarray:
    """(predicted tier index, margin) -> logits whose softmax picks that tier."""
    out = np.zeros((len(rows), 3))
    for i, (tier, margin) in enumerate(rows):
        out[i, tier] = margin
    return out


def _data(split: str, rows: list[tuple[int, float]], labels: list[int]) -> dict[str, np.ndarray]:
    n = len(rows)
    return {
        "query_id": np.array([f"{split}-{i}" for i in range(n)]),
        "split": np.array([split] * n),
        "tier": np.array(labels),
        "logits_tier": _logits(rows),
        "checkpoint_config_sha256": np.array("cfg"),
    }


def _records(ids: list[str], local_ok: list[bool], mid_ok: list[bool]) -> tuple[list, list]:
    runs, verdicts = [], []
    for qid, lo, mo in zip(ids, local_ok, mid_ok, strict=True):
        for tier in TIER_ORDER:
            runs.append(
                TierRun(
                    query_id=qid, tier=tier, model=f"m-{tier.value}", completion="a",
                    prompt_tokens=100, completion_tokens=100, cost_usd=0.0, latency_s=0.0,
                )
            )  # fmt: skip
        for tier, ok in ((L, lo), (M, mo)):
            verdicts.append(
                JudgeVerdict(
                    query_id=qid, candidate_tier=tier, verdict="equivalent" if ok else "worse",
                    reason="r", judge_model="judge",
                )
            )  # fmt: skip
    return runs, verdicts


def _ledger() -> list[LedgerEntry]:
    now = datetime.now(UTC)
    return [
        LedgerEntry(
            ts=now, tier=t.value, model=f"m-{t.value}", prompt_tokens=p, completion_tokens=c,
            usd=(p + c) * PRICE[t], latency_ms=1.0,
        )
        for t in TIER_ORDER
        for p, c in ((100, 50), (30, 200))
    ]  # fmt: skip


def test_tier_probabilities_softmax_with_temperature() -> None:
    p = ev.tier_probabilities(np.array([[0.0, np.log(3.0), 0.0]]), 1.0)
    assert p.tolist()[0] == pytest.approx([0.2, 0.6, 0.2])
    hotter = ev.tier_probabilities(np.array([[0.0, np.log(3.0), 0.0]]), 2.0)
    assert hotter[0, 1] == pytest.approx(np.sqrt(3) / (2 + np.sqrt(3)))


def test_reliability_bins_count_every_row_and_reproduce_ece() -> None:
    conf = np.array([0.05, 0.5, 0.52, 0.99, 1.0])
    correct = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    bins = ev.reliability_bins(conf, correct)
    assert len(bins) == 15 and sum(b.count for b in bins) == 5
    first = bins[0]
    assert (first.count, first.mean_confidence, first.accuracy) == (1, 0.05, 0.0)
    assert all(b.mean_confidence is None for b in bins if b.count == 0)
    ece = sum(b.count / 5 * abs(b.mean_confidence - b.accuracy) for b in bins if b.count)
    assert ece == pytest.approx(expected_calibration_error(conf, correct))


def test_confusion_rows_are_true_tiers() -> None:
    assert ev.confusion_counts([0, 0, 2, 1], [0, 1, 2, 2]) == [[1, 1, 0], [0, 0, 1], [0, 0, 1]]


def test_risk_coverage_curve_admits_ties_together() -> None:
    conf = np.array([0.9, 0.9, 0.6, 0.3])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    points = [(p.threshold, p.coverage, p.risk) for p in ev.risk_coverage_curve(conf, correct)]
    assert points == [(0.9, 0.5, 0.5), (0.6, 0.75, pytest.approx(1 / 3)), (0.3, 1.0, 0.5)]


def test_candidate_taus_span_never_and_always_escalate() -> None:
    taus = ev.candidate_taus(np.array([0.7, 0.4, 0.7]))
    assert taus[0] == 0.0 and taus[1:3] == [0.4, 0.7] and taus[-1] > 1.0


def _results(
    cal_rows: list[tuple[int, float]], cal_ok: list[bool], cal_mid_ok: bool = True
) -> ev.Results:
    test = _data("test", [(0, 3.0), (0, 0.2), (1, 2.0), (2, 1.0)], [0, 1, 1, 2])
    cal = _data("calibration", cal_rows, [0] * len(cal_rows))
    ids = [*test["query_id"].tolist(), *cal["query_id"].tolist()]
    runs, verdicts = _records(
        ids,
        local_ok=[True, False, False, False, *cal_ok],
        mid_ok=[True, True, True, False, *([cal_mid_ok] * len(cal_ok))],
    )
    return ev.build_results(
        run="unit", test=test, calibration=cal, temperature=1.0, calibrated=True,
        runs=runs, verdicts=verdicts, ledger=_ledger(),
        latency=ev.latency_summary([5.0, 6.0, 7.0, 100.0], "cpu", 2), target_quality=0.95,
    )  # fmt: skip


def test_build_results_classification_and_cost_curves() -> None:
    r = _results([(0, 3.0), (0, 0.1)], [True, False])
    c = r.classification
    assert (r.n_test, r.n_calibration, c.accuracy) == (4, 2, 0.75)
    assert c.support == {L: 1, M: 2, F: 1}
    assert r.confusion == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert sum(b.count for b in r.reliability) == 4

    # Always-frontier: 200 tokens * $4e-6 per query.
    assert r.cost_curves.always_frontier.usd_per_1k == pytest.approx(200 * 4e-6 * 1000)
    # tau = 0 never escalates; a tau above 1 escalates every non-frontier prediction.
    never, always = r.cost_curves.tollgate[0], r.cost_curves.tollgate[-1]
    assert never.quality_retained == 0.75  # test-1: local predicted, local judged worse
    assert always.quality_retained == 1.0
    # Random router at tau = 0 has the same tier mix (2 local, 1 mid, 1 frontier) but blind:
    # 2/4 * mean local ok (1/4) + 1/4 * mean mid ok (3/4) + 1/4 * 1
    rand = r.cost_curves.random_router[0]
    assert rand.quality_retained == pytest.approx(0.5 * 0.25 + 0.25 * 0.75 + 0.25)
    assert rand.usd_per_1k == pytest.approx(never.usd_per_1k)  # same mix, same flat costs


def test_operating_point_is_chosen_on_calibration_not_test() -> None:
    # Calibration: the low-confidence row's local answer is worse, so tau must escalate it.
    r = _results([(0, 3.0), (0, 0.1)], [True, False])
    op = r.operating_point
    assert op.selected_on is Split.CALIBRATION and op.target_met
    assert op.calibration.quality_retained >= 0.95
    cal_conf = ev.tier_probabilities(_logits([(0, 3.0), (0, 0.1)]), 1.0).max(axis=1)
    assert cal_conf.min() < op.tau <= cal_conf.max()
    # A calibration split where nothing ever fails picks tau = 0, whatever test would prefer.
    assert _results([(0, 3.0), (0, 0.1)], [True, True]).operating_point.tau == 0.0


def test_unmet_target_falls_back_to_best_quality_at_least_cost() -> None:
    # Local and mid both fail, and escalation only goes one tier up: no tau reaches the target,
    # every tau retains 0, so the cheapest (never escalate) is chosen and flagged.
    op = _results([(0, 3.0)], [False], cal_mid_ok=False).operating_point
    assert (op.target_met, op.calibration.quality_retained, op.tau) == (False, 0.0, 0.0)


def test_latency_percentiles() -> None:
    lat = ev.latency_summary([float(x) for x in range(1, 101)], "cpu", 2)
    assert (lat.p50_ms, lat.p95_ms, lat.p99_ms) == pytest.approx((50.5, 95.05, 99.01))
    with pytest.raises(ValueError, match="no latency samples"):
        ev.latency_summary([], "cpu", 2)


def test_measure_latency_times_each_call_after_warmup() -> None:
    seen: list[str] = []
    rows = [
        LabeledExample(
            query_id=f"q{i}", prompt=f"p{i}", source="s", split=Split.TEST, tier=Tier.MID_TIER,
            needs_tools=False, needs_rag=False,
        )
        for i in range(3)
    ]  # fmt: skip
    timings = ev.measure_latency(lambda r: seen.append(r.query_id), rows, samples=4, warmup=2)
    assert len(timings) == 4 and all(t >= 0 for t in timings)
    assert seen == ["q0", "q1", "q0", "q1", "q2", "q0"]


def test_results_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    r = _results([(0, 3.0), (0, 0.1)], [True, False])
    path = tmp_path / "results.json"  # type: ignore[operator]
    path.write_text(r.model_dump_json(), encoding="utf-8")
    assert ev.read_results(path) == r
