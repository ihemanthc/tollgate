"""tollgate.eval.metrics against hand-computed values (worked arithmetic in the comments)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from laya.common import ece_score
from sklearn.metrics import f1_score

from tollgate.collect.runner import LedgerEntry
from tollgate.eval import metrics as m
from tollgate.schema import JudgeVerdict, Tier, TierRun

L, M, F = Tier.LOCAL_SMALL, Tier.MID_TIER, Tier.FRONTIER

# --- accuracy / macro-F1 ------------------------------------------------------------------------

Y_TRUE = [0, 1, 2, 2, 1]
Y_PRED = [0, 2, 2, 2, 1]


def test_accuracy() -> None:
    assert m.accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 5)  # only index 1 is wrong


def test_macro_f1() -> None:
    # class 0: TP1 FP0 FN0 -> F1 1
    # class 1: TP1 FP0 FN1 -> P 1, R 1/2 -> F1 2/3
    # class 2: TP2 FP1 FN0 -> P 2/3, R 1 -> F1 4/5
    # macro: (1 + 2/3 + 4/5) / 3 = 37/45
    assert m.macro_f1(Y_TRUE, Y_PRED) == pytest.approx(37 / 45)


def test_macro_f1_counts_predicted_only_class_as_zero() -> None:
    # class 0: TP1 FN1 -> 2/3; class 1: 1; class 2 predicted once, never true -> 0
    assert m.macro_f1([0, 0, 1, 1], [0, 2, 1, 1]) == pytest.approx(5 / 9)


def test_macro_f1_explicit_labels_include_absent_class_as_zero() -> None:
    # same per-class F1s as test_macro_f1 plus class 3 (no support, never predicted) -> 0
    assert m.macro_f1(Y_TRUE, Y_PRED, labels=[0, 1, 2, 3]) == pytest.approx(37 / 60)


def test_macro_f1_matches_sklearn_on_random_labels() -> None:
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
    expected = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert m.macro_f1(y_true.tolist(), y_pred.tolist()) == pytest.approx(expected)


def test_tier_labels_work_directly() -> None:
    assert m.accuracy([L, M, F], [L, F, F]) == pytest.approx(2 / 3)


# --- Brier --------------------------------------------------------------------------------------


def test_brier_multiclass() -> None:
    probs = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]]
    # row 0 (y=0): 0.3^2 + 0.2^2 + 0.1^2 = 0.14
    # row 1 (y=2): 0.1^2 + 0.6^2 + 0.7^2 = 0.86
    # row 2 (y=2): 0.2^2 + 0.2^2 + 0.4^2 = 0.24
    # mean: 1.24 / 3
    assert m.brier_score(probs, [0, 2, 2]) == pytest.approx(1.24 / 3)


def test_brier_two_columns_is_twice_binary_brier() -> None:
    # row 0 (y=0): 0.2^2 + 0.2^2 = 0.08; row 1 (y=0): 0.7^2 + 0.7^2 = 0.98; mean 0.53
    assert m.brier_score([[0.8, 0.2], [0.3, 0.7]], [0, 0]) == pytest.approx(0.53)


# --- ECE (15 bins) ------------------------------------------------------------------------------


def test_ece_15_bins() -> None:
    conf = [0.95, 0.95, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35]
    correct = [1, 0, 1, 1, 1, 0, 0, 0]
    # bin (14/15, 1]:    n 2, acc 1/2, conf 0.95 -> 2/8 * 0.45 = 0.1125
    # bin (9/15, 10/15]: n 4, acc 3/4, conf 0.65 -> 4/8 * 0.10 = 0.0500
    # bin (5/15, 6/15]:  n 2, acc 0,   conf 0.35 -> 2/8 * 0.35 = 0.0875
    assert m.expected_calibration_error(conf, correct) == pytest.approx(0.25)


def test_ece_includes_both_ends_of_the_unit_interval() -> None:
    # conf 1.0 lands in the last bin, 0.0 in the first: each off by 1, weight 1/2
    assert m.expected_calibration_error([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_ece_matches_laya_definition() -> None:
    rng = np.random.default_rng(1)
    conf, correct = rng.uniform(0, 1, 500), rng.integers(0, 2, 500).astype(float)
    assert m.expected_calibration_error(conf, correct) == pytest.approx(ece_score(conf, correct))


# --- risk-coverage AUC --------------------------------------------------------------------------


def test_risk_coverage_auc() -> None:
    # risk of top-k by confidence: k1 0, k2 1/2, k3 1/3, k4 1/4 -> mean 13/48
    assert m.risk_coverage_auc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 1]) == pytest.approx(13 / 48)


def test_risk_coverage_auc_perfect_ranking() -> None:
    # the only error is least confident: 0, 0, 1/3 -> 1/9
    assert m.risk_coverage_auc([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1 / 9)


@pytest.mark.parametrize("order", [[0, 1, 2], [1, 0, 2], [2, 1, 0]])
def test_risk_coverage_auc_ties_are_order_independent(order: list[int]) -> None:
    # a threshold admits both 0.9s together: risks 1/2, 1/2, then 1/3 -> 4/9
    conf, correct = [0.9, 0.9, 0.5], [1, 0, 1]
    assert m.risk_coverage_auc([conf[i] for i in order], [correct[i] for i in order]) == (
        pytest.approx(4 / 9)
    )


@pytest.mark.parametrize(
    "fn",
    [
        lambda: m.accuracy([], []),
        lambda: m.expected_calibration_error([], []),
        lambda: m.risk_coverage_auc([0.5], [1, 0]),
        lambda: m.brier_score([[0.5, 0.5]], [2]),
    ],
)
def test_bad_inputs_raise(fn: object) -> None:
    with pytest.raises(ValueError):
        fn()  # type: ignore[operator]


# --- per-token rates from the cost ledger -------------------------------------------------------


def _entry(tier: str, prompt: int, completion: int, usd: float) -> LedgerEntry:
    return LedgerEntry(
        ts=datetime.now(UTC),
        tier=tier,
        model=f"{tier}-model",
        prompt_tokens=prompt,
        completion_tokens=completion,
        usd=usd,
        latency_ms=1.0,
    )


LEDGER = [
    _entry("local_small", 100, 50, 0.0),
    # mid: $1/M prompt, $4/M completion
    _entry("mid_tier", 100, 50, 100 * 1e-6 + 50 * 4e-6),
    _entry("mid_tier", 200, 10, 200 * 1e-6 + 10 * 4e-6),
    # frontier: $10/M prompt, $50/M completion
    _entry("frontier", 100, 50, 100 * 1e-5 + 50 * 5e-5),
    _entry("frontier", 300, 20, 300 * 1e-5 + 20 * 5e-5),
    _entry("judge", 999, 999, 1.0),  # not a tier: ignored
]


def test_rates_recover_prompt_and_completion_prices() -> None:
    rates = m.rates_from_ledger(LEDGER)
    assert set(rates) == {L, M, F}
    assert (rates[L].prompt_usd, rates[L].completion_usd) == (0.0, 0.0)
    assert rates[M].prompt_usd == pytest.approx(1e-6)
    assert rates[M].completion_usd == pytest.approx(4e-6)
    assert rates[F].prompt_usd == pytest.approx(1e-5)
    assert rates[F].completion_usd == pytest.approx(5e-5)


def test_rates_fall_back_to_blended_when_underdetermined() -> None:
    # one call: 200 tokens for $0.002 -> $1e-5 per token, either kind
    rates = m.rates_from_ledger([_entry("mid_tier", 100, 100, 0.002)])
    assert rates[M].prompt_usd == pytest.approx(1e-5)
    assert rates[M].completion_usd == pytest.approx(1e-5)


# --- cost_retained / quality_retained -----------------------------------------------------------

RATES = m.rates_from_ledger(LEDGER)
TOKENS_100 = {L: (100, 100), M: (100, 100), F: (100, 100)}
# per query at (100, 100): local $0, mid 100*1e-6 + 100*4e-6 = $5e-4, frontier 1e-3 + 5e-3 = $6e-3


def _q(qid: str, predicted: Tier, conf: float, sufficient: dict[Tier, bool | None]) -> m.EvalQuery:
    return m.EvalQuery(
        query_id=qid,
        predicted=predicted,
        confidence=conf,
        tokens=TOKENS_100,
        sufficient=sufficient,
    )


QUERIES = [
    _q("q1", L, 0.9, {L: True, M: True}),
    _q("q2", L, 0.4, {L: False, M: True}),
    _q("q3", M, 0.6, {L: False, M: False}),
    _q("q4", F, 0.5, {L: False, M: False}),
]
ROUTER = m.RoutingEval(QUERIES, RATES)
FRONTIER_USD = 4 * 6e-3


def test_tau_zero_is_the_bare_router() -> None:
    # served L, L, M, F -> $0 + $0 + $5e-4 + $6e-3 = $6.5e-3; sufficient q1, q4 -> 2/4
    out = ROUTER.outcome(0.0)
    assert out.total_usd == pytest.approx(6.5e-3)
    assert out.frontier_usd == pytest.approx(FRONTIER_USD)
    assert ROUTER.cost_retained(0.0) == pytest.approx(6.5e-3 / FRONTIER_USD)
    assert ROUTER.quality_retained(0.0) == pytest.approx(0.5)
    assert out.escalated == 0


def test_below_tau_goes_up_one_tier() -> None:
    # tau 0.55 escalates q2 (0.4: L -> M) and q4 (0.5, already frontier: stays F)
    # served L, M, M, F -> $0 + $5e-4 + $5e-4 + $6e-3 = $7e-3; sufficient q1, q2, q4 -> 3/4
    assert ROUTER.cost_retained(0.55) == pytest.approx(7e-3 / FRONTIER_USD)
    assert ROUTER.quality_retained(0.55) == pytest.approx(0.75)
    assert ROUTER.outcome(0.55).escalated == 1  # q4 had nowhere to go


def test_confidence_equal_to_tau_is_not_escalated() -> None:
    # q3's 0.6 is not below 0.6, so it stays on mid (judged worse)
    assert ROUTER.served_tier(QUERIES[2], 0.6) is M
    assert ROUTER.served_tier(QUERIES[2], 0.6001) is F


def test_tau_above_every_confidence_escalates_all() -> None:
    # served M, M, F, F -> $5e-4 + $5e-4 + $6e-3 + $6e-3 = $1.3e-2; all sufficient
    assert ROUTER.cost_retained(0.95) == pytest.approx(1.3e-2 / FRONTIER_USD)
    assert ROUTER.quality_retained(0.95) == pytest.approx(1.0)


def test_cost_uses_the_served_tiers_own_token_counts() -> None:
    q = m.EvalQuery(
        query_id="q",
        predicted=L,
        confidence=0.1,
        tokens={L: (10, 10), M: (100, 200), F: (300, 400)},
        sufficient={L: False, M: True},
    )
    out = m.RoutingEval([q], RATES).outcome(0.5)
    assert out.total_usd == pytest.approx(100 * 1e-6 + 200 * 4e-6)  # mid: $9e-4
    assert out.frontier_usd == pytest.approx(300 * 1e-5 + 400 * 5e-5)  # $2.3e-2


def test_unjudged_served_answer_counts_as_not_sufficient() -> None:
    q = _q("q5", L, 0.3, {L: False, M: None})
    out = m.RoutingEval([q], RATES).outcome(0.5)
    assert (out.quality_retained, out.unknown_quality) == (0.0, 1)


def test_sweep_is_monotone_in_cost_here() -> None:
    taus = [0.0, 0.45, 0.55, 0.65, 0.95]
    outs = ROUTER.sweep(taus)
    assert [o.tau for o in outs] == taus
    costs = [o.total_usd for o in outs]
    assert costs == sorted(costs)


def test_missing_tier_rate_is_an_error() -> None:
    with pytest.raises(ValueError, match="frontier"):
        m.RoutingEval(QUERIES, {L: RATES[L], M: RATES[M]})


# --- building EvalQuery from runs + verdicts ----------------------------------------------------


def _run(tier: Tier, prompt: int, completion: int, error: str | None = None) -> TierRun:
    return TierRun(
        query_id="q",
        tier=tier,
        model="x",
        completion="",
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=0.0,
        latency_s=0.0,
        error=error,
    )


def _verdict(tier: Tier, verdict: str) -> JudgeVerdict:
    return JudgeVerdict(
        query_id="q", candidate_tier=tier, verdict=verdict, reason="r", judge_model="j"
    )


def test_eval_query_from_runs_and_verdicts() -> None:
    q = m.eval_query(
        "q",
        predicted=M,
        confidence=0.7,
        runs=[_run(L, 1, 2), _run(M, 3, 4), _run(F, 5, 6)],
        verdicts=[_verdict(L, "worse"), _verdict(M, "acceptable")],
    )
    assert q.tokens == {L: (1, 2), M: (3, 4), F: (5, 6)}
    assert q.sufficient == {L: False, M: True, F: True}


def test_eval_query_without_mid_verdict_marks_it_unknown() -> None:
    q = m.eval_query(
        "q", L, 0.7, [_run(L, 1, 2), _run(M, 3, 4), _run(F, 5, 6)], [_verdict(L, "equivalent")]
    )
    assert q.sufficient == {L: True, M: None, F: True}


def test_eval_query_needs_a_successful_run_per_tier() -> None:
    with pytest.raises(ValueError, match="mid_tier"):
        m.eval_query(
            "q", L, 0.7, [_run(L, 1, 2), _run(M, 0, 0, error="Timeout"), _run(F, 5, 6)], []
        )
