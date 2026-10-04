"""Evaluation metrics: classification, calibration, selective risk, and cost/quality routing.

Each definition is pinned by hand-computed values in tests/test_metrics.py:

- accuracy: fraction of predictions that are exactly right.
- macro_f1: unweighted mean of per-class F1 = 2TP / (2TP + FP + FN) over `labels` (default: every
  class seen in y_true or y_pred). A class with no true or predicted support scores 0.
- brier_score: mean over rows of sum_k (p_k - y_k)^2. With two columns this is twice the
  one-probability binary Brier; it is the form calibrate.py reports.
- expected_calibration_error: 15 equal-width bins on [0, 1], first bin closed at 0 and the rest
  right-closed -- laya.common.ece_score's binning, so calibration and eval numbers agree.
- risk_coverage_auc: mean over items of the error rate among all items at least as confident.
  Ties are admitted together, as a threshold would admit them. Lower is better.

Routing (RoutingEval): a query whose calibrated confidence is below tau is served one tier above
its prediction (frontier stays frontier). Cost prices each served tier's own token counts at
per-token rates fitted from the cost ledger -- not TierRun.cost_usd, which is 0 on cache hits.
Quality is the fraction of served answers the judge rated equivalent/acceptable; frontier is the
reference and always counts, and an unjudged served answer counts as not sufficient.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tollgate.collect.runner import LedgerEntry
from tollgate.schema import TIER_ORDER, JudgeVerdict, Probability, Tier, TierRun, TollgateModel

ECE_BINS = 15
_RATE_TOLERANCE = 1e-12  # lstsq noise on exact prices; anything more negative is a bad fit


def _labels(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> tuple[list, list]:
    true, pred = list(y_true), list(y_pred)
    if not true or len(true) != len(pred):
        raise ValueError(f"need equal, non-empty label lists; got {len(true)} and {len(pred)}")
    return true, pred


def _scored(confidences: Any, correct: Any) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=float).reshape(-1)
    hit = np.asarray(correct, dtype=float).reshape(-1)
    if conf.size == 0 or conf.size != hit.size:
        raise ValueError(f"need equal, non-empty arrays; got {conf.size} and {hit.size}")
    if ((conf < 0) | (conf > 1)).any() or not np.isin(hit, (0.0, 1.0)).all():
        raise ValueError("confidences must lie in [0, 1] and correct must be 0/1")
    return conf, hit


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    true, pred = _labels(y_true, y_pred)
    return sum(t == p for t, p in zip(true, pred, strict=True)) / len(true)


def macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable] | None = None,
) -> float:
    true, pred = _labels(y_true, y_pred)
    classes = list(labels) if labels is not None else list(dict.fromkeys(true + pred))
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(true, pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(true, pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(true, pred, strict=True))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def brier_score(probs: Any, y_true: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    if p.ndim != 2 or len(p) != len(y) or len(y) == 0:
        raise ValueError(
            f"need an (n, k) probability matrix and n labels; got {p.shape}, {y.shape}"
        )
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError(f"labels must index the {p.shape[1]} probability columns")
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(axis=1).mean())


def expected_calibration_error(confidences: Any, correct: Any, n_bins: int = ECE_BINS) -> float:
    conf, hit = _scored(confidences, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(pairwise(edges)):
        in_bin = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(conf[in_bin].mean() - hit[in_bin].mean())
    return float(total)


def risk_coverage_auc(confidences: Any, correct: Any) -> float:
    conf, hit = _scored(confidences, correct)
    order = np.argsort(-conf, kind="stable")
    conf, errors = conf[order], 1.0 - hit[order]
    group_ends = np.flatnonzero(np.r_[conf[1:] != conf[:-1], True])
    admitted = group_ends[np.searchsorted(group_ends, np.arange(len(conf)))]  # last tied index
    return float((np.cumsum(errors)[admitted] / (admitted + 1)).mean())


# --- cost ledger rates --------------------------------------------------------------------------


class TokenRates(TollgateModel):
    prompt_usd: float = Field(ge=0.0, description="USD per prompt token.")
    completion_usd: float = Field(ge=0.0, description="USD per completion token.")

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.prompt_usd + completion_tokens * self.completion_usd


def read_ledger(path: Path) -> list[LedgerEntry]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LedgerEntry.model_validate_json(line) for line in lines if line.strip()]


def rates_from_ledger(entries: Iterable[LedgerEntry]) -> dict[Tier, TokenRates]:
    """Per-tier USD per prompt and per completion token, least-squares fitted to logged calls.

    Exact provider prices are recovered when the ledger has calls with at least two different
    prompt/completion mixes; otherwise (or if the fit goes negative) one blended rate is used for
    both. Rows that are not a tier, such as the judge's, are ignored.
    """
    by_tier: dict[Tier, list[LedgerEntry]] = defaultdict(list)
    tier_values = {t.value: t for t in TIER_ORDER}
    for entry in entries:
        if entry.tier in tier_values:
            by_tier[tier_values[entry.tier]].append(entry)
    rates: dict[Tier, TokenRates] = {}
    for tier, rows in by_tier.items():
        tokens = np.array([[e.prompt_tokens, e.completion_tokens] for e in rows], dtype=float)
        usd = np.array([e.usd for e in rows], dtype=float)
        if np.linalg.matrix_rank(tokens) == 2:
            coef = np.linalg.lstsq(tokens, usd, rcond=None)[0]
            if coef.min() >= -_RATE_TOLERANCE:
                coef = coef.clip(min=0.0)
                rates[tier] = TokenRates(prompt_usd=coef[0], completion_usd=coef[1])
                continue
        blended = float(usd.sum() / tokens.sum()) if tokens.sum() else 0.0
        rates[tier] = TokenRates(prompt_usd=blended, completion_usd=blended)
    return rates


# --- routing: cost and quality retained ---------------------------------------------------------


class EvalQuery(TollgateModel):
    """One test query: the router's prediction, and what each tier would cost and be worth."""

    query_id: str
    predicted: Tier
    confidence: Probability = Field(description="Calibrated probability of `predicted`.")
    tokens: dict[Tier, tuple[int, int]] = Field(
        description="(prompt, completion) tokens of each tier's own run of this query."
    )
    sufficient: dict[Tier, bool | None] = Field(
        description="Judge verdict per tier (equivalent/acceptable); None if never judged."
    )

    @model_validator(mode="after")
    def _every_tier_priced(self) -> EvalQuery:
        missing = [t.value for t in TIER_ORDER if t not in self.tokens]
        if missing:
            raise ValueError(f"{self.query_id}: no token counts for {missing}")
        if any(n < 0 for pair in self.tokens.values() for n in pair):
            raise ValueError(f"{self.query_id}: negative token count")
        return self

    def is_sufficient(self, tier: Tier) -> bool | None:
        return True if tier is Tier.FRONTIER else self.sufficient.get(tier)


class RoutingOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    tau: float
    n: int
    escalated: int = Field(description="Queries actually moved up a tier.")
    total_usd: float
    frontier_usd: float = Field(description="Cost of sending every query to frontier.")
    cost_retained: float = Field(description="total_usd / frontier_usd.")
    quality_retained: float = Field(description="Fraction served a judged-sufficient answer.")
    unknown_quality: int = Field(description="Served answers never judged; counted as failures.")


def eval_query(
    query_id: str,
    predicted: Tier,
    confidence: float,
    runs: Sequence[TierRun],
    verdicts: Sequence[JudgeVerdict],
) -> EvalQuery:
    by_tier = {r.tier: r for r in runs if r.query_id == query_id}
    tokens: dict[Tier, tuple[int, int]] = {}
    for tier in TIER_ORDER:
        run = by_tier.get(tier)
        if run is None or run.error is not None:
            raise ValueError(f"{query_id}: no successful {tier.value} run to price")
        tokens[tier] = (run.prompt_tokens, run.completion_tokens)
    sufficient: dict[Tier, bool | None] = {tier: None for tier in TIER_ORDER}
    sufficient[Tier.FRONTIER] = True
    for verdict in verdicts:
        if verdict.query_id == query_id:
            sufficient[verdict.candidate_tier] = verdict.sufficient
    return EvalQuery(
        query_id=query_id,
        predicted=predicted,
        confidence=confidence,
        tokens=tokens,
        sufficient=sufficient,
    )


class RoutingEval:
    """Cost and quality of serving a fixed test set at abstention threshold tau."""

    def __init__(self, queries: Sequence[EvalQuery], rates: Mapping[Tier, TokenRates]) -> None:
        if not queries:
            raise ValueError("no queries to evaluate")
        missing = [t.value for t in TIER_ORDER if t not in rates]
        if missing:
            raise ValueError(f"no per-token rates for {missing}; the ledger has no calls for them")
        self.queries = list(queries)
        self.rates = dict(rates)
        self.frontier_usd = sum(self._cost(q, Tier.FRONTIER) for q in self.queries)

    def _cost(self, query: EvalQuery, tier: Tier) -> float:
        return self.rates[tier].cost(*query.tokens[tier])

    @staticmethod
    def served_tier(query: EvalQuery, tau: float) -> Tier:
        if query.confidence < tau and query.predicted is not Tier.FRONTIER:
            return TIER_ORDER[query.predicted.rank + 1]
        return query.predicted

    def outcome(self, tau: float) -> RoutingOutcome:
        served = [(q, self.served_tier(q, tau)) for q in self.queries]
        total = sum(self._cost(q, tier) for q, tier in served)
        judged = [q.is_sufficient(tier) for q, tier in served]
        n = len(served)
        return RoutingOutcome(
            tau=tau,
            n=n,
            escalated=sum(tier is not q.predicted for q, tier in served),
            total_usd=total,
            frontier_usd=self.frontier_usd,
            cost_retained=total / self.frontier_usd if self.frontier_usd else float("nan"),
            quality_retained=sum(v is True for v in judged) / n,
            unknown_quality=sum(v is None for v in judged),
        )

    def cost_retained(self, tau: float) -> float:
        return self.outcome(tau).cost_retained

    def quality_retained(self, tau: float) -> float:
        return self.outcome(tau).quality_retained

    def sweep(self, taus: Iterable[float]) -> list[RoutingOutcome]:
        return [self.outcome(tau) for tau in taus]
