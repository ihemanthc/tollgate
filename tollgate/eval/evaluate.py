"""Evaluate a trained run on the test split and write reports/results.json.

Inputs, all produced elsewhere and only read here:
- checkpoints/<run>/logits_test.npz and logits_calibration.npz: raw marker logits from the GPU
  run (tollgate.train.infer). Nothing is re-scored on this machine.
- checkpoints/<run>/best/calibration.json: temperatures fitted on the calibration split only.
- data/tier_runs.jsonl, data/verdicts.jsonl, data/cost_ledger.jsonl: what each tier answered,
  whether the judge found it sufficient, and what tokens cost.

The one thing measured here is router latency, because it only means something on the machine
that serves: the checkpoint is loaded and timed on single queries.

The routing threshold tau (below it, a query goes one tier up) is chosen on the calibration
split -- the cheapest tau whose quality retained reaches --target-quality -- and then reported
on the test split. The test split never picks its own operating point.

Every number in the README and in docs/assets/ comes from the file this writes.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from pydantic import Field

from tollgate import paths
from tollgate.collect.build_dataset import read_split
from tollgate.collect.judge import read_verdicts
from tollgate.collect.runner import LedgerEntry, read_runs
from tollgate.eval.metrics import (
    ECE_BINS,
    EvalQuery,
    RoutingEval,
    RoutingOutcome,
    accuracy,
    brier_score,
    eval_query,
    expected_calibration_error,
    macro_f1,
    rates_from_ledger,
    read_ledger,
    risk_coverage_auc,
)
from tollgate.schema import (
    TIER_ORDER,
    JudgeVerdict,
    LabeledExample,
    Split,
    Tier,
    TierRun,
    TollgateModel,
)

RESULTS_FILE = "results.json"
RESULTS_VERSION = 1
DEFAULT_LIMIT = 64
DEFAULT_TARGET_QUALITY = 0.95
DEFAULT_LATENCY_SAMPLES = 30
LATENCY_WARMUP = 2
CHOICE_TYPE = 0  # laya.common.QTYPES["choice"]: the tier question's temperature slot


# --- results.json schema ------------------------------------------------------------------------


class ReliabilityBin(TollgateModel):
    lo: float
    hi: float
    count: int = Field(ge=0)
    mean_confidence: float | None = Field(description="None for an empty bin.")
    accuracy: float | None = Field(description="None for an empty bin.")


class CoveragePoint(TollgateModel):
    threshold: float = Field(description="Queries at or above this confidence are covered.")
    coverage: float
    risk: float = Field(description="Error rate among covered queries.")


class Classification(TollgateModel):
    n: int
    accuracy: float
    macro_f1: float
    brier: float
    ece: float
    support: dict[Tier, int] = Field(description="True labels per tier in the test split.")


class OperatingPoint(TollgateModel):
    tau: float
    selected_on: Split = Split.CALIBRATION
    target_quality: float
    target_met: bool = Field(description="Whether any tau reached the target on calibration.")
    calibration: RoutingOutcome
    test: RoutingOutcome
    coverage: float = Field(description="Test queries served at their predicted tier.")
    risk: float | None = Field(description="Error rate among them; None if none are covered.")


class CostPoint(TollgateModel):
    tau: float | None = Field(description="Threshold; None for a fixed policy.")
    quality_retained: float
    usd_per_1k: float


class CostCurves(TollgateModel):
    tollgate: list[CostPoint]
    random_router: list[CostPoint] = Field(
        description="Each tier mix of the tollgate curve, assigned to queries at random."
    )
    always_frontier: CostPoint


class Latency(TollgateModel):
    device: str
    warmup: int
    samples_ms: list[float]
    p50_ms: float
    p95_ms: float
    p99_ms: float


class Results(TollgateModel):
    version: int = RESULTS_VERSION
    created_at: datetime
    run: str
    checkpoint_config_sha256: str
    calibrated: bool = Field(description="False: temperatures were NOT fitted (smoke run only).")
    temperature: float = Field(description="Temperature applied to the tier logits.")
    n_test: int
    n_calibration: int
    classification: Classification
    reliability: list[ReliabilityBin]
    confusion: list[list[int]] = Field(description="Counts; rows = true tier, cols = predicted.")
    risk_coverage: list[CoveragePoint]
    risk_coverage_auc: float
    operating_point: OperatingPoint
    cost_curves: CostCurves
    latency: Latency


def read_results(path: Path | None = None) -> Results:
    path = path or paths.reports_dir() / RESULTS_FILE
    return Results.model_validate_json(path.read_text(encoding="utf-8"))


# --- pure computations (tested in tests/test_evaluate.py) ---------------------------------------


def tier_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Row-wise softmax of (n, 3) tier logits at one temperature."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS
) -> list[ReliabilityBin]:
    """The bins expected_calibration_error sums over, with their counts."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        inside = ((confidences >= lo) if i == 0 else (confidences > lo)) & (confidences <= hi)
        n = int(inside.sum())
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=n,
                mean_confidence=float(confidences[inside].mean()) if n else None,
                accuracy=float(correct[inside].mean()) if n else None,
            )
        )
    return bins


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    counts = np.zeros((len(TIER_ORDER), len(TIER_ORDER)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        counts[t, p] += 1
    return counts.tolist()


def risk_coverage_curve(confidences: np.ndarray, correct: np.ndarray) -> list[CoveragePoint]:
    """One point per distinct confidence, highest first; ties are admitted together."""
    points = []
    for threshold in np.unique(confidences)[::-1]:
        covered = confidences >= threshold
        points.append(
            CoveragePoint(
                threshold=float(threshold),
                coverage=float(covered.mean()),
                risk=float(1.0 - correct[covered].mean()),
            )
        )
    return points


def candidate_taus(confidences: np.ndarray) -> list[float]:
    """Every threshold that changes routing: never escalate, each confidence, always escalate."""
    return [0.0, *np.unique(confidences).tolist(), float(np.nextafter(1.0, 2.0))]


def select_tau(evaluator: RoutingEval, taus: Sequence[float], target: float) -> tuple[float, bool]:
    """Cheapest tau whose quality retained reaches `target`; else the best quality, cheapest."""
    outcomes = evaluator.sweep(taus)
    meeting = [o for o in outcomes if o.quality_retained >= target]
    if meeting:
        return min(meeting, key=lambda o: (o.cost_retained, o.tau)).tau, True
    best = max(o.quality_retained for o in outcomes)
    tied = [o for o in outcomes if o.quality_retained == best]
    return min(tied, key=lambda o: (o.cost_retained, o.tau)).tau, False


def _usd_per_1k(total_usd: float, n: int) -> float:
    return total_usd / n * 1000.0


def cost_curves(evaluator: RoutingEval, taus: Sequence[float]) -> CostCurves:
    """Tollgate at every tau, a query-blind router with the same tier mix, and always-frontier.

    The random router's points are expectations, not samples: with tier shares s_t it costs
    sum_t s_t * mean cost of tier t, and retains sum_t s_t * mean sufficiency of tier t.
    """
    queries = evaluator.queries
    n = len(queries)
    mean_cost = {
        t: float(np.mean([evaluator.rates[t].cost(*q.tokens[t]) for q in queries]))
        for t in TIER_ORDER
    }
    mean_ok = {t: float(np.mean([q.is_sufficient(t) is True for q in queries])) for t in TIER_ORDER}
    tollgate, random_router = [], []
    for tau in taus:
        outcome = evaluator.outcome(tau)
        tollgate.append(
            CostPoint(
                tau=tau,
                quality_retained=outcome.quality_retained,
                usd_per_1k=_usd_per_1k(outcome.total_usd, n),
            )
        )
        shares = Counter(RoutingEval.served_tier(q, tau) for q in queries)
        random_router.append(
            CostPoint(
                tau=tau,
                quality_retained=sum(shares[t] / n * mean_ok[t] for t in TIER_ORDER),
                usd_per_1k=sum(shares[t] / n * mean_cost[t] for t in TIER_ORDER) * 1000.0,
            )
        )
    return CostCurves(
        tollgate=tollgate,
        random_router=random_router,
        always_frontier=CostPoint(
            tau=None, quality_retained=1.0, usd_per_1k=_usd_per_1k(evaluator.frontier_usd, n)
        ),
    )


def latency_summary(samples_ms: Sequence[float], device: str, warmup: int) -> Latency:
    if not samples_ms:
        raise ValueError("no latency samples")
    p50, p95, p99 = np.percentile(np.asarray(samples_ms, dtype=float), [50, 95, 99])
    return Latency(
        device=device,
        warmup=warmup,
        samples_ms=[float(s) for s in samples_ms],
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
    )


def _queries(
    data: dict[str, np.ndarray],
    temperature: float,
    runs: Sequence[TierRun],
    verdicts: Sequence[JudgeVerdict],
) -> list[EvalQuery]:
    probs = tier_probabilities(data["logits_tier"], temperature)
    return [
        eval_query(str(qid), TIER_ORDER[int(p.argmax())], float(p.max()), runs, verdicts)
        for qid, p in zip(data["query_id"], probs, strict=True)
    ]


def build_results(
    *,
    run: str,
    test: dict[str, np.ndarray],
    calibration: dict[str, np.ndarray],
    temperature: float,
    calibrated: bool,
    runs: Sequence[TierRun],
    verdicts: Sequence[JudgeVerdict],
    ledger: Sequence[LedgerEntry],
    latency: Latency,
    target_quality: float = DEFAULT_TARGET_QUALITY,
) -> Results:
    """Everything in results.json, from logits and collection records alone (no model)."""
    probs = tier_probabilities(test["logits_tier"], temperature)
    y_true = [int(t) for t in test["tier"]]
    y_pred = [int(i) for i in probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(float)

    # Prices only from the models that actually answered these queries.
    models = {(r.tier.value, r.model) for r in runs}
    rates = rates_from_ledger(e for e in ledger if (e.tier, e.model) in models)
    test_eval = RoutingEval(_queries(test, temperature, runs, verdicts), rates)
    cal_eval = RoutingEval(_queries(calibration, temperature, runs, verdicts), rates)
    cal_conf = tier_probabilities(calibration["logits_tier"], temperature).max(axis=1)
    tau, met = select_tau(cal_eval, candidate_taus(cal_conf), target_quality)
    covered = conf >= tau

    return Results(
        created_at=datetime.now(UTC),
        run=run,
        checkpoint_config_sha256=str(test["checkpoint_config_sha256"]),
        calibrated=calibrated,
        temperature=temperature,
        n_test=len(y_true),
        n_calibration=len(calibration["query_id"]),
        classification=Classification(
            n=len(y_true),
            accuracy=accuracy(y_true, y_pred),
            macro_f1=macro_f1(y_true, y_pred, labels=range(len(TIER_ORDER))),
            brier=brier_score(probs, y_true),
            ece=expected_calibration_error(conf, correct),
            support={t: y_true.count(i) for i, t in enumerate(TIER_ORDER)},
        ),
        reliability=reliability_bins(conf, correct),
        confusion=confusion_counts(y_true, y_pred),
        risk_coverage=risk_coverage_curve(conf, correct),
        risk_coverage_auc=risk_coverage_auc(conf, correct),
        operating_point=OperatingPoint(
            tau=tau,
            target_quality=target_quality,
            target_met=met,
            calibration=cal_eval.outcome(tau),
            test=test_eval.outcome(tau),
            coverage=float(covered.mean()),
            risk=float(1.0 - correct[covered].mean()) if covered.any() else None,
        ),
        cost_curves=cost_curves(test_eval, candidate_taus(conf)),
        latency=latency,
    )


# --- the parts that need a checkpoint -----------------------------------------------------------


def checkpoint_temperature(checkpoint: Path, *, allow_uncalibrated: bool) -> tuple[float, bool]:
    """(tier temperature, calibrated?) from calibration.json, or the raw config if allowed."""
    from laya.common import clamp_temperature

    from tollgate.train import read_calibration

    try:
        payload = read_calibration(checkpoint)
    except FileNotFoundError:
        if not allow_uncalibrated:
            raise
        payload = json.loads((checkpoint / "rl_agent_config.json").read_text(encoding="utf-8"))
        calibrated = False
    else:
        calibrated = True
    if payload.get("temperature_by_options"):
        raise ValueError(
            f"{checkpoint}: per-option-count temperatures are not supported; Tollgate fits one"
            " temperature per question type"
        )
    return clamp_temperature(payload["temperature"][CHOICE_TYPE]), calibrated


def measure_latency(
    route: Callable[[LabeledExample], Any],
    rows: Sequence[LabeledExample],
    samples: int,
    warmup: int = LATENCY_WARMUP,
) -> list[float]:
    """Milliseconds per single-query routing decision, cycling through `rows`."""
    usable = [r for r in rows if r.prompt]
    if not usable:
        raise ValueError("no rows with query text to time the router on")
    for i in range(warmup):
        route(usable[i % len(usable)])
    timings = []
    for i in range(samples):
        start = time.perf_counter()
        route(usable[i % len(usable)])
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _router(checkpoint: Path, device: str, temperature: float) -> Callable[[LabeledExample], Any]:
    """One routing decision as serving makes it: encode, forward, temperature-scaled softmax."""
    import laya

    from tollgate.train.infer import compute_logits

    agent = laya.load(str(checkpoint), device=device)

    def route(row: LabeledExample) -> np.ndarray:
        logits = compute_logits(agent, [row], batch_size=1)["logits_tier"]
        return tier_probabilities(logits, temperature)

    return route


def evaluate(
    run: Annotated[str, typer.Option(help="Run dir under <out dir>/checkpoints.")] = "laya",
    limit: Annotated[int, typer.Option(min=1, help="First N test rows.")] = DEFAULT_LIMIT,
    target_quality: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Quality retained tau must reach.")
    ] = DEFAULT_TARGET_QUALITY,
    latency_samples: Annotated[
        int, typer.Option(min=1, help="Timed single-query routing calls.")
    ] = DEFAULT_LATENCY_SAMPLES,
    device: Annotated[str, typer.Option(help="auto | cuda | mps | cpu (serving device)")] = "cpu",
    allow_uncalibrated: Annotated[
        bool,
        typer.Option(help="Smoke runs only: evaluate without calibration.json, marked as such."),
    ] = False,
    out: Annotated[Path | None, typer.Option(help="[default: reports/results.json]")] = None,
) -> None:
    """Test-split metrics, calibration, routing cost and router latency -> results.json."""
    from tollgate.device import resolve
    from tollgate.train.infer import config_sha256, load_logits

    run_dir = paths.checkpoints_dir() / run
    checkpoint = run_dir / "best"
    temperature, calibrated = checkpoint_temperature(
        checkpoint, allow_uncalibrated=allow_uncalibrated
    )
    test, cal = (load_logits(run_dir / f"logits_{s}.npz") for s in ("test", "calibration"))
    for name, data in (("test", test), ("calibration", cal)):
        if str(data["checkpoint_config_sha256"]) != config_sha256(checkpoint):
            raise typer.BadParameter(f"logits_{name}.npz was computed by a different checkpoint")
        if set(str(s) for s in data["split"]) != {name}:
            raise typer.BadParameter(f"logits_{name}.npz holds rows from other splits")
    test = {k: (v[:limit] if v.ndim else v) for k, v in test.items()}

    serving = str(resolve(device))
    route = _router(checkpoint, serving, temperature)
    timings = measure_latency(route, read_split(Split.TEST)[:limit], latency_samples)
    results = build_results(
        run=run,
        test=test,
        calibration=cal,
        temperature=temperature,
        calibrated=calibrated,
        runs=read_runs(paths.tier_runs_path()),
        verdicts=read_verdicts(paths.verdicts_path()),
        ledger=read_ledger(paths.ledger_path()),
        latency=latency_summary(timings, serving, LATENCY_WARMUP),
        target_quality=target_quality,
    )
    out = out or paths.reports_dir() / RESULTS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(results.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, out)
    c, op = results.classification, results.operating_point
    typer.echo(
        f"{'UNCALIBRATED (smoke) ' if not calibrated else ''}{results.n_test} test rows -> {out}\n"
        f"accuracy {c.accuracy:.3f}  macro-F1 {c.macro_f1:.3f}  ECE {c.ece:.3f}  "
        f"Brier {c.brier:.3f}\n"
        f"tau {op.tau:.3f} (target {op.target_quality} {'met' if op.target_met else 'NOT met'} "
        f"on calibration): test cost retained {op.test.cost_retained:.3f}, "
        f"quality retained {op.test.quality_retained:.3f}\n"
        f"router latency on {serving}: p50 {results.latency.p50_ms:.0f} ms, "
        f"p99 {results.latency.p99_ms:.0f} ms"
    )


if __name__ == "__main__":
    typer.run(evaluate)
