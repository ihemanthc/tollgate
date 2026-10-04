"""Per-question-type temperature scaling, fitted ONLY on the calibration split.

Writes <checkpoint>/calibration.json in laya's own format, so `laya.load(..., calibration=)` and
`tollgate.train.load_calibrated` apply it, plus a `tollgate` block recording what it was fitted on
and NLL / ECE / Brier before and after. Those numbers are in-sample on the calibration split;
held-out calibration quality is measured on the test split by tollgate.eval.

One temperature per laya question type: `tier` is the choice type, `tools` and `rag` share the
noul type. Per-option-count buckets are deliberately not fitted.

Fitting on anything but calibration rows is refused at every layer:
- nothing takes a split argument; the CLI reads read_split(Split.CALIBRATION) and nothing else;
- fit_calibration() rejects any row whose split is not CALIBRATION, before loading weights;
- it also rejects any row the checkpoint trained on (train_query_ids.json), which catches a
  dataset rebuilt after training whose splits moved underneath the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer
from laya.calibrate import (
    MIN_TYPE_N,
    calibration_payload,
    fit_temperature_map,
    records_from_labeled,
)
from laya.common import QTYPE_NAMES, QTYPES, clamp_temperature, ece_score
from pydantic import BaseModel

from tollgate import paths
from tollgate.collect.build_dataset import read_split
from tollgate.device import resolve
from tollgate.schema import LabeledExample, Split
from tollgate.train import CALIBRATION_FILE, FITTED_ON
from tollgate.train.infer import CONFIG_FILE, config_sha256, label_index, load_logits
from tollgate.train.train import (
    QUESTIONS,
    TRAIN_IDS_FILE,
    build_state,
    target_options,
)

DEFAULT_LIMIT = 64
MIN_ROWS = MIN_TYPE_N  # below laya's type-level floor the fit silently stays at 1.0

Record = tuple[int, np.ndarray, np.ndarray, int]


class CalibrationLeakError(ValueError):
    """Rows that are not unseen calibration rows reached temperature fitting."""


class TypeReport(BaseModel):
    n: int
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float


def read_train_ids(checkpoint: Path) -> set[str]:
    path = checkpoint / TRAIN_IDS_FILE
    if not path.exists():
        raise CalibrationLeakError(
            f"{path} not found: cannot prove the calibration rows were unseen in training. "
            "Retrain with the current train.py."
        )
    return set(json.loads(path.read_text(encoding="utf-8")))


def check_calibration_ids(ids: Sequence[str], splits: Sequence[str], train_ids: set[str]) -> None:
    """Raise unless every row is a calibration-split row the checkpoint never trained on."""
    wrong = [i for i, s in zip(ids, splits, strict=True) if s != Split.CALIBRATION.value]
    if wrong:
        raise CalibrationLeakError(
            f"{len(wrong)} of {len(ids)} rows are not from the calibration split "
            f"(e.g. {wrong[:3]}); temperatures are fitted on calibration rows only."
        )
    seen = [i for i in ids if i in train_ids]
    if seen:
        raise CalibrationLeakError(
            f"{len(seen)} calibration rows were in this checkpoint's training data (e.g. "
            f"{seen[:3]}); the dataset was rebuilt after training. Retrain on the current build."
        )
    if len(ids) < MIN_ROWS:
        raise ValueError(f"{len(ids)} calibration rows; need at least {MIN_ROWS} to fit.")


def check_calibration_rows(rows: Sequence[LabeledExample], train_ids: set[str]) -> None:
    check_calibration_ids(
        [r.query_id for r in rows], [Split(r.split).value for r in rows], train_ids
    )


def _one_hot(k: int, index: int) -> np.ndarray:
    return np.eye(k, dtype=np.float32)[index]


def _option_count(qid: str) -> int:
    spec = QUESTIONS[qid]
    return len(spec["criteria"]) if spec["type"] == "choice" else 2


def collect_records(agent: Any, rows: Sequence[LabeledExample]) -> list[Record]:
    """Raw marker logits in canonical option order: exactly what inference divides by T."""
    pairs = [
        (
            build_state(agent.tok, r.prompt, r.conversation_summary),
            QUESTIONS,
            {qid: _one_hot(_option_count(qid), i) for qid, i in target_options(r).items()},
        )
        for r in rows
    ]
    with torch.inference_mode():
        return records_from_labeled(agent, pairs)


def fit_type_temperatures(records: Sequence[Record]) -> list[float]:
    """One temperature per laya question type (choice, score, noul); clamped by laya."""
    return [float(t) for t in fit_temperature_map(records)["temperature"]]


def _scaled(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits.astype(np.float64) / temperature
    p = np.exp(z - z.max())
    return p / p.sum()


def type_reports(
    records: Sequence[Record], before: Sequence[float], after: Sequence[float]
) -> dict[str, TypeReport]:
    by_type: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        by_type[record[0]].append(record)
    reports = {}
    for qtype, recs in sorted(by_type.items()):
        stats: dict[str, float] = {}
        for tag, temps in (("before", before), ("after", after)):
            nll, conf, correct, brier = [], [], [], []
            for _, logits, target, _ in recs:
                p, y = _scaled(logits, temps[qtype]), int(target.argmax())
                nll.append(-np.log(max(p[y], 1e-12)))
                conf.append(p.max())
                correct.append(float(p.argmax() == y))
                brier.append(((p - target) ** 2).sum())
            stats[f"nll_{tag}"] = float(np.mean(nll))
            stats[f"ece_{tag}"] = float(ece_score(np.array(conf), np.array(correct)))
            stats[f"brier_{tag}"] = float(np.mean(brier))
        reports[QTYPE_NAMES[qtype]] = TypeReport(n=len(recs), temperature=after[qtype], **stats)
    return reports


def fit_calibration(
    checkpoint: str | Path, rows: Sequence[LabeledExample], device: str = "cpu"
) -> Path:
    """Fit per-type temperatures on calibration rows and write <checkpoint>/calibration.json."""
    import laya

    path = Path(checkpoint).resolve()
    check_calibration_rows(rows, read_train_ids(path))  # before any weights are loaded
    agent = laya.load(str(path), device=device)
    records = collect_records(agent, rows)
    before = [float(t) for t in agent.temperature]
    ids = [r.query_id for r in rows]
    return _write_calibration(path, agent.cfg, records, before, ids, origin="model forward")


def records_from_logits(data: dict[str, np.ndarray]) -> list[Record]:
    """Calibration records straight from an exported logits file: no model, no GPU."""
    records: list[Record] = []
    labels = {qid: label_index(data, qid) for qid in QUESTIONS}
    for row in range(len(data["query_id"])):
        for qid, spec in QUESTIONS.items():
            k = _option_count(qid)
            target = _one_hot(k, int(labels[qid][row]))
            records.append((QTYPES[spec["type"]], data[f"logits_{qid}"][row], target, k))
    return records


def fit_calibration_from_logits(checkpoint: str | Path, logits: Path) -> Path:
    """Fit temperatures from logits_calibration.npz (written by a GPU run) on the CPU.

    Same guards as fit_calibration: calibration rows only, none the checkpoint trained on, and
    the logits must come from this exact checkpoint.
    """
    path = Path(checkpoint).resolve()
    data = load_logits(logits)
    if str(data["checkpoint_config_sha256"]) != config_sha256(path):
        raise CalibrationLeakError(
            f"{logits} was computed by a different checkpoint than {path}; export it again."
        )
    ids = [str(i) for i in data["query_id"]]
    check_calibration_ids(ids, [str(s) for s in data["split"]], read_train_ids(path))
    cfg = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
    before = [clamp_temperature(t) for t in cfg.get("temperature", [1.0, 1.0, 1.0])]
    records = records_from_logits(data)
    return _write_calibration(path, cfg, records, before, ids, origin=f"logits:{logits.name}")


def _write_calibration(
    path: Path,
    cfg: dict[str, Any],
    records: Sequence[Record],
    before: Sequence[float],
    ids: Sequence[str],
    origin: str,
) -> Path:
    after = fit_type_temperatures(records)
    payload = calibration_payload(after, {}, model_id_or_path=str(path), subfolder=None, config=cfg)
    payload["tollgate"] = {
        "fitted_on": FITTED_ON,
        "logits_from": origin,
        "n_rows": len(ids),
        "query_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
        "fitted_at": datetime.now(UTC).isoformat(),
        "in_sample": {k: v.model_dump() for k, v in type_reports(records, before, after).items()},
    }
    out = path / CALIBRATION_FILE
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    return out


def calibrate(
    checkpoint: Annotated[
        Path | None, typer.Option(help="[default: <out dir>/checkpoints/laya/best]")
    ] = None,
    dataset: Annotated[
        Path | None, typer.Option(help="[default: <data dir>/dataset.parquet]")
    ] = None,
    limit: Annotated[int, typer.Option(help="First N calibration rows.")] = DEFAULT_LIMIT,
    device: Annotated[str, typer.Option(help="auto | cuda | mps | cpu")] = "auto",
    logits: Annotated[
        Path | None,
        typer.Option(help="logits_calibration.npz from a GPU run: fit on CPU, no model load."),
    ] = None,
) -> None:
    """Fit per-question-type temperatures on the calibration split (and only that split)."""
    checkpoint = checkpoint or paths.checkpoints_dir() / "laya" / "best"
    if logits is not None:
        out = fit_calibration_from_logits(checkpoint, logits)
    else:
        dataset = dataset or paths.dataset_path()
        if not dataset.exists():
            raise typer.BadParameter(f"{dataset} not found; run build-dataset first.")
        rows = read_split(Split.CALIBRATION, dataset)[:limit]
        out = fit_calibration(checkpoint, rows, device=str(resolve(device)))
    tollgate = json.loads(out.read_text(encoding="utf-8"))["tollgate"]
    report = tollgate["in_sample"]
    typer.echo(
        f"{tollgate['n_rows']} calibration rows -> {out}  (in-sample; test split is for eval)"
    )
    typer.echo("type     n     T  nll before->after  ece before->after  brier before->after")
    for name, r in report.items():
        typer.echo(
            f"{name:<6} {r['n']:>4} {r['temperature']:>5.2f}"
            f"  {r['nll_before']:.4f}->{r['nll_after']:.4f}"
            f"    {r['ece_before']:.4f}->{r['ece_after']:.4f}"
            f"      {r['brier_before']:.4f}->{r['brier_after']:.4f}"
        )


if __name__ == "__main__":
    typer.run(calibrate)
