"""calibrate.py: refuses anything but unseen calibration rows; fits, saves, and loads temps."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import laya
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from tests.conftest import TINY_HEAD_MAX_LEN, TINY_MAX_LEN, TinyModelFactory
from tollgate.cli import app
from tollgate.schema import LabeledExample, Split
from tollgate.train import CALIBRATION_FILE, config_identity, load_calibrated
from tollgate.train import calibrate as cal
from tollgate.train import train as tt

TRAINED = tt.synthetic_examples(24)  # ids synthetic-0 .. synthetic-23
UNSEEN = tt.synthetic_examples(60)[30:]  # ids synthetic-30 .. synthetic-59, never trained on


def _as(rows: list[LabeledExample], split: Split) -> list[LabeledExample]:
    return [r.model_copy(update={"split": split}) for r in rows]


@pytest.fixture(scope="module")
def trained_checkpoint(
    tok: Any, tiny_model: TinyModelFactory, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    model, base_cfg = tiny_model()
    fit_rows, holdout = tt.holdout_split(TRAINED, 0.2, seed=0)
    cfg = tt.TrainConfig(
        epochs=1, batch_size=8, max_len=TINY_MAX_LEN, head_max_len=TINY_HEAD_MAX_LEN
    )
    run_dir = tmp_path_factory.mktemp("run")
    tt.fit(
        model, tok, base_cfg, fit_rows, holdout, cfg, torch.device("cpu"), run_dir,
        synthetic=True, echo=lambda _: None,
    )  # fmt: skip
    return run_dir / "best"


@pytest.fixture
def checkpoint(trained_checkpoint: Path, tmp_path: Path) -> Path:
    """A private copy, so tests that write calibration.json do not see each other's files."""
    return Path(shutil.copytree(trained_checkpoint, tmp_path / "best"))


@pytest.fixture
def no_weight_loads(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def boom(*_: Any, **__: Any) -> None:
        raise AssertionError("weights were loaded before the split check")

    monkeypatch.setattr(laya, "load", boom)
    yield


@pytest.mark.parametrize(
    "rows",
    [
        _as(UNSEEN, Split.TRAIN),
        _as(UNSEEN, Split.TEST),
        _as(UNSEEN, Split.CALIBRATION)[:-1] + _as(UNSEEN[-1:], Split.TRAIN),  # one stray row
    ],
    ids=["all-train", "all-test", "one-train-row"],
)
def test_fitting_on_train_data_raises(
    checkpoint: Path, rows: list[LabeledExample], no_weight_loads: None
) -> None:
    with pytest.raises(cal.CalibrationLeakError, match="not from the calibration split"):
        cal.fit_calibration(checkpoint, rows)
    assert not (checkpoint / CALIBRATION_FILE).exists()


def test_calibration_rows_the_checkpoint_trained_on_raise(
    checkpoint: Path, no_weight_loads: None
) -> None:
    # A rebuilt dataset can relabel a row the checkpoint trained on as "calibration".
    rows = _as(UNSEEN[:20] + TRAINED[:1], Split.CALIBRATION)
    with pytest.raises(cal.CalibrationLeakError, match="training data"):
        cal.fit_calibration(checkpoint, rows)


def test_checkpoint_without_train_ids_cannot_be_calibrated(
    checkpoint: Path, no_weight_loads: None
) -> None:
    (checkpoint / tt.TRAIN_IDS_FILE).unlink()
    with pytest.raises(cal.CalibrationLeakError, match="cannot prove"):
        cal.fit_calibration(checkpoint, _as(UNSEEN, Split.CALIBRATION))


def test_too_few_calibration_rows_raise(checkpoint: Path, no_weight_loads: None) -> None:
    with pytest.raises(ValueError, match="need at least"):
        cal.fit_calibration(checkpoint, _as(UNSEEN[: cal.MIN_ROWS - 1], Split.CALIBRATION))


def test_cli_offers_no_way_to_pick_a_split() -> None:
    result = CliRunner().invoke(app, ["calibrate", "--split", "train"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_fit_type_temperatures_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    records = []
    for qtype, k in ((0, 3), (2, 2)):
        for _ in range(3000):
            true_logits = rng.normal(0, 1.5, size=k)
            p = np.exp(true_logits) / np.exp(true_logits).sum()
            target = np.eye(k, dtype=np.float32)[rng.choice(k, p=p)]
            records.append((qtype, (3.0 * true_logits).astype(np.float32), target, k))
    temps = cal.fit_type_temperatures(records)
    assert temps[0] == pytest.approx(3.0, rel=0.1)  # choice
    assert temps[1] == 1.0  # score: no records, stays neutral
    assert temps[2] == pytest.approx(3.0, rel=0.1)  # noul


def test_fit_writes_calibration_and_loader_applies_it(checkpoint: Path) -> None:
    out = cal.fit_calibration(checkpoint, _as(UNSEEN, Split.CALIBRATION))
    payload = json.loads(out.read_text())
    cfg = json.loads((checkpoint / "rl_agent_config.json").read_text())

    assert out == checkpoint.resolve() / CALIBRATION_FILE
    assert payload["tollgate"]["fitted_on"] == "calibration"
    assert payload["tollgate"]["n_rows"] == 30
    assert payload["temperature_by_options"] == {}  # per question type only
    assert payload["temperature"][1] == 1.0  # no score questions
    assert all(0.5 <= t <= 5.0 for t in payload["temperature"])
    assert payload["config"] == config_identity(cfg)
    in_sample = payload["tollgate"]["in_sample"]
    assert (in_sample["choice"]["n"], in_sample["noul"]["n"]) == (30, 60)
    for report in in_sample.values():
        assert report["nll_after"] <= report["nll_before"] + 1e-6

    agent = load_calibrated(checkpoint, device="cpu")
    assert agent.temperature == pytest.approx(payload["temperature"])
    assert agent.temperature_by_options == {}


def test_loader_refuses_uncalibrated_checkpoint(checkpoint: Path) -> None:
    with pytest.raises(FileNotFoundError, match="tollgate calibrate"):
        load_calibrated(checkpoint)


def test_loader_refuses_calibration_from_another_checkpoint(checkpoint: Path) -> None:
    cal.fit_calibration(checkpoint, _as(UNSEEN, Split.CALIBRATION))
    cfg_path = checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tollgate"]["step"] += 1  # a different checkpoint, e.g. a later save into the same dir
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="different checkpoint"):
        load_calibrated(checkpoint)


def test_loader_refuses_temperatures_not_fitted_on_calibration(checkpoint: Path) -> None:
    out = cal.fit_calibration(checkpoint, _as(UNSEEN, Split.CALIBRATION))
    payload = json.loads(out.read_text())
    payload["tollgate"]["fitted_on"] = "train"
    out.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not the calibration split"):
        load_calibrated(checkpoint)
