"""export_logits on a tiny checkpoint: the same logits calibrate.py computes, and CPU-side
calibration from them matches calibration through the model, with every leak guard intact."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import laya
import numpy as np
import pytest
import torch

from tests.conftest import TINY_HEAD_MAX_LEN, TINY_MAX_LEN, TinyModelFactory
from tollgate import paths
from tollgate.collect.build_dataset import write_dataset
from tollgate.schema import TIER_ORDER, LabeledExample, Split
from tollgate.train import calibrate as cal
from tollgate.train import infer
from tollgate.train import train as tt

TRAINED = tt.synthetic_examples(24)
CAL = [r.model_copy(update={"split": Split.CALIBRATION}) for r in tt.synthetic_examples(60)[30:]]
TEST = [r.model_copy(update={"split": Split.TEST}) for r in tt.synthetic_examples(80)[60:]]


@pytest.fixture(scope="module")
def trained(
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
    return run_dir / tt.BEST_DIR


@pytest.fixture
def checkpoint(trained: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    write_dataset([*TRAINED, *CAL, *TEST])
    return Path(shutil.copytree(trained, tmp_path / "run" / "best"))


def _export(checkpoint: Path, split: Split) -> dict[str, np.ndarray]:
    path = infer.export_logits(checkpoint, split, device="cpu", limit=1000, batch_size=7)
    assert path == checkpoint.parent / f"logits_{split.value}.npz"
    return infer.load_logits(path)


def test_export_writes_labels_logits_and_identity_but_no_text(checkpoint: Path) -> None:
    data = _export(checkpoint, Split.TEST)
    n = len(TEST)
    assert data["logits_tier"].shape == (n, 3)
    assert data["logits_tools"].shape == data["logits_rag"].shape == (n, 2)
    assert list(data["query_id"]) == [r.query_id for r in TEST]
    assert set(data["split"]) == {"test"}
    assert list(data["tier"]) == [TIER_ORDER.index(r.tier) for r in TEST]
    assert list(data["needs_rag"]) == [int(r.needs_rag) for r in TEST]
    assert str(data["checkpoint_config_sha256"]) == infer.config_sha256(checkpoint)
    strings = " ".join(str(v) for k, a in data.items() if a.dtype.kind == "U" for v in a.ravel())
    assert not any(r.prompt in strings for r in TEST)


def test_exported_logits_are_the_ones_calibration_uses(checkpoint: Path) -> None:
    data = _export(checkpoint, Split.CALIBRATION)
    records = cal.collect_records(laya.load(str(checkpoint), device="cpu"), CAL)
    for row in range(len(CAL)):
        for j, qid in enumerate(tt.QUESTIONS):
            assert np.allclose(data[f"logits_{qid}"][row], records[row * 3 + j][1], atol=1e-5)


def test_cpu_calibration_from_logits_matches_calibration_through_the_model(
    checkpoint: Path, tmp_path: Path
) -> None:
    logits = infer.logits_path(checkpoint.parent, Split.CALIBRATION)
    infer.export_logits(checkpoint, Split.CALIBRATION, device="cpu", limit=1000)
    twin = Path(shutil.copytree(checkpoint, tmp_path / "twin" / "best"))

    via_logits = json.loads(cal.fit_calibration_from_logits(checkpoint, logits).read_text())
    via_model = json.loads(cal.fit_calibration(twin, CAL).read_text())
    assert via_logits["temperature"] == pytest.approx(via_model["temperature"], rel=1e-4)
    assert via_logits["tollgate"]["fitted_on"] == "calibration"
    assert via_logits["tollgate"]["n_rows"] == len(CAL)
    assert via_logits["tollgate"]["logits_from"] == "logits:logits_calibration.npz"


def test_logits_from_non_calibration_rows_are_refused(checkpoint: Path) -> None:
    infer.export_logits(checkpoint, Split.TEST, device="cpu", limit=1000)
    with pytest.raises(cal.CalibrationLeakError, match="not from the calibration split"):
        cal.fit_calibration_from_logits(
            checkpoint, infer.logits_path(checkpoint.parent, Split.TEST)
        )


def test_logits_from_another_checkpoint_are_refused(checkpoint: Path) -> None:
    logits = infer.export_logits(checkpoint, Split.CALIBRATION, device="cpu", limit=1000)
    cfg_path = checkpoint / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tollgate"]["step"] += 1
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(cal.CalibrationLeakError, match="different checkpoint"):
        cal.fit_calibration_from_logits(checkpoint, logits)


def test_logits_of_trained_rows_are_refused(checkpoint: Path) -> None:
    logits = infer.export_logits(checkpoint, Split.CALIBRATION, device="cpu", limit=1000)
    data = infer.load_logits(logits)
    data["query_id"] = np.array([TRAINED[0].query_id, *data["query_id"][1:]])
    np.savez(logits, **data)
    with pytest.raises(cal.CalibrationLeakError, match="training data"):
        cal.fit_calibration_from_logits(checkpoint, logits)


def test_export_of_an_empty_split_fails(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_dataset([r for r in TRAINED])
    with pytest.raises(ValueError, match="no test rows"):
        infer.export_logits(checkpoint, Split.TEST, device="cpu")


def test_rows_are_example_objects() -> None:
    assert all(isinstance(r, LabeledExample) for r in [*CAL, *TEST])
