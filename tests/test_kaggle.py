"""tollgate.train.kaggle: GPU check, secrets, install commit, metadata, push, manifest."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tollgate.collect.publish import RestoreReport
from tollgate.train import kaggle as tk
from tollgate.train.train import TrainResult


def _fake_gpu(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (7, 5))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


def test_gpu_info_without_a_gpu_says_how_to_attach_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="GPU T4 x2"):
        tk.gpu_info()


def test_t4_is_expected(monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder) -> None:
    _fake_gpu(monkeypatch, "Tesla T4")
    info = tk.gpu_info()
    assert (info.name, info.count, info.capability, info.expected) == ("Tesla T4", 2, "7.5", True)
    assert not recwarn.list


def test_other_gpus_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gpu(monkeypatch, "NVIDIA A10G")
    with pytest.warns(UserWarning, match="not a T4 or P100"):
        assert not tk.gpu_info().expected


def _secrets_module(monkeypatch: pytest.MonkeyPatch, get_secret: Any) -> None:
    module = ModuleType("kaggle_secrets")
    module.UserSecretsClient = lambda: SimpleNamespace(get_secret=get_secret)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)


def test_secret_outside_kaggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kaggle_secrets", None)
    with pytest.raises(RuntimeError, match="run this on Kaggle"):
        tk.kaggle_secret("HF_TOKEN")


@pytest.mark.parametrize("behaviour", ["raises", "empty"])
def test_missing_secret_explains_the_fix(monkeypatch: pytest.MonkeyPatch, behaviour: str) -> None:
    def get_secret(name: str) -> str:
        if behaviour == "raises":
            raise ConnectionError("no such secret")
        return ""

    _secrets_module(monkeypatch, get_secret)
    with pytest.raises(RuntimeError, match=r"Add-ons > Secrets with the label HF_TOKEN"):
        tk.kaggle_secret("HF_TOKEN")


def test_secret_value_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    _secrets_module(monkeypatch, lambda name: f"hf_value_for_{name}")
    assert tk.kaggle_secret("HF_TOKEN") == "hf_value_for_HF_TOKEN"


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (
            {"url": "https://github.com/a/b", "vcs_info": {"vcs": "git", "commit_id": "abc123"}},
            "abc123",
        ),
        ({"url": "file:///F:/tollgate", "dir_info": {"editable": True}}, None),
        (None, None),
    ],
)
def test_installed_commit(
    monkeypatch: pytest.MonkeyPatch, direct_url: dict[str, Any] | None, expected: str | None
) -> None:
    text = None if direct_url is None else json.dumps(direct_url)
    dist = SimpleNamespace(read_text=lambda name: text if name == "direct_url.json" else None)
    monkeypatch.setattr(tk.metadata, "distribution", lambda _: dist)
    assert tk.installed_commit() == expected


def _run_dir(tmp_path: Path, *, last: bool = True) -> Path:
    run = tmp_path / "laya"
    (run / "best" / "tokenizer").mkdir(parents=True)
    cfg = {"tollgate": {"train_config": {"epochs": 3, "lr": 2e-5, "seed": 7}}}
    (run / "best" / "rl_agent_config.json").write_text(json.dumps(cfg))
    (run / "best" / "model.safetensors").write_bytes(b"w" * 10)
    (run / "best" / "tokenizer" / "tokenizer.json").write_text("{}")
    if last:
        (run / "last").mkdir()
        (run / "last" / "trainer_state.pt").write_bytes(b"s" * 20)
    (run / "history.json").write_text("{}")
    for split, n in (("calibration", 3), ("test", 2)):
        np.savez(run / f"logits_{split}.npz", query_id=np.array([f"q{i}" for i in range(n)]))
    return run


def test_run_files_layout(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    extra = [run / "logits_calibration.npz", run / "history.json"]
    assert sorted(tk.run_files(run, extra, include_last=False)) == [
        "runs/laya/best/model.safetensors",
        "runs/laya/best/rl_agent_config.json",
        "runs/laya/best/tokenizer/tokenizer.json",
        "runs/laya/history.json",
        "runs/laya/logits_calibration.npz",
    ]
    with_last = tk.run_files(run, extra, include_last=True)
    assert "runs/laya/last/trainer_state.pt" in with_last


def test_run_files_needs_best(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tk.run_files(tmp_path / "nothing", [], include_last=False)


def test_push_run_is_one_private_model_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tollgate import hub

    calls: list[dict[str, Any]] = []

    def fake_push(files: dict[str, Path], **kwargs: Any) -> str:
        calls.append({"files": files, **kwargs})
        return "alice/tollgate-router"

    monkeypatch.setattr(hub, "push_files", fake_push)
    run = _run_dir(tmp_path)
    uploads = tk.push_run(run, [run / "history.json"])
    (call,) = calls
    assert call["repo_type"] == "model" and call["private"] is True
    by_path = {u.path_in_repo: u for u in uploads}
    assert by_path["runs/laya/best/model.safetensors"].bytes == 10
    assert set(by_path) == set(call["files"])
    manifest = tk.format_manifest(uploads, {"dataset": "alice/data@abc"})
    assert "https://huggingface.co/alice/tollgate-router" in manifest
    assert (
        "https://huggingface.co/alice/tollgate-router/blob/main/runs/laya/history.json" in manifest
    )
    assert "input  dataset: alice/data@abc" in manifest
    assert tk.format_manifest([]) == "nothing was uploaded"


def test_run_metadata_records_what_reproduces_the_run(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    started = datetime(2026, 10, 4, 8, 0, tzinfo=UTC)
    result = TrainResult(
        run_dir=run, synthetic=False, train=[], val=[], best_step=40, best_val_loss=1.5,
        stopped_early=False, budget_exhausted=True, seconds=3600.0,
    )  # fmt: skip
    meta = tk.run_metadata(
        run, result,
        gpu=tk.GpuInfo(name="Tesla T4", count=2, capability="7.5", cuda="12.4", expected=True),
        dataset_repo="alice/data", dataset_revision="abc123",
        restore=RestoreReport(rows=10, withheld=4, recovered=4, dropped=0),
        logits=[run / "logits_calibration.npz", run / "logits_test.npz"],
        started_at=started, finished_at=started + timedelta(hours=11), session_train_s=36000.0,
    )  # fmt: skip
    assert meta.seed == 7 and meta.hyperparameters["lr"] == 2e-5
    assert meta.base_revision == tk.LAYA_REVISION
    assert meta.rows_scored == {"logits_calibration.npz": 3, "logits_test.npz": 2}
    assert meta.train["budget_exhausted"] is True and meta.train["best_step"] == 40
    assert meta.durations_s == {"train_this_session": 36000.0, "notebook_total": 39600.0}
    assert set(meta.library_versions) == {"laya", "torch", "transformers"}
    written = tk.write_metadata(meta, run)
    assert json.loads(written.read_text())["dataset_revision"] == "abc123"
