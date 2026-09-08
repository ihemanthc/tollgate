"""tollgate.paths: local defaults, env overrides read at call time, and no stray hardcoded paths."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from tollgate import device, paths
from tollgate.cli import app
from tollgate.collect.prompts import write_seed
from tollgate.config import RunnerConfig
from tollgate.schema import QueryRecord

PACKAGE = Path(__file__).resolve().parents[1] / "tollgate"


@pytest.fixture(autouse=True)
def no_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(paths.DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(paths.OUT_DIR_ENV, raising=False)


def test_defaults_are_the_local_repo_layout() -> None:
    assert paths.data_dir() == Path("data")
    assert paths.seed_path() == Path("data/seed.parquet")
    assert paths.cache_dir() == Path("data/cache")
    assert paths.ledger_path() == Path("data/cost_ledger.jsonl")
    assert paths.tier_runs_path() == Path("data/tier_runs.jsonl")
    assert paths.verdicts_path() == Path("data/verdicts.jsonl")
    assert paths.dataset_path() == Path("data/dataset.parquet")
    assert paths.checkpoints_dir() == Path("checkpoints")
    assert paths.reports_dir() == Path("reports")
    assert paths.assets_dir() == Path("docs/assets")


def test_env_overrides_are_read_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    before = paths.dataset_path()
    monkeypatch.setenv(paths.DATA_DIR_ENV, "/kaggle/working/data")
    monkeypatch.setenv(paths.OUT_DIR_ENV, "/kaggle/working")
    assert before == Path("data/dataset.parquet")
    assert paths.dataset_path() == Path("/kaggle/working/data/dataset.parquet")
    assert paths.cache_dir() == Path("/kaggle/working/data/cache")
    assert paths.checkpoints_dir() == Path("/kaggle/working/checkpoints")
    assert paths.reports_dir() == Path("/kaggle/working/reports")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_env_means_default(monkeypatch: pytest.MonkeyPatch, blank: str) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, blank)
    assert paths.data_dir() == Path("data")


def test_home_is_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.OUT_DIR_ENV, "~/tollgate-out")
    assert paths.out_dir() == Path.home() / "tollgate-out"


def test_runner_and_writers_follow_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "d"))
    cfg = RunnerConfig()
    assert cfg.cache_dir == tmp_path / "d" / "cache"
    assert cfg.ledger_path == tmp_path / "d" / "cost_ledger.jsonl"
    written = write_seed([QueryRecord(query_id="q", prompt="p", source="s")])
    assert written == tmp_path / "d" / "seed.parquet" and written.exists()


def test_cli_defaults_follow_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "elsewhere"))
    result = CliRunner().invoke(app, ["build-dataset"])
    assert result.exit_code != 0
    assert "elsewhere" in result.output  # the missing seed it reports is under the env dir


def test_no_hardcoded_relative_paths_outside_paths_module() -> None:
    offenders = [
        f"{py.relative_to(PACKAGE)}:{n}"
        for py in PACKAGE.rglob("*.py")
        if py.name != "paths.py"
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"""Path\(\s*["']""", line)
    ]
    assert offenders == []


# --- tollgate.device.resolve --------------------------------------------------------------------


def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "_mps_available", lambda: False)


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_gpu(monkeypatch)
    assert device.resolve("auto") == torch.device("cpu")


def test_auto_prefers_cuda_then_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.resolve() == torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "_mps_available", lambda: True)
    assert device.resolve() == torch.device("mps")


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_device_raises(
    monkeypatch: pytest.MonkeyPatch, requested: str
) -> None:
    _no_gpu(monkeypatch)
    with pytest.raises(ValueError, match="not available"):
        device.resolve(requested)


def test_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="one of"):
        device.resolve("tpu")


def test_cpu_is_always_available() -> None:
    assert device.resolve(" CPU ") == torch.device("cpu")
