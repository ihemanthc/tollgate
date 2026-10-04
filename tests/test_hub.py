"""tollgate.hub against an autospec'd HfApi: no network, real HfApi method signatures enforced."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from huggingface_hub import HfApi

from tollgate import config, hub, paths

SECRET = "hf_SuperSecretToken123"


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> mock.MagicMock:
    """Patch hub.HfApi with an autospec'd instance; record the token it was built with."""
    instance = mock.create_autospec(HfApi, instance=True)
    instance.whoami.return_value = {"name": "alice"}
    factory = mock.Mock(return_value=instance)
    monkeypatch.setattr(hub, "HfApi", factory)
    monkeypatch.setenv(hub.TOKEN_ENV, SECRET)
    monkeypatch.delenv(config.HF_REPO_MODEL_ENV, raising=False)
    monkeypatch.delenv(config.HF_REPO_DATA_ENV, raising=False)
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.OUT_DIR_ENV, str(tmp_path / "out"))
    instance.factory = factory
    return instance


def _checkpoint(root: Path) -> Path:
    (root / "tokenizer").mkdir(parents=True)
    (root / hub.CHECKPOINT_MARKER).write_text("{}")
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer" / "tokenizer.json").write_text("{}")
    return root


def _fake_snapshot(files: dict[str, str]) -> Any:
    def snapshot_download(**kwargs: Any) -> str:
        local = Path(kwargs["local_dir"])
        for rel, text in files.items():
            (local / rel).parent.mkdir(parents=True, exist_ok=True)
            (local / rel).write_text(text)
        (local / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        return str(local)

    return snapshot_download


# --- token handling -----------------------------------------------------------------------------


def test_missing_token_fails_before_any_hub_call(
    monkeypatch: pytest.MonkeyPatch, api: mock.MagicMock, tmp_path: Path
) -> None:
    monkeypatch.delenv(hub.TOKEN_ENV)
    with pytest.raises(hub.HubError, match="HF_TOKEN is not set"):
        hub.pull_dataset()
    api.factory.assert_not_called()


def test_token_reaches_hfapi_and_nothing_else(
    api: mock.MagicMock,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    data = tmp_path / "data" / "dataset.parquet"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"x")
    api.hf_hub_download.return_value = str(data)
    api.snapshot_download.side_effect = _fake_snapshot({})  # nothing there: error path too

    hub.push_dataset()
    hub.pull_dataset()
    hub.push_checkpoint(_checkpoint(tmp_path / "ckpt"), path_in_repo="runs/laya/last")
    with pytest.raises(hub.HubError) as missing:
        hub.pull_checkpoint("runs/laya/last")

    api.factory.assert_called_with(token=SECRET)
    out, err = capsys.readouterr()
    for text in (caplog.text, out, err, str(missing.value)):
        assert SECRET not in text
    hub_calls = [c for c in api.mock_calls if not c[0].startswith("factory")]
    assert hub_calls  # the uploads and downloads above
    for call in hub_calls:  # only the constructor sees the token, never a Hub call argument
        assert SECRET not in repr(call)


# --- datasets -----------------------------------------------------------------------------------


def test_push_dataset_creates_private_repo_in_token_namespace(
    api: mock.MagicMock, tmp_path: Path
) -> None:
    data = tmp_path / "data" / "dataset.parquet"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"x")
    assert hub.push_dataset() == "alice/tollgate-routing-data/dataset.parquet"
    api.create_repo.assert_called_once_with(
        "alice/tollgate-routing-data", repo_type="dataset", private=True, exist_ok=True
    )
    upload = api.upload_file.call_args.kwargs
    assert (upload["path_or_fileobj"], upload["path_in_repo"]) == (str(data), "dataset.parquet")
    assert (upload["repo_id"], upload["repo_type"]) == ("alice/tollgate-routing-data", "dataset")


def test_repo_ids_come_from_env_and_skip_whoami(
    monkeypatch: pytest.MonkeyPatch, api: mock.MagicMock, tmp_path: Path
) -> None:
    monkeypatch.setenv(config.HF_REPO_DATA_ENV, "acme/routing")
    data = tmp_path / "x.parquet"
    data.write_bytes(b"x")
    hub.push_dataset(data, private=False)
    api.whoami.assert_not_called()
    api.create_repo.assert_called_once_with(
        "acme/routing", repo_type="dataset", private=False, exist_ok=True
    )


def test_push_dataset_missing_file(api: mock.MagicMock) -> None:
    with pytest.raises(FileNotFoundError):
        hub.push_dataset()
    api.upload_file.assert_not_called()


def test_pull_dataset_lands_in_data_dir(api: mock.MagicMock, tmp_path: Path) -> None:
    expected = tmp_path / "data" / "dataset.parquet"
    api.hf_hub_download.return_value = str(expected)
    assert hub.pull_dataset() == expected
    kwargs = api.hf_hub_download.call_args.kwargs
    assert kwargs["repo_id"] == "alice/tollgate-routing-data"
    assert kwargs["filename"] == "dataset.parquet"
    assert kwargs["repo_type"] == "dataset"
    assert Path(kwargs["local_dir"]) == tmp_path / "data"


# --- checkpoints --------------------------------------------------------------------------------


def test_push_checkpoint_uploads_folder_privately(api: mock.MagicMock, tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path / "ckpt")
    assert hub.push_checkpoint(ckpt, path_in_repo="/runs/laya/last/") == (
        "alice/tollgate-router/runs/laya/last"
    )
    api.create_repo.assert_called_once_with(
        "alice/tollgate-router", repo_type="model", private=True, exist_ok=True
    )
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["folder_path"] == str(ckpt)
    assert kwargs["path_in_repo"] == "runs/laya/last"
    assert kwargs["repo_type"] == "model"


def test_push_checkpoint_to_root(api: mock.MagicMock, tmp_path: Path) -> None:
    assert hub.push_checkpoint(_checkpoint(tmp_path / "ckpt")) == "alice/tollgate-router"
    assert api.upload_folder.call_args.kwargs["path_in_repo"] is None


def test_push_checkpoint_refuses_non_checkpoint(api: mock.MagicMock, tmp_path: Path) -> None:
    (tmp_path / "junk").mkdir()
    with pytest.raises(FileNotFoundError, match="not a checkpoint"):
        hub.push_checkpoint(tmp_path / "junk")
    api.upload_folder.assert_not_called()


def test_pull_run_checkpoint_defaults_next_to_local_runs(
    api: mock.MagicMock, tmp_path: Path
) -> None:
    api.snapshot_download.side_effect = _fake_snapshot(
        {
            "runs/laya/last/rl_agent_config.json": "{}",
            "runs/laya/last/model.safetensors": "w",
            "runs/laya/last/tokenizer/tokenizer.json": "{}",
        }
    )
    dest = hub.pull_checkpoint("runs/laya/last")
    assert dest == tmp_path / "out" / "checkpoints" / "laya" / "last"
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
        "model.safetensors",
        "rl_agent_config.json",
        "tokenizer/tokenizer.json",
    ]
    kwargs = api.snapshot_download.call_args.kwargs
    assert kwargs["allow_patterns"] == ["runs/laya/last/*"]
    assert kwargs["repo_id"] == "alice/tollgate-router"
    assert not list(dest.parent.glob(".pull-*"))  # staging cleaned up


def test_pull_root_checkpoint_skips_training_runs(api: mock.MagicMock, tmp_path: Path) -> None:
    api.snapshot_download.side_effect = _fake_snapshot(
        {"rl_agent_config.json": "{}", "calibration.json": "{}"}
    )
    dest = hub.pull_checkpoint(dest=tmp_path / "router")
    assert (dest / "calibration.json").exists()
    assert not (dest / ".cache").exists()
    assert api.snapshot_download.call_args.kwargs["ignore_patterns"] == ["runs/*"]


def test_pull_replaces_an_existing_dest(api: mock.MagicMock, tmp_path: Path) -> None:
    dest = tmp_path / "last"
    (dest / "stale").mkdir(parents=True)
    api.snapshot_download.side_effect = _fake_snapshot({"x/rl_agent_config.json": "{}"})
    hub.pull_checkpoint("x", dest=dest)
    assert not (dest / "stale").exists() and (dest / "rl_agent_config.json").exists()


def test_pull_missing_checkpoint_leaves_dest_untouched(api: mock.MagicMock, tmp_path: Path) -> None:
    dest = tmp_path / "last"
    dest.mkdir()
    (dest / "keep.txt").write_text("previous session")
    api.snapshot_download.side_effect = _fake_snapshot({})
    with pytest.raises(hub.HubError, match="no checkpoint"):
        hub.pull_checkpoint("runs/laya/last", dest=dest)
    assert (dest / "keep.txt").read_text() == "previous session"


def test_push_dataset_files_is_one_private_commit(api: mock.MagicMock, tmp_path: Path) -> None:
    files = {}
    for name in ("dataset.parquet", "splits/train.txt", "README.md"):
        path = tmp_path / name.replace("/", "_")
        path.write_text("x")
        files[name] = path
    assert hub.push_dataset_files(files) == "alice/tollgate-routing-data"
    api.create_repo.assert_called_once_with(
        "alice/tollgate-routing-data", repo_type="dataset", private=True, exist_ok=True
    )
    api.create_commit.assert_called_once()
    (repo_id, operations), kwargs = api.create_commit.call_args
    assert repo_id == "alice/tollgate-routing-data" and kwargs["repo_type"] == "dataset"
    assert sorted(op.path_in_repo for op in operations) == sorted(files)


def test_push_dataset_files_checks_files_before_touching_the_hub(
    api: mock.MagicMock, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        hub.push_dataset_files({"README.md": tmp_path / "missing.md"})
    api.create_repo.assert_not_called()


def test_push_files_to_the_model_repo(api: mock.MagicMock, tmp_path: Path) -> None:
    f = tmp_path / "logits.npz"
    f.write_bytes(b"x")
    assert hub.push_files({"runs/laya/logits.npz": f}, repo_type="model") == "alice/tollgate-router"
    api.create_repo.assert_called_once_with(
        "alice/tollgate-router", repo_type="model", private=True, exist_ok=True
    )
    assert api.create_commit.call_args.kwargs["repo_type"] == "model"


def test_dataset_revision_and_repo_resolution(api: mock.MagicMock) -> None:
    api.dataset_info.return_value = mock.Mock(sha="abc123def")
    assert hub.dataset_revision() == "abc123def"
    api.dataset_info.assert_called_once_with("alice/tollgate-routing-data")
    assert hub.resolve_repo_id("tollgate-router") == "alice/tollgate-router"
    assert hub.resolve_repo_id("acme/x") == "acme/x"


def test_pull_run_fetches_best_and_files_but_not_last(api: mock.MagicMock, tmp_path: Path) -> None:
    api.snapshot_download.side_effect = _fake_snapshot(
        {
            "runs/laya/best/rl_agent_config.json": "{}",
            "runs/laya/logits_calibration.npz": "n",
            "runs/laya/run_metadata.json": "{}",
        }
    )
    dest = hub.pull_run("laya")
    assert dest == tmp_path / "out" / "checkpoints" / "laya"
    assert sorted(p.name for p in dest.iterdir()) == [
        "best",
        "logits_calibration.npz",
        "run_metadata.json",
    ]
    kwargs = api.snapshot_download.call_args.kwargs
    assert kwargs["allow_patterns"] == ["runs/laya/*"]
    assert kwargs["ignore_patterns"] == ["runs/laya/last/*"]
    hub.pull_run("laya", include_last=True)
    assert api.snapshot_download.call_args.kwargs["ignore_patterns"] is None


def test_pull_run_without_best_fails(api: mock.MagicMock) -> None:
    api.snapshot_download.side_effect = _fake_snapshot({"runs/laya/history.json": "{}"})
    with pytest.raises(hub.HubError, match="no run with a best/"):
        hub.pull_run("laya")
