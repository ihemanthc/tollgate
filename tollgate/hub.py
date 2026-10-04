"""Artifact transport over the Hugging Face Hub: how heavy and light machines hand off work.

- model repo   (HF_REPO_MODEL, default tollgate-router): checkpoints. The serving model lives at
  the repo root; training runs push to runs/<run>/best and runs/<run>/last.
- dataset repo (HF_REPO_DATA, default tollgate-routing-data): dataset.parquet, predictions.parquet.

A bare repo name resolves to the token owner's namespace. Repos are created private unless asked
otherwise. The token is read from HF_TOKEN only and is never logged, printed, or put into an
error message; it is handed to HfApi and nothing else.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from huggingface_hub import CommitOperationAdd, HfApi

from tollgate import paths
from tollgate.config import hf_repo_data, hf_repo_model

log = logging.getLogger(__name__)

TOKEN_ENV = "HF_TOKEN"
DATASET_FILE = "dataset.parquet"
CHECKPOINT_MARKER = "rl_agent_config.json"
RUNS_PREFIX = "runs"


class HubError(RuntimeError):
    """A Hub transfer that cannot proceed (no token, nothing at the requested path)."""


def _api() -> HfApi:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise HubError(
            f"{TOKEN_ENV} is not set; create a token with write access at "
            "https://huggingface.co/settings/tokens"
        )
    return HfApi(token=token)


def _repo_id(api: HfApi, name: str) -> str:
    """'owner/name' unchanged; a bare name goes under the token owner's namespace."""
    if "/" in name:
        return name
    return f"{api.whoami()['name']}/{name}"


def push_dataset(
    path: Path | None = None,
    *,
    repo: str | None = None,
    path_in_repo: str | None = None,
    private: bool = True,
) -> str:
    """Upload one data file (default <data dir>/dataset.parquet). Returns 'repo_id/path'."""
    path = path or paths.dataset_path()
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    api = _api()
    repo_id = _repo_id(api, repo or hf_repo_data())
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    target = path_in_repo or path.name
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=target,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"tollgate: upload {target}",
    )
    log.info("pushed %s -> dataset %s/%s", path, repo_id, target)
    return f"{repo_id}/{target}"


def pull_dataset(
    filename: str = DATASET_FILE,
    *,
    repo: str | None = None,
    dest_dir: Path | None = None,
    revision: str | None = None,
) -> Path:
    """Download one file of the dataset repo into dest_dir (default <data dir>)."""
    api = _api()
    repo_id = _repo_id(api, repo or hf_repo_data())
    dest_dir = dest_dir or paths.data_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = api.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest_dir),
    )
    log.info("pulled dataset %s/%s -> %s", repo_id, filename, local)
    return Path(local)


def push_files(
    files: Mapping[str, Path],
    *,
    repo_type: Literal["model", "dataset"],
    repo: str | None = None,
    private: bool = True,
    commit_message: str = "tollgate: upload",
) -> str:
    """Upload {path_in_repo: local file} as one atomic commit. Returns the repo id."""
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"not found: {missing}")
    api = _api()
    default = hf_repo_model() if repo_type == "model" else hf_repo_data()
    repo_id = _repo_id(api, repo or default)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.create_commit(
        repo_id,
        [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()],
        commit_message=commit_message,
        repo_type=repo_type,
    )
    log.info("pushed %d files -> %s %s", len(files), repo_type, repo_id)
    return repo_id


def push_dataset_files(
    files: Mapping[str, Path],
    *,
    repo: str | None = None,
    private: bool = True,
    commit_message: str = "tollgate: publish dataset",
) -> str:
    """Upload {path_in_repo: local file} to the dataset repo as one atomic commit."""
    return push_files(
        files, repo_type="dataset", repo=repo, private=private, commit_message=commit_message
    )


def dataset_revision(repo: str | None = None) -> str:
    """Commit sha the dataset repo's main branch points at now (record it next to a run)."""
    api = _api()
    return str(api.dataset_info(_repo_id(api, repo or hf_repo_data())).sha)


def push_checkpoint(
    local_dir: Path,
    *,
    path_in_repo: str = "",
    repo: str | None = None,
    private: bool = True,
) -> str:
    """Upload a checkpoint dir to the model repo: its root, or e.g. runs/<run>/last."""
    if not (local_dir / CHECKPOINT_MARKER).is_file():
        raise FileNotFoundError(f"{local_dir} is not a checkpoint (no {CHECKPOINT_MARKER})")
    api = _api()
    repo_id = _repo_id(api, repo or hf_repo_model())
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    prefix = path_in_repo.strip("/")
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=prefix or None,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"tollgate: upload {prefix or 'checkpoint'}",
    )
    log.info("pushed %s -> model %s/%s", local_dir, repo_id, prefix)
    return f"{repo_id}/{prefix}" if prefix else repo_id


def pull_checkpoint(
    path_in_repo: str = "",
    *,
    dest: Path | None = None,
    repo: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a checkpoint dir from the model repo and swap it in at `dest`.

    Default dest: <out dir>/checkpoints/<run>/<dir> for runs/<run>/<dir>, so a pulled last/
    lands where `tollgate train --resume-from` and calibrate expect it; the root (serving)
    checkpoint goes to <out dir>/checkpoints/router. Root pulls skip the runs/ tree.
    """
    prefix = path_in_repo.strip("/")
    if dest is None:
        relative = prefix.removeprefix(f"{RUNS_PREFIX}/") if prefix else "router"
        dest = paths.checkpoints_dir() / relative
    api = _api()
    repo_id = _repo_id(api, repo or hf_repo_model())
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pull-", dir=dest.parent))
    try:
        api.snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(staging),
            allow_patterns=[f"{prefix}/*"] if prefix else None,
            ignore_patterns=None if prefix else [f"{RUNS_PREFIX}/*"],
        )
        source = staging / prefix if prefix else staging
        if not (source / CHECKPOINT_MARKER).is_file():
            raise HubError(f"no checkpoint at {repo_id}/{prefix or '(root)'}")
        shutil.rmtree(source / ".cache", ignore_errors=True)  # hub download bookkeeping
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(source), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info("pulled model %s/%s -> %s", repo_id, prefix, dest)
    return dest


def resolve_repo_id(name: str) -> str:
    """'owner/name' for a configured repo name (a bare name resolves to the token owner)."""
    return _repo_id(_api(), name)


def pull_run(
    run_name: str,
    *,
    dest: Path | None = None,
    include_last: bool = False,
    repo: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download runs/<run> from the model repo (best/, logits, metadata) into dest.

    Default dest is <out dir>/checkpoints/<run>, the layout `tollgate train` writes, so
    calibrate --logits and load_calibrated work on it directly. last/ (weights + optimizer
    state, ~5 GB) is skipped unless asked for.
    """
    prefix = f"{RUNS_PREFIX}/{run_name}"
    dest = dest or paths.checkpoints_dir() / run_name
    api = _api()
    repo_id = _repo_id(api, repo or hf_repo_model())
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pull-", dir=dest.parent))
    try:
        api.snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(staging),
            allow_patterns=[f"{prefix}/*"],
            ignore_patterns=None if include_last else [f"{prefix}/last/*"],
        )
        source = staging / prefix
        if not (source / "best" / CHECKPOINT_MARKER).is_file():
            raise HubError(f"no run with a best/ checkpoint at {repo_id}/{prefix}")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(source), str(dest))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log.info("pulled run %s/%s -> %s", repo_id, prefix, dest)
    return dest
