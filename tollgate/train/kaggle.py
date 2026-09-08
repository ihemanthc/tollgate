"""What notebooks/kaggle_train.ipynb runs, kept in the package so it is tested.

The notebook is a short sequence of calls into this module and into `tollgate train` /
`export_logits`; anything with a branch or a failure mode lives here.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tollgate.collect.publish import RestoreReport
from tollgate.train.infer import load_logits
from tollgate.train.train import BEST_DIR, LAST_DIR, LAYA_REPO, LAYA_REVISION, TrainResult

EXPECTED_GPUS = ("T4", "P100")
METADATA_FILE = "run_metadata.json"
RUNS_PREFIX = "runs"


class GpuInfo(BaseModel):
    name: str
    count: int
    capability: str
    cuda: str | None
    expected: bool = Field(description="One of the GPUs this notebook is sized for.")


def gpu_info() -> GpuInfo:
    """The GPU this session runs on. Raises without one; warns if it is not a T4 or P100."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU in this session. On Kaggle: Settings > Accelerator > GPU T4 x2, then restart."
        )
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    expected = any(tag in name for tag in EXPECTED_GPUS)
    if not expected:
        warnings.warn(
            f"GPU is {name!r}, not a T4 or P100: batch size and the 10 h budget were sized for "
            "those, so timing and memory headroom will differ.",
            stacklevel=2,
        )
    return GpuInfo(
        name=name,
        count=torch.cuda.device_count(),
        capability=f"{major}.{minor}",
        cuda=torch.version.cuda,
        expected=expected,
    )


def kaggle_secret(name: str) -> str:
    """A Kaggle notebook secret, with an error that says how to add it. Never printed."""
    try:
        from kaggle_secrets import UserSecretsClient  # only importable on Kaggle
    except ImportError:
        raise RuntimeError(
            f"kaggle_secrets is not available, so {name} cannot be read: run this on Kaggle."
        ) from None
    try:
        value = UserSecretsClient().get_secret(name)
    except Exception:
        value = None
    if not value:
        raise RuntimeError(
            f"Kaggle secret {name!r} is missing or not attached to this notebook. Add it under "
            f"Add-ons > Secrets with the label {name}, tick it for this notebook, then re-run."
        )
    return value


def installed_commit(dist: str = "tollgate") -> str | None:
    """The git commit pip installed `dist` from (its direct_url.json); None if not a VCS install."""
    try:
        raw = metadata.distribution(dist).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    return (json.loads(raw).get("vcs_info") or {}).get("commit_id")


class RunMetadata(BaseModel):
    run_name: str
    started_at: datetime
    finished_at: datetime
    tollgate_version: str
    tollgate_commit: str | None
    base_model: str
    base_revision: str
    library_versions: dict[str, str]
    gpu: GpuInfo
    dataset_repo: str
    dataset_revision: str
    dataset: RestoreReport
    hyperparameters: dict[str, Any]
    seed: int
    train: dict[str, Any]
    rows_scored: dict[str, int]
    durations_s: dict[str, float]


def run_metadata(
    run_dir: Path,
    result: TrainResult,
    *,
    gpu: GpuInfo,
    dataset_repo: str,
    dataset_revision: str,
    restore: RestoreReport,
    logits: Sequence[Path],
    started_at: datetime,
    finished_at: datetime,
    session_train_s: float,
) -> RunMetadata:
    """Everything needed to reproduce or audit a run; hyperparameters come from the checkpoint."""
    checkpoint_cfg = json.loads((run_dir / BEST_DIR / "rl_agent_config.json").read_text())
    train_cfg = checkpoint_cfg["tollgate"]["train_config"]
    versions = {d: metadata.version(d) for d in ("laya", "torch", "transformers")}
    return RunMetadata(
        run_name=run_dir.name,
        started_at=started_at,
        finished_at=finished_at,
        tollgate_version=metadata.version("tollgate"),
        tollgate_commit=installed_commit(),
        base_model=LAYA_REPO,
        base_revision=LAYA_REVISION,
        library_versions=versions,
        gpu=gpu,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        dataset=restore,
        hyperparameters=train_cfg,
        seed=train_cfg["seed"],
        train={
            "steps": len(result.train),
            "best_step": result.best_step,
            "best_holdout_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "budget_exhausted": result.budget_exhausted,
            "train_seconds_all_sessions": result.seconds,
        },
        rows_scored={p.name: len(load_logits(p)["query_id"]) for p in logits},
        durations_s={
            "train_this_session": round(session_train_s, 1),
            "notebook_total": round((finished_at - started_at).total_seconds(), 1),
        },
    )


def write_metadata(meta: RunMetadata, run_dir: Path) -> Path:
    path = run_dir / METADATA_FILE
    path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    return path


class Upload(BaseModel):
    repo_id: str
    path_in_repo: str
    bytes: int

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/blob/main/{self.path_in_repo}"


def run_files(run_dir: Path, extra: Sequence[Path], *, include_last: bool) -> dict[str, Path]:
    """{path_in_repo: local file}: best/ (+ last/ when the run must be resumed) and extras."""
    prefix = f"{RUNS_PREFIX}/{run_dir.name}"
    files: dict[str, Path] = {}
    for sub in (BEST_DIR, LAST_DIR) if include_last else (BEST_DIR,):
        root = run_dir / sub
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            files[f"{prefix}/{sub}/{path.relative_to(root).as_posix()}"] = path
    for path in extra:
        files[f"{prefix}/{path.name}"] = path
    return files


def push_run(
    run_dir: Path,
    extra: Sequence[Path],
    *,
    include_last: bool = False,
    repo: str | None = None,
    private: bool = True,
) -> list[Upload]:
    """One commit to the model repo under runs/<run>/; returns exactly what was uploaded."""
    from tollgate import hub

    files = run_files(run_dir, extra, include_last=include_last)
    repo_id = hub.push_files(
        files,
        repo_type="model",
        repo=repo,
        private=private,
        commit_message=f"tollgate: run {run_dir.name}",
    )
    return [
        Upload(repo_id=repo_id, path_in_repo=k, bytes=v.stat().st_size) for k, v in files.items()
    ]


def format_manifest(uploads: Sequence[Upload], pulled: Mapping[str, str] | None = None) -> str:
    """Human-readable list of every uploaded file, its size and its URL."""
    if not uploads:
        return "nothing was uploaded"
    repo_id = uploads[0].repo_id
    total = sum(u.bytes for u in uploads)
    lines = [
        f"uploaded {len(uploads)} files ({total / 1e9:.2f} GB) to model repo "
        f"https://huggingface.co/{repo_id}",
    ]
    width = max(len(u.path_in_repo) for u in uploads)
    lines += [f"  {u.path_in_repo:<{width}}  {u.bytes:>14,} B  {u.url}" for u in uploads]
    for name, value in (pulled or {}).items():
        lines.append(f"input  {name}: {value}")
    return "\n".join(lines)
