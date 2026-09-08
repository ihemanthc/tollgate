"""Every filesystem location Tollgate reads or writes, resolved when called.

TOLLGATE_DATA_DIR  data artifacts: seed, cache, ledger, tier runs, verdicts, dataset (default data/)
TOLLGATE_OUT_DIR   outputs: checkpoints/, reports/, docs/ (default: the current directory)

Unset or empty means the default, which is the local repo layout. These are functions rather than
constants so a notebook can set the variables after importing tollgate (e.g. to /kaggle/working).
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "TOLLGATE_DATA_DIR"
OUT_DIR_ENV = "TOLLGATE_OUT_DIR"


def _dir(var: str, default: str) -> Path:
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser() if value else Path(default)


def data_dir() -> Path:
    return _dir(DATA_DIR_ENV, "data")


def out_dir() -> Path:
    return _dir(OUT_DIR_ENV, ".")


def seed_path() -> Path:
    return data_dir() / "seed.parquet"


def cache_dir() -> Path:
    return data_dir() / "cache"


def ledger_path() -> Path:
    return data_dir() / "cost_ledger.jsonl"


def tier_runs_path() -> Path:
    return data_dir() / "tier_runs.jsonl"


def verdicts_path() -> Path:
    return data_dir() / "verdicts.jsonl"


def dataset_path() -> Path:
    return data_dir() / "dataset.parquet"


def checkpoints_dir() -> Path:
    return out_dir() / "checkpoints"


def reports_dir() -> Path:
    return out_dir() / "reports"


def docs_dir() -> Path:
    return out_dir() / "docs"


def assets_dir() -> Path:
    return docs_dir() / "assets"
