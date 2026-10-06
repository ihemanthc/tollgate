"""What notebooks/kaggle_collect.ipynb runs, kept in the package so it is tested.

Every role (local_small, mid_tier, frontier, judge) is an open model served by one Ollama server
on the Kaggle GPUs, so collection needs no API key. Two T4s cannot hold all four models at once,
so the notebook runs one model over every query, removes it, and moves on to the next. Each
stage is an ordinary `tollgate` command: the runner's cache makes the later ones replay the
earlier answers instead of calling again.

Imports nothing heavy: the [collect] extra is enough (no torch, no laya).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict, Field

from tollgate import paths
from tollgate.config import (
    JUDGE_MODEL_ENV,
    MODEL_ENV_VARS,
    OLLAMA_BASE_ENV,
    OLLAMA_PREFIXES,
    ROLE_PREFIXES,
)

OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"
# Files a previous session's output can hand to this one. The seed is left out on purpose: it
# holds source text (LMSYS-Chat-1M forbids redistribution) and rebuilds deterministically.
STATE_FILES = ("cache", "cost_ledger.jsonl", "tier_runs.jsonl", "verdicts.jsonl")


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


# --- roles --------------------------------------------------------------------------------------


def role_env(
    models: Mapping[str, str], prices: Mapping[str, tuple[float, float] | None]
) -> dict[str, str]:
    """Environment for tollgate.config from the notebook's settings.

    `models` maps 'mid_tier' / 'frontier' / 'judge' to a litellm model string (local_small is
    fixed in config.LOCAL_SMALL_MODEL). `prices` maps a role to (USD per M input, per M output)
    tokens, or None to leave it at $0.
    """
    model_vars = {**{t.value: v for t, v in MODEL_ENV_VARS.items()}, "judge": JUDGE_MODEL_ENV}
    unknown = sorted((set(models) - set(model_vars)) | (set(prices) - set(ROLE_PREFIXES)))
    if unknown:
        raise ValueError(f"unknown roles {unknown}; use {sorted(model_vars)}")
    missing = sorted(set(model_vars) - set(models))
    if missing:
        raise ValueError(f"no model set for {missing}")
    env = {model_vars[role]: model for role, model in models.items()}
    for role, price in prices.items():
        if price is None:
            continue
        price_in, price_out = price
        prefix = ROLE_PREFIXES[role]
        env[f"{prefix}_USD_PER_MTOK_IN"] = str(price_in)
        env[f"{prefix}_USD_PER_MTOK_OUT"] = str(price_out)
    return env


def ollama_tag(model: str) -> str:
    """'ollama/qwen2.5:14b' -> 'qwen2.5:14b'. Raises for a model Ollama does not serve."""
    for prefix in OLLAMA_PREFIXES:
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    raise ValueError(f"{model!r} is not an Ollama model ({' or '.join(OLLAMA_PREFIXES)}<tag>)")


# --- Ollama server ------------------------------------------------------------------------------


class OllamaSettings(BaseModel):
    """One `ollama serve` for every stage; only one model is resident at a time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = "127.0.0.1:11434"
    parallel: int = Field(default=4, ge=1, description="Requests one model serves at once.")
    context_length: int = Field(
        default=8192,
        ge=2048,
        description="Per request. A judge prompt holds the query and two answers; Ollama's"
        " default would silently cut it.",
    )
    models_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "ollama-models",
        description="Off /kaggle/working, whose 20 GB would not hold the weights.",
    )
    log_path: Path = Field(
        default_factory=lambda: paths.data_dir() / "ollama.log",
        description="Kept with the data, so a saved version shows why a server failed.",
    )
    start_timeout_s: float = Field(default=120.0, gt=0.0)
    pull_timeout_s: float = Field(default=3600.0, gt=0.0)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}"

    def env(self) -> dict[str, str]:
        return {
            "OLLAMA_HOST": self.host,
            "OLLAMA_NUM_PARALLEL": str(self.parallel),
            "OLLAMA_CONTEXT_LENGTH": str(self.context_length),
            # Loading the next stage's model evicts the last one instead of sharing the GPUs.
            "OLLAMA_MAX_LOADED_MODELS": "1",
            # 8-bit KV cache (needs flash attention) lets a 32B model keep `parallel` slots of
            # `context_length` on 2x T4.
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
            "OLLAMA_MODELS": str(self.models_dir),
        }


def install_ollama() -> str:
    """Install the Ollama binary (no-op if present). Returns `ollama --version`."""
    if shutil.which("ollama") is None:
        if shutil.which("zstd") is None:  # the install script unpacks a .tar.zst
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)
        with urllib.request.urlopen(OLLAMA_INSTALL_URL, timeout=60) as response:
            script = response.read()
        subprocess.run(["sh"], input=script, check=True)
    done = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
    return (done.stdout or done.stderr).strip()


def _api(
    settings: OllamaSettings,
    path: str,
    body: Mapping[str, Any] | None = None,
    *,
    method: str = "GET",
    timeout: float = 10.0,
) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"{settings.base_url}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw.strip() else None


def ollama_running(settings: OllamaSettings) -> bool:
    try:
        _api(settings, "/api/tags", timeout=2)
    except (urllib.error.URLError, OSError):
        return False
    return True


def _log_tail(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return "(no log)"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def start_ollama(settings: OllamaSettings) -> subprocess.Popen[bytes] | None:
    """Start `ollama serve` and wait until it answers. None if one is already up (a re-run)."""
    if ollama_running(settings):
        return None
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.log_path.open("ab") as log:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            env={**os.environ, **settings.env()},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + settings.start_timeout_s
    while not ollama_running(settings):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(
                f"ollama serve did not come up at {settings.base_url}; last log lines "
                f"({settings.log_path}):\n{_log_tail(settings.log_path)}"
            )
        time.sleep(1)
    return proc


def pull(model: str, settings: OllamaSettings) -> None:
    """Download a model's weights into settings.models_dir (a no-op if already there)."""
    tag = ollama_tag(model)
    reply = _api(
        settings,
        "/api/pull",
        {"model": tag, "stream": False},
        method="POST",
        timeout=settings.pull_timeout_s,
    )
    if not reply or reply.get("status") != "success":
        raise RuntimeError(f"ollama pull {tag} failed: {reply}")


def remove(model: str, settings: OllamaSettings) -> None:
    """Delete a model's weights, so the next stage's model fits on disk."""
    _api(settings, "/api/delete", {"model": ollama_tag(model)}, method="DELETE")


# --- running commands from a notebook -----------------------------------------------------------


def call(command: Callable[..., object], /, **kwargs: Any) -> int:
    """Run a `tollgate` command function from a notebook cell. Returns its exit code.

    In a worker thread: Jupyter's own event loop runs on the main thread, and the commands call
    asyncio.run(), which refuses to start inside a running loop. Errors other than a clean
    typer.Exit propagate.
    """

    def target() -> int:
        try:
            command(**kwargs)
        except typer.Exit as exc:
            return int(exc.exit_code)
        return 0

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(target).result()


# --- state across sessions ----------------------------------------------------------------------


def restore_state(src: Path, dest: Path | None = None) -> list[str]:
    """Copy a previous session's cache, ledger, runs and verdicts into the data dir.

    `src` is that session's data dir as attached input (e.g. /kaggle/input/<notebook>/data).
    Every finished model call is then a cache hit, so the run continues where it stopped.
    Returns the names copied.
    """
    dest = dest or paths.data_dir()
    if not src.is_dir():
        raise FileNotFoundError(
            f"{src} not found: add the previous version's output as input (Add Input > Your Work"
            " > this notebook) and point RESUME_FROM at its data/ folder"
        )
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in STATE_FILES:
        source = src / name
        if source.is_dir():
            shutil.copytree(source, dest / name, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copyfile(source, dest / name)
        else:
            continue
        copied.append(name)
    return copied


def ollama_env(settings: OllamaSettings) -> dict[str, str]:
    """Where tollgate.config sends every ollama/ model."""
    return {OLLAMA_BASE_ENV: settings.base_url}
