"""Tier -> concrete model wiring, plus runner knobs. Read from the environment at call time.

local_small is a fixed Ollama model; mid_tier, frontier and the judge are litellm model
strings taken from TOLLGATE_{MID_TIER,FRONTIER,JUDGE}_MODEL so swapping providers needs no code.
Any of them may be an `ollama/<tag>` model too (kaggle_collect.ipynb serves every role from
Ollama on the GPU); those go to OLLAMA_API_BASE and need no confirmation to run.

OpenAI-compatible providers: an `openai/<name>` model is sent to OPENAI_API_BASE (or
OPENAI_BASE_URL) and litellm reads OPENAI_API_KEY. Any role can override both with
TOLLGATE_<ROLE>_API_BASE / TOLLGATE_<ROLE>_API_KEY. litellm has no prices for most such models, so
set TOLLGATE_<ROLE>_USD_PER_MTOK_IN / _OUT (USD per million prompt / completion tokens), or the
cost ledger records $0 for every call. An Ollama role is $0 unless the same variables are set: they
then price it at what a hosted provider charges for that model, so the cost curve reflects what
routing would save in production. All of these may live in a gitignored .env (load_env).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tollgate import paths
from tollgate.schema import TIER_ORDER, Tier

LOCAL_SMALL_MODEL = "ollama/qwen2.5:7b"
MODEL_ENV_VARS: dict[Tier, str] = {
    Tier.MID_TIER: "TOLLGATE_MID_TIER_MODEL",
    Tier.FRONTIER: "TOLLGATE_FRONTIER_MODEL",
}
JUDGE_MODEL_ENV = "TOLLGATE_JUDGE_MODEL"
OLLAMA_BASE_ENV = "OLLAMA_API_BASE"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"
OLLAMA_PREFIXES = ("ollama/", "ollama_chat/")
ROLE_PREFIXES: dict[str, str] = {
    Tier.MID_TIER.value: "TOLLGATE_MID_TIER",
    Tier.FRONTIER.value: "TOLLGATE_FRONTIER",
    "judge": "TOLLGATE_JUDGE",
}
OPENAI_BASE_ENVS = ("OPENAI_API_BASE", "OPENAI_BASE_URL")
ENV_FILE = ".env"

# Hugging Face Hub artifact repos (CLAUDE.md). A bare name resolves to the token's namespace.
HF_REPO_MODEL_ENV = "HF_REPO_MODEL"
HF_REPO_DATA_ENV = "HF_REPO_DATA"
DEFAULT_HF_REPO_MODEL = "tollgate-router"
DEFAULT_HF_REPO_DATA = "tollgate-routing-data"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(description="litellm model string, e.g. 'ollama/qwen2.5:7b'.")
    api_base: str | None = None
    api_key: str | None = Field(default=None, repr=False)
    local: bool = Field(
        default=False,
        description="Served by Ollama: needs no confirmation and is $0 unless prices are set.",
    )
    usd_per_mtok_in: float | None = Field(default=None, ge=0.0)
    usd_per_mtok_out: float | None = Field(default=None, ge=0.0)

    def price(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        """USD from configured prices; else $0 for a local model; else None (unpriced)."""
        if self.usd_per_mtok_in is None or self.usd_per_mtok_out is None:
            return 0.0 if self.local else None
        return (
            prompt_tokens * self.usd_per_mtok_in + completion_tokens * self.usd_per_mtok_out
        ) / 1e6


class TierConfig(ModelConfig):
    tier: Tier


class RunnerConfig(BaseModel):
    """Knobs for collect.runner. Generation params are NOT part of the cache key."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    concurrency: int = Field(default=4, ge=1, description="In-flight hosted-model calls.")
    local_concurrency: int = Field(
        default=1, ge=1, description="In-flight local (Ollama) calls; Ollama on CPU is serial."
    )
    max_retries: int = Field(default=4, ge=0)
    backoff_base_s: float = Field(default=1.0, gt=0.0)
    backoff_max_s: float = Field(default=30.0, gt=0.0)
    timeout_s: float = Field(default=180.0, gt=0.0)
    local_timeout_s: float = Field(default=900.0, gt=0.0, description="A CPU 7B is slow.")
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Output cap for every role. Reasoning models spend much of it thinking, and"
        " at 1024 their answers were often cut off mid-sentence.",
    )
    cache_dir: Path = Field(default_factory=paths.cache_dir)
    ledger_path: Path = Field(default_factory=paths.ledger_path)


def load_env(path: Path | None = None) -> bool:
    """Read .env (gitignored) into os.environ; variables already set in the shell win.

    A no-op when python-dotenv is absent (e.g. a [gpu]-only install, which needs no API keys).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(path or Path(ENV_FILE), override=False)


def _env(var: str) -> str | None:
    return os.environ.get(var, "").strip() or None


def _required_env(var: str, what: str) -> str:
    value = _env(var)
    if value is None:
        raise ValueError(f"{var} is not set; it must be a litellm model string for {what}.")
    return value


def _price(var: str) -> float | None:
    raw = _env(var)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{var}={raw!r} is not a number (USD per million tokens)") from None


def _prices(prefix: str) -> dict[str, float | None]:
    price_in, price_out = _price(f"{prefix}_USD_PER_MTOK_IN"), _price(f"{prefix}_USD_PER_MTOK_OUT")
    if (price_in is None) != (price_out is None):
        raise ValueError(f"set both {prefix}_USD_PER_MTOK_IN and _OUT, or neither")
    return {"usd_per_mtok_in": price_in, "usd_per_mtok_out": price_out}


def _role(role: str, model: str) -> dict[str, object]:
    """Endpoint, key and prices for a configurable role.

    Ollama models go to OLLAMA_API_BASE. Hosted ones use per-role variables, then shared ones.
    """
    prefix = ROLE_PREFIXES[role]
    if model.startswith(OLLAMA_PREFIXES):
        return {
            "model": model,
            "api_base": os.environ.get(OLLAMA_BASE_ENV, DEFAULT_OLLAMA_BASE),
            "local": True,
            **_prices(prefix),
        }
    api_base = _env(f"{prefix}_API_BASE")
    if api_base is None and model.startswith("openai/"):
        api_base = next((v for v in map(_env, OPENAI_BASE_ENVS) if v), None)
    return {
        "model": model,
        "api_base": api_base,
        "api_key": _env(f"{prefix}_API_KEY"),
        **_prices(prefix),
    }


def judge_config() -> ModelConfig:
    model = _required_env(JUDGE_MODEL_ENV, "the judge")
    return ModelConfig.model_validate(_role("judge", model))


def tier_config(tier: Tier) -> TierConfig:
    """Resolve one tier. Raises if a remote tier's env var is unset."""
    if tier is Tier.LOCAL_SMALL:
        return TierConfig(
            tier=tier,
            model=LOCAL_SMALL_MODEL,
            api_base=os.environ.get(OLLAMA_BASE_ENV, DEFAULT_OLLAMA_BASE),
            local=True,
        )
    model = _required_env(MODEL_ENV_VARS[tier], tier.value)
    return TierConfig.model_validate({"tier": tier, **_role(tier.value, model)})


def hf_repo_model() -> str:
    return os.environ.get(HF_REPO_MODEL_ENV, "").strip() or DEFAULT_HF_REPO_MODEL


def hf_repo_data() -> str:
    return os.environ.get(HF_REPO_DATA_ENV, "").strip() or DEFAULT_HF_REPO_DATA


def base_model_name(model: str) -> str:
    """'openai/gpt-4o' and 'azure/gpt-4o' are the same model behind different providers."""
    return model.rsplit("/", 1)[-1].strip().lower()


def check_judge_independent(judge_model: str, frontier_models: Iterable[str]) -> None:
    """The judge must not be the frontier model: it would be grading against its own answers.

    A model scoring a candidate against a reference it wrote itself favours the reference
    (self-preference), which pushes labels up to frontier and makes the router overspend.
    """
    clashes = sorted(
        {m for m in frontier_models if base_model_name(m) == base_model_name(judge_model)}
    )
    if clashes:
        raise ValueError(
            f"judge model {judge_model!r} is the frontier model ({clashes}); "
            "set TOLLGATE_JUDGE_MODEL to a different model"
        )


def parse_tiers(spec: str) -> tuple[Tier, ...]:
    """'all' or a comma list like 'local_small,frontier', returned cheapest-first."""
    if spec.strip().lower() == "all":
        return TIER_ORDER
    wanted = {Tier(part.strip()) for part in spec.split(",") if part.strip()}
    if not wanted:
        raise ValueError("no tiers given")
    return tuple(t for t in TIER_ORDER if t in wanted)
