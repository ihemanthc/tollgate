"""Tier -> concrete model wiring, plus runner knobs. Read from the environment at call time.

local_small is a fixed Ollama model; mid_tier, frontier and the judge are litellm model
strings taken from TOLLGATE_{MID_TIER,FRONTIER,JUDGE}_MODEL so swapping providers needs no code.
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

# Hugging Face Hub artifact repos (CLAUDE.md). A bare name resolves to the token's namespace.
HF_REPO_MODEL_ENV = "HF_REPO_MODEL"
HF_REPO_DATA_ENV = "HF_REPO_DATA"
DEFAULT_HF_REPO_MODEL = "tollgate-router"
DEFAULT_HF_REPO_DATA = "tollgate-routing-data"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(description="litellm model string, e.g. 'ollama/qwen2.5:7b'.")
    api_base: str | None = None
    local: bool = Field(default=False, description="Local models are always costed at $0.")


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
    max_tokens: int = Field(default=1024, ge=1)
    cache_dir: Path = Field(default_factory=paths.cache_dir)
    ledger_path: Path = Field(default_factory=paths.ledger_path)


def _required_env(var: str, what: str) -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        raise ValueError(f"{var} is not set; it must be a litellm model string for {what}.")
    return value


def judge_config() -> ModelConfig:
    model = _required_env(JUDGE_MODEL_ENV, "the judge")
    if model.startswith(("ollama/", "ollama_chat/")):
        return ModelConfig(
            model=model, api_base=os.environ.get(OLLAMA_BASE_ENV, DEFAULT_OLLAMA_BASE), local=True
        )
    return ModelConfig(model=model)


def tier_config(tier: Tier) -> TierConfig:
    """Resolve one tier. Raises if a remote tier's env var is unset."""
    if tier is Tier.LOCAL_SMALL:
        return TierConfig(
            tier=tier,
            model=LOCAL_SMALL_MODEL,
            api_base=os.environ.get(OLLAMA_BASE_ENV, DEFAULT_OLLAMA_BASE),
            local=True,
        )
    return TierConfig(tier=tier, model=_required_env(MODEL_ENV_VARS[tier], tier.value))


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
