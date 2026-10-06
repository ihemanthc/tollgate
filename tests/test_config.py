"""OpenAI-compatible endpoints, per-role prices, and .env loading (shell wins)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tollgate import config
from tollgate.schema import Tier

ROLE_VARS = [
    f"{prefix}_{suffix}"
    for prefix in config.ROLE_PREFIXES.values()
    for suffix in ("MODEL", "API_BASE", "API_KEY", "USD_PER_MTOK_IN", "USD_PER_MTOK_OUT")
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in [*ROLE_VARS, *config.OPENAI_BASE_ENVS, "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)


def test_openai_model_goes_to_the_shared_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_FRONTIER_MODEL", "openai/claude-sonnet")
    monkeypatch.setenv("OPENAI_API_BASE", "https://llm.example.com/v1")
    cfg = config.tier_config(Tier.FRONTIER)
    assert (cfg.model, cfg.api_base, cfg.api_key) == (
        "openai/claude-sonnet",
        "https://llm.example.com/v1",
        None,  # litellm reads OPENAI_API_KEY itself
    )


def test_openai_base_url_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_MID_TIER_MODEL", "openai/Qwen3.8-27B")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://alias.example.com/v1")
    assert config.tier_config(Tier.MID_TIER).api_base == "https://alias.example.com/v1"


def test_non_openai_models_keep_their_provider_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_FRONTIER_MODEL", "anthropic/claude-x")
    monkeypatch.setenv("OPENAI_API_BASE", "https://llm.example.com/v1")
    assert config.tier_config(Tier.FRONTIER).api_base is None


def test_per_role_overrides_win_and_key_stays_out_of_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOLLGATE_JUDGE_MODEL", "openai/deepseek-v4.1-flash")
    monkeypatch.setenv("OPENAI_API_BASE", "https://shared.example.com/v1")
    monkeypatch.setenv("TOLLGATE_JUDGE_API_BASE", "https://judge.example.com/v1")
    monkeypatch.setenv("TOLLGATE_JUDGE_API_KEY", "sk-judge-secret")
    cfg = config.judge_config()
    assert (cfg.api_base, cfg.api_key) == ("https://judge.example.com/v1", "sk-judge-secret")
    assert "sk-judge-secret" not in repr(cfg)


def test_prices_are_usd_per_million_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_MID_TIER_MODEL", "openai/Qwen3.8-27B")
    monkeypatch.setenv("TOLLGATE_MID_TIER_USD_PER_MTOK_IN", "0.5")
    monkeypatch.setenv("TOLLGATE_MID_TIER_USD_PER_MTOK_OUT", "2")
    # 1000 * $0.5/M + 500 * $2/M
    assert config.tier_config(Tier.MID_TIER).price(1000, 500) == pytest.approx(0.0015)


def test_unpriced_is_none_and_local_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_FRONTIER_MODEL", "openai/claude-sonnet")
    assert config.tier_config(Tier.FRONTIER).price(1000, 1000) is None
    assert config.tier_config(Tier.LOCAL_SMALL).price(1000, 1000) == 0.0


def test_ollama_tiers_are_local_and_free_unless_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_MID_TIER_MODEL", "ollama/qwen2.5:14b")
    monkeypatch.setenv("TOLLGATE_FRONTIER_MODEL", "ollama_chat/qwen2.5:32b")
    monkeypatch.setenv("OLLAMA_API_BASE", "http://127.0.0.1:11434")
    monkeypatch.setenv("TOLLGATE_FRONTIER_USD_PER_MTOK_IN", "0.8")
    monkeypatch.setenv("TOLLGATE_FRONTIER_USD_PER_MTOK_OUT", "0.8")
    mid, frontier = config.tier_config(Tier.MID_TIER), config.tier_config(Tier.FRONTIER)
    assert (mid.local, mid.api_base) == (True, "http://127.0.0.1:11434")
    assert mid.price(1000, 1000) == 0.0
    # A reference price for what a hosted provider charges for the same model.
    assert frontier.local and frontier.price(1000, 1000) == pytest.approx(0.0016)


def test_ollama_judge_takes_prices_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_JUDGE_MODEL", "ollama/mistral-small:24b")
    monkeypatch.setenv("TOLLGATE_JUDGE_USD_PER_MTOK_IN", "1")
    monkeypatch.setenv("TOLLGATE_JUDGE_USD_PER_MTOK_OUT", "1")
    judge = config.judge_config()
    assert judge.local and judge.price(500, 500) == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("price_in", "price_out", "match"),
    [("1.0", "", "or neither"), ("", "1.0", "or neither"), ("cheap", "1.0", "not a number")],
)
def test_bad_prices_raise(
    monkeypatch: pytest.MonkeyPatch, price_in: str, price_out: str, match: str
) -> None:
    monkeypatch.setenv("TOLLGATE_FRONTIER_MODEL", "openai/claude-sonnet")
    monkeypatch.setenv("TOLLGATE_FRONTIER_USD_PER_MTOK_IN", price_in)
    monkeypatch.setenv("TOLLGATE_FRONTIER_USD_PER_MTOK_OUT", price_out)
    with pytest.raises(ValueError, match=match):
        config.tier_config(Tier.FRONTIER)


def test_blank_model_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOLLGATE_MID_TIER_MODEL", "  ")
    with pytest.raises(ValueError, match="TOLLGATE_MID_TIER_MODEL is not set"):
        config.tier_config(Tier.MID_TIER)


def test_load_env_fills_gaps_but_the_shell_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "TOLLGATE_FRONTIER_MODEL=openai/from-file\nTOLLGATE_MID_TIER_MODEL=openai/from-file\n"
    )
    monkeypatch.setenv("TOLLGATE_MID_TIER_MODEL", "openai/from-shell")
    assert config.load_env(env)
    assert os.environ["TOLLGATE_FRONTIER_MODEL"] == "openai/from-file"
    assert os.environ["TOLLGATE_MID_TIER_MODEL"] == "openai/from-shell"


def test_tests_never_read_the_real_env_file() -> None:
    assert config.ENV_FILE.endswith("absent.env")
