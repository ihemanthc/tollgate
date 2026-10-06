"""tollgate.collect.kaggle: roles from settings, Ollama control, running commands in a notebook."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import typer

from tollgate.collect import kaggle as ck

MODELS = {
    "mid_tier": "ollama/qwen2.5:14b",
    "frontier": "ollama/qwen2.5:32b",
    "judge": "ollama/mistral-small:24b",
}


def test_role_env_sets_models_and_only_the_given_prices() -> None:
    env = ck.role_env(MODELS, {"mid_tier": (0.2, 0.6), "frontier": None})
    assert env == {
        "TOLLGATE_MID_TIER_MODEL": "ollama/qwen2.5:14b",
        "TOLLGATE_FRONTIER_MODEL": "ollama/qwen2.5:32b",
        "TOLLGATE_JUDGE_MODEL": "ollama/mistral-small:24b",
        "TOLLGATE_MID_TIER_USD_PER_MTOK_IN": "0.2",
        "TOLLGATE_MID_TIER_USD_PER_MTOK_OUT": "0.6",
    }


@pytest.mark.parametrize(
    ("models", "prices", "match"),
    [
        ({**MODELS, "local_small": "ollama/x"}, {}, "unknown roles"),
        (MODELS, {"frontrunner": (1.0, 1.0)}, "unknown roles"),
        ({"mid_tier": "ollama/x"}, {}, "no model set"),
    ],
)
def test_role_env_rejects_typos_and_gaps(
    models: dict[str, str], prices: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ck.role_env(models, prices)


def test_ollama_tag() -> None:
    assert ck.ollama_tag("ollama/qwen2.5:14b") == "qwen2.5:14b"
    assert ck.ollama_tag("ollama_chat/gemma2:27b") == "gemma2:27b"
    with pytest.raises(ValueError, match="not an Ollama model"):
        ck.ollama_tag("openai/gpt-x")


def test_server_settings_keep_one_model_resident(tmp_path: Path) -> None:
    settings = ck.OllamaSettings(parallel=3, context_length=4096, models_dir=tmp_path)
    env = settings.env()
    assert env["OLLAMA_MAX_LOADED_MODELS"] == "1"
    assert (env["OLLAMA_NUM_PARALLEL"], env["OLLAMA_CONTEXT_LENGTH"]) == ("3", "4096")
    assert env["OLLAMA_MODELS"] == str(tmp_path)
    assert ck.ollama_env(settings) == {"OLLAMA_API_BASE": "http://127.0.0.1:11434"}


def test_pull_and_remove_use_the_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any, str]] = []

    def fake_api(_: ck.OllamaSettings, path: str, body: Any = None, **kw: Any) -> Any:
        calls.append((path, body, kw.get("method", "GET")))
        return {"status": "success"} if path == "/api/pull" else None

    monkeypatch.setattr(ck, "_api", fake_api)
    settings = ck.OllamaSettings()
    ck.pull("ollama/qwen2.5:7b", settings)
    ck.remove("ollama/qwen2.5:7b", settings)
    assert calls == [
        ("/api/pull", {"model": "qwen2.5:7b", "stream": False}, "POST"),
        ("/api/delete", {"model": "qwen2.5:7b"}, "DELETE"),
    ]


def test_failed_pull_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ck, "_api", lambda *_, **__: {"error": "pull model manifest: not found"})
    with pytest.raises(RuntimeError, match="ollama pull nope:1b failed"):
        ck.pull("ollama/nope:1b", ck.OllamaSettings())


def test_call_runs_asyncio_commands_inside_a_running_loop() -> None:
    def command(limit: int) -> None:
        assert asyncio.run(asyncio.sleep(0, result=limit)) == 3

    async def notebook_cell() -> int:  # Jupyter runs cells while its own loop is running
        return ck.call(command, limit=3)

    assert asyncio.run(notebook_cell()) == 0


def test_call_returns_exit_codes_and_raises_real_errors() -> None:
    def exits(code: int) -> None:
        raise typer.Exit(code=code)

    def breaks() -> None:
        raise ValueError("boom")

    assert ck.call(exits, code=1) == 1
    assert ck.call(exits, code=0) == 0
    with pytest.raises(ValueError, match="boom"):
        ck.call(breaks)


def test_restore_state_copies_everything_but_the_seed(tmp_path: Path) -> None:
    src, dest = tmp_path / "previous", tmp_path / "data"
    (src / "cache").mkdir(parents=True)
    (src / "cache" / "abc.json").write_text("{}")
    (src / "cost_ledger.jsonl").write_text("{}\n")
    (src / "seed.parquet").write_bytes(b"source text")
    assert ck.restore_state(src, dest) == ["cache", "cost_ledger.jsonl"]
    assert (dest / "cache" / "abc.json").exists() and not (dest / "seed.parquet").exists()


def test_restore_state_explains_a_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Add Input"):
        ck.restore_state(tmp_path / "missing", tmp_path / "data")
