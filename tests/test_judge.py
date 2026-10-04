"""Judge parsing, one retry on bad JSON, and caching -- against a scripted fake LLM."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tollgate.collect.judge import JudgeError, judge, judge_query, parse_judge_output
from tollgate.collect.runner import Runner
from tollgate.config import ModelConfig, RunnerConfig
from tollgate.schema import QueryRecord, Tier, TierRun

JUDGE = ModelConfig(model="openai/fake-judge")
RECORD = QueryRecord(query_id="q1", prompt="Capital of France?", source="test")


def _run(tier: Tier, completion: str = "Paris", error: str | None = None) -> TierRun:
    return TierRun(
        query_id="q1",
        tier=tier,
        model=f"{tier}-model",
        completion=completion,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        latency_s=0.0,
        error=error,
    )


class ScriptedLLM:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _runner(tmp_path: Path, llm: ScriptedLLM) -> Runner:
    cfg = RunnerConfig(cache_dir=tmp_path / "cache", ledger_path=tmp_path / "ledger.jsonl")
    return Runner(cfg, completion_fn=llm, cost_fn=lambda _: 0.001)


GOOD = json.dumps({"verdict": "acceptable", "reason": "Correct but terse."})


@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        f"```json\n{GOOD}\n```",
        f"Here you go: {GOOD}",
        json.dumps({"verdict": " Acceptable ", "reason": "Correct but terse.\nMore.", "x": 1}),
    ],
)
def test_parse_tolerates_fences_case_and_extra_lines(text: str) -> None:
    out = parse_judge_output(text)
    assert (out.verdict, out.reason) == ("acceptable", "Correct but terse.")


@pytest.mark.parametrize(
    "text", ["not json", '{"verdict": "better", "reason": "x"}', '{"verdict": "worse"}']
)
def test_parse_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_judge_output(text)


def test_judge_happy_path_uses_json_mode_and_ledgers(tmp_path: Path) -> None:
    llm = ScriptedLLM(GOOD)
    verdict = asyncio.run(
        judge(_runner(tmp_path, llm), JUDGE, RECORD, _run(Tier.FRONTIER), _run(Tier.LOCAL_SMALL))
    )
    assert (verdict.verdict, verdict.sufficient, verdict.candidate_tier) == (
        "acceptable",
        True,
        Tier.LOCAL_SMALL,
    )
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    ledger = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [x["tier"] for x in ledger] == ["judge"]


def test_retries_once_then_reruns_fully_cached(tmp_path: Path) -> None:
    llm = ScriptedLLM("sorry, I think it's fine", GOOD)
    args = (JUDGE, RECORD, _run(Tier.FRONTIER), _run(Tier.MID_TIER))
    first = asyncio.run(judge(_runner(tmp_path, llm), *args))
    assert len(llm.calls) == 2
    assert "sorry, I think it's fine" in llm.calls[1]["messages"][0]["content"]

    again = asyncio.run(judge(_runner(tmp_path, llm), *args))  # replays both cache entries
    assert len(llm.calls) == 2
    assert again == first


def test_gives_up_after_one_retry(tmp_path: Path) -> None:
    llm = ScriptedLLM("nope", '{"verdict": "great"}')
    with pytest.raises(JudgeError):
        asyncio.run(
            judge(_runner(tmp_path, llm), JUDGE, RECORD, _run(Tier.FRONTIER), _run(Tier.MID_TIER))
        )
    assert len(llm.calls) == 2


def test_judge_query_skips_errored_and_unparseable(tmp_path: Path) -> None:
    llm = ScriptedLLM("bad", "still bad")  # only the mid_tier candidate gets judged
    runs = [
        _run(Tier.FRONTIER),
        _run(Tier.LOCAL_SMALL, completion="", error="ConnectionError: down"),
        _run(Tier.MID_TIER),
    ]
    assert asyncio.run(judge_query(_runner(tmp_path, llm), JUDGE, RECORD, runs)) == []
    assert len(llm.calls) == 2


def test_judge_query_without_frontier_makes_no_calls(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    runs = [_run(Tier.FRONTIER, completion="", error="Timeout"), _run(Tier.LOCAL_SMALL)]
    assert asyncio.run(judge_query(_runner(tmp_path, llm), JUDGE, RECORD, runs)) == []
    assert llm.calls == []
