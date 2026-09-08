"""load_seed_prompts: dedup, length filter, round-robin interleave, schema."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from tollgate.collect import prompts
from tollgate.schema import QueryRecord


def _fake(*rows: tuple[str, str, str]) -> prompts.Loader:
    def loader() -> Iterator[tuple[str, str, str]]:
        yield from rows

    return loader


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prompts,
        "_load_lmsys",
        _fake(
            ("lmsys-chat-1m", "What is the capital of France?", "chat"),
            ("lmsys-chat-1m", "  what IS   the Capital of France? ", "chat"),  # dup normalized
            ("lmsys-chat-1m", "x" * (prompts.MAX_CHARS + 1), "chat"),  # too long
            ("lmsys-chat-1m", "   ", "chat"),  # empty after strip
            ("lmsys-chat-1m", "Write me a haiku.", "chat"),
        ),
    )
    monkeypatch.setattr(
        prompts, "_load_gsm8k", _fake(("gsm8k", "Jan has 3 apples, eats 1. How many?", "math"))
    )
    monkeypatch.setattr(
        prompts, "_load_mmlu", _fake(("mmlu", "Which organ pumps blood?\nA. Heart", "anatomy"))
    )


def test_dedup_and_length_filter(patched: None) -> None:
    records = prompts.load_seed_prompts(limit=50)
    assert [r.prompt for r in records] == [
        "What is the capital of France?",
        "Jan has 3 apples, eats 1. How many?",
        "Which organ pumps blood?\nA. Heart",
        "Write me a haiku.",
    ]
    assert all(0 < len(r.prompt) <= prompts.MAX_CHARS for r in records)
    assert len({prompts.normalize(r.prompt) for r in records}) == len(records)


def test_schema_and_source_tagging(patched: None) -> None:
    records = prompts.load_seed_prompts(limit=50)
    assert all(isinstance(r, QueryRecord) for r in records)
    assert {(r.source, r.domain) for r in records} == {
        ("lmsys-chat-1m", "chat"),
        ("gsm8k", "math"),
        ("mmlu", "anatomy"),
    }
    for record in records:
        assert record.query_id == prompts.query_id(record.prompt)
        assert record.collected_at is not None
        assert QueryRecord.model_validate(record.model_dump(mode="json")) == record


def test_limit_truncates_after_interleaving(patched: None) -> None:
    records = prompts.load_seed_prompts(limit=2)
    assert [r.source for r in records] == ["lmsys-chat-1m", "gsm8k"]


def test_sources_subset_skips_other_loaders(patched: None) -> None:
    records = prompts.load_seed_prompts(limit=50, sources=["gsm8k", "mmlu"])
    assert [r.source for r in records] == ["gsm8k", "mmlu"]


@pytest.mark.parametrize("sources", [["sharegpt"], []])
def test_bad_sources_rejected(sources: list[str]) -> None:
    with pytest.raises(ValueError):
        prompts.load_seed_prompts(limit=1, sources=sources)


def test_write_seed_roundtrip(patched: None, tmp_path: Path) -> None:
    records = prompts.load_seed_prompts(limit=3)
    frame = pd.read_parquet(prompts.write_seed(records, tmp_path / "seed.parquet"))
    assert len(frame) == 3
    assert set(QueryRecord.model_fields) == set(frame.columns)
