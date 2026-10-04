"""Seed prompts from LMSYS-Chat-1M chat, GSM8K math and MMLU knowledge, streamed
and interleaved round-robin so a small --limit still draws from all three."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from tollgate import paths
from tollgate.schema import QueryRecord

MAX_CHARS = 1500
DEFAULT_LIMIT = 32
_WHITESPACE = re.compile(r"\s+")

# Loaders yield (source, prompt, domain): the tag travels with the row.
Loader = Callable[[], Iterator[tuple[str, str, str]]]


def normalize(text: str) -> str:
    """Dedup key: whitespace collapsed, case folded."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


def query_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# Hub revisions each source is read at: the ones whose licenses were reviewed (publish.py) and
# the ones rebuild_dataset.py streams, so withheld text can be recovered byte for byte.
SOURCE_REVISIONS: dict[str, str] = {
    "lmsys/lmsys-chat-1m": "200748d9d3cddcc9d782887541057aca0b18c5da",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
}


def _stream(path: str, *args: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    revision = SOURCE_REVISIONS[path]
    return iter(load_dataset(path, *args, split=split, streaming=True, revision=revision))


def first_user_turn(conversation: Any) -> str | None:
    """LMSYS prompt text: the first user message of a conversation, unstripped."""
    turn = next((t for t in conversation or [] if t.get("role") == "user"), None)
    return turn["content"] if turn and turn.get("content") else None


def _load_lmsys() -> Iterator[tuple[str, str, str]]:
    for row in _stream("lmsys/lmsys-chat-1m", split="train"):
        text = first_user_turn(row.get("conversation"))
        if text:
            yield "lmsys-chat-1m", text, "chat"


def _load_gsm8k() -> Iterator[tuple[str, str, str]]:
    for row in _stream("openai/gsm8k", "main", split="test"):
        yield "gsm8k", row["question"], "math"


def _load_mmlu() -> Iterator[tuple[str, str, str]]:
    for row in _stream("cais/mmlu", "all", split="validation"):
        choices = "\n".join(
            f"{letter}. {text}" for letter, text in zip("ABCD", row["choices"], strict=False)
        )
        yield "mmlu", f"{row['question']}\n{choices}", row.get("subject") or "mmlu"


SOURCES: tuple[str, ...] = ("lmsys-chat-1m", "gsm8k", "mmlu")


def _loaders(sources: Sequence[str] = SOURCES) -> tuple[Loader, ...]:
    # Resolved per call so tests can monkeypatch individual loaders.
    available = {"lmsys-chat-1m": _load_lmsys, "gsm8k": _load_gsm8k, "mmlu": _load_mmlu}
    unknown = [s for s in sources if s not in available]
    if unknown or not sources:
        raise ValueError(f"unknown or empty sources {unknown}; choose from {list(available)}")
    return tuple(available[s] for s in sources)


def load_seed_prompts(
    limit: int = DEFAULT_LIMIT, sources: Sequence[str] = SOURCES
) -> list[QueryRecord]:
    """Interleave the sources, drop long and duplicate prompts, stop at `limit`."""
    streams = [load() for load in _loaders(sources)]
    collected_at = datetime.now(UTC)
    seen: set[str] = set()
    records: list[QueryRecord] = []
    for group in zip_longest(*streams):
        for item in group:
            if item is None:
                continue
            source, raw, domain = item
            prompt = raw.strip()
            key = normalize(prompt)
            if not prompt or len(prompt) > MAX_CHARS or key in seen:
                continue
            seen.add(key)
            records.append(
                QueryRecord(
                    query_id=query_id(prompt),
                    prompt=prompt,
                    source=source,
                    domain=domain,
                    token_estimate=len(prompt) // 4,
                    collected_at=collected_at,
                )
            )
            if len(records) >= limit:
                return records
    return records


def write_seed(records: list[QueryRecord], path: Path | None = None) -> Path:
    path = path or paths.seed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.model_dump(mode="json") for r in records]).to_parquet(path, index=False)
    return path


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    limit: Annotated[int, typer.Option(help="Maximum prompts to collect.")] = DEFAULT_LIMIT,
    out: Annotated[
        Path | None, typer.Option(help="Destination parquet. [default: <data dir>/seed.parquet]")
    ] = None,
    sources: Annotated[
        str, typer.Option(help="Comma list. lmsys-chat-1m is gated: needs HF login + access.")
    ] = ",".join(SOURCES),
) -> None:
    records = load_seed_prompts(limit, [s.strip() for s in sources.split(",") if s.strip()])
    path = write_seed(records, out)
    typer.echo(f"{len(records)} prompts -> {path} {dict(Counter(r.source for r in records))}")


if __name__ == "__main__":
    app()
