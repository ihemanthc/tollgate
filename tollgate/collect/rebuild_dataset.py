"""Rebuild the query text withheld from the Tollgate routing dataset.

Some seed sources forbid redistribution (LMSYS-Chat-1M: "You should not distribute, copy,
disclose, assign, sublicense, embed, host, or otherwise transfer the dataset to any third
party"). Rows from those sources are published with `prompt` empty and `text_sha256` set to
sha256 of the exact query text. This script streams each such source yourself -- you must accept
its license first -- extracts text exactly the way Tollgate did, and fills `prompt` back in for
every hash it finds.

Standalone on purpose (pandas + datasets only, no tollgate import), so it can be run from a copy
of the published dataset repo.

    pip install pandas pyarrow datasets
    huggingface-cli login     # with access granted to the gated sources
    python rebuild_dataset.py --dataset dataset.parquet --out dataset.rebuilt.parquet --limit 0

--limit caps how many source rows are scanned per source (0 = the whole source). The default is
small, for a smoke test; the full LMSYS-Chat-1M scan streams ~1.5 GB.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd

DEFAULT_LIMIT = 1000


def first_user_turn(conversation: Any) -> str | None:
    """LMSYS prompt text: the first user message of a conversation, unstripped."""
    turn = next((t for t in conversation or [] if t.get("role") == "user"), None)
    return turn["content"] if turn and turn.get("content") else None


def _lmsys_texts(limit: int) -> Iterator[str]:
    from datasets import load_dataset

    rows = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        revision="200748d9d3cddcc9d782887541057aca0b18c5da",
    )
    for i, row in enumerate(rows):
        if limit and i >= limit:
            return
        text = first_user_turn(row.get("conversation"))
        if text:
            yield text.strip()  # Tollgate strips before hashing


TextStream = Callable[[int], Iterator[str]]

# Withheld sources: source tag -> stream of candidate query texts, as Tollgate extracted them.
SOURCES: dict[str, TextStream] = {"lmsys-chat-1m": _lmsys_texts}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rebuild(
    frame: pd.DataFrame,
    limit: int,
    sources: dict[str, TextStream] = SOURCES,
) -> tuple[pd.DataFrame, int, int]:
    """Fill `prompt` for withheld rows whose hash is found. Returns (frame, recovered, missing)."""
    frame = frame.copy()
    withheld = frame["prompt"].isna()
    recovered = 0
    for source, texts in sources.items():
        wanted = set(frame.loc[withheld & (frame["source"] == source), "text_sha256"])
        found: dict[str, str] = {}
        for text in texts(limit):
            if not wanted:
                break
            digest = sha256(text)
            if digest in wanted:
                found[digest] = text
                wanted.discard(digest)
        rows = withheld & (frame["source"] == source) & frame["text_sha256"].isin(found)
        frame.loc[rows, "prompt"] = frame.loc[rows, "text_sha256"].map(found)
        recovered += int(rows.sum())
    return frame, recovered, int(frame["prompt"].isna().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="dataset.parquet")
    parser.add_argument("--out", default="dataset.rebuilt.parquet")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="0 = whole source")
    args = parser.parse_args()
    frame, recovered, missing = rebuild(pd.read_parquet(args.dataset), args.limit)
    frame.to_parquet(args.out, index=False)
    print(f"recovered {recovered} withheld prompts, {missing} still missing -> {args.out}")
    if missing and args.limit:
        print("raise --limit (0 scans the whole source) to find the rest")


if __name__ == "__main__":
    main()
