"""Raw logits for a split: what a GPU run hands to the CPU for calibration and metrics.

One inference pass in fp32, options in canonical order -- the same numerics CPU serving uses and
the same rows calibrate.py would compute itself -- writes logits_<split>.npz:

  query_id, source, split       (n,)   str
  tier                          (n,)   int, index into TIER_ORDER (the label)
  needs_tools, needs_rag        (n,)   int 0/1 (the heuristic flags)
  logits_tier                   (n, 3) raw marker logits in TIER_ORDER
  logits_tools, logits_rag      (n, 2) raw marker logits, [false, true]
  checkpoint_config_sha256      ()     sha256 of the checkpoint's rl_agent_config.json

No query text is stored, so the file can go wherever the labels can.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import typer

from tollgate import paths
from tollgate.collect.build_dataset import read_split
from tollgate.device import resolve
from tollgate.schema import LabeledExample, Split
from tollgate.train.train import QUESTIONS, build_state, target_options

LOGITS_TEMPLATE = "logits_{split}.npz"
CONFIG_FILE = "rl_agent_config.json"
DEFAULT_LIMIT = 64


def config_sha256(checkpoint: Path) -> str:
    """Identity of a checkpoint: its config records the run, step and holdout loss."""
    return hashlib.sha256((checkpoint / CONFIG_FILE).read_bytes()).hexdigest()


def logits_path(directory: Path, split: Split) -> Path:
    return directory / LOGITS_TEMPLATE.format(split=Split(split).value)


def compute_logits(
    agent: Any, rows: Sequence[LabeledExample], batch_size: int = 32, progress: bool = False
) -> dict[str, np.ndarray]:
    """Raw per-question marker logits for `rows`, batched, through laya's own decode path."""
    from laya.agent import _option_logits
    from laya.common import collate_items

    agent.amp_enabled, agent.dtype = False, torch.float32  # fp32, as CPU serving computes them
    qids = list(QUESTIONS)
    internal = {q: agent._to_internal(QUESTIONS[q]) for q in qids}
    collected: dict[str, list[np.ndarray]] = {q: [] for q in qids}
    starts: Any = range(0, len(rows), batch_size)
    if progress:
        from tqdm.auto import tqdm

        starts = tqdm(starts, desc="logits", unit="batch")
    with torch.inference_mode():
        for start in starts:
            chunk = rows[start : start + batch_size]
            per_row = [
                agent._encode_state(
                    build_state(agent.tok, r.prompt, r.conversation_summary), qids, internal
                )
                for r in chunk
            ]
            raw, _ = agent._forward(collate_items(per_row, agent.tok.pad_token_id))
            flat = _option_logits(raw, [item for items in per_row for item in items], 0)
            for k in range(len(chunk)):
                for j, q in enumerate(qids):
                    collected[q].append(np.asarray(flat[k * len(qids) + j], dtype=np.float32))
    return {f"logits_{q}": np.stack(v) for q, v in collected.items()}


def export_logits(
    checkpoint: Path,
    split: Split,
    *,
    dataset: Path | None = None,
    out_dir: Path | None = None,
    device: str = "auto",
    batch_size: int = 32,
    limit: int = DEFAULT_LIMIT,
    progress: bool = False,
) -> Path:
    """Write logits_<split>.npz for the first `limit` rows of `split` (default: next to best/)."""
    import laya

    rows = read_split(split, dataset)[:limit]
    if not rows:
        raise ValueError(f"no {Split(split).value} rows to score")
    agent = laya.load(str(checkpoint), device=str(resolve(device)))
    arrays = compute_logits(agent, rows, batch_size, progress)
    targets = [target_options(r) for r in rows]
    out = logits_path(out_dir or checkpoint.parent, split)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        query_id=np.array([r.query_id for r in rows]),
        source=np.array([r.source for r in rows]),
        split=np.array([r.split.value for r in rows]),
        tier=np.array([t["tier"] for t in targets], dtype=np.int64),
        needs_tools=np.array([t["tools"] for t in targets], dtype=np.int64),
        needs_rag=np.array([t["rag"] for t in targets], dtype=np.int64),
        checkpoint_config_sha256=np.array(config_sha256(checkpoint)),
        **arrays,
    )
    return out


def load_logits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def label_index(data: dict[str, np.ndarray], qid: str) -> np.ndarray:
    """The correct option per row for one question, in that question's canonical order."""
    return data["tier"] if qid == "tier" else data[f"needs_{qid}"]


def export_logits_cmd(
    checkpoint: Annotated[
        Path | None, typer.Option(help="[default: <out dir>/checkpoints/laya/best]")
    ] = None,
    splits: Annotated[str, typer.Option(help="Comma list of splits.")] = "calibration,test",
    dataset: Annotated[Path | None, typer.Option(help="[default: <data dir>]")] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="[default: the checkpoint's run dir]")
    ] = None,
    device: Annotated[str, typer.Option(help="auto | cuda | mps | cpu")] = "auto",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    limit: Annotated[int, typer.Option(min=1, help="First N rows per split.")] = DEFAULT_LIMIT,
) -> None:
    """Score splits with a checkpoint and save raw logits for CPU-side calibration and metrics."""
    checkpoint = checkpoint or paths.checkpoints_dir() / "laya" / "best"
    for name in (s.strip() for s in splits.split(",") if s.strip()):
        out = export_logits(
            checkpoint, Split(name), dataset=dataset, out_dir=out_dir, device=device,
            batch_size=batch_size, limit=limit,
        )  # fmt: skip
        n = len(load_logits(out)["query_id"])
        typer.echo(f"{name}: {n} rows -> {out}")
