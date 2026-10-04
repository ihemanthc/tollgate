"""Full fine-tune of Laya on Tollgate's three typed questions.

Every (query, question) pair is one sequence in laya's own `build_sequence` layout, so a checkpoint
saved here loads with `laya.load(path)` and answers the way it was trained:

- tier  (choice): cross-entropy over the option-marker logits.
- tools (noul):   BCE on logit(true) - logit(false); over two markers that is laya's P(true).
- rag   (noul):   same.

Option order is reshuffled for every row on every pass, so no option owns a position.

Early stopping watches a stratified holdout carved from TRAIN. The calibration split is never read
here: it is reserved for temperature fitting. Saved checkpoints reset temperatures to 1.0, because
the shipped ones were fitted to the base model, not to this one.

Runs the same code on cpu, mps and cuda (tollgate.device.resolve). On CUDA it adds fp16 autocast
with a GradScaler -- fp16, not bf16, because the target GPU is a T4 (Turing has no bf16). The
backbone always runs sdpa attention; flash-attention-2 does not exist on Turing.

Long runs survive a session cap: --save-every N writes <run>/last (weights plus optimizer,
scheduler, scaler and RNG state) and --resume-from <dir> continues from it. Batch order and option
shuffles are seeded per (epoch, micro-batch), so a resumed run replays exactly what an
uninterrupted one would have seen.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import torch
import torch.nn.functional as F
import typer
from laya.agent import Agent
from laya.common import (
    QTYPES,
    DecisionModel,
    build_sequence,
    collate_items,
    encode_text,
    render_options,
    serialize_state,
)
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import save_file

from tollgate import paths
from tollgate.collect.build_dataset import read_split, stratified_assign
from tollgate.device import resolve
from tollgate.schema import TIER_ORDER, LabeledExample, Split

log = logging.getLogger(__name__)

LAYA_REPO = "convaiinnovations/laya"
# Reviewed commit; the same SHA laya 0.3.25 lists in laya.revisions.PINNED_REVISIONS.
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
TRAIN_IDS_FILE = "train_query_ids.json"
BEST_DIR = "best"
LAST_DIR = "last"
STATE_JSON = "trainer_state.json"
STATE_TENSORS = "trainer_state.pt"
SUMMARY_MAX_TOKENS = 128
DEFAULT_LIMIT = 32

# The multi-question spec. Option order of `tier` must stay TIER_ORDER (cheapest first).
QUESTIONS: dict[str, dict[str, Any]] = {
    "tier": {
        "type": "choice",
        "instructions": (
            "What is the cheapest model tier that answers `query` as well as a frontier model?"
        ),
        "criteria": {
            "local_small": "a small local model is enough: short, common or simple",
            "mid_tier": "needs a capable hosted model, but not the strongest one",
            "frontier": "needs the strongest model: hard reasoning, niche expertise, long work",
        },
    },
    "tools": {
        "type": "noul",
        "instructions": (
            "Does answering `query` require calling tools, such as running code or acting on "
            "external systems?"
        ),
    },
    "rag": {
        "type": "noul",
        "instructions": (
            "Does answering `query` require retrieving current or external information the "
            "model may not already know?"
        ),
    },
}
_INTERNAL = {qid: Agent._to_internal(spec) for qid, spec in QUESTIONS.items()}

Item = dict[str, Any]
Logger = Callable[[str], None]


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=8, ge=1, description="Queries per micro-batch (x3 rows).")
    grad_accum: int = Field(default=2, ge=1)
    lr: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    max_len: int = Field(default=512, ge=64)
    head_max_len: int = Field(default=192, ge=32)
    patience: int = Field(default=2, ge=1, description="Evals without improvement before stop.")
    eval_every: int = Field(default=0, ge=0, description="Optimizer steps; 0 = once per epoch.")
    holdout_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    grad_checkpointing: bool = True
    amp: bool = Field(default=True, description="fp16 autocast + GradScaler on CUDA only.")
    save_every: int = Field(default=0, ge=0, description="Steps between resumable saves; 0 = off.")
    seed: int = 0


# Settings a resumed run may change; every other TrainConfig field must match the saved run.
RESUME_MUTABLE = frozenset({"patience", "eval_every", "save_every", "grad_checkpointing"})


class LossPoint(BaseModel):
    step: int
    epoch: float
    loss: float
    tier: float
    tools: float
    rag: float
    tier_acc: float | None = None
    lr: float | None = None


class TrainResult(BaseModel):
    run_dir: Path
    synthetic: bool
    train: list[LossPoint]
    val: list[LossPoint]
    best_step: int | None
    best_val_loss: float | None
    stopped_early: bool
    budget_exhausted: bool = Field(
        default=False, description="Stopped by max_seconds; resume from last/ to continue."
    )
    seconds: float


class TrainerState(BaseModel):
    """Everything but tensors needed to continue a run exactly where `last/` was written."""

    step: int
    epoch: int = Field(description="Epoch to continue in.")
    next_chunk: int = Field(description="First micro-batch of `epoch` not yet trained on.")
    best_loss: float | None
    best_step: int | None
    bad_evals: int
    train: list[LossPoint]
    val: list[LossPoint]
    config: TrainConfig
    rows_sha256: str = Field(description="Fingerprint of the fit and holdout query ids.")
    elapsed_s: float


def use_amp(device: torch.device, cfg: TrainConfig) -> bool:
    """fp16 mixed precision runs only on CUDA; cpu and mps take the same path in fp32."""
    return cfg.amp and device.type == "cuda"


def force_sdpa(model: DecisionModel) -> None:
    """Pin the backbone to sdpa attention: ModernBERT may otherwise pick flash-attention-2."""
    encoder = model.encoder
    if getattr(encoder.config, "_attn_implementation", None) != "sdpa":
        encoder.set_attn_implementation("sdpa")


def build_state(tok: Any, query: str, summary: str | None) -> dict[str, str]:
    """Query first: laya keeps the start of a dict state, so overflow only ever cuts the summary."""
    text = (summary or "").strip()
    if text:
        ids = encode_text(tok, text, add_special_tokens=False)["input_ids"]
        if len(ids) > SUMMARY_MAX_TOKENS:
            text = tok.decode(ids[:SUMMARY_MAX_TOKENS]).strip()
    return {"query": query, "conversation_summary": text}


def target_options(example: LabeledExample) -> dict[str, int]:
    """Index of the correct option per question; noul options are [false, true]."""
    return {
        "tier": TIER_ORDER.index(example.tier),
        "tools": int(example.needs_tools),
        "rag": int(example.needs_rag),
    }


def encode_example(
    tok: Any, example: LabeledExample, rng: random.Random, cfg: TrainConfig
) -> list[Item]:
    """One sequence per question, each with a fresh random option order."""
    state = build_state(tok, example.prompt, example.conversation_summary)
    state_text = serialize_state(state).replace(tok.mask_token, " ")
    state_ids = encode_text(tok, state_text, add_special_tokens=False)["input_ids"]
    items: list[Item] = []
    for qid, target in target_options(example).items():
        q = _INTERNAL[qid]
        k = len(render_options(q))
        order = rng.sample(range(k), k)  # slot s shows option order[s]
        ids, markers = build_sequence(
            tok, state, q, cfg.max_len, cfg.head_max_len, option_order=order, state_ids=state_ids
        )
        if len(markers) != k:
            raise ValueError(f"{qid}: only {len(markers)} of {k} options fit in max_len")
        noul = q["t"] == "noul"
        items.append(
            {
                "ids": ids,
                "markers": markers,
                "qtype": QTYPES[q["t"]],
                "qid": qid,
                "order": order,
                "slot": order.index(target),
                "true_slot": order.index(1) if noul else -1,
                "false_slot": order.index(0) if noul else -1,
                "y": float(target),
            }
        )
    return items


def question_losses(logits: torch.Tensor, meta: Sequence[Item]) -> dict[str, torch.Tensor]:
    """Mean loss per question: CE over markers for choice, BCE on the true-false margin for noul."""
    losses: dict[str, torch.Tensor] = {}
    for qid, spec in QUESTIONS.items():
        rows = [i for i, m in enumerate(meta) if m["qid"] == qid]
        if not rows:
            continue
        sub = logits[rows]
        if spec["type"] == "choice":
            target = torch.tensor([meta[i]["slot"] for i in rows], device=logits.device)
            losses[qid] = F.cross_entropy(sub, target)
        else:
            true_idx = torch.tensor([[meta[i]["true_slot"]] for i in rows], device=logits.device)
            false_idx = torch.tensor([[meta[i]["false_slot"]] for i in rows], device=logits.device)
            margin = (sub.gather(1, true_idx) - sub.gather(1, false_idx)).squeeze(1)
            y = torch.tensor([meta[i]["y"] for i in rows], device=logits.device)
            losses[qid] = F.binary_cross_entropy_with_logits(margin, y.to(margin.dtype))
    return losses


def tier_correct(logits: torch.Tensor, meta: Sequence[Item]) -> tuple[int, int]:
    rows = [i for i, m in enumerate(meta) if m["qid"] == "tier"]
    target = torch.tensor([meta[i]["slot"] for i in rows], device=logits.device)
    return int((logits[rows].argmax(-1) == target).sum()), len(rows)


def _forward(
    model: DecisionModel, batch: dict[str, Any], device: torch.device, amp: bool = False
) -> torch.Tensor:
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
    with autocast:
        logits, _ = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["marker_pos"].to(device),
            batch["marker_mask"].to(device),
            batch["qtype"].to(device),
        )
    return logits.float()


def _chunks(rows: Sequence[LabeledExample], size: int) -> Iterator[Sequence[LabeledExample]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


@torch.no_grad()
def evaluate(
    model: DecisionModel,
    tok: Any,
    rows: Sequence[LabeledExample],
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Row-weighted holdout losses; option orders use a fixed seed, so evals are comparable."""
    was_training = model.training
    model.eval()
    rng = random.Random(f"eval:{cfg.seed}")
    amp = use_amp(device, cfg)
    sums: dict[str, float] = defaultdict(float)
    correct = total = 0
    for chunk in _chunks(rows, cfg.batch_size):
        batch = collate_items([encode_example(tok, ex, rng, cfg) for ex in chunk], tok.pad_token_id)
        logits = _forward(model, batch, device, amp)
        for qid, value in question_losses(logits, batch["meta"]).items():
            sums[qid] += value.item() * len(chunk)
        c, n = tier_correct(logits, batch["meta"])
        correct, total = correct + c, total + n
    model.train(was_training)
    out = {qid: sums[qid] / len(rows) for qid in QUESTIONS}
    return {**out, "loss": sum(out.values()), "tier_acc": correct / max(total, 1)}


def save_checkpoint(
    model: DecisionModel,
    tok: Any,
    base_cfg: dict[str, Any],
    path: Path,
    meta: dict[str, Any],
    *,
    train_ids: Sequence[str],
    write_extra: Callable[[Path], None] | None = None,
) -> Path:
    """Write a directory `laya.load(path)` accepts; swapped in atomically over any previous one.

    `train_ids` (fit + early-stop holdout) lets calibrate.py prove its rows were never trained on.
    `write_extra` adds files (the resumable trainer state) inside the same atomic swap.
    """
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / TRAIN_IDS_FILE).write_text(json.dumps(sorted(train_ids)), encoding="utf-8")
    if write_extra is not None:
        write_extra(tmp)
    cfg = {
        **base_cfg,
        "temperature": [1.0, 1.0, 1.0],  # refit on the calibration split, never inherited
        "temperature_by_options": {},
        "tollgate": {**meta, "questions": QUESTIONS},
    }
    (tmp / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(tmp / "model.safetensors"))
    tok.save_pretrained(str(tmp / "tokenizer"))
    model.encoder.config.save_pretrained(str(tmp / "encoder"))
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)
    return path


def _rows_fingerprint(
    fit_rows: Sequence[LabeledExample], holdout_rows: Sequence[LabeledExample]
) -> str:
    ids = [sorted(r.query_id for r in fit_rows), sorted(r.query_id for r in holdout_rows)]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _epoch_order(
    rows: Sequence[LabeledExample], cfg: TrainConfig, epoch: int
) -> list[LabeledExample]:
    """Fixed by (seed, epoch), so a resumed run replays exactly the batches it would have seen."""
    order = list(rows)
    random.Random(f"order:{cfg.seed}:{epoch}").shuffle(order)
    return order


def load_trainer_state(path: Path) -> tuple[TrainerState, dict[str, Any]]:
    """Read a last/ dir. Tensors load with weights_only=True: resume dirs may come from the Hub."""
    if not (path / STATE_JSON).exists() or not (path / STATE_TENSORS).exists():
        raise FileNotFoundError(
            f"{path} is not resumable (no {STATE_JSON}); resume from a {LAST_DIR}/ dir that "
            "`tollgate train --save-every N` wrote."
        )
    state = TrainerState.model_validate_json((path / STATE_JSON).read_text(encoding="utf-8"))
    tensors = torch.load(path / STATE_TENSORS, map_location="cpu", weights_only=True)
    return state, tensors


def _check_resumable(state: TrainerState, cfg: TrainConfig, fingerprint: str) -> None:
    saved = state.config.model_dump(exclude=set(RESUME_MUTABLE))
    current = cfg.model_dump(exclude=set(RESUME_MUTABLE))
    changed = sorted(k for k in current if saved.get(k) != current[k])
    if changed:
        raise ValueError(f"cannot resume: settings differ from the saved run: {changed}")
    if state.rows_sha256 != fingerprint:
        raise ValueError(
            "cannot resume: the train rows differ from the saved run (dataset, --limit or --seed)"
        )


def fit(
    model: DecisionModel,
    tok: Any,
    base_cfg: dict[str, Any],
    fit_rows: Sequence[LabeledExample],
    holdout_rows: Sequence[LabeledExample],
    cfg: TrainConfig,
    device: torch.device,
    run_dir: Path,
    *,
    synthetic: bool,
    echo: Logger = log.info,
    resume: Path | None = None,
    on_save: Callable[[Path], None] | None = None,
    stop_after_steps: int | None = None,
    max_seconds: float | None = None,
    progress: bool = False,
) -> TrainResult:
    """Train, early-stop on the holdout, write best/ (and last/ every cfg.save_every steps).

    `resume` continues from a last/ dir; `model` must already hold that dir's weights.
    `on_save` is called with every directory written (used to push to the Hub).
    `stop_after_steps` ends the run early without finishing, as a killed session would.
    `max_seconds` is a wall-clock budget for this session: once spent, last/ is written and
    training stops cleanly, so a capped session (Kaggle: 12 h) can be resumed.
    `progress` shows a tqdm bar over optimizer steps.
    """
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    force_sdpa(model)
    model.to(device).float().train()
    if cfg.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.head_checkpointing = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg.weight_decay},
            {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    micro_per_epoch = math.ceil(len(fit_rows) / cfg.batch_size)
    total_steps = math.ceil(micro_per_epoch / cfg.grad_accum) * cfg.epochs
    warmup = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp = use_amp(device, cfg)
    # Disabled (cpu, mps, --no-amp) the scaler is a pass-through, so every device runs this code.
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    meta = {
        "base_model": LAYA_REPO,
        "base_revision": LAYA_REVISION,
        "synthetic": synthetic,
        "train_config": cfg.model_dump(),
    }
    train_ids = [r.query_id for r in [*fit_rows, *holdout_rows]]
    fingerprint = _rows_fingerprint(fit_rows, holdout_rows)
    train_points: list[LossPoint] = []
    val_points: list[LossPoint] = []
    pending: list[dict[str, float]] = []
    best_loss, best_step, bad_evals, step, stopped = math.inf, None, 0, 0, False
    start_epoch, start_chunk, elapsed_before = 0, 0, 0.0

    if resume is not None:
        state, tensors = load_trainer_state(resume)
        _check_resumable(state, cfg, fingerprint)
        optimizer.load_state_dict(tensors["optimizer"])
        scheduler.load_state_dict(tensors["scheduler"])
        if tensors["scaler"]:  # empty when saved by a run without fp16 (e.g. on cpu)
            scaler.load_state_dict(tensors["scaler"])
        torch.set_rng_state(tensors["torch_rng"])
        if device.type == "cuda" and tensors["cuda_rng"]:
            torch.cuda.set_rng_state_all(tensors["cuda_rng"])
        step, bad_evals, best_step = state.step, state.bad_evals, state.best_step
        best_loss = math.inf if state.best_loss is None else state.best_loss
        train_points, val_points = list(state.train), list(state.val)
        start_epoch, start_chunk, elapsed_before = state.epoch, state.next_chunk, state.elapsed_s
        echo(f"resumed {resume} at step {step} (epoch {start_epoch}, batch {start_chunk})")

    def elapsed() -> float:
        return elapsed_before + time.perf_counter() - started

    def save_last(next_epoch: int, next_chunk: int) -> None:
        state = TrainerState(
            step=step,
            epoch=next_epoch,
            next_chunk=next_chunk,
            best_loss=None if best_step is None else best_loss,
            best_step=best_step,
            bad_evals=bad_evals,
            train=train_points,
            val=val_points,
            config=cfg,
            rows_sha256=fingerprint,
            elapsed_s=round(elapsed(), 1),
        )
        tensors = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }

        def write(tmp: Path) -> None:
            (tmp / STATE_JSON).write_text(state.model_dump_json(indent=2), encoding="utf-8")
            torch.save(tensors, tmp / STATE_TENSORS)

        path = save_checkpoint(
            model, tok, base_cfg, run_dir / LAST_DIR, {**meta, "step": step},
            train_ids=train_ids, write_extra=write,
        )  # fmt: skip
        echo(f"step {step} resumable checkpoint -> {path}")
        if on_save is not None:
            on_save(path)

    echo(
        f"{len(fit_rows)} fit rows, {total_steps} optimizer steps, device={device}, "
        f"precision={'fp16 amp' if amp else 'fp32'}"
    )
    interrupted = budget_exhausted = False
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=total_steps, initial=step, unit="step", desc="train")
    for epoch in range(start_epoch, cfg.epochs):
        chunks = list(_chunks(_epoch_order(fit_rows, cfg, epoch), cfg.batch_size))
        first = start_chunk if epoch == start_epoch else 0
        for i in range(first, len(chunks)):
            rng = random.Random(f"rows:{cfg.seed}:{epoch}:{i}")
            batch = collate_items(
                [encode_example(tok, ex, rng, cfg) for ex in chunks[i]], tok.pad_token_id
            )
            losses = question_losses(_forward(model, batch, device, amp), batch["meta"])
            loss = torch.stack(list(losses.values())).sum()
            window_start = (i // cfg.grad_accum) * cfg.grad_accum
            scaler.scale(loss / min(cfg.grad_accum, len(chunks) - window_start)).backward()
            pending.append({"loss": loss.item(), **{k: v.item() for k, v in losses.items()}})
            last_in_epoch = i + 1 == len(chunks)
            if (i + 1) % cfg.grad_accum and not last_in_epoch:
                continue

            scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(optimizer)  # skipped by the scaler if fp16 gradients overflowed
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            averaged = {k: sum(p[k] for p in pending) / len(pending) for k in pending[0]}
            pending.clear()
            point = LossPoint(
                step=step,
                epoch=round(epoch + (i + 1) / len(chunks), 3),
                lr=scheduler.get_last_lr()[0],
                **averaged,
            )
            train_points.append(point)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(loss=f"{point.loss:.4f}")
            echo(
                f"step {step}/{total_steps} train loss {point.loss:.4f} "
                f"(tier {point.tier:.3f} tools {point.tools:.3f} rag {point.rag:.3f}) "
                f"{elapsed():.0f}s"
            )

            if (cfg.eval_every and step % cfg.eval_every == 0) or last_in_epoch:
                metrics = evaluate(model, tok, holdout_rows, cfg, device)
                val_points.append(LossPoint(step=step, epoch=point.epoch, **metrics))
                echo(
                    f"step {step} holdout loss {metrics['loss']:.4f} "
                    f"tier_acc {metrics['tier_acc']:.3f}"
                )
                if metrics["loss"] < best_loss:
                    best_loss, best_step, bad_evals = metrics["loss"], step, 0
                    best = save_checkpoint(
                        model, tok, base_cfg, run_dir / BEST_DIR,
                        {**meta, "step": step, "holdout_loss": metrics["loss"]},
                        train_ids=train_ids,
                    )  # fmt: skip
                    if on_save is not None:
                        on_save(best)
                else:
                    bad_evals += 1
                    stopped = bad_evals >= cfg.patience
                    if stopped:
                        echo(f"early stop: {bad_evals} evals without improvement")
            if stopped:
                break
            position = (epoch + 1, 0) if last_in_epoch else (epoch, i + 1)
            saved_now = bool(cfg.save_every) and step % cfg.save_every == 0
            if saved_now:
                save_last(*position)
            if max_seconds is not None and time.perf_counter() - started >= max_seconds:
                if not saved_now:
                    save_last(*position)
                budget_exhausted = True
                echo(
                    f"time budget of {max_seconds / 3600:.2f} h spent at step {step}; "
                    "resume from last/"
                )
                break
            if stop_after_steps is not None and step >= stop_after_steps:
                interrupted = True
                break
        if stopped or interrupted or budget_exhausted:
            break
    if bar is not None:
        bar.close()

    if best_step is not None and not (run_dir / BEST_DIR).exists():
        echo(f"warning: best step {best_step} was saved in an earlier session; fetch its best/ dir")
    result = TrainResult(
        run_dir=run_dir,
        synthetic=synthetic,
        train=train_points,
        val=val_points,
        best_step=best_step,
        best_val_loss=None if best_step is None else best_loss,
        stopped_early=stopped,
        budget_exhausted=budget_exhausted,
        seconds=round(elapsed(), 1),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


_SYNTHETIC_WORDS = (
    "invoice", "gradient", "kernel", "poem", "tax", "recipe", "proof", "sql", "translate",
    "summarize", "contract", "orbit", "enzyme", "haiku", "refactor", "budget", "weather", "stock",
)  # fmt: skip


def synthetic_examples(n: int, seed: int = 0) -> list[LabeledExample]:
    """Plumbing-only TRAIN rows with random labels. Never written to data/; losses mean nothing."""
    rng = random.Random(f"synthetic:{seed}")
    rows = []
    for i in range(n):
        words = " ".join(rng.choices(_SYNTHETIC_WORDS, k=rng.randint(4, 40)))
        summary = " ".join(rng.choices(_SYNTHETIC_WORDS, k=rng.randint(5, 300)))
        rows.append(
            LabeledExample(
                query_id=f"synthetic-{i}",
                prompt=f"[synthetic smoke row {i}] {words}",
                conversation_summary=f"[synthetic] {summary}" if rng.random() < 0.5 else None,
                source=rng.choice(("synthetic-a", "synthetic-b")),
                split=Split.TRAIN,
                tier=rng.choice(TIER_ORDER),
                needs_tools=rng.random() < 0.3,
                needs_rag=rng.random() < 0.3,
            )
        )
    return rows


def load_train_rows(
    dataset: Path, limit: int, *, synthetic: bool, seed: int
) -> list[LabeledExample]:
    """The first `limit` TRAIN rows. Only Split.TRAIN is ever requested from the dataset."""
    rows = (
        synthetic_examples(limit, seed) if synthetic else read_split(Split.TRAIN, dataset)[:limit]
    )
    leaked = [r.query_id for r in rows if r.split is not Split.TRAIN]
    if leaked:
        raise ValueError(f"non-train rows reached training: {leaked[:3]}")
    return rows


def holdout_split(
    rows: Sequence[LabeledExample], fraction: float, seed: int
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Stratified (tier, source) early-stopping holdout carved out of TRAIN."""
    buckets = stratified_assign(
        {r.query_id: (r.tier.value, r.source) for r in rows},
        (("fit", 1.0 - fraction), ("holdout", fraction)),
        seed,
    )
    fit_rows = [r for r in rows if buckets[r.query_id] == "fit"]
    holdout = [r for r in rows if buckets[r.query_id] == "holdout"]
    if not fit_rows or not holdout:
        raise ValueError(f"{len(rows)} train rows is too few to hold out {fraction:.0%}")
    return fit_rows, holdout


def load_base(resume: Path | None = None) -> tuple[DecisionModel, Any, dict[str, Any]]:
    """The pinned base Laya, or the weights of the last/ dir a resumed run continues from."""
    import laya

    if resume is not None:
        agent = laya.load(str(resume), device="cpu")
    else:
        agent = laya.load(LAYA_REPO, revision=LAYA_REVISION, device="cpu")
    return agent.model, agent.tok, dict(agent.cfg)


def train_laya(
    limit: Annotated[int, typer.Option(help="First N train rows.")] = DEFAULT_LIMIT,
    epochs: Annotated[int, typer.Option(min=1)] = 3,
    batch_size: Annotated[int, typer.Option(min=1, help="Queries per micro-batch.")] = 8,
    grad_accum: Annotated[int, typer.Option(min=1)] = 2,
    lr: Annotated[float, typer.Option()] = 2e-5,
    patience: Annotated[int, typer.Option(min=1)] = 2,
    eval_every: Annotated[int, typer.Option(min=0, help="Steps; 0 = per epoch.")] = 0,
    holdout: Annotated[float, typer.Option(help="Early-stop holdout, from train.")] = 0.1,
    max_len: Annotated[int, typer.Option()] = 512,
    device: Annotated[str, typer.Option(help="auto | cuda | mps | cpu")] = "auto",
    amp: Annotated[bool, typer.Option(help="fp16 autocast + GradScaler on CUDA.")] = True,
    seed: Annotated[int, typer.Option()] = 0,
    grad_checkpointing: Annotated[bool, typer.Option()] = True,
    save_every: Annotated[
        int, typer.Option(min=0, help="Write a resumable last/ every N steps; 0 = off.")
    ] = 0,
    resume_from: Annotated[
        Path | None, typer.Option(help="A last/ dir written by --save-every.")
    ] = None,
    push_to_hub: Annotated[
        bool, typer.Option(help="Upload best/ and last/ to HF_REPO_MODEL as they are written.")
    ] = False,
    synthetic: Annotated[bool, typer.Option(help="Random-label plumbing smoke test.")] = False,
    dataset: Annotated[
        Path | None, typer.Option(help="[default: <data dir>/dataset.parquet]")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="[default: <out dir>/checkpoints]")] = None,
    run_name: Annotated[str | None, typer.Option(help="Default: synthetic-smoke | laya")] = None,
    max_hours: Annotated[
        float, typer.Option(min=0, help="Wall-clock budget for this session; 0 = none.")
    ] = 0.0,
    progress: Annotated[bool, typer.Option(help="Show a tqdm progress bar.")] = False,
) -> TrainResult:
    """Fine-tune Laya on the train split; early-stop on a holdout carved from train."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    dataset = dataset or paths.dataset_path()
    run_dir = (out or paths.checkpoints_dir()) / (
        run_name or ("synthetic-smoke" if synthetic else "laya")
    )
    cfg = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        patience=patience,
        eval_every=eval_every,
        holdout_fraction=holdout,
        max_len=max_len,
        grad_checkpointing=grad_checkpointing,
        amp=amp,
        save_every=save_every,
        seed=seed,
    )
    if not synthetic and not dataset.exists():
        raise typer.BadParameter(f"{dataset} not found; run build-dataset or pass --synthetic.")
    torch_device = resolve(device)
    rows = load_train_rows(dataset, limit, synthetic=synthetic, seed=seed)
    fit_rows, holdout_rows = holdout_split(rows, cfg.holdout_fraction, seed)
    typer.echo(
        f"{len(fit_rows)} fit / {len(holdout_rows)} holdout rows (calibration split untouched)"
        + ("  [SYNTHETIC LABELS: plumbing test only]" if synthetic else "")
    )
    on_save = None
    if push_to_hub:
        from tollgate import hub

        def on_save(path: Path) -> None:
            hub.push_checkpoint(path, path_in_repo=f"runs/{run_dir.name}/{path.name}")

    model, tok, base_cfg = load_base(resume_from)
    result = fit(
        model, tok, base_cfg, fit_rows, holdout_rows, cfg, torch_device, run_dir,
        synthetic=synthetic, resume=resume_from, on_save=on_save,
        max_seconds=max_hours * 3600 if max_hours else None, progress=progress,
    )  # fmt: skip
    val_by_step = {p.step: p for p in result.val}
    typer.echo("\nstep  train_loss  holdout_loss  holdout_tier_acc")
    for p in result.train:
        v = val_by_step.get(p.step)
        typer.echo(
            f"{p.step:>4}  {p.loss:>10.4f}  "
            + (f"{v.loss:>12.4f}  {v.tier_acc:>16.3f}" if v and v.tier_acc is not None else "")
        )
    typer.echo(
        f"best step {result.best_step} (holdout {result.best_val_loss}) -> "
        f"{run_dir / BEST_DIR}; {result.seconds:.0f}s"
        + ("; time budget spent, resume from last/" if result.budget_exhausted else "")
    )
    return result


if __name__ == "__main__":
    typer.run(train_laya)
