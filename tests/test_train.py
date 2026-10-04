"""train.py on a tiny random ModernBERT: spec, shuffling, losses, splits, fit, laya round-trip."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import laya
import pytest
import torch
import torch.nn.functional as F
from laya.agent import Agent
from laya.revisions import PINNED_REVISIONS
from transformers import PreTrainedTokenizerFast

from tests.conftest import TINY_HEAD_MAX_LEN, TINY_MAX_LEN, TinyModelFactory
from tollgate.collect.build_dataset import write_dataset
from tollgate.schema import TIER_ORDER, LabeledExample, Split
from tollgate.train import train as tt

CFG = tt.TrainConfig(
    epochs=2,
    batch_size=4,
    grad_accum=2,
    max_len=TINY_MAX_LEN,
    head_max_len=TINY_HEAD_MAX_LEN,
    eval_every=1,
    patience=10,
)


def test_question_spec_is_valid_laya_and_tier_ordered() -> None:
    for qid, spec in tt.QUESTIONS.items():
        Agent._check_question(qid, spec)
    assert list(tt.QUESTIONS["tier"]["criteria"]) == [t.value for t in TIER_ORDER]
    assert {q["type"] for q in tt.QUESTIONS.values()} == {"choice", "noul"}


def test_base_model_pinned_to_reviewed_sha() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", tt.LAYA_REVISION)
    assert PINNED_REVISIONS[tt.LAYA_REPO] == tt.LAYA_REVISION


def test_encode_example_shuffles_and_tracks_targets(tok: PreTrainedTokenizerFast) -> None:
    rng = random.Random(0)
    tier_orders = set()
    for example in tt.synthetic_examples(60):
        items = tt.encode_example(tok, example, rng, CFG)
        targets = tt.target_options(example)
        assert [i["qid"] for i in items] == list(tt.QUESTIONS)
        for item in items:
            assert len(item["markers"]) == len(item["order"])
            assert item["order"][item["slot"]] == targets[item["qid"]]
            if item["qid"] != "tier":
                assert item["order"][item["true_slot"]] == 1
                assert item["order"][item["false_slot"]] == 0
            else:
                tier_orders.add(tuple(item["order"]))
    assert len(tier_orders) == 6  # every permutation of 3 options shows up


def test_build_state_truncates_summary_not_query(tok: PreTrainedTokenizerFast) -> None:
    query = "translate this poem"
    state = tt.build_state(tok, query, " ".join(["kernel"] * 1000))
    assert state["query"] == query
    assert len(tok(state["conversation_summary"], add_special_tokens=False)["input_ids"]) == (
        tt.SUMMARY_MAX_TOKENS
    )
    assert tt.build_state(tok, query, None) == {"query": query, "conversation_summary": ""}


def test_noul_bce_on_margin_equals_two_way_cross_entropy() -> None:
    torch.manual_seed(0)
    logits = torch.randn(6, 3)
    meta = []
    for i in range(6):
        order = [1, 0] if i % 2 else [0, 1]
        y = i % 3 == 0
        meta.append(
            {
                "qid": "tools",
                "true_slot": order.index(1),
                "false_slot": order.index(0),
                "y": float(y),
            }
        )
    loss = tt.question_losses(logits, meta)["tools"]
    two_way = logits[:, :2]
    target = torch.tensor([m["true_slot"] if m["y"] else m["false_slot"] for m in meta])
    assert torch.allclose(loss, F.cross_entropy(two_way, target))


def test_choice_loss_is_cross_entropy_on_shuffled_slot() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]])
    meta = [{"qid": "tier", "slot": 0}, {"qid": "tier", "slot": 2}]
    expected = F.cross_entropy(logits, torch.tensor([0, 2]))
    assert torch.allclose(tt.question_losses(logits, meta)["tier"], expected)
    assert tt.tier_correct(logits, meta) == (1, 2)


def test_holdout_is_stratified_from_train() -> None:
    rows = tt.synthetic_examples(300)
    fit_rows, holdout = tt.holdout_split(rows, 0.1, seed=0)
    assert {r.query_id for r in fit_rows}.isdisjoint(r.query_id for r in holdout)
    assert len(fit_rows) + len(holdout) == 300
    assert abs(len(holdout) - 30) <= 6  # within one row per (tier, source) stratum
    with pytest.raises(ValueError):
        tt.holdout_split(rows[:1], 0.1, seed=0)


def test_load_train_rows_never_returns_calibration(tmp_path: Path) -> None:
    splits = [Split.TRAIN] * 5 + [Split.CALIBRATION] * 3 + [Split.TEST] * 2
    rows = [
        r.model_copy(update={"split": s})
        for r, s in zip(tt.synthetic_examples(10), splits, strict=True)
    ]
    path = write_dataset(rows, tmp_path / "dataset.parquet")
    loaded = tt.load_train_rows(path, limit=100, synthetic=False, seed=0)
    assert [r.query_id for r in loaded] == [r.query_id for r in rows[:5]]


def test_fit_end_to_end_and_checkpoint_loads_in_laya(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    model, base_cfg = tiny_model()
    fit_rows, holdout = tt.holdout_split(tt.synthetic_examples(24), 0.2, seed=0)
    result = tt.fit(
        model, tok, base_cfg, fit_rows, holdout, CFG, torch.device("cpu"), tmp_path,
        synthetic=True, echo=lambda _: None,
    )  # fmt: skip

    steps_per_epoch = math.ceil(math.ceil(len(fit_rows) / CFG.batch_size) / CFG.grad_accum)
    assert len(result.train) == len(result.val) == steps_per_epoch * CFG.epochs
    assert all(torch.isfinite(torch.tensor(p.loss)) for p in result.train + result.val)
    assert result.best_step is not None
    assert json.loads((tmp_path / "history.json").read_text())["synthetic"] is True

    saved_cfg = json.loads((tmp_path / "best" / "rl_agent_config.json").read_text())
    assert saved_cfg["temperature"] == [1.0, 1.0, 1.0]
    assert saved_cfg["temperature_by_options"] == {}
    assert saved_cfg["tollgate"]["synthetic"] is True
    assert saved_cfg["tollgate"]["questions"] == tt.QUESTIONS

    agent = laya.load(str(tmp_path / "best"), device="cpu")
    answers = agent.predict_batch([{"query": "poem", "conversation_summary": ""}], tt.QUESTIONS)
    answer = answers[0]["answers"]
    assert answer["tier"]["choice"] in {t.value for t in TIER_ORDER}
    assert 0.0 <= answer["tools"]["noul"] <= 1.0


def test_early_stopping_on_holdout(
    tok: PreTrainedTokenizerFast,
    tiny_model: TinyModelFactory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holdout_losses: Iterator[float] = iter([1.0, 0.8, 0.9, 0.95, 0.5, 0.4])
    saved_at: list[int] = []

    def fake_evaluate(*_: Any) -> dict[str, float]:
        loss = next(holdout_losses)
        return {"tier": loss, "tools": 0.0, "rag": 0.0, "loss": loss, "tier_acc": 0.5}

    monkeypatch.setattr(tt, "evaluate", fake_evaluate)
    monkeypatch.setattr(tt, "save_checkpoint", lambda *a, **_: saved_at.append(a[4]["step"]))
    model, base_cfg = tiny_model()
    cfg = CFG.model_copy(update={"grad_accum": 1, "patience": 2, "epochs": 3})
    result = tt.fit(
        model, tok, base_cfg, tt.synthetic_examples(40), [], cfg, torch.device("cpu"), tmp_path,
        synthetic=True, echo=lambda _: None,
    )  # fmt: skip
    assert result.stopped_early
    assert [p.loss for p in result.val] == [1.0, 0.8, 0.9, 0.95]
    assert result.best_step == 2
    assert saved_at == [1, 2]


def test_label_example_with_summary_roundtrips(tmp_path: Path) -> None:
    row = tt.synthetic_examples(1)[0].model_copy(update={"conversation_summary": "earlier turns"})
    path = write_dataset([row], tmp_path / "d.parquet")
    assert tt.load_train_rows(path, 1, synthetic=False, seed=0) == [row]
    assert isinstance(row, LabeledExample)


# --- device parity, fp16 AMP, sdpa ------------------------------------------------------------

CPU = torch.device("cpu")


def _quiet(_: str) -> None:
    pass


@pytest.mark.parametrize(
    ("device", "amp", "expected"),
    [("cuda", True, True), ("cuda", False, False), ("cpu", True, False), ("mps", True, False)],
)
def test_amp_only_on_cuda(device: str, amp: bool, expected: bool) -> None:
    cfg = CFG.model_copy(update={"amp": amp})
    assert tt.use_amp(torch.device(device), cfg) is expected


def test_amp_autocasts_to_fp16_never_bf16(
    tok: PreTrainedTokenizerFast,
    tiny_model: TinyModelFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, torch.dtype]] = []
    real_autocast = torch.autocast

    def recording_autocast(device_type: str, dtype: torch.dtype) -> Any:
        seen.append((device_type, dtype))
        return real_autocast("cpu", enabled=False)  # run the real forward on this cpu box

    monkeypatch.setattr(tt.torch, "autocast", recording_autocast)
    model, _ = tiny_model()
    example = tt.synthetic_examples(1)[0]
    batch = tt.collate_items([tt.encode_example(tok, example, random.Random(0), CFG)], 0)
    tt._forward(model, batch, CPU, amp=True)
    assert seen == [("cuda", torch.float16)]
    seen.clear()
    tt._forward(model, batch, CPU, amp=False)
    assert seen == []


def test_force_sdpa_pins_the_backbone(tiny_model: TinyModelFactory) -> None:
    model, _ = tiny_model()
    model.encoder.set_attn_implementation("eager")
    tt.force_sdpa(model)
    assert model.encoder.config._attn_implementation == "sdpa"


# --- save-every / resume -----------------------------------------------------------------------


def _rows() -> tuple[list[LabeledExample], list[LabeledExample]]:
    return tt.holdout_split(tt.synthetic_examples(24), 0.2, seed=0)


@pytest.mark.parametrize("stop_after", [2, 3], ids=["mid-epoch", "epoch-boundary"])
def test_resumed_run_matches_uninterrupted_run(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path, stop_after: int
) -> None:
    fit_rows, holdout = _rows()
    cfg = CFG.model_copy(update={"save_every": 1})

    model_a, base_cfg = tiny_model()
    full = tt.fit(
        model_a, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path / "a",
        synthetic=True, echo=_quiet,
    )  # fmt: skip

    model_b, base_cfg = tiny_model()
    cut = tt.fit(
        model_b, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path / "b",
        synthetic=True, echo=_quiet, stop_after_steps=stop_after,
    )  # fmt: skip
    assert len(cut.train) == stop_after < len(full.train)

    last = tmp_path / "b" / tt.LAST_DIR
    model_c, tok_c, cfg_c = tt.load_base(last)  # a fresh session: weights from last/
    resumed = tt.fit(
        model_c, tok_c, cfg_c, fit_rows, holdout, cfg, CPU, tmp_path / "b",
        synthetic=True, echo=_quiet, resume=last,
    )  # fmt: skip

    assert [p.loss for p in resumed.train] == [p.loss for p in full.train]
    assert [p.loss for p in resumed.val] == [p.loss for p in full.val]
    assert resumed.best_step == full.best_step
    full_state, resumed_state = model_a.state_dict(), model_c.state_dict()
    assert full_state.keys() == resumed_state.keys()
    for name, tensor in full_state.items():
        assert torch.equal(tensor, resumed_state[name]), name


def test_last_dir_is_laya_loadable_and_resumable(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    model, base_cfg = tiny_model()
    fit_rows, holdout = _rows()
    cfg = CFG.model_copy(update={"epochs": 1, "save_every": 2})
    tt.fit(
        model, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path,
        synthetic=True, echo=_quiet,
    )  # fmt: skip
    last = tmp_path / tt.LAST_DIR
    for name in (tt.STATE_JSON, tt.STATE_TENSORS, tt.TRAIN_IDS_FILE, "model.safetensors"):
        assert (last / name).exists(), name
    state, tensors = tt.load_trainer_state(last)
    assert state.step == 2 and (state.epoch, state.next_chunk) == (0, 4)
    assert set(tensors) == {"optimizer", "scheduler", "scaler", "torch_rng", "cuda_rng"}
    assert tensors["scaler"] == {}  # fp16 scaling is off on cpu
    laya.load(str(last), device="cpu")


def test_on_save_sees_every_directory_written(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    written: list[str] = []
    model, base_cfg = tiny_model()
    fit_rows, holdout = _rows()
    cfg = CFG.model_copy(update={"epochs": 1, "save_every": 1})
    tt.fit(
        model, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path,
        synthetic=True, echo=_quiet, on_save=lambda p: written.append(p.name),
    )  # fmt: skip
    assert written.count(tt.LAST_DIR) == 3  # one per optimizer step
    assert tt.BEST_DIR in written


def _cut_run(tok: Any, tiny_model: TinyModelFactory, run_dir: Path) -> Path:
    model, base_cfg = tiny_model()
    fit_rows, holdout = _rows()
    tt.fit(
        model, tok, base_cfg, fit_rows, holdout, CFG.model_copy(update={"save_every": 1}),
        CPU, run_dir, synthetic=True, echo=_quiet, stop_after_steps=1,
    )  # fmt: skip
    return run_dir / tt.LAST_DIR


def test_resume_refuses_changed_settings_or_rows(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    last = _cut_run(tok, tiny_model, tmp_path)
    state, _ = tt.load_trainer_state(last)
    fit_rows, holdout = _rows()
    fingerprint = tt._rows_fingerprint(fit_rows, holdout)
    saved = state.config

    tt._check_resumable(
        state, saved.model_copy(update={"patience": 99, "save_every": 7}), fingerprint
    )
    with pytest.raises(ValueError, match=r"settings differ.*\['lr'\]"):
        tt._check_resumable(state, saved.model_copy(update={"lr": 1e-3}), fingerprint)
    with pytest.raises(ValueError, match="train rows differ"):
        tt._check_resumable(state, saved, tt._rows_fingerprint(fit_rows[1:], holdout))


def test_resume_needs_a_last_dir(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    _cut_run(tok, tiny_model, tmp_path)
    with pytest.raises(FileNotFoundError, match="not resumable"):
        tt.load_trainer_state(tmp_path / tt.BEST_DIR)


def test_time_budget_stops_cleanly_and_resumes_identically(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    fit_rows, holdout = _rows()
    cfg = CFG.model_copy(update={"save_every": 0})  # the guard saves last/ on its own

    model_a, base_cfg = tiny_model()
    full = tt.fit(
        model_a, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path / "a",
        synthetic=True, echo=_quiet,
    )  # fmt: skip

    model_b, base_cfg = tiny_model()
    capped = tt.fit(
        model_b, tok, base_cfg, fit_rows, holdout, cfg, CPU, tmp_path / "b",
        synthetic=True, echo=_quiet, max_seconds=0.0,  # budget spent after the first step
    )  # fmt: skip
    assert capped.budget_exhausted and not capped.stopped_early
    assert len(capped.train) == 1
    last = tmp_path / "b" / tt.LAST_DIR
    assert (last / tt.STATE_JSON).exists()

    model_c, tok_c, cfg_c = tt.load_base(last)
    resumed = tt.fit(
        model_c, tok_c, cfg_c, fit_rows, holdout, cfg, CPU, tmp_path / "b",
        synthetic=True, echo=_quiet, resume=last,
    )  # fmt: skip
    assert not resumed.budget_exhausted
    assert [p.loss for p in resumed.train] == [p.loss for p in full.train]
    for name, tensor in model_a.state_dict().items():
        assert torch.equal(tensor, model_c.state_dict()[name]), name


def test_progress_bar_runs(
    tok: PreTrainedTokenizerFast, tiny_model: TinyModelFactory, tmp_path: Path
) -> None:
    model, base_cfg = tiny_model()
    fit_rows, holdout = _rows()
    result = tt.fit(
        model, tok, base_cfg, fit_rows, holdout, CFG.model_copy(update={"epochs": 1}), CPU,
        tmp_path, synthetic=True, echo=_quiet, progress=True,
    )  # fmt: skip
    assert len(result.train) == 3
