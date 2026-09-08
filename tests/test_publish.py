"""push-dataset: restricted text never leaves, split indexes, judge rule, card, rebuild script."""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from tollgate import paths
from tollgate.cli import app
from tollgate.collect import prompts, publish, rebuild_dataset
from tollgate.collect.build_dataset import read_split, write_dataset
from tollgate.collect.judge import write_verdicts
from tollgate.collect.runner import write_runs
from tollgate.schema import JudgeVerdict, LabeledExample, Split, Tier, TierRun

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "docs" / publish.CARD_NAME
FRONTIER, JUDGE = "openai/front", "anthropic/judge"


def _ex(text: str, source: str, split: Split, tier: Tier) -> LabeledExample:
    return LabeledExample(
        query_id=hashlib.sha256(text.encode()).hexdigest(),
        prompt=text,
        conversation_summary="earlier turns" if source == "lmsys-chat-1m" else None,
        source=source,
        split=split,
        tier=tier,
        needs_tools=False,
        needs_rag=source == "lmsys-chat-1m",
        judged_tiers=(Tier.LOCAL_SMALL, Tier.MID_TIER),
    )


S, T = Split, Tier
EXAMPLES = [
    _ex("SECRET lmsys prompt about my landlord", "lmsys-chat-1m", S.TRAIN, T.MID_TIER),
    _ex("SECRET lmsys prompt: write a poem", "lmsys-chat-1m", S.TRAIN, T.LOCAL_SMALL),
    _ex("SECRET lmsys prompt: debug my code", "lmsys-chat-1m", S.CALIBRATION, T.FRONTIER),
    _ex("SECRET lmsys prompt: plan a trip", "lmsys-chat-1m", S.TEST, T.MID_TIER),
    _ex("Janet has 3 apples. How many?", "gsm8k", S.TRAIN, T.LOCAL_SMALL),
    _ex("A train leaves at 3pm...", "gsm8k", S.TEST, T.MID_TIER),
    _ex("Which organ pumps blood?\nA. Heart", "mmlu", S.TRAIN, T.LOCAL_SMALL),
    _ex("Order of Z_24 subgroup?\nA. 4", "mmlu", S.CALIBRATION, T.FRONTIER),
    _ex("SECRET text from a source nobody reviewed", "new-source", S.TRAIN, T.MID_TIER),
]
WITHHELD = [e for e in EXAMPLES if e.source in ("lmsys-chat-1m", "new-source")]


def _write_provenance(judge: str = JUDGE, frontier: str = FRONTIER) -> None:
    models = {
        Tier.LOCAL_SMALL: "ollama/small",
        Tier.MID_TIER: "openai/mid",
        Tier.FRONTIER: frontier,
    }
    write_runs(
        [
            TierRun(
                query_id=e.query_id, tier=t, model=m, completion="a", prompt_tokens=1,
                completion_tokens=1, cost_usd=0.0, latency_s=0.0,
            )
            for e in EXAMPLES
            for t, m in models.items()
        ],
        paths.tier_runs_path(),
    )  # fmt: skip
    write_verdicts(
        [
            JudgeVerdict(
                query_id=e.query_id, candidate_tier=t, verdict="worse", reason="r",
                judge_model=judge,
            )
            for e in EXAMPLES
            for t in (Tier.LOCAL_SMALL, Tier.MID_TIER)
        ],
        paths.verdicts_path(),
    )  # fmt: skip


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))
    monkeypatch.setenv(paths.OUT_DIR_ENV, str(tmp_path / "out"))
    _write_provenance()
    return tmp_path


def _stage(tmp_path: Path) -> dict[str, Path]:
    files, _ = publish.stage(
        EXAMPLES,
        CARD.read_text(encoding="utf-8"),
        paths.tier_runs_path(),
        paths.verdicts_path(),
        tmp_path / "staged",
    )
    return files


# --- what is published --------------------------------------------------------------------------


def test_restricted_and_unknown_sources_lose_their_text() -> None:
    frame, inclusion = publish.public_frame(EXAMPLES)
    by_id = frame.set_index("query_id")
    for e in EXAMPLES:
        row = by_id.loc[e.query_id]
        assert row["text_sha256"] == e.query_id
        if e in WITHHELD:
            assert not row["text_included"]
            assert pd.isna(row["prompt"]) and pd.isna(row["conversation_summary"])
        else:
            assert row["text_included"] and row["prompt"] == e.prompt
    assert {i.source: (i.rows, i.text_rows, i.withheld_rows) for i in inclusion} == {
        "gsm8k": (2, 2, 0),
        "lmsys-chat-1m": (4, 0, 4),
        "mmlu": (2, 2, 0),
        "new-source": (1, 0, 1),  # fail closed: never reviewed, so never published
    }


def test_no_withheld_text_in_any_staged_file(tmp_path: Path) -> None:
    files = _stage(tmp_path)
    secrets = [e.prompt for e in WITHHELD] + ["earlier turns"]
    table = pd.read_parquet(files["dataset.parquet"])
    cells = [str(v) for v in table.astype(str).to_numpy().ravel()]
    for secret in secrets:
        assert not any(secret in cell for cell in cells), secret
    for name, path in files.items():
        if name != "dataset.parquet":
            text = path.read_text(encoding="utf-8")
            assert not any(secret in text for secret in secrets), name
    assert "SECRET" not in files["dataset.parquet"].read_bytes().decode("latin-1")


def test_columns_are_exactly_the_documented_ones() -> None:
    frame, _ = publish.public_frame(EXAMPLES)
    assert list(frame.columns) == list(publish.COLUMNS)
    assert set(publish.COLUMNS) == set(LabeledExample.model_fields) | {
        "text_sha256",
        "text_included",
    }


def test_refuses_rows_whose_id_is_not_the_text_hash() -> None:
    bad = EXAMPLES[0].model_copy(update={"query_id": "synthetic-0"})
    with pytest.raises(ValueError, match="not sha256"):
        publish.public_frame([bad])


def test_split_index_files_match_the_split_column(tmp_path: Path) -> None:
    files = _stage(tmp_path)
    seen: set[str] = set()
    for split in Split:
        ids = files[f"splits/{split.value}.txt"].read_text().split()
        assert ids == sorted(e.query_id for e in EXAMPLES if e.split is split)
        assert seen.isdisjoint(ids)
        seen |= set(ids)
    assert seen == {e.query_id for e in EXAMPLES}


def test_staged_upload_set(tmp_path: Path) -> None:
    files = _stage(tmp_path)
    assert set(files) == {
        "dataset.parquet",
        "splits/train.txt",
        "splits/calibration.txt",
        "splits/test.txt",
        "README.md",
        "rebuild_dataset.py",
    }


def test_no_rebuild_script_when_nothing_is_withheld(tmp_path: Path) -> None:
    open_rows = [e for e in EXAMPLES if e not in WITHHELD]
    files, _ = publish.stage(
        open_rows, CARD.read_text(), paths.tier_runs_path(), paths.verdicts_path(), tmp_path / "s"
    )
    assert publish.REBUILD_SCRIPT not in files


# --- judge independence -------------------------------------------------------------------------


@pytest.mark.parametrize("judge", [FRONTIER, "azure/front", "OPENAI/FRONT"])
def test_refuses_when_the_judge_was_the_frontier_model(tmp_path: Path, judge: str) -> None:
    _write_provenance(judge=judge)
    with pytest.raises(ValueError, match="is the frontier model"):
        _stage(tmp_path)


def test_refuses_without_verdicts_to_name_the_judge(tmp_path: Path) -> None:
    paths.verdicts_path().write_text("")
    with pytest.raises(ValueError, match="cannot name the judge"):
        _stage(tmp_path)


# --- the card -----------------------------------------------------------------------------------


def test_rendered_card_reports_the_data(tmp_path: Path) -> None:
    card = _stage(tmp_path)["README.md"].read_text(encoding="utf-8")
    assert card.startswith("---\npretty_name: Tollgate routing labels")
    assert publish.PENDING not in card
    assert f"| judge | `{JUDGE}` |" in card
    assert f"| frontier | `{FRONTIER}` |" in card
    # labels over the 9 rows: local 3, mid 4, frontier 2
    assert "| local_small | 3 | 33.3% |" in card
    assert "| mid_tier | 4 | 44.4% |" in card
    assert "| frontier | 2 | 22.2% |" in card
    assert "| lmsys-chat-1m | 4 | 0 | 4 |" in card
    assert "| **all** | 9 | 4 | 5 |" in card
    # train rows: lmsys mid + local, gsm8k local, mmlu local, new-source mid -> 3 / 2 / 0 of 9
    assert "| train | 3 | 2 | 0 | 5 | 55.6% |" in card


def test_committed_card_is_in_step_with_the_code() -> None:
    text = CARD.read_text(encoding="utf-8")
    blocks = dict(
        re.findall(r"<!-- tollgate:begin (\w+) -->\n(.*?)<!-- tollgate:end \1 -->", text, re.S)
    )
    for name, body in publish.static_blocks().items():
        assert blocks[name] == body, f"{name}: re-render docs/DATASET_CARD.md"
    for name in ("contents", "models", "labels", "splits", "generated"):
        assert blocks[name] == publish.PENDING, f"{name} must stay generated, not hand-written"


def test_card_states_both_licenses_and_the_rule() -> None:
    text = CARD.read_text(encoding="utf-8")
    for needle in (
        "Copyright (c) 2021 OpenAI",
        "Copyright (c) 2020 Dan Hendrycks",
        "Prohibited Transfers",
        "The judge is never the frontier model",
        "Known judge biases",
        "Not included",
    ):
        assert needle in text, needle


def test_render_card_rejects_unknown_blocks() -> None:
    with pytest.raises(ValueError, match="no blocks named"):
        publish.render_card("no blocks here", {"labels": "x"})


# --- rebuild script -----------------------------------------------------------------------------


def test_rebuild_restores_exactly_the_withheld_text() -> None:
    frame, _ = publish.public_frame(EXAMPLES)
    lmsys = [e.prompt for e in EXAMPLES if e.source == "lmsys-chat-1m"]

    def stream(limit: int) -> Iterator[str]:
        yield from ["unrelated", *lmsys][: limit or None]

    rebuilt, recovered, missing = rebuild_dataset.rebuild(frame, 0, {"lmsys-chat-1m": stream})
    assert (recovered, missing) == (4, 1)  # the unreviewed source has no rebuild path
    by_id = rebuilt.set_index("query_id")["prompt"]
    for e in EXAMPLES:
        if e.source != "new-source":
            assert by_id[e.query_id] == e.prompt

    _, recovered, _ = rebuild_dataset.rebuild(frame, 2, {"lmsys-chat-1m": stream})
    assert recovered == 1  # --limit 2 scans only "unrelated" and the first prompt


@pytest.mark.parametrize(
    "conversation",
    [
        [{"role": "user", "content": "  hi there \n"}, {"role": "assistant", "content": "x"}],
        [{"role": "assistant", "content": "x"}, {"role": "user", "content": "second"}],
        [{"role": "user", "content": ""}],
        [],
        None,
    ],
)
def test_rebuild_extracts_text_exactly_like_the_seed_loader(conversation: Any) -> None:
    assert rebuild_dataset.first_user_turn(conversation) == prompts.first_user_turn(conversation)


def test_rebuilt_hash_matches_the_seed_query_id(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = "  What is the capital of France?\n"
    monkeypatch.setattr(prompts, "_load_lmsys", lambda: iter([("lmsys-chat-1m", raw, "chat")]))
    (record,) = prompts.load_seed_prompts(limit=1, sources=["lmsys-chat-1m"])
    assert rebuild_dataset.sha256(raw.strip()) == record.query_id


def test_rebuild_script_is_standalone() -> None:
    tree = ast.parse(Path(rebuild_dataset.__file__).read_text(encoding="utf-8"))
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not any(m.split(".")[0] == "tollgate" for m in imported), imported


def test_rebuild_pins_the_reviewed_revision() -> None:
    source = Path(rebuild_dataset.__file__).read_text(encoding="utf-8")
    assert prompts.SOURCE_REVISIONS["lmsys/lmsys-chat-1m"] in source


# --- CLI ----------------------------------------------------------------------------------------


def _cli_setup() -> None:
    write_dataset(EXAMPLES)
    paths.docs_dir().mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CARD, paths.docs_dir() / publish.CARD_NAME)


def test_push_dataset_without_yes_only_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_setup()
    from tollgate import hub

    monkeypatch.setattr(hub, "push_dataset_files", lambda *a, **k: pytest.fail("uploaded"))
    result = CliRunner().invoke(app, ["push-dataset"])
    assert result.exit_code == 0, result.output
    assert "dry run: nothing uploaded" in result.output
    assert "lmsys-chat-1m    4 rows: 0 with text, 4 as sha256 + labels only" in result.output
    assert (paths.reports_dir() / "dataset_publish" / "README.md").exists()


def test_push_dataset_with_yes_uploads_privately(monkeypatch: pytest.MonkeyPatch) -> None:
    _cli_setup()
    from tollgate import hub

    calls: list[dict[str, Any]] = []

    def fake_push(files: dict[str, Path], **kwargs: Any) -> str:
        calls.append({"files": sorted(files), **kwargs})
        return "alice/tollgate-routing-data"

    monkeypatch.setattr(hub, "push_dataset_files", fake_push)
    result = CliRunner().invoke(app, ["push-dataset", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls[0]["private"] is True and "README.md" in calls[0]["files"]


# --- restore (the inverse, run wherever the dataset is opened, e.g. Kaggle) ---------------------


def test_restore_rebuilds_withheld_text_into_a_trainable_dataset(tmp_path: Path) -> None:
    frame, _ = publish.public_frame(EXAMPLES)
    public = tmp_path / "public.parquet"
    frame.to_parquet(public, index=False)
    lmsys = [e.prompt for e in EXAMPLES if e.source == "lmsys-chat-1m"]

    def stream(limit: int) -> Iterator[str]:
        yield from lmsys

    out = tmp_path / "restored.parquet"
    report = publish.restore_dataset(public, out, sources={"lmsys-chat-1m": stream})
    assert (report.withheld, report.recovered, report.dropped) == (5, 4, 1)
    assert report.rows == len(EXAMPLES) - 1  # the unreviewed source cannot be rebuilt: dropped
    restored = [r for split in Split for r in read_split(split, out)]
    by_id = {r.query_id: r for r in restored}
    for e in EXAMPLES:
        if e.source != "new-source":
            assert by_id[e.query_id].prompt == e.prompt
            assert by_id[e.query_id].tier is e.tier and by_id[e.query_id].split is e.split


def test_restore_passes_an_unredacted_dataset_through(tmp_path: Path) -> None:
    path = write_dataset(EXAMPLES, tmp_path / "internal.parquet")
    report = publish.restore_dataset(path, tmp_path / "copy.parquet")
    assert (report.rows, report.withheld, report.dropped) == (len(EXAMPLES), 0, 0)
    assert (tmp_path / "copy.parquet").exists()
