"""`tollgate push-dataset`: publish the labelled dataset without redistributing restricted text.

Each seed source's terms were reviewed at the Hub revision prompts.py reads (SOURCE_REVISIONS).
Sources whose license allows redistribution ship their query text. The rest ship only
`text_sha256` (sha256 of the exact query text, equal to `query_id`) plus Tollgate's own labels,
and rebuild_dataset.py restores the text for anyone who has accepted that source's license.
A source missing from SOURCE_TERMS is withheld: this fails closed.

docs/DATASET_CARD.md is prose plus generated blocks. Static blocks (licenses, columns) render
from the registries below; data blocks (contents, models, labels, splits) render from the data at
push time, so no number in the card is ever typed by hand.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from pydantic import BaseModel

from tollgate import paths
from tollgate.collect import rebuild_dataset
from tollgate.collect.build_dataset import read_split
from tollgate.collect.judge import read_verdicts
from tollgate.collect.prompts import SOURCE_REVISIONS
from tollgate.collect.runner import read_runs
from tollgate.config import check_judge_independent, hf_repo_data
from tollgate.schema import TIER_ORDER, LabeledExample, Split, Tier

CARD_NAME = "DATASET_CARD.md"
SPLIT_INDEX_DIR = "splits"
REBUILD_SCRIPT = "rebuild_dataset.py"
TEXT_COLUMNS = ("prompt", "conversation_summary")  # the only columns holding source text
_BLOCK = re.compile(r"(<!-- tollgate:begin (\w+) -->\n)(.*?)(<!-- tollgate:end \2 -->)", re.S)
PENDING = "_Generated from the data by `tollgate push-dataset`; nothing has been published yet._\n"


class SourceTerms(BaseModel):
    source: str
    dataset: str
    license: str
    license_url: str
    copyright: str | None
    redistribute_text: bool
    basis: str
    citation: str


SOURCE_TERMS: dict[str, SourceTerms] = {
    "lmsys-chat-1m": SourceTerms(
        source="lmsys-chat-1m",
        dataset="lmsys/lmsys-chat-1m",
        license="LMSYS-Chat-1M Dataset License Agreement",
        license_url=(
            "https://huggingface.co/datasets/lmsys/lmsys-chat-1m"
            "#lmsys-chat-1m-dataset-license-agreement"
        ),
        copyright=None,
        redistribute_text=False,
        basis=(
            '"Prohibited Transfers: You should not distribute, copy, disclose, assign, '
            'sublicense, embed, host, or otherwise transfer the dataset to any third party."'
        ),
        citation="Zheng et al., LMSYS-Chat-1M (2023), arXiv:2309.11998",
    ),
    "gsm8k": SourceTerms(
        source="gsm8k",
        dataset="openai/gsm8k",
        license="MIT",
        license_url="https://github.com/openai/grade-school-math/blob/master/LICENSE",
        copyright="Copyright (c) 2021 OpenAI",
        redistribute_text=True,
        basis="MIT permits redistribution with the copyright and permission notice (below).",
        citation="Cobbe et al., Training Verifiers to Solve Math Word Problems (2021), "
        "arXiv:2110.14168",
    ),
    "mmlu": SourceTerms(
        source="mmlu",
        dataset="cais/mmlu",
        license="MIT",
        license_url="https://github.com/hendrycks/test/blob/master/LICENSE",
        copyright="Copyright (c) 2020 Dan Hendrycks",
        redistribute_text=True,
        basis="MIT permits redistribution with the copyright and permission notice (below).",
        citation="Hendrycks et al., Measuring Massive Multitask Language Understanding (2021), "
        "arXiv:2009.03300",
    ),
}

# Every published column and what it holds. A test keeps this in step with the frame.
COLUMNS: dict[str, str] = {
    "query_id": "sha256 of the exact query text (UTF-8); the row key.",
    "text_sha256": "Same value as query_id, named for what it is: how withheld text is matched.",
    "text_included": "True when `prompt` carries the text; False when the source forbids it.",
    "prompt": "Query text; empty (null) for rows from a source that forbids redistribution.",
    "conversation_summary": "Prior-turn summary; empty for withheld sources and single-turn rows.",
    "source": "Seed source tag (see Seed sources).",
    "split": "train / calibration / test.",
    "tier": "Label: minimum sufficient tier (local_small, mid_tier or frontier).",
    "needs_tools": "Heuristic flag from the frontier answer's text; placeholder, not judged.",
    "needs_rag": "Heuristic flag from the frontier answer's text; placeholder, not judged.",
    "judged_tiers": "Cheaper tiers that received a judge verdict for this query.",
    "frontier_cost_usd": "What the frontier tier's answer cost.",
    "label_cost_usd": "What the answer at the label tier cost.",
}

MIT_PERMISSION_NOTICE = """\
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


def text_published(source: str) -> bool:
    terms = SOURCE_TERMS.get(source)
    return terms is not None and terms.redistribute_text


class SourceInclusion(BaseModel):
    source: str
    rows: int
    text_rows: int
    withheld_rows: int


def public_frame(examples: Sequence[LabeledExample]) -> tuple[pd.DataFrame, list[SourceInclusion]]:
    """The publishable table: source text removed for every row whose license forbids sharing."""
    rows = []
    for example in examples:
        if hashlib.sha256(example.prompt.encode("utf-8")).hexdigest() != example.query_id:
            raise ValueError(
                f"{example.query_id}: query_id is not sha256(prompt), so withheld text could "
                "not be rebuilt from its hash; refusing to publish"
            )
        include = text_published(example.source)
        row = example.model_dump(mode="json")
        row |= {"text_sha256": example.query_id, "text_included": include}
        if not include:
            row |= dict.fromkeys(TEXT_COLUMNS)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    counts = Counter((e.source, text_published(e.source)) for e in examples)
    inclusion = [
        SourceInclusion(
            source=source,
            rows=counts[(source, True)] + counts[(source, False)],
            text_rows=counts[(source, True)],
            withheld_rows=counts[(source, False)],
        )
        for source in sorted({e.source for e in examples})
    ]
    return frame, inclusion


def split_index(frame: pd.DataFrame) -> dict[Split, list[str]]:
    return {s: sorted(frame.loc[frame["split"] == s.value, "query_id"]) for s in Split}


# --- card ---------------------------------------------------------------------------------------


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def static_blocks() -> dict[str, str]:
    """Blocks rendered from code registries only (no data): licenses and columns."""
    licenses = _table(
        ["Source", "Hub dataset @ revision", "License", "Query text published?", "Basis"],
        [
            [
                f"`{t.source}`",
                f"[`{t.dataset}`](https://huggingface.co/datasets/{t.dataset}) @ "
                f"`{SOURCE_REVISIONS[t.dataset][:12]}`",
                f"[{t.license}]({t.license_url})",
                "**yes**" if t.redistribute_text else "**no**: sha256 + labels only",
                t.basis,
            ]
            for t in SOURCE_TERMS.values()
        ],
    )
    notices = [t.copyright for t in SOURCE_TERMS.values() if t.copyright]
    licenses += (
        "\nAny source not listed here is treated as non-redistributable: its text is withheld."
        "\n\nMIT notice for the GSM8K and MMLU text included in `prompt`:\n\n```\n"
        + "\n".join(notices)
        + "\n\n"
        + MIT_PERMISSION_NOTICE
        + "```\n\nCite the sources you use:\n\n"
        + "".join(f"- {t.citation}\n" for t in SOURCE_TERMS.values())
    )
    columns = _table(["Column", "Contents"], [[f"`{k}`", v] for k, v in COLUMNS.items()])
    return {"licenses": licenses, "columns": columns}


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "-"


def data_blocks(
    frame: pd.DataFrame,
    inclusion: Sequence[SourceInclusion],
    tier_models: Mapping[str, Sequence[str]],
    judge_models: Sequence[str],
) -> dict[str, str]:
    total = len(frame)
    tiers = [t.value for t in TIER_ORDER]
    contents = _table(
        ["Source", "Rows", "Rows with text", "Rows as sha256 + labels only"],
        [[i.source, i.rows, i.text_rows, i.withheld_rows] for i in inclusion]
        + [
            [
                "**all**",
                total,
                sum(i.text_rows for i in inclusion),
                sum(i.withheld_rows for i in inclusion),
            ]
        ],
    )
    models = _table(
        ["Role", "Model(s)"],
        [[t, ", ".join(f"`{m}`" for m in tier_models.get(t, [])) or "unknown"] for t in tiers]
        + [["judge", ", ".join(f"`{m}`" for m in judge_models) or "unknown"]],
    )
    counts = frame["tier"].value_counts()
    labels = _table(
        ["Label (minimum sufficient tier)", "Rows", "Share"],
        [[t, int(counts.get(t, 0)), _pct(int(counts.get(t, 0)), total)] for t in tiers],
    )
    labels += "\n" + _table(
        ["Heuristic flag", "Rows true", "Share"],
        [
            [c, int(frame[c].sum()), _pct(int(frame[c].sum()), total)]
            for c in ("needs_tools", "needs_rag")
        ],
    )
    by_source = pd.crosstab(frame["source"], frame["tier"]).reindex(columns=tiers, fill_value=0)
    labels += "\nLabel by source:\n\n" + _table(
        ["Source", *tiers, "total"],
        [[s, *map(int, row), int(row.sum())] for s, row in by_source.iterrows()],
    )
    by_split = pd.crosstab(frame["split"], frame["tier"]).reindex(
        index=[s.value for s in Split], columns=tiers, fill_value=0
    )
    splits = _table(
        ["Split", *tiers, "total", "Share of rows"],
        [
            [s, *map(int, row), int(row.sum()), _pct(int(row.sum()), total)]
            for s, row in by_split.iterrows()
        ],
    )
    generated = (
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by tollgate "
        f"{metadata.version('tollgate')} from {total} labelled rows.\n"
    )
    return {
        "contents": contents,
        "models": models,
        "labels": labels,
        "splits": splits,
        "generated": generated,
    }


def render_card(template: str, blocks: Mapping[str, str]) -> str:
    """Replace the body of every `<!-- tollgate:begin NAME -->` block named in `blocks`."""
    present = {m.group(2) for m in _BLOCK.finditer(template)}
    missing = sorted(set(blocks) - present)
    if missing:
        raise ValueError(f"card template has no blocks named {missing}")

    def fill(match: re.Match[str]) -> str:
        name = match.group(2)
        body = blocks.get(name, match.group(3))
        return f"{match.group(1)}{body}{match.group(4)}"

    return _BLOCK.sub(fill, template)


# --- push ---------------------------------------------------------------------------------------


def _models_for(
    ids: set[str], runs_path: Path, verdicts_path: Path
) -> tuple[dict[str, list[str]], list[str]]:
    tier_models: dict[str, set[str]] = {t.value: set() for t in TIER_ORDER}
    for run in read_runs(runs_path):
        if run.query_id in ids and run.error is None:
            tier_models[run.tier.value].add(run.model)
    judges = {v.judge_model for v in read_verdicts(verdicts_path) if v.query_id in ids}
    return {k: sorted(v) for k, v in tier_models.items()}, sorted(judges)


def stage(
    examples: Sequence[LabeledExample],
    template: str,
    runs_path: Path,
    verdicts_path: Path,
    out: Path,
) -> tuple[dict[str, Path], list[SourceInclusion]]:
    """Write everything that would be uploaded into `out`. Returns {path_in_repo: file}."""
    frame, inclusion = public_frame(examples)
    tier_models, judges = _models_for(set(frame["query_id"]), runs_path, verdicts_path)
    if not judges:
        raise ValueError(f"no verdicts for these rows in {verdicts_path}; cannot name the judge")
    for judge in judges:
        check_judge_independent(judge, tier_models[Tier.FRONTIER.value])
    shutil.rmtree(out, ignore_errors=True)
    (out / SPLIT_INDEX_DIR).mkdir(parents=True)
    files: dict[str, Path] = {"dataset.parquet": out / "dataset.parquet"}
    frame.to_parquet(files["dataset.parquet"], index=False)
    for split, ids in split_index(frame).items():
        name = f"{SPLIT_INDEX_DIR}/{split.value}.txt"
        files[name] = out / name
        files[name].write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    card = render_card(
        template, {**static_blocks(), **data_blocks(frame, inclusion, tier_models, judges)}
    )
    files["README.md"] = out / "README.md"
    files["README.md"].write_text(card, encoding="utf-8")
    if any(i.withheld_rows for i in inclusion):
        files[REBUILD_SCRIPT] = out / REBUILD_SCRIPT
        shutil.copyfile(rebuild_dataset.__file__, files[REBUILD_SCRIPT])
    return files, inclusion


def push_dataset_cmd(
    repo: Annotated[str | None, typer.Option(help="[default: $HF_REPO_DATA]")] = None,
    public: Annotated[bool, typer.Option("--public", help="Create the repo public.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Upload; without it, stage only.")] = False,
    dataset: Annotated[Path | None, typer.Option(help="[default: <data dir>]")] = None,
) -> None:
    """Publish dataset.parquet (restricted text removed), split indexes, card, rebuild script."""
    dataset = dataset or paths.dataset_path()
    template_path = paths.docs_dir() / CARD_NAME
    for path in (dataset, paths.tier_runs_path(), paths.verdicts_path()):
        if not path.exists():
            raise typer.BadParameter(
                f"{path} not found: there is no labelled dataset to publish yet. "
                "Build it first with `tollgate collect-all --limit N` (then --yes)."
            )
    if not template_path.exists():
        raise typer.BadParameter(f"{template_path} not found (the dataset card template).")
    examples = [row for split in Split for row in read_split(split, dataset)]
    out = paths.reports_dir() / "dataset_publish"
    try:
        files, inclusion = stage(
            examples,
            template_path.read_text(encoding="utf-8"),
            paths.tier_runs_path(),
            paths.verdicts_path(),
            out,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(
        f"staged in {out} for {'public' if public else 'private'} repo {repo or hf_repo_data()}:"
    )
    for name, path in files.items():
        typer.echo(f"  {name:<28} {path.stat().st_size:>10,} bytes")
    for i in inclusion:
        typer.echo(
            f"  {i.source:<16} {i.rows} rows: {i.text_rows} with text, "
            f"{i.withheld_rows} as sha256 + labels only"
        )
    if not yes:
        typer.echo("dry run: nothing uploaded. Review the staged files, then re-run with --yes.")
        raise typer.Exit(code=0)
    from tollgate import hub

    repo_id = hub.push_dataset_files(files, repo=repo, private=not public)
    typer.echo(f"pushed {len(files)} files to https://huggingface.co/datasets/{repo_id}")


class RestoreReport(BaseModel):
    rows: int
    withheld: int
    recovered: int
    dropped: int


def restore_dataset(
    public: Path,
    out: Path,
    *,
    limit: int = 0,
    sources: Mapping[str, rebuild_dataset.TextStream] | None = None,
) -> RestoreReport:
    """Turn a published dataset.parquet back into a trainable one, wherever it is opened.

    Withheld text is rebuilt from the original sources with the caller's own access (their
    license, their download), so restricted text never passes through a Tollgate repo. Rows whose
    text cannot be found are dropped and counted. `limit` caps source rows scanned (0 = all).
    """
    frame = pd.read_parquet(public)
    if "text_included" not in frame.columns:  # not redacted: already trainable
        out.parent.mkdir(parents=True, exist_ok=True)
        if public.resolve() != out.resolve():
            shutil.copyfile(public, out)
        return RestoreReport(rows=len(frame), withheld=0, recovered=0, dropped=0)
    withheld = int((~frame["text_included"].astype(bool)).sum())
    rebuilt, recovered, missing = rebuild_dataset.rebuild(
        frame, limit, dict(sources) if sources is not None else rebuild_dataset.SOURCES
    )
    kept = rebuilt[rebuilt["prompt"].notna()].drop(columns=["text_sha256", "text_included"])
    mismatched = [
        q for q, text in zip(kept["query_id"], kept["prompt"], strict=True)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != q
    ]  # fmt: skip
    if mismatched:
        raise ValueError(f"{len(mismatched)} restored prompts do not hash to their query_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out, index=False)
    return RestoreReport(rows=len(kept), withheld=withheld, recovered=recovered, dropped=missing)
