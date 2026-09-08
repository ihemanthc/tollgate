"""tollgate.eval.plots: five 300-dpi PNGs from a results file, none blank, none clipped."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from tollgate.collect.runner import LedgerEntry
from tollgate.eval import evaluate as ev
from tollgate.eval import plots
from tollgate.schema import TIER_ORDER, JudgeVerdict, TierRun

NAMES = ["reliability.png", "risk_coverage.png", "cost_curve.png", "confusion.png", "latency.png"]


def _split(name: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    labels = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    logits = rng.normal(0, 1, size=(n, 3))
    logits[np.arange(n), labels] += rng.uniform(0, 3, size=n)  # informative, not perfect
    return {
        "query_id": np.array([f"{name}-{i}" for i in range(n)]),
        "split": np.array([name] * n),
        "tier": labels,
        "logits_tier": logits,
        "checkpoint_config_sha256": np.array("cfg"),
    }


def _results(calibrated: bool = True, n: int = 300) -> ev.Results:
    rng = np.random.default_rng(0)
    test, cal = _split("test", n, rng), _split("calibration", n // 2, rng)
    runs, verdicts = [], []
    for data in (test, cal):
        for qid, label in zip(data["query_id"], data["tier"], strict=True):
            for rank, tier in enumerate(TIER_ORDER):
                tokens = int(rng.integers(50, 800))
                runs.append(
                    TierRun(
                        query_id=str(qid), tier=tier, model=f"m-{tier.value}", completion="a",
                        prompt_tokens=tokens, completion_tokens=tokens, cost_usd=0.0,
                        latency_s=0.0,
                    )
                )  # fmt: skip
                if rank < 2:
                    verdicts.append(
                        JudgeVerdict(
                            query_id=str(qid), candidate_tier=tier, reason="r", judge_model="j",
                            verdict="acceptable" if rank >= label else "worse",
                        )
                    )  # fmt: skip
    now = datetime.now(UTC)
    ledger = [
        LedgerEntry(
            ts=now, tier=t.value, model=f"m-{t.value}", prompt_tokens=p, completion_tokens=c,
            usd=(p + c) * rate, latency_ms=1.0,
        )
        for t, rate in zip(TIER_ORDER, (0.0, 3e-7, 4e-6), strict=True)
        for p, c in ((100, 50), (30, 200))
    ]  # fmt: skip
    latency = ev.latency_summary(list(rng.lognormal(3.0, 0.4, size=200)), "cpu", 2)
    return ev.build_results(
        run="synthetic", test=test, calibration=cal, temperature=1.0, calibrated=calibrated,
        runs=runs, verdicts=verdicts, ledger=ledger, latency=latency,
    )  # fmt: skip


@pytest.mark.parametrize("calibrated", [True, False])
def test_writes_five_300_dpi_pngs_none_blank(tmp_path: Path, calibrated: bool) -> None:
    written = plots.write_figures(_results(calibrated), tmp_path)
    assert [p.name for p in written] == NAMES
    for path in written:
        with Image.open(path) as im:
            dpi = im.info["dpi"]
            assert round(dpi[0]) == round(dpi[1]) == plots.DPI, path.name
            assert im.size[0] == round(plots.FIGSIZE[0] * plots.DPI), path.name
        pixels = mpimg.imread(path)[..., :3]
        assert (pixels.reshape(-1, 3) < 0.9).any(axis=1).mean() > 0.02, f"{path.name} is blank"
        corners = pixels[[0, 0, -1, -1], [0, -1, 0, -1]]
        assert np.allclose(corners, 1.0), f"{path.name}: background is not white"


def test_regenerates_from_results_json(tmp_path: Path) -> None:
    results = tmp_path / "results.json"
    results.write_text(_results(n=60).model_dump_json(), encoding="utf-8")
    plots.main(results=results, out_dir=tmp_path / "assets")
    assert sorted(p.name for p in (tmp_path / "assets").iterdir()) == sorted(NAMES)


def test_clipped_text_is_refused() -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.text(1.4, 0.5, "far outside", transform=ax.transAxes)
    try:
        with pytest.raises(plots.ClippedFigureError, match="canvas"):
            plots.check_layout(fig)
    finally:
        plt.close(fig)


def test_legend_over_figure_text_is_refused() -> None:
    fig, ax = plt.subplots(figsize=(4, 3), layout="constrained")
    ax.plot([0, 1], [0, 1], label="a line with a long legend label")
    fig.legend(loc="lower center")
    fig.text(0.5, 0.02, "a caption placed under the legend", ha="center")
    try:
        with pytest.raises(plots.ClippedFigureError, match="legend overlaps"):
            plots.check_layout(fig)
    finally:
        plt.close(fig)


def test_palette_is_okabe_ito() -> None:
    okabe_ito = {"#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"}
    used = {plots.BLUE, plots.ORANGE, plots.GREEN, plots.VERMILLION, plots.PURPLE, plots.SKY}
    assert used <= okabe_ito
