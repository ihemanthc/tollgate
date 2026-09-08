"""The five README figures, drawn only from reports/results.json (written by tollgate evaluate).

    uv run python -m tollgate.eval.plots            # what `make figures` runs

reliability.png    reliability diagram with per-bin counts, ECE in the title
risk_coverage.png  risk vs coverage, the selected operating point marked
cost_curve.png     quality retained vs USD per 1k queries: tollgate, always-frontier, random router
confusion.png      confusion matrix over the three tiers, normalized per true tier
latency.png        p50 / p95 / p99 router overhead, log x-axis

One style for all five: Okabe-Ito colours (colourblind-safe), white background, no seaborn,
sized to stay legible when shown 800 px wide, saved at 300 dpi. Every figure is checked before
it is written -- nothing outside the canvas, no legend over text or data -- so a clipped or
covered label fails the run instead of reaching the README. A results file from an
uncalibrated smoke run says so on every figure.

No --limit: this reads one JSON file and processes no rows.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")  # files only; never needs a display

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tollgate import paths
from tollgate.eval.evaluate import RESULTS_FILE, Results, read_results
from tollgate.schema import TIER_ORDER

DPI = 300
FIGSIZE = (6.4, 4.4)  # inches: 1920 px wide at 300 dpi, text still legible scaled to 800 px

# Okabe & Ito (2008), "Color Universal Design".
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#7F7F7F",
)
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "lines.linewidth": 2.0,
}
TIER_LABELS = [t.value.replace("_", " ") for t in TIER_ORDER]
MAX_MARKED_POINTS = 30  # beyond this, per-threshold markers smear into a thick line


class ClippedFigureError(RuntimeError):
    """Some text of a figure lies outside its canvas."""


SMOKE_WARNING = "smoke run: UNCALIBRATED temperatures, not a result"


def _set_subtitle(ax: Axes, results: Results, text: str) -> None:
    """The line under the figure title; an uncalibrated run adds a red warning line to it."""
    if results.calibrated:
        ax.set_title(text)
    else:
        ax.set_title(f"{text}\n{SMOKE_WARNING}", color=VERMILLION)


def _figure(title: str, results: Results, **kwargs: object) -> tuple[Figure, object]:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained", **kwargs)  # type: ignore[arg-type]
    fig.suptitle(title, fontweight="bold")
    return fig, ax


def _subtitle(results: Results) -> str:
    return f"test split, n = {results.n_test}"


def check_layout(fig: Figure, tolerance_px: float = 1.0) -> None:
    """Raise if anything drawn leaves the canvas, or a figure legend covers text or a plot.

    The canvas test uses the figure's tight bounding box, which counts only what is actually
    drawn: a tick label beyond the axis limits exists as an object but is never rendered.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    drawn = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)  # inches -> pixels
    width, height = fig.bbox.width, fig.bbox.height
    if (
        drawn.x0 < -tolerance_px
        or drawn.y0 < -tolerance_px
        or drawn.x1 > width + tolerance_px
        or drawn.y1 > height + tolerance_px
    ):
        raise ClippedFigureError(
            f"drawn content spans ({drawn.x0:.0f}, {drawn.y0:.0f})-({drawn.x1:.0f}, "
            f"{drawn.y1:.0f}) px on a {width:.0f}x{height:.0f} px canvas"
        )
    candidates = [*fig.texts, fig._suptitle, fig._supxlabel]
    texts = list({id(t): t for t in candidates if t is not None}.values())
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    others = [(f"text {t.get_text()!r}", t.get_window_extent(renderer)) for t in texts]
    others += [(f"axes {i}", ax.get_tightbbox(renderer)) for i, ax in enumerate(fig.axes)]
    for legend in fig.legends:
        box = legend.get_window_extent(renderer).padded(-tolerance_px)
        hits = [name for name, other in others if other is not None and box.overlaps(other)]
        if hits:
            raise ClippedFigureError(f"figure legend overlaps {hits}")


def reliability(results: Results) -> Figure:
    ece = results.classification.ece
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=FIGSIZE, layout="constrained", sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
    )  # fmt: skip
    fig.suptitle(f"Reliability of the tier prediction (ECE {ece:.3f})", fontweight="bold")
    filled = [b for b in results.reliability if b.count]
    width = results.reliability[0].hi - results.reliability[0].lo
    top.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1.2, label="perfect calibration")
    top.bar(
        [(b.lo + b.hi) / 2 for b in filled],
        [b.accuracy for b in filled],
        width=width * 0.9,
        color=BLUE,
        alpha=0.85,
        label="accuracy in bin",
    )
    top.plot(
        [b.mean_confidence for b in filled],
        [b.accuracy for b in filled],
        "o",
        color=ORANGE,
        markeredgecolor="black",
        markeredgewidth=0.6,
        label="mean confidence",
    )
    top.set(ylim=(0, 1.05), ylabel="accuracy")
    _set_subtitle(top, results, _subtitle(results))
    top.legend(loc="upper left", frameon=False)

    counts = [b.count for b in filled]
    bars = bottom.bar([(b.lo + b.hi) / 2 for b in filled], counts, width=width * 0.9, color=SKY)
    bottom.bar_label(bars, labels=[str(c) for c in counts], padding=2, fontsize=9)
    bottom.set(
        xlim=(0, 1),
        ylim=(0, max(counts) * 1.35),
        xlabel="confidence (probability of the predicted tier)",
        ylabel="queries",
    )
    return fig


def risk_coverage(results: Results) -> Figure:
    fig, ax = _figure("Selective risk: abstaining on low confidence", results)
    pts = results.risk_coverage
    coverage = [0.0, *[p.coverage for p in pts]]
    risk = [pts[0].risk, *[p.risk for p in pts]]
    ax.step(coverage, risk, where="post", color=BLUE, label="tollgate")
    ax.fill_between(coverage, risk, step="post", color=BLUE, alpha=0.12)
    op = results.operating_point
    if op.risk is not None:
        ax.plot(
            op.coverage, op.risk, marker="*", markersize=16, color=VERMILLION,
            markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
            label=f"operating point (τ = {op.tau:.2f})", clip_on=False, zorder=5,
        )  # fmt: skip
    top = max(0.1, max(risk) * 1.2)
    ax.set(
        xlim=(-0.02, 1.02),
        ylim=(-0.03 * top, top),
        xlabel="coverage (share served at the predicted tier)",
        ylabel="risk (error rate among covered)",
    )
    _set_subtitle(ax, results, f"{_subtitle(results)}, AURC {results.risk_coverage_auc:.3f}")
    ax.legend(loc="upper left", frameon=False)
    return fig


def cost_curve(results: Results) -> Figure:
    fig, ax = _figure("Cost vs. quality: what routing saves", results)
    curves = results.cost_curves
    for points, label, color, style in (
        (curves.random_router, "random router (same tier mix)", ORANGE, "--"),
        (curves.tollgate, "tollgate", BLUE, "-"),
    ):
        ordered = sorted(points, key=lambda p: (p.quality_retained, p.usd_per_1k))
        ax.plot(
            [p.quality_retained for p in ordered],
            [p.usd_per_1k for p in ordered],
            linestyle=style,
            marker="o" if len(ordered) <= MAX_MARKED_POINTS else None,
            markersize=4,
            color=color,
            label=label,
        )
    frontier = curves.always_frontier
    ax.axhline(frontier.usd_per_1k, color=VERMILLION, linestyle=":", label="always frontier")
    ax.plot(
        frontier.quality_retained, frontier.usd_per_1k, "s", color=VERMILLION, markersize=7,
        clip_on=False, zorder=4,
    )  # fmt: skip
    op = results.operating_point.test
    ax.plot(
        op.quality_retained, op.total_usd / op.n * 1000, marker="*", markersize=16,
        color=VERMILLION, markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
        label=f"operating point (τ = {op.tau:.2f})", clip_on=False, zorder=5,
    )  # fmt: skip
    top = max(frontier.usd_per_1k, *[p.usd_per_1k for p in curves.tollgate])
    lowest_q = min(p.quality_retained for p in [*curves.tollgate, *curves.random_router])
    left = max(0.0, lowest_q - 0.05)
    ax.set(
        xlim=(left, 1.0 + 0.02 * (1.0 - left)),  # quality cannot exceed 1
        ylim=(-0.03 * top, top * 1.1),
        xlabel="quality retained (share of answers judged sufficient)",
        ylabel="USD per 1k queries",
    )
    _set_subtitle(ax, results, _subtitle(results))
    # Below the axes: the curves and the frontier line can sit anywhere inside them.
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    return fig


def confusion(results: Results) -> Figure:
    fig, ax = _figure("Confusion matrix (rows sum to 1)", results)
    counts = np.asarray(results.confusion, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(totals > 0, counts / totals, np.nan)
    # A true tier with no test rows has no share to show: light grey, not the colour of 0.
    cmap = matplotlib.colormaps["cividis"].with_extremes(bad="#EEEEEE")
    image = ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, vmin=0, vmax=1)
    for i in range(len(TIER_ORDER)):
        for j in range(len(TIER_ORDER)):
            if totals[i, 0] == 0:
                text, color = "—", "black"
            else:
                text = f"{norm[i, j]:.2f}\n({int(counts[i, j])})"
                color = "black" if norm[i, j] > 0.6 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    ax.set(
        xticks=range(len(TIER_ORDER)),
        yticks=range(len(TIER_ORDER)),
        xticklabels=TIER_LABELS,
        yticklabels=[
            f"{t}\n(n = {int(n)})" for t, n in zip(TIER_LABELS, totals[:, 0], strict=True)
        ],
        xlabel="predicted tier",
        ylabel="true tier (minimum sufficient)",
    )
    _set_subtitle(ax, results, _subtitle(results))
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="share of true tier")
    return fig


def latency(results: Results) -> Figure:
    lat = results.latency
    fig, ax = _figure("Router overhead per query", results)
    names, values = ["p50", "p95", "p99"], [lat.p50_ms, lat.p95_ms, lat.p99_ms]
    bars = ax.barh(names, values, color=[GREEN, ORANGE, VERMILLION], height=0.6)
    ax.bar_label(bars, labels=[f"{v:,.0f} ms" for v in values], padding=4)
    ax.set_xscale("log")
    lo = min([*values, *lat.samples_ms])
    ax.set(
        xlim=(lo / 2, max(values) * 4),
        xlabel="milliseconds (log scale)",
    )
    _set_subtitle(
        ax,
        results,
        f"{len(lat.samples_ms)} single-query calls on {lat.device} after {lat.warmup} warm-up",
    )
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    return fig


FIGURES: dict[str, Callable[[Results], Figure]] = {
    "reliability.png": reliability,
    "risk_coverage.png": risk_coverage,
    "cost_curve.png": cost_curve,
    "confusion.png": confusion,
    "latency.png": latency,
}


def write_figures(results: Results, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(STYLE):
        for name, draw in FIGURES.items():
            fig = draw(results)
            try:
                check_layout(fig)
                path = out_dir / name
                fig.savefig(path, dpi=DPI)
            finally:
                plt.close(fig)
            written.append(path)
    return written


def main(
    results: Annotated[
        Path | None, typer.Option(help=f"[default: <out dir>/reports/{RESULTS_FILE}]")
    ] = None,
    out_dir: Annotated[Path | None, typer.Option(help="[default: docs/assets]")] = None,
) -> None:
    """Regenerate every figure from results.json."""
    loaded = read_results(results)
    for path in write_figures(loaded, out_dir or paths.assets_dir()):
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    typer.run(main)
