"""`tollgate` console entry point."""

from __future__ import annotations

from typing import Annotated

import typer

from tollgate.collect.build_dataset import build_dataset
from tollgate.collect.pipeline import collect_all, judge_cmd
from tollgate.collect.publish import push_dataset_cmd
from tollgate.collect.runner import run_tiers

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Tollgate: a calibrated LLM cost router. Reads .env; shell variables win."""
    from tollgate.config import load_env

    load_env()


app.command("run-tiers")(run_tiers)
app.command("judge")(judge_cmd)
app.command("build-dataset")(build_dataset)
app.command("collect-all")(collect_all)
app.command("push-dataset")(push_dataset_cmd)


@app.command("pull-run")
def pull_run_cmd(
    run: Annotated[str, typer.Option(help="Run name under runs/ in HF_REPO_MODEL.")] = "laya",
    include_last: Annotated[bool, typer.Option(help="Also fetch last/ (~5 GB).")] = False,
) -> None:
    """Fetch a GPU run (best/, logits_*.npz, run_metadata.json) for CPU-side calibration."""
    from tollgate import hub

    dest = hub.pull_run(run, include_last=include_last)
    typer.echo(f"{dest}: " + ", ".join(sorted(p.name for p in dest.iterdir())))


try:  # torch + laya come with the [gpu] or [serve] extra
    from tollgate.eval.evaluate import evaluate
    from tollgate.train.calibrate import calibrate
    from tollgate.train.infer import export_logits_cmd
    from tollgate.train.train import train_laya
except ImportError:
    pass
else:
    app.command("train")(train_laya)
    app.command("export-logits")(export_logits_cmd)
    app.command("calibrate")(calibrate)
    app.command("evaluate")(evaluate)
