"""notebooks/*.ipynb: generated, valid, plain Python, and in step with the package."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import nbformat
import pytest

from tollgate import hub
from tollgate.collect import kaggle as collect_kaggle
from tollgate.collect import pipeline, publish, runner
from tollgate.train import infer, kaggle, train

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "kaggle_train.ipynb"
COLLECT_NOTEBOOK = REPO / "notebooks" / "kaggle_collect.ipynb"
ALL_NOTEBOOKS = [NOTEBOOK, COLLECT_NOTEBOOK]


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_notebook", REPO / "scripts/make_notebook.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> nbformat.NotebookNode:
    raw = path.read_text(encoding="utf-8")
    json.loads(raw)
    return nbformat.reads(raw, as_version=4)


@pytest.fixture(scope="module")
def nb() -> nbformat.NotebookNode:
    return _read(NOTEBOOK)


@pytest.fixture(scope="module")
def collect_nb() -> nbformat.NotebookNode:
    return _read(COLLECT_NOTEBOOK)


@pytest.fixture(scope="module", params=ALL_NOTEBOOKS, ids=lambda p: p.stem)
def any_nb(request: pytest.FixtureRequest) -> nbformat.NotebookNode:
    return _read(request.param)


def _code(nb: nbformat.NotebookNode) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.stem)
def test_committed_notebook_is_exactly_the_generated_one(path: Path) -> None:
    assert path.read_text(encoding="utf-8") == _generator().render(path.name)


def test_notebook_passes_nbformat_validation(any_nb: nbformat.NotebookNode) -> None:
    nbformat.validate(any_nb)
    assert (any_nb.nbformat, any_nb.nbformat_minor) == (4, 5)
    assert len({c.id for c in any_nb.cells}) == len(any_nb.cells)
    assert all(not c.get("outputs") for c in any_nb.cells if c.cell_type == "code")


def test_every_code_cell_is_plain_python(any_nb: nbformat.NotebookNode) -> None:
    for i, source in enumerate(_code(any_nb)):
        compile(source, f"cell-{i}", "exec")
        assert not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())


def test_train_steps_run_in_the_required_order(nb: nbformat.NotebookNode) -> None:
    headings = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown"][1:]
    assert headings == [
        "## 1. GPU",
        "## 2. Install `tollgate[gpu]`",
        "## 3. Hugging Face token and dataset",
        "## 4. Train",
        "## 5. Raw logits for the calibration and test splits",
        "## 6. Push to the Hub",
        "## 7. What was uploaded, and where",
    ]
    settings, gpu, install, secrets, *_ = _code(nb)
    assert "BUDGET_HOURS = 10.0" in settings and "SAVE_EVERY" in settings
    assert "assert torch.cuda.is_available()" in gpu and '"T4", "P100"' in gpu
    assert "tollgate[gpu] @ git+" in install
    assert 'tk.kaggle_secret("HF_TOKEN")' in secrets


# Calls the notebook makes into the package, by the name it binds them to.
TARGETS = {
    "train_laya": train.train_laya,
    "export_logits": infer.export_logits,
    "load_logits": infer.load_logits,
    "restore_dataset": publish.restore_dataset,
}
MODULES = {"tk": kaggle, "hub": hub}


def _check_calls(
    nb: nbformat.NotebookNode,
    targets: dict[str, object],
    modules: dict[str, ModuleType],
    wrapped: dict[str, object] | None = None,
) -> int:
    """Every keyword a call passes exists on its target. Returns the number of calls checked.

    `wrapped` names command functions passed as the first argument of a runner (ck.call),
    whose keywords belong to the command, not to the runner.
    """
    checked = 0
    for source in _code(nb):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in targets:
                target = targets[func.id]
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in modules
            ):
                target = getattr(modules[func.value.id], func.attr)  # AttributeError = drift
            else:
                continue
            first = node.args[0] if node.args else None
            if wrapped and isinstance(first, ast.Name) and first.id in wrapped:
                target = wrapped[first.id]
            params = inspect.signature(target).parameters
            for kw in node.keywords:
                assert kw.arg in params, f"{getattr(func, 'id', None) or func.attr}({kw.arg}=)"
            checked += 1
    return checked


def test_notebook_calls_match_package_signatures(nb: nbformat.NotebookNode) -> None:
    assert _check_calls(nb, TARGETS, MODULES) >= 12


def test_train_call_uses_fp16_cuda_budget_and_progress(nb: nbformat.NotebookNode) -> None:
    train_cell = next(s for s in _code(nb) if "train_laya(" in s)
    for needle in ('device="cuda"', "amp=True", "max_hours=BUDGET_HOURS", "progress=True"):
        assert needle in train_cell


def test_generator_check_mode_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _generator()
    monkeypatch.setattr("sys.argv", ["make_notebook.py", "--check"])
    gen.main()  # exits non-zero if stale


# --- kaggle_collect.ipynb ------------------------------------------------------------------------


def test_collect_steps_run_in_the_required_order(collect_nb: nbformat.NotebookNode) -> None:
    headings = [c.source.splitlines()[0] for c in collect_nb.cells if c.cell_type == "markdown"]
    assert headings[1:] == [
        "## 1. GPUs",
        "## 2. Install `tollgate[collect]` and Ollama",
        "## 3. Token, models and previous state",
        "## 4. Seed prompts",
        "## 5. Answer every prompt at each tier",
        "## 6. Judge and assemble the dataset",
        "## 7. Publish to the Hub",
        "## 8. What was built",
    ]
    settings, _, install, secrets, *_ = _code(collect_nb)
    assert "LIMIT = 300" in settings and "local_small" in settings
    assert "[collect]" in install and "ck.install_ollama()" in install
    assert 'ck.kaggle_secret("HF_TOKEN")' in secrets


def test_collect_settings_are_valid_roles(collect_nb: nbformat.NotebookNode) -> None:
    namespace: dict[str, Any] = {}
    exec(_code(collect_nb)[0], namespace)  # the settings cell is plain assignments
    env = collect_kaggle.role_env(namespace["MODELS"], namespace["PRICES"])
    for model in namespace["MODELS"].values():
        collect_kaggle.ollama_tag(model)  # every role is served by Ollama, so no API key
    assert env["TOLLGATE_JUDGE_MODEL"] != env["TOLLGATE_FRONTIER_MODEL"]


def test_collect_answers_each_tier_before_judging(collect_nb: nbformat.NotebookNode) -> None:
    source = "\n".join(_code(collect_nb))
    tiers, judge = source.index("run_tiers, limit="), source.index("ck.call(collect_all")
    assert tiers < judge < source.index("ck.call(push_dataset_cmd")
    assert source.count("local_concurrency=PARALLEL") == 2
    # source text with a no-redistribution license never stays in the saved output
    assert "paths.seed_path().unlink" in source


COLLECT_TARGETS: dict[str, object] = {
    "ensure_seed": pipeline.ensure_seed,
    "load_records": runner.load_records,
}
COLLECT_WRAPPED: dict[str, object] = {
    "run_tiers": runner.run_tiers,
    "collect_all": pipeline.collect_all,
    "push_dataset_cmd": publish.push_dataset_cmd,
}


def test_collect_calls_match_package_signatures(collect_nb: nbformat.NotebookNode) -> None:
    checked = _check_calls(
        collect_nb, COLLECT_TARGETS, {"ck": collect_kaggle, "hub": hub}, COLLECT_WRAPPED
    )
    assert checked >= 12
