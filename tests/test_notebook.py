"""notebooks/kaggle_train.ipynb: generated, valid, plain Python, and in step with the package."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType

import nbformat
import pytest

from tollgate import hub
from tollgate.collect import publish
from tollgate.train import infer, kaggle, train

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "kaggle_train.ipynb"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_notebook", REPO / "scripts/make_notebook.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nb() -> nbformat.NotebookNode:
    raw = NOTEBOOK.read_text(encoding="utf-8")
    json.loads(raw)
    return nbformat.reads(raw, as_version=4)


def _code(nb: nbformat.NotebookNode) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_committed_notebook_is_exactly_the_generated_one() -> None:
    assert NOTEBOOK.read_text(encoding="utf-8") == _generator().render(NOTEBOOK.name)


def test_notebook_passes_nbformat_validation(nb: nbformat.NotebookNode) -> None:
    nbformat.validate(nb)
    assert (nb.nbformat, nb.nbformat_minor) == (4, 5)
    assert len({c.id for c in nb.cells}) == len(nb.cells)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def test_every_code_cell_is_plain_python(nb: nbformat.NotebookNode) -> None:
    for i, source in enumerate(_code(nb)):
        compile(source, f"cell-{i}", "exec")
        assert not any(line.lstrip().startswith(("!", "%")) for line in source.splitlines())


def test_steps_run_in_the_required_order(nb: nbformat.NotebookNode) -> None:
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


def test_notebook_calls_match_package_signatures(nb: nbformat.NotebookNode) -> None:
    checked = 0
    for source in _code(nb):
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in TARGETS:
                target = TARGETS[func.id]
            elif (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in MODULES
            ):
                target = getattr(MODULES[func.value.id], func.attr)  # AttributeError = drift
            else:
                continue
            params = inspect.signature(target).parameters
            for kw in node.keywords:
                assert kw.arg in params, f"{getattr(func, 'id', None) or func.attr}({kw.arg}=)"
            checked += 1
    assert checked >= 12


def test_train_call_uses_fp16_cuda_budget_and_progress(nb: nbformat.NotebookNode) -> None:
    train_cell = next(s for s in _code(nb) if "train_laya(" in s)
    for needle in ('device="cuda"', "amp=True", "max_hours=BUDGET_HOURS", "progress=True"):
        assert needle in train_cell


def test_generator_check_mode_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = _generator()
    monkeypatch.setattr("sys.argv", ["make_notebook.py", "--check"])
    gen.main()  # exits non-zero if stale
