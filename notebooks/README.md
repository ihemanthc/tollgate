# Notebooks

`kaggle_train.ipynb` fine-tunes the Laya router on a Kaggle GPU. It also scores the calibration
and test splits so that calibration and every metric run afterwards on a CPU.

Notebooks here are generated. Edit `scripts/make_notebook.py` and run
`uv run python scripts/make_notebook.py`; never edit an `.ipynb` by hand. A test fails if a
committed notebook is out of date.

## Before you open Kaggle

1. **Push this repo to GitHub.** The notebook installs it with pip from `REPO_URL`. A private
   repo works too: see step 5.
2. **Publish the dataset to the Hub.** On your machine, after `tollgate collect-all`:
   ```
   uv run tollgate push-dataset          # stages files for review
   uv run tollgate push-dataset --yes    # uploads to the private HF_REPO_DATA repo
   ```
3. **Use a Hugging Face token with write access**
   (https://huggingface.co/settings/tokens). The notebook uses it to pull the dataset, push the
   run, and rebuild LMSYS-Chat-1M text. The account behind the token must have accepted the
   [LMSYS-Chat-1M license](https://huggingface.co/datasets/lmsys/lmsys-chat-1m): the dataset on
   the Hub carries only hashes for those rows, so the notebook rebuilds their text from the
   official source under that account's access.

## Kaggle setup

1. **Create the notebook.** kaggle.com > **Code** > **New Notebook**, then **File** >
   **Import Notebook** and upload `notebooks/kaggle_train.ipynb`.
2. **Select the T4 x2 accelerator.** In the right-hand panel, open **Session options** (or the
   **Settings** menu) > **Accelerator** > **GPU T4 x2**. Kaggle restarts the session. The
   notebook stops at its first cell if no GPU is attached, and warns if the GPU is not a T4 or
   P100. Training uses one of the two T4s.
3. **Turn Internet on.** Same panel: **Internet** > **On**. pip, the Hub and LMSYS-Chat-1M all
   need it. Kaggle only offers this toggle on phone-verified accounts (**Settings** >
   **Phone verification** on your Kaggle profile).
4. **Add the `HF_TOKEN` secret.** **Add-ons** > **Secrets** > **Add a new secret**. Label it
   exactly `HF_TOKEN` and paste the token as the value. Then **tick the checkbox next to it** so
   it is attached to this notebook; an unattached secret reads as missing. If it is missing,
   step 3 of the notebook stops with this same instruction.
5. **Private GitHub repo only: add `GITHUB_TOKEN`** the same way. Use a fine-grained token with
   read-only *Contents* access to the one repo. It is passed to git through environment
   variables and never appears in pip's output.
6. **Edit the settings cell.** Set `REPO_URL` (and `REPO_REF` to pin a branch, tag or commit),
   `HF_REPO_DATA` and `HF_REPO_MODEL` (`owner/name`, or a bare name to use your own namespace),
   and `RUN_NAME`. The training settings default to the real run.

## Running it

- **Use Save Version > Save & Run All (Commit) for the real run.** It runs in the background
  and does not depend on your browser staying open. Use **Run All** in the interactive editor
  only for a short check.
- **Time.** Kaggle ends a session at 12 hours. Training stops cleanly after `BUDGET_HOURS`
  (10 h), which leaves time to score the splits and upload. GPU hours also count against your
  weekly quota, shown in the session panel.
- **If training runs out of budget,** the notebook still scores and pushes the best checkpoint
  so far, plus `last/` (weights with optimizer state, about 5 GB). Set `RESUME = True` and run
  it again: it pulls `last/` and continues exactly where it stopped.
- **Disk.** `/kaggle/working` holds about 20 GB. A run uses about 12 GB at peak while saving
  `last/` next to `best/`.

## What it uploads

Everything goes to the model repo `HF_REPO_MODEL`, under `runs/<RUN_NAME>/`, as one commit:

| Path | Contents |
|---|---|
| `best/` | The best checkpoint by holdout loss, loadable with `laya.load`; temperatures reset to 1.0 |
| `last/` | Only if the time budget ran out: the resumable state |
| `logits_calibration.npz`, `logits_test.npz` | fp32 raw logits per question, labels, query ids, and the checkpoint's config hash. No query text |
| `run_metadata.json` | Base model and revision, installed tollgate commit, hyperparameters, seed, GPU, library versions, dataset revision, durations |
| `history.json` | Train and holdout loss curves |

The last cell prints every file with its size and URL.

## Back on your machine (CPU only)

```
uv run tollgate pull-run --run laya
uv run tollgate calibrate --checkpoint checkpoints/laya/best --logits checkpoints/laya/logits_calibration.npz
```

`calibrate --logits` fits the temperatures from the exported logits without loading the model.
It refuses logits that came from a different checkpoint, rows that are not from the calibration
split, and rows the checkpoint trained on.
