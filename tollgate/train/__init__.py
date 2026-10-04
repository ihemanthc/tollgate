"""Laya fine-tuning, calibration, and the calibrated-model loader used at inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laya.agent import Agent

CALIBRATION_FILE = "calibration.json"
FITTED_ON = "calibration"
_TEMPERATURE_KEYS = ("temperature", "temperature_by_options")


def config_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    """A checkpoint config minus the temperatures; what calibration.json records as `config`."""
    return {str(k): v for k, v in cfg.items() if k not in _TEMPERATURE_KEYS}


def load_calibrated(checkpoint: str | Path, device: str | None = None) -> Agent:
    """Load a fine-tuned checkpoint with its calibration.json temperatures applied.

    Refuses, rather than warns, when there is no calibration.json, when it was not fitted on the
    calibration split, or when it was fitted for a different checkpoint: uncalibrated confidences
    served as calibrated ones are the failure Tollgate exists to prevent.
    """
    import laya

    path = Path(checkpoint).resolve()
    calibration = path / CALIBRATION_FILE
    if not calibration.exists():
        raise FileNotFoundError(
            f"{calibration} not found; run `tollgate calibrate --checkpoint {checkpoint}` first."
        )
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    fitted_on = (payload.get("tollgate") or {}).get("fitted_on")
    if fitted_on != FITTED_ON:
        raise ValueError(f"{calibration} was fitted on {fitted_on!r}, not the calibration split.")
    cfg = json.loads((path / "rl_agent_config.json").read_text(encoding="utf-8"))
    if payload.get("config") != config_identity(cfg):
        raise ValueError(f"{calibration} was fitted for a different checkpoint than {path}.")
    return laya.load(str(path), device=device, calibration=str(calibration))
