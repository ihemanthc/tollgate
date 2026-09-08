"""The one place a torch device is chosen. Code must never assume CUDA."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEVICES = ("auto", "cuda", "mps", "cpu")


def _mps_available() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return backend is not None and backend.is_available()


def resolve(requested: str = "auto") -> torch.device:
    """cuda | mps | cpu. 'auto' takes the best available; an unavailable explicit choice raises."""
    import torch

    choice = requested.strip().lower()
    if choice not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if _mps_available() else torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but CUDA is not available")
    if choice == "mps" and not _mps_available():
        raise ValueError("device 'mps' requested but MPS is not available")
    return torch.device(choice)
