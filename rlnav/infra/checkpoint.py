from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(path, map_location=map_location)
