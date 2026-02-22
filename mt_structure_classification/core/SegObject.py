from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SegMethod = Literal["cellpose", "circles", "combined"]

@dataclass
class SegObject:
    # common identity
    method: str  # "cellpose" or "circles"
    score: float | None  # optional (circle: can be None)
    # geometry
    cx: float
    cy: float
    radius: float | None  # circle has it; cellpose may not
    # masks
    mask: np.ndarray | None  # HxW bool or 0/1; optional for circle-only pipeline
    bbox: tuple[int, int, int, int]  # (y0, x0, y1, x1)
