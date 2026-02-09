from __future__ import annotations

from typing import Any, Optional
import numpy as np

from cellpose import models


def get_cellpose_model(
    model_type: str = "cyto3",
    gpu: bool = False,
    device: Optional[str] = None,  # cellpose uses gpu flag; device mostly torch-side
) -> models.Cellpose:
    return models.Cellpose(gpu=gpu, model_type=model_type)


def segment_with_cellpose(
    model: models.Cellpose,
    img_2d: np.ndarray,
    diameter: Optional[float] = None,
    channels: tuple[int, int] = (0, 0),
    **kwargs: Any,
) -> np.ndarray:
    """
    Returns label mask (H,W).
    img_2d: 2D float array. You can normalize before calling.
    """
    masks, flows, styles, diams = model.eval(
        img_2d,
        diameter=diameter,
        channels=channels,
        **kwargs,
    )
    return masks.astype(np.int32, copy=False)
