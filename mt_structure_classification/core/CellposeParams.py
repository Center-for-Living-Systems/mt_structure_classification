from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mt_structure_classification.core.SegObject import SegObject


@dataclass
class CellposeParams:
    model_type: str = "cyto2"   # or your pretrained model
    diameter: float | None = None
    channels: tuple[int, int] = (0, 0)  # adjust if using multi-channel


_DEFAULT_CELLPOSE_PARAMS = CellposeParams()


def segment_cellpose(
    guv_img: np.ndarray,
    params: CellposeParams | None = None,
) -> list[SegObject]:
    """
    Return instance masks as SegObject list.
    """
    from cellpose import models
    from skimage.measure import regionprops

    if params is None:
        params = _DEFAULT_CELLPOSE_PARAMS

    model = models.Cellpose(model_type=params.model_type)
    masks, _flows, _styles, _diams = model.eval(
        guv_img,
        diameter=params.diameter,
        channels=params.channels,
    )

    objs: list[SegObject] = []
    # masks is HxW labeled (0=background, 1..N instances)
    for rp in regionprops(masks):
        y0, x0, y1, x1 = rp.bbox
        cy, cx = rp.centroid
        inst_mask = masks == rp.label

        # approximate radius from area (optional)
        radius = float(np.sqrt(rp.area / np.pi))

        objs.append(
            SegObject(
                method="cellpose",
                score=None,
                cx=float(cx),
                cy=float(cy),
                radius=radius,
                mask=inst_mask,
                bbox=(y0, x0, y1, x1),
            )
        )
    return objs
