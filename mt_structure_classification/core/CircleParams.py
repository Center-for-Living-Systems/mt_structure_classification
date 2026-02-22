from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage import filters

from mt_structure_classification.core.SegObject import SegObject


def _percentile_normalize(x, pmin=2, pmax=99.8):
    """Percentile-based normalization (replaces csbdeep.utils.normalize)."""
    lo, hi = np.percentile(x, pmin), np.percentile(x, pmax)
    return (x - lo) / (hi - lo + 1e-20)


@dataclass
class CircleParams:
    sigma: float = 1.0
    # Hough bands: (minDist, param1, param2, minR, maxR, dp)
    bands: tuple[tuple[float, float, float, int, int, float], ...] = (
        (20, 30, 30, 10, 30, 1.1),  # small
        (40, 25, 40, 30, 60, 1.1),  # medium
        (60, 20, 60, 60, 90, 1.1),  # large
    )
    # simple filters
    min_radius: int = 10
    max_radius: int = 120
    allow_border_touch: bool = False


_DEFAULT_CIRCLE_PARAMS = CircleParams()


def _hough_multi_band(img_u8: np.ndarray, params: CircleParams) -> np.ndarray:
    all_circles = []
    for minDist, param1, param2, minR, maxR, dp in params.bands:
        c = cv2.HoughCircles(
            img_u8, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=minDist,
            param1=param1, param2=param2,
            minRadius=minR, maxRadius=maxR
        )
        if c is not None and c.shape[1] > 0:
            all_circles.append(c[0])
    if not all_circles:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(all_circles, axis=0).astype(np.float32)


def segment_circles(
    guv_img: np.ndarray,
    mt_img: np.ndarray | None = None,
    params: CircleParams | None = None,
    mt_bg_int: float | None = None,
) -> list[SegObject]:
    """
    Returns circles as SegObject list. If mt_img is provided, you can later add
    MT-based filtering here (std/intensity filters like your notebook).
    """
    if params is None:
        params = _DEFAULT_CIRCLE_PARAMS

    # smooth + normalize
    guv_smooth = filters.gaussian(guv_img, params.sigma)
    guv_norm = _percentile_normalize(guv_smooth, 1, 99)
    guv_u8 = cv2.normalize(guv_norm, None, 0, 255, cv2.NORM_MINMAX)
    guv_u8 = cv2.GaussianBlur(guv_u8, (5, 5), sigmaX=1)
    guv_u8 = np.uint8(guv_u8)

    circles = _hough_multi_band(guv_u8, params)

    H, W = guv_img.shape[-2], guv_img.shape[-1]
    out: list[SegObject] = []
    for x, y, r in circles:
        r_i = float(r)
        if r_i < params.min_radius or r_i > params.max_radius:
            continue

        x0 = int(np.floor(x - r_i))
        x1 = int(np.ceil(x + r_i))
        y0 = int(np.floor(y - r_i))
        y1 = int(np.ceil(y + r_i))

        # boundary rule
        if not params.allow_border_touch and (x0 < 0 or y0 < 0 or x1 >= W or y1 >= H):
            continue

        # circle mask (optional but very useful downstream)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= (r_i - 1) ** 2

        out.append(
            SegObject(
                method="circles",
                score=None,
                cx=float(x),
                cy=float(y),
                radius=r_i,
                mask=mask,
                bbox=(max(0, y0), max(0, x0), min(H, y1), min(W, x1)),
            )
        )
    return out
