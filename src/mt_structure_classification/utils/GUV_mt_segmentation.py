# src/mt_structure_classification/utils/GUV_mt_segmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

# circle detection / image ops
import cv2
from scipy.spatial.distance import cdist
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage.filters.rank import median as median_rank

# cellpose
from cellpose import models

# saving
from PIL import Image
import tifffile


SegMethod = Literal["cellpose", "circle", "combined"]


# ============================================================
#                         CELLPOSE
# ============================================================
def get_cellpose_model(
    model_type: str = "cyto3",
    gpu: bool = False,
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
    Returns label mask (H,W) with integer labels 0..K.
    img_2d: 2D float array.
    """
    masks, flows, styles, diams = model.eval(
        img_2d,
        diameter=diameter,
        channels=channels,
        **kwargs,
    )
    return masks.astype(np.int32, copy=False)


def segment_guv_cellpose(
    guv_img: np.ndarray,
    model_type: str = "cyto3",
    gpu: bool = True,
    diameter: Optional[float] = None,
    channels: tuple[int, int] = (0, 0),
    **kwargs: Any,
) -> np.ndarray:
    """
    Convenience wrapper: build model + run eval.
    """
    model = get_cellpose_model(model_type=model_type, gpu=gpu)
    # cellpose works fine on float; but ensure 2D
    guv_2d = np.asarray(guv_img)
    if guv_2d.ndim != 2:
        raise ValueError(f"segment_guv_cellpose expects 2D array, got shape {guv_2d.shape}")
    return segment_with_cellpose(model, guv_2d, diameter=diameter, channels=channels, **kwargs)


# ============================================================
#                   NORMALIZATION HELPERS
# ============================================================
def percentile_normalize(img: np.ndarray, p1: float = 1, p99: float = 99) -> np.ndarray:
    """
    Normalize to [0,1] using percentiles (like your notebook).
    """
    x = np.asarray(img, dtype=np.float32)
    lo = np.percentile(x, p1)
    hi = np.percentile(x, p99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y[y < 0] = 0
    y[y > 1] = 1
    return y.astype(np.float32, copy=False)


def to_uint8_for_cv2(img01: np.ndarray, blur_ksize: tuple[int, int] = (5, 5), sigmaX: float = 1.0) -> np.ndarray:
    """
    Convert [0,1] float -> uint8 and blur.
    """
    img01 = np.asarray(img01, dtype=np.float32)
    u8 = cv2.normalize(img01, None, 0, 255, cv2.NORM_MINMAX)
    u8 = cv2.GaussianBlur(u8, blur_ksize, sigmaX=sigmaX)
    return np.uint8(u8)


# ============================================================
#                     CIRCLE DETECTION
# ============================================================
@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    r: int


from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class HoughCircleConfig:
    minDist: int
    param1: int
    param2: int
    minRadius: int
    maxRadius: int

DEFAULT_HOUGH_SCALES: tuple[HoughCircleConfig, ...] = (
    HoughCircleConfig(minDist=20, param1=30, param2=30, minRadius=10, maxRadius=30),  # small
    HoughCircleConfig(minDist=40, param1=25, param2=40, minRadius=30, maxRadius=60),  # medium
    HoughCircleConfig(minDist=60, param1=20, param2=60, minRadius=60, maxRadius=90),  # large
)

def _hough_multi_scale(
    img_u8: np.ndarray,
    *,
    dp: float,
    scales: Sequence[HoughCircleConfig],
) -> np.ndarray:
    """
    Returns circles array shape (N,3) float32 [x,y,r] (can be empty).
    """
    found: list[np.ndarray] = []

    for cfg in scales:
        res = cv2.HoughCircles(
            img_u8,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=cfg.minDist,
            param1=cfg.param1,
            param2=cfg.param2,
            minRadius=cfg.minRadius,
            maxRadius=cfg.maxRadius,
        )
        if res is not None and res.shape[1] > 0:
            found.append(res[0, :].astype(np.float32, copy=False))

    if not found:
        return np.empty((0, 3), dtype=np.float32)

    return np.concatenate(found, axis=0).astype(np.float32, copy=False)


def _bad_region_flags(
    circles_u16: np.ndarray,
    mt_img: np.ndarray,
    mt_bg_int: float,
    mt_std_threshold: float,
) -> np.ndarray:
    """
    Mimics your notebook filters:
      1) circle out of bounds
      2) too small/big
      3) MT too dim (no pixels > mt_bg_int inside)
      4) MT std too low (adaptive by pixel count)

    Returns flags array length N, where 0=good, else nonzero reason code.
    """
    h, w = mt_img.shape[:2]
    flags = np.zeros((circles_u16.shape[0],), dtype=np.int32)

    # median filter MT (like your notebook)
    mt_med = median_rank(np.asarray(mt_img, dtype=np.uint16), disk(3))
    mt_med = np.asarray(mt_med, dtype=np.float32)

    for idx, (x, y, r) in enumerate(circles_u16):
        # bounds check
        if (x - r < 0) or (y - r < 0) or (x + r > w) or (y + r > h):
            flags[idx] = 1
            continue

        if r < 10 or r > 120:
            flags[idx] = 2
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(r - 1), 255, thickness=-1)

        content = mt_med[mask > 0]
        content = content[content > mt_bg_int]
        if content.size == 0:
            flags[idx] = 3
            continue

        stdv = float(content.std())
        npx = int(content.size)

        # your adaptive std rules
        if stdv < mt_std_threshold and npx <= 3000:
            flags[idx] = 4
            continue
        if stdv < 10 and 3000 < npx <= 5000:
            flags[idx] = 4
            continue
        if stdv < 5 and npx > 5000:
            flags[idx] = 4
            continue

    return flags


def _overlap_redetect(
    circles: np.ndarray,
    flags: np.ndarray,
    img_u8: np.ndarray,
    dist_thresh: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Your notebook logic:
      - find close centers (<15 px) in two radius bands
      - mark as bad (5 or 6)
      - crop region and re-run Hough on crop to pick a better single circle
      - append best circle and flag=0

    Returns (circles_new, flags_new)
    """
    if circles.shape[0] == 0:
        return circles, flags

    circles_u16 = np.uint16(np.around(circles))
    centers = np.array([[c[0], c[1]] for c in circles_u16], dtype=np.float32)
    radii = np.array([c[2] for c in circles_u16], dtype=np.float32)

    dists = cdist(centers, centers)
    pairs: list[tuple[int, int, int]] = []  # (i,j, band_code)

    for i in range(centers.shape[0] - 1):
        for j in range(i + 1, centers.shape[0]):
            if dists[i, j] < dist_thresh:
                ri, rj = radii[i], radii[j]
                # band 1 (your ~30px radius group)
                if (23 < ri < 37) and (23 < rj < 37):
                    flags[i] = 5
                    flags[j] = 5
                    pairs.append((i, j, 5))
                # band 2 (your ~60px radius group)
                if (53 < ri < 67) and (53 < rj < 67):
                    flags[i] = 6
                    flags[j] = 6
                    pairs.append((i, j, 6))

    if not pairs:
        return circles, flags

    h, w = img_u8.shape[:2]
    circles_acc = circles.copy().astype(np.float32, copy=False)
    flags_acc = flags.copy().astype(np.int32, copy=False)

    for (i, j, code) in pairs:
        x1, y1, r1 = circles_u16[i]
        x2, y2, r2 = circles_u16[j]

        crop_x_l = max(0, min(x1 - r1, x2 - r2) - 10)
        crop_x_r = min(w - 1, max(x1 + r1, x2 + r2) + 10)
        crop_y_l = max(0, min(y1 - r1, y2 - r2) - 10)
        crop_y_r = min(h - 1, max(y1 + r1, y2 + r2) + 10)

        crop = img_u8[int(crop_y_l): int(crop_y_r), int(crop_x_l): int(crop_x_r)]

        if code == 5:
            dup = cv2.HoughCircles(
                crop, cv2.HOUGH_GRADIENT,
                dp=1.1, minDist=20, param1=25, param2=30,
                minRadius=25, maxRadius=35,
            )
        else:
            dup = cv2.HoughCircles(
                crop, cv2.HOUGH_GRADIENT,
                dp=1.1, minDist=40, param1=20, param2=40,
                minRadius=55, maxRadius=65,
            )

        if dup is None or dup.shape[1] == 0:
            # fallback: keep larger radius one
            if r1 >= r2:
                best = np.array([[float(x1), float(y1), float(r1)]], dtype=np.float32)
            else:
                best = np.array([[float(x2), float(y2), float(r2)]], dtype=np.float32)
        else:
            best = np.around(dup[0, 0:1]).astype(np.float32)
            best[0, 0] += float(crop_x_l)
            best[0, 1] += float(crop_y_l)

        circles_acc = np.concatenate([circles_acc, best], axis=0)
        flags_acc = np.concatenate([flags_acc, np.array([0], dtype=np.int32)], axis=0)

    return circles_acc, flags_acc


def segment_guv_hough_circles(
    guv_img: np.ndarray,
    *,
    mt_img: np.ndarray,
    sigma_smooth: float = 1.0,
    dp: float = 1.1,
    hough_scales: Sequence[HoughCircleConfig] = DEFAULT_HOUGH_SCALES,
    mt_bg_int: float = 120.0,
    mt_std_threshold: float = 15.0,
    overlap_dist_thresh: float = 15.0,
) -> dict[str, Any]:
    """
    Returns dict with:
      - circles_all: (N,3) float32 [x,y,r] before filtering (after overlap redetect)
      - good_circles: (M,3) float32 only good circles
      - flags: (N,) int32, 0=good else reason code
    """
    if guv_img.ndim != 2 or mt_img.ndim != 2:
        raise ValueError("segment_guv_hough_circles expects 2D guv_img and 2D mt_img")

    # smooth + percentile normalize (like your notebook)
    guv_s = gaussian(np.asarray(guv_img, dtype=np.float32), sigma=sigma_smooth, preserve_range=True)
    mt_med = median_rank(np.asarray(mt_img, dtype=np.uint16), disk(3))
    mt_s = gaussian(np.asarray(mt_med, dtype=np.float32), sigma=sigma_smooth, preserve_range=True)

    guv_n = percentile_normalize(guv_s, 1, 99)
    mt_n = percentile_normalize(mt_s, 1, 99)

    guv_u8 = to_uint8_for_cv2(guv_n, blur_ksize=(5, 5), sigmaX=1.0)

    circles = _hough_multi_scale(guv_u8, dp=dp, scales=hough_scales)

    if circles.shape[0] == 0:
        return {
            "circles_all": circles,
            "good_circles": circles,
            "flags": np.zeros((0,), dtype=np.int32),
            "guv_norm": guv_n,
            "mt_norm": mt_n,
        }

    circles_u16 = np.uint16(np.around(circles))
    flags = _bad_region_flags(circles_u16, mt_img=np.asarray(mt_img), mt_bg_int=mt_bg_int, mt_std_threshold=mt_std_threshold)

    # overlap redetect (adds circles; flags appended with 0 for new)
    circles2, flags2 = _overlap_redetect(circles.astype(np.float32), flags, guv_u8, dist_thresh=overlap_dist_thresh)

    # final “good circles” are those with flag==0 (and also not overlap-flagged 5/6)
    # Note: we keep original flags, so overlap originals remain flagged.
    good = circles2[flags2 == 0] if circles2.shape[0] == flags2.shape[0] else circles2

    return {
        "circles_all": circles2.astype(np.float32, copy=False),
        "good_circles": good.astype(np.float32, copy=False),
        "flags": flags2.astype(np.int32, copy=False),
        "guv_norm": guv_n,
        "mt_norm": mt_n,
    }


# ============================================================
#                 COMBINE CELLPOSE + CIRCLES
# ============================================================
def combine_segmentations(
    *,
    masks: Optional[np.ndarray],
    circles: Optional[dict[str, Any]],
    method: SegMethod,
) -> dict[str, Any]:
    """
    Returns a unified object dict used by the cropping stage.
    """
    out: dict[str, Any] = {"method": method}

    if method == "cellpose":
        if masks is None:
            raise ValueError("combine_segmentations(method='cellpose') requires masks")
        out["masks"] = masks

    elif method == "circle":
        if circles is None:
            raise ValueError("combine_segmentations(method='circle') requires circles dict")
        out["circles"] = circles["good_circles"]

    elif method == "combined":
        # simple strategy:
        # - if circles exist use them for cropping, else fallback to masks
        # you can later make this “union” smarter if you want
        if circles is not None and circles.get("good_circles", None) is not None and len(circles["good_circles"]) > 0:
            out["circles"] = circles["good_circles"]
        elif masks is not None:
            out["masks"] = masks
        else:
            out["circles"] = np.empty((0, 3), dtype=np.float32)
    else:
        raise ValueError(f"Unknown method: {method}")

    return out


# ============================================================
#                      CROPPING HELPERS
# ============================================================
def _safe_crop(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, min(w, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1), dtype=img.dtype)
    return img[y0:y1, x0:x1]


def _save_png_gray(img: np.ndarray, out_path: Path) -> None:
    """
    Save 2D image as PNG (8-bit).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(img)
    if x.ndim != 2:
        raise ValueError("PNG saver expects 2D grayscale")
    # normalize to 0..255 for display purposes
    x01 = percentile_normalize(x, 1, 99)
    u8 = (x01 * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_path)


def _save_tif(img: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), np.asarray(img))


def crop_objects_from_masks_or_circles(
    *,
    objects: dict[str, Any],
    guv_img: np.ndarray,
    mt_img: np.ndarray,
    crops_dir: str | Path,
    crops_tif_dir: str | Path,
    debug_dir: Optional[str | Path],
    source_row: Any,
    image_index: int,
    crop_pad: int = 10,
) -> pd.DataFrame:
    """
    Create per-object crops and a metadata DataFrame.

    Output crops:
      - PNG: MT channel by default (matches how you later trained on MT crops)
      - TIF: also saves MT crop (float/uint16 preserved)

    You can change what gets saved (GUV vs MT) later; for now this mirrors your workflow.
    """
    crops_dir = Path(crops_dir)
    crops_tif_dir = Path(crops_tif_dir)
    if debug_dir is not None:
        debug_dir = Path(debug_dir)

    rows: list[dict[str, Any]] = []

    method = objects.get("method", "unknown")

    if "circles" in objects:
        circles = np.asarray(objects["circles"], dtype=np.float32)
        for k in range(circles.shape[0]):
            x, y, r = circles[k]
            x = int(round(x))
            y = int(round(y))
            r = int(round(r))

            x0 = x - r - crop_pad
            x1 = x + r + crop_pad
            y0 = y - r - crop_pad
            y1 = y + r + crop_pad

            mt_crop = _safe_crop(mt_img, x0, x1, y0, y1)
            guv_crop = _safe_crop(guv_img, x0, x1, y0, y1)

            # filenames
            base = f"img{image_index:05d}_obj{k+1:03d}_x{x}_y{y}_r{r}"
            png_path = crops_dir / f"{base}.png"
            tif_path = crops_tif_dir / f"{base}.tif"

            _save_png_gray(mt_crop, png_path)
            _save_tif(mt_crop, tif_path)

            row_dict = {
                "filename": png_path.name,
                "tif_filename": tif_path.name,
                "image_index": image_index,
                "object_index": k + 1,
                "method": method,
                "cx": x,
                "cy": y,
                "radius": r,
            }

            # carry over useful columns if present
            for col in ["GUV_folder_path", "GUV_file_name", "MT_folder_path", "MT_file_name", "condition", "date"]:
                if hasattr(source_row, "__getitem__") and col in getattr(source_row, "index", []):
                    row_dict[col] = source_row[col]
                else:
                    # try dict-like access
                    try:
                        row_dict[col] = source_row[col]
                    except Exception:
                        pass

            rows.append(row_dict)

    elif "masks" in objects:
        masks = np.asarray(objects["masks"], dtype=np.int32)
        if masks.ndim != 2:
            raise ValueError(f"masks should be (H,W); got {masks.shape}")

        labels = np.unique(masks)
        labels = labels[labels != 0]

        for idx, lab in enumerate(labels, start=1):
            ys, xs = np.where(masks == lab)
            if ys.size == 0:
                continue

            y0 = int(max(0, ys.min() - crop_pad))
            y1 = int(min(masks.shape[0], ys.max() + crop_pad + 1))
            x0 = int(max(0, xs.min() - crop_pad))
            x1 = int(min(masks.shape[1], xs.max() + crop_pad + 1))

            mt_crop = _safe_crop(mt_img, x0, x1, y0, y1)

            base = f"img{image_index:05d}_mask{lab:03d}"
            png_path = crops_dir / f"{base}.png"
            tif_path = crops_tif_dir / f"{base}.tif"

            _save_png_gray(mt_crop, png_path)
            _save_tif(mt_crop, tif_path)

            rows.append(
                {
                    "filename": png_path.name,
                    "tif_filename": tif_path.name,
                    "image_index": image_index,
                    "object_index": idx,
                    "method": method,
                    "mask_label": int(lab),
                    "bbox_x0": x0,
                    "bbox_x1": x1,
                    "bbox_y0": y0,
                    "bbox_y1": y1,
                }
            )
    else:
        # no objects
        pass

    return pd.DataFrame(rows)