from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from scipy.spatial.distance import cdist
from skimage.filters import gaussian
from skimage.filters.rank import median as median_rank
from skimage.measure import regionprops
from skimage.morphology import disk

from mt_structure_classification.utils.image_processing import crop_circle_patch, crop_mask_patch

SegMethod = Literal["cellpose", "circle", "combined"]


# ============================================================
#                     HOUGH CONFIG
# ============================================================

@dataclass(frozen=True)
class HoughCircleConfig:
    """Parameters for one HoughCircles pass (e.g. small / medium / large)."""
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


# ============================================================
#                         CELLPOSE
# ============================================================

def get_cellpose_model(
    model_type: str = "cyto3",
    gpu: bool = False,
):
    from cellpose import models
    return models.CellposeModel(gpu=gpu, model_type=model_type)  


def _upsample_labels(labels_small: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Upsample a label mask from downsampled (::2) space back to full resolution
    using nearest-neighbour repeat (matches original notebook behaviour).

    labels_small : (H//2, W//2) int
    Returns      : (H, W) int32
    """
    out = np.zeros((target_h, target_w), dtype=np.int32)
    h2, w2 = labels_small.shape
    out[0:h2*2:2,   0:w2*2:2] = labels_small
    out[1:h2*2:2,   0:w2*2:2] = labels_small
    out[0:h2*2:2,   1:w2*2:2] = labels_small
    out[1:h2*2:2,   1:w2*2:2] = labels_small
    return out


def filter_cellpose_masks(
    label_mask: np.ndarray,
    mt_img: np.ndarray,
    mt_bg_int: float = 0.0,   # set to 1st percentile of raw MT stack
    mt_std_threshold: float = 15.0,
    max_eccentricity: float = 0.5,
    min_area: int = 1000,
    max_area: int = 40000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter cellpose label mask using regionprops criteria matching the
    original notebook:
      - eccentricity > max_eccentricity  → reject
      - area < min_area or > max_area    → reject
      - no MT pixels above mt_bg_int     → reject
      - MT std < mt_std_threshold        → reject

    Returns
    -------
    filtered_mask : (H, W) int32  — bad labels set to 0
    bad_flags     : (N,)   int32  — 0=good, 1=eccentricity/area,
                                    2=no MT signal, 3=low MT std
    """
    from skimage.filters.rank import median as median_rank

    label_mask = label_mask.copy().astype(np.int32)
    n = int(label_mask.max())
    bad_flags = np.zeros(n, dtype=np.int32)

    mt_med = median_rank(np.asarray(mt_img, dtype=np.uint16), disk(3))
    mt_med = np.asarray(mt_med, dtype=np.float32)

    props = regionprops(label_mask, intensity_image=mt_img.astype(np.float32))

    for region in props:
        idx = region.label - 1  # 0-based index into bad_flags

        # shape filters
        if (region.eccentricity > max_eccentricity
                or region.area < min_area
                or region.area > max_area):
            bad_flags[idx] = 1
            label_mask[label_mask == region.label] = 0
            continue

        # MT signal filter
        img_content = mt_med[label_mask == region.label]
        img_content = img_content[img_content > mt_bg_int]
        if img_content.size == 0:
            bad_flags[idx] = 2
            label_mask[label_mask == region.label] = 0
            continue

        # MT std filter (fixed threshold — matches notebook)
        if img_content.std() < mt_std_threshold:
            bad_flags[idx] = 3
            label_mask[label_mask == region.label] = 0

    return label_mask, bad_flags


def segment_guv_cellpose(
    guv_img_norm: np.ndarray,
    mt_img_norm: np.ndarray,
    mt_img_raw: np.ndarray,
    model_type: str = "cyto3",
    gpu: bool = True,
    diameter: float | None = None,
    channels: tuple[int, int] = (0, 0),
    mt_bg_int: float = 0.0,   # set to 1st percentile of raw MT stack
    mt_std_threshold: float = 15.0,
    max_eccentricity: float = 0.5,
    min_area: int = 1000,
    max_area: int = 40000,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Cellpose on a downsampled (::2) GUV+MT stack, upsample the result
    back to full resolution, then filter masks using regionprops criteria.

    Downsampling matches the original notebook: `model.eval(img[:,::2,::2])`.
    Upsampling uses nearest-neighbour pixel repeat (not interpolation).

    Parameters
    ----------
    guv_img_norm : np.ndarray, shape (H, W), float32 — percentile-normalized GUV (for cellpose)
    mt_img_norm  : np.ndarray, shape (H, W), float32 — percentile-normalized MT (for cellpose)
    mt_img_raw   : np.ndarray, shape (H, W), float32 — raw MT in original intensity units
                   (used for filtering — mt_bg_int and mt_std_threshold are in raw units)
    model_type : str
    gpu : bool
    diameter : float or None
    channels : (int, int)
    mt_bg_int : float
        MT background intensity for mask filtering.
    mt_std_threshold : float
    max_eccentricity : float
    min_area, max_area : int

    Returns
    -------
    label_mask_filtered : (H, W) int32 — 0=background, bad labels zeroed
    bad_flags           : (N,)   int32 — one per original label
    """
    if guv_img_norm.shape != mt_img_norm.shape or guv_img_norm.shape != mt_img_raw.shape:
        raise ValueError(
            f"guv_img_norm, mt_img_norm and mt_img_raw must all have the same shape, "
            f"got {guv_img_norm.shape}, {mt_img_norm.shape}, {mt_img_raw.shape}"
        )

    h, w = guv_img_norm.shape[:2]

    # stack normalized channels and downsample by 2 for cellpose (matches notebook)
    guv_mt_img = np.zeros((2, h, w), dtype=np.float32)
    guv_mt_img[0] = guv_img_norm
    guv_mt_img[1] = mt_img_norm

    
    model = get_cellpose_model(model_type=model_type, gpu=gpu)
    result = model.eval(
        guv_mt_img[:, ::2, ::2],
        diameter=diameter,
        channels=channels,
        **kwargs,
    )
    
    # Handle both old (4 values) and new (3 values) API
    if len(result) == 4:
        masks_small, _, _, _ = result  # v2.x
    else:
        masks_small, _, _ = result      # v4.x



    # upsample back to full resolution
    label_mask = _upsample_labels(masks_small, h, w)

    # filter using raw MT image — thresholds are in raw intensity units
    label_mask_filtered, bad_flags = filter_cellpose_masks(
        label_mask,
        mt_img=mt_img_raw,
        mt_bg_int=mt_bg_int,
        mt_std_threshold=mt_std_threshold,
        max_eccentricity=max_eccentricity,
        min_area=min_area,
        max_area=max_area,
    )

    return label_mask_filtered, bad_flags


# ============================================================
#                   NORMALIZATION HELPERS
# ============================================================

def percentile_normalize(img: np.ndarray, p1: float = 1, p99: float = 99) -> np.ndarray:
    """Normalize to [0,1] using percentiles."""
    x = np.asarray(img, dtype=np.float32)
    lo = np.percentile(x, p1)
    hi = np.percentile(x, p99)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def to_uint8_for_cv2(
    img01: np.ndarray,
    blur_ksize: tuple[int, int] = (5, 5),
    sigmaX: float = 1.0,
) -> np.ndarray:
    """Convert [0,1] float -> uint8 and apply Gaussian blur."""
    u8 = cv2.normalize(np.asarray(img01, dtype=np.float32), None, 0, 255, cv2.NORM_MINMAX)
    u8 = cv2.GaussianBlur(u8, blur_ksize, sigmaX=sigmaX)
    return np.uint8(u8)


# ============================================================
#                     CIRCLE DETECTION
# ============================================================

def _hough_multi_scale(
    img_u8: np.ndarray,
    *,
    dp: float,
    scales: Sequence[HoughCircleConfig],
) -> np.ndarray:
    """
    Run HoughCircles at multiple scales and concatenate results.
    Returns array shape (N, 3) float32 [x, y, r], or empty (0, 3).
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
            found.append(res[0].astype(np.float32, copy=False))

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
    Filter circles based on MT channel content.

    Reason codes:
      1 = out of bounds
      2 = radius out of range (< 10 or > 120)
      3 = no MT signal above background
      4 = MT std too low (adaptive by pixel count)

    Returns flags array length N, 0 = good.
    """
    h, w = mt_img.shape[:2]
    flags = np.zeros((circles_u16.shape[0],), dtype=np.int32)

    mt_med = median_rank(np.asarray(mt_img, dtype=np.uint16), disk(3))
    mt_med = np.asarray(mt_med, dtype=np.float32)

    for idx, (x, y, r) in enumerate(circles_u16):
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
        npx  = int(content.size)

        if (stdv < mt_std_threshold and npx <= 3000) or (stdv < 10 and 3000 < npx <= 5000) or (stdv < 5 and npx > 5000):
            flags[idx] = 4

    return flags


def _overlap_redetect(
    circles: np.ndarray,
    flags: np.ndarray,
    img_u8: np.ndarray,
    dist_thresh: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find close circle pairs in two radius bands, mark them bad,
    crop the region and re-run Hough to find a single best circle.

    Reason codes added:
      5 = overlap in small band (~30px radius)
      6 = overlap in medium band (~60px radius)

    Returns (circles_new, flags_new) with re-detected circles appended
    with flag=0.
    """
    if circles.shape[0] == 0:
        return circles, flags

    circles_u16 = np.uint16(np.around(circles))
    centers = circles_u16[:, :2].astype(np.float32)
    radii   = circles_u16[:, 2].astype(np.float32)
    dists   = cdist(centers, centers)

    h, w = img_u8.shape[:2]
    circles_acc = circles.copy().astype(np.float32)
    flags_acc   = flags.copy().astype(np.int32)

    for i in range(len(circles_u16) - 1):
        for j in range(i + 1, len(circles_u16)):
            if dists[i, j] >= dist_thresh:
                continue

            ri, rj = radii[i], radii[j]
            if (23 < ri < 37) and (23 < rj < 37):
                band_code = 5
                hough_params = dict(dp=1.1, minDist=20, param1=25, param2=30,
                                    minRadius=25, maxRadius=35)
            elif (53 < ri < 67) and (53 < rj < 67):
                band_code = 6
                hough_params = dict(dp=1.1, minDist=40, param1=20, param2=40,
                                    minRadius=55, maxRadius=65)
            else:
                continue

            flags_acc[i] = band_code
            flags_acc[j] = band_code

            x1, y1, r1 = circles_u16[i]
            x2, y2, r2 = circles_u16[j]
            crop_x0 = max(0, int(min(x1 - r1, x2 - r2)) - 10)
            crop_x1 = min(w - 1, int(max(x1 + r1, x2 + r2)) + 10)
            crop_y0 = max(0, int(min(y1 - r1, y2 - r2)) - 10)
            crop_y1 = min(h - 1, int(max(y1 + r1, y2 + r2)) + 10)

            crop = img_u8[crop_y0:crop_y1, crop_x0:crop_x1]
            dup  = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, **hough_params)

            if dup is None or dup.shape[1] == 0:
                best = np.array([[float(x1 if r1 >= r2 else x2),
                                  float(y1 if r1 >= r2 else y2),
                                  float(max(r1, r2))]], dtype=np.float32)
            else:
                best = np.around(dup[0, 0:1]).astype(np.float32)
                best[0, 0] += float(crop_x0)
                best[0, 1] += float(crop_y0)

            circles_acc = np.concatenate([circles_acc, best], axis=0)
            flags_acc   = np.concatenate([flags_acc, np.array([0], dtype=np.int32)], axis=0)

    return circles_acc, flags_acc


def segment_guv_hough_circles(
    guv_img: np.ndarray,
    *,
    mt_img: np.ndarray,
    sigma_smooth: float = 1.0,
    dp: float = 1.1,
    hough_scales: Sequence[HoughCircleConfig] = DEFAULT_HOUGH_SCALES,
    mt_bg_int: float = 0.0,   # set to 1st percentile of raw MT stack
    mt_std_threshold: float = 15.0,
    overlap_dist_thresh: float = 15.0,
) -> dict[str, Any]:
    """
    Full Hough circle detection pipeline on a GUV image.

    Steps:
      1. Smooth + percentile-normalize GUV and MT
      2. Multi-scale HoughCircles on GUV
      3. Filter bad circles based on MT content
      4. Re-detect overlapping circles

    Parameters
    ----------
    guv_img : np.ndarray, shape (H, W)
    mt_img  : np.ndarray, shape (H, W)
    sigma_smooth : float
    dp : float
        Inverse ratio of accumulator resolution.
    hough_scales : sequence of HoughCircleConfig
    mt_bg_int : float
        MT background intensity threshold — use the 1st percentile of the
        raw MT stack (compute_background_intensity(mt_stack)).
    mt_std_threshold : float
        MT std threshold for adaptive filtering.
    overlap_dist_thresh : float
        Max center distance (px) to consider two circles overlapping.

    Returns
    -------
    dict with keys:
      circles_all  : (N, 3) float32 [x, y, r]
      good_circles : (M, 3) float32 flag==0 only
      flags        : (N,)   int32 reason codes (0 = good)
      guv_norm     : (H, W) float32
      mt_norm      : (H, W) float32
    """
    if guv_img.ndim != 2 or mt_img.ndim != 2:
        raise ValueError("segment_guv_hough_circles expects 2D guv_img and mt_img")

    guv_s  = gaussian(guv_img.astype(np.float32), sigma=sigma_smooth, preserve_range=True)
    mt_med = median_rank(mt_img.astype(np.uint16), disk(3))
    mt_s   = gaussian(mt_med.astype(np.float32),   sigma=sigma_smooth, preserve_range=True)

    guv_n = percentile_normalize(guv_s, 1, 99)
    mt_n  = percentile_normalize(mt_s,  1, 99)
    guv_u8 = to_uint8_for_cv2(guv_n)

    circles = _hough_multi_scale(guv_u8, dp=dp, scales=hough_scales)

    if circles.shape[0] == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return {"circles_all": empty, "good_circles": empty,
                "flags": np.zeros((0,), dtype=np.int32),
                "guv_norm": guv_n, "mt_norm": mt_n}

    circles_u16 = np.uint16(np.around(circles))
    flags = _bad_region_flags(
        circles_u16, mt_img=mt_img.astype(np.float32),
        mt_bg_int=mt_bg_int, mt_std_threshold=mt_std_threshold,
    )

    circles2, flags2 = _overlap_redetect(
        circles.astype(np.float32), flags, guv_u8,
        dist_thresh=overlap_dist_thresh,
    )

    good = circles2[flags2 == 0]

    return {
        "circles_all":  circles2.astype(np.float32, copy=False),
        "good_circles": good.astype(np.float32,   copy=False),
        "flags":        flags2.astype(np.int32,    copy=False),
        "guv_norm":     guv_n,
        "mt_norm":      mt_n,
    }


# ============================================================
#                 COMBINE CELLPOSE + CIRCLES
# ============================================================

def _circle_to_mask(
    cx: int, cy: int, r: int, h: int, w: int,
) -> np.ndarray:
    """Rasterize a circle into a binary mask of shape (H, W)."""
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 1, thickness=-1)
    return m


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def match_circles_to_masks(
    good_circles: np.ndarray,
    label_mask: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Match Hough circles to Cellpose masks using IoU.

    For each circle, rasterize it and compute IoU against every cellpose
    mask label. A match is recorded when IoU >= iou_threshold.
    Each mask label can only be matched to one circle (best IoU wins).

    Parameters
    ----------
    good_circles : np.ndarray, shape (N, 3) float32 [x, y, r]
    label_mask   : np.ndarray, shape (H, W) int32, cellpose label output
    iou_threshold : float

    Returns
    -------
    List of (circle_index, mask_label) pairs that exceed the threshold.
    """
    if good_circles.shape[0] == 0:
        return []

    h, w = label_mask.shape
    labels = np.unique(label_mask)
    labels = labels[labels != 0]

    if labels.size == 0:
        return []

    # precompute binary mask per label
    label_masks: dict[int, np.ndarray] = {
        int(lab): (label_mask == lab).astype(np.uint8)
        for lab in labels
    }

    # (circle_idx, label, iou) candidates
    candidates: list[tuple[int, int, float]] = []

    for ci in range(good_circles.shape[0]):
        cx = round(float(good_circles[ci, 0]))
        cy = round(float(good_circles[ci, 1]))
        r  = round(float(good_circles[ci, 2]))
        circle_mask = _circle_to_mask(cx, cy, r, h, w)

        for lab, lmask in label_masks.items():
            iou = _iou(circle_mask, lmask)
            if iou >= iou_threshold:
                candidates.append((ci, lab, iou))

    if not candidates:
        return []

    # greedy match: for each mask label keep the circle with highest IoU
    # (each label matched at most once)
    candidates.sort(key=lambda t: t[2], reverse=True)
    matched_circles: set[int] = set()
    matched_labels:  set[int] = set()
    matches: list[tuple[int, int]] = []

    for ci, lab, _score in candidates:
        if ci in matched_circles or lab in matched_labels:
            continue
        matches.append((ci, lab))
        matched_circles.add(ci)
        matched_labels.add(lab)

    return matches


def combine_segmentations(
    *,
    masks: np.ndarray | None,
    circles: dict[str, Any] | None,
    method: SegMethod,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Unify cellpose masks and/or Hough circles into a single object dict
    used by the cropping stage.

    method='cellpose' : use cellpose masks only
    method='circle'   : use Hough circles only
    method='combined' : strict intersection — keep only objects detected
                        by both methods with IoU >= iou_threshold.
                        Crop is defined by the cellpose mask.

    Returns
    -------
    dict with keys:
      'method'  : SegMethod
      'masks'   : (H,W) int32 label mask  — present for cellpose / combined
      'circles' : (N,3) float32 [x,y,r]  — present for circle only
      'matches' : list of (circle_idx, mask_label) — present for combined
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
        if masks is None or circles is None:
            raise ValueError(
                "combine_segmentations(method='combined') requires both masks and circles"
            )
        good = circles.get("good_circles", np.empty((0, 3), dtype=np.float32))
        matches = match_circles_to_masks(good, masks, iou_threshold=iou_threshold)

        # build a filtered label mask containing only matched labels
        matched_labels = {lab for _, lab in matches}
        filtered_masks = np.where(np.isin(masks, list(matched_labels)), masks, 0).astype(np.int32)

        out["masks"]   = filtered_masks
        out["matches"] = matches  # (circle_idx, mask_label) pairs

    else:
        raise ValueError(f"Unknown segmentation method: {method!r}")

    return out


# ============================================================
#                 CROPPING — MT CHANNEL ONLY
# ============================================================

def crop_objects_from_masks_or_circles(
    *,
    objects: dict[str, Any],
    mt_img: np.ndarray,
    mt_bg_intensity: float,
    crops_dir: str | Path,
    source_row: Any,
    image_index: int,
    patch_size: int = 96,
) -> pd.DataFrame:
    """
    Crop 96x96 MT patches for each detected object and save as TIF.

    Circle objects: center and radius from Hough detection.
    Mask objects: bounding box center and half-diagonal used as radius.

    Parameters
    ----------
    objects : dict
        Output of combine_segmentations().
    mt_img : np.ndarray, shape (H, W)
        MT channel (float32, background-subtracted).
    mt_bg_intensity : float
        Scalar fill value for pixels outside the circle mask.
    crops_dir : str or Path
        Directory to save TIF patches.
    source_row : pd.Series
        Row from pairs dataframe (for metadata columns).
    image_index : int
        Index of this image in the stack.
    patch_size : int
        Fixed output patch size in pixels (default 96).

    Returns
    -------
    pd.DataFrame with one row per object.
    """
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    method = objects.get("method", "unknown")

    meta_cols = ["GUV_folder_path", "GUV_file_name", "MT_folder_path",
                 "MT_file_name", "condition", "date"]

    meta: dict[str, Any] = {}
    for col in meta_cols:
        with contextlib.suppress(Exception):
            meta[col] = source_row[col]

    if "circles" in objects:
        circles = np.asarray(objects["circles"], dtype=np.float32)

        for k in range(circles.shape[0]):
            cx = round(float(circles[k, 0]))
            cy = round(float(circles[k, 1]))
            radius = round(float(circles[k, 2]))

            patch = crop_circle_patch(
                mt_img, cx=cx, cy=cy, radius=radius,
                bg_intensity=mt_bg_intensity, patch_size=patch_size,
            )

            fname = f"img{image_index:05d}_obj{k+1:03d}_x{cx}_y{cy}_r{radius}.tif"
            tifffile.imwrite(str(crops_dir / fname), patch)

            rows.append({
                "filename":     fname,
                "image_index":  image_index,
                "object_index": k + 1,
                "method":       method,
                "cx":           cx,
                "cy":           cy,
                "radius":       radius,
                **meta,
            })

    elif "masks" in objects:
        masks = np.asarray(objects["masks"], dtype=np.int32)
        if masks.ndim != 2:
            raise ValueError(f"masks should be (H,W); got {masks.shape}")

        for idx, lab in enumerate(np.unique(masks)[1:], start=1):  # skip 0
            single_mask = (masks == lab).astype(np.uint8)
            ys, xs = np.where(single_mask > 0)
            if ys.size == 0:
                continue

            cy = round(float(ys.mean()))
            cx = round(float(xs.mean()))
            # radius stored for metadata only (not used for cropping)
            radius = round(max(ys.max() - ys.min(),
                               xs.max() - xs.min()) / 2)

            patch = crop_mask_patch(
                mt_img,
                mask=single_mask,
                bg_intensity=mt_bg_intensity,
                patch_size=patch_size,
            )

            fname = f"img{image_index:05d}_mask{lab:03d}.tif"
            tifffile.imwrite(str(crops_dir / fname), patch)

            rows.append({
                "filename":     fname,
                "image_index":  image_index,
                "object_index": idx,
                "method":       method,
                "cx":           cx,
                "cy":           cy,
                "radius":       radius,
                "mask_label":   int(lab),
                **meta,
            })

    return pd.DataFrame(rows)
