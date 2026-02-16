from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.filters import median
from skimage.morphology import disk
from tifffile import imread

# ============================================================
#                     LOADING
# ============================================================

def load_tiff_2d_max(path: str | Path) -> np.ndarray:
    """
    Read a TIFF. If it is 3D (Z,Y,X), do max projection over axis=0.
    Returns float32 array.
    """
    img = imread(str(path))
    if img.ndim == 3:
        img = img.max(axis=0)
    return img.astype(np.float32, copy=False)


def load_pair_image_2d(
    guv_path: str | Path,
    mt_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GUV/MT image pair, applying max projection if 3D.
    Returns (guv_img, mt_img) as float32 arrays.
    """
    guv_img = load_tiff_2d_max(guv_path)
    mt_img  = load_tiff_2d_max(mt_path)
    return guv_img, mt_img


# ============================================================
#                     STACKING
# ============================================================

def pad_or_crop_to_shape(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Pads with zeros (bottom/right) or crops (top-left region) to target_shape.
    """
    th, tw = target_shape
    out = np.zeros((th, tw), dtype=img.dtype)
    h = min(th, img.shape[0])
    w = min(tw, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def stack_pairs_to_arrays(
    df,
    target_shape: tuple[int, int] = (512, 512),
    nan_for_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (N,H,W) float32 stacks for GUV and MT channels.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: GUV_folder_path, GUV_file_name, MT_folder_path, MT_file_name
    target_shape : (H, W)
        All images are padded or cropped to this shape.
    nan_for_zero : bool
        If True, zero pixels are replaced with NaN (for robust statistics downstream).

    Returns
    -------
    guv_stack, mt_stack : np.ndarray shape (N, H, W)
    """
    n = len(df)
    guv = np.zeros((n, *target_shape), dtype=np.float32)
    mt  = np.zeros((n, *target_shape), dtype=np.float32)

    for i in range(n):
        row = df.iloc[i]
        guv_path = Path(row["GUV_folder_path"]) / row["GUV_file_name"]
        mt_path  = Path(row["MT_folder_path"])  / row["MT_file_name"]

        guv_img = load_tiff_2d_max(guv_path)
        mt_img  = load_tiff_2d_max(mt_path)

        guv[i] = pad_or_crop_to_shape(guv_img, target_shape)
        mt[i]  = pad_or_crop_to_shape(mt_img,  target_shape)

    if nan_for_zero:
        guv[guv == 0] = np.nan
        mt[mt == 0]   = np.nan

    return guv, mt


# ============================================================
#                     BACKGROUND
# ============================================================

def compute_background_median(
    stack: np.ndarray,
    disk_radius: int = 5,
) -> np.ndarray:
    """
    Compute a background image by taking the per-pixel median across the stack,
    then applying a median filter to smooth it.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Stack of images with NaNs for missing pixels.
    disk_radius : int
        Radius of the median filter disk footprint.

    Returns
    -------
    bg : np.ndarray, shape (H, W), float32
    """
    bg = np.nanmedian(stack, axis=0)
    bg_filtered = median(bg, footprint=disk(disk_radius))
    return bg_filtered.astype(np.float32, copy=False)


def compute_background_intensity(
    stack: np.ndarray,
) -> float:
    """
    Compute a single scalar background intensity for the stack.
    Uses the 1st percentile of all non-NaN pixels across the full stack.
    Used as:
      - the offset added back after background subtraction
      - the fill value for pixels outside circle/mask crops

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)

    Returns
    -------
    bg_intensity : float
    """
    flat = stack.ravel()
    flat = flat[~np.isnan(flat)]
    return float(np.percentile(flat, 1))


def subtract_background(
    stack: np.ndarray,
    bg_image: np.ndarray,
    bg_intensity: float = 0.0,
) -> np.ndarray:
    """
    Subtract background image pixel-wise and add back bg_intensity offset:
        corrected = image - bg_image + bg_intensity

    bg_intensity (1st percentile of the full stack) is added back so that
    near-background pixels retain a small positive baseline rather than
    being clipped to zero.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
    bg_image : np.ndarray, shape (H, W)
    bg_intensity : float
        Scalar offset added back after subtraction (use 1st percentile
        of the full stack, from compute_background_intensity()).

    Returns
    -------
    corrected : np.ndarray, shape (N, H, W), float32
    """
    corrected = stack - bg_image[np.newaxis, :, :] + bg_intensity
    # restore NaNs that were in the original stack
    corrected[np.isnan(stack)] = np.nan
    return corrected.astype(np.float32, copy=False)


def remove_background_and_pad(
    guv_stack: np.ndarray,
    mt_stack: np.ndarray,
    guv_bg: np.ndarray,
    mt_bg: np.ndarray,
    guv_bg_intensity: float = 0.0,
    mt_bg_intensity: float = 0.0,
    pad: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract background images from GUV and MT stacks, add back the 1st
    percentile offset, and replace NaNs with bg_intensity for downstream.

    Formula per pixel: corrected = image - bg_image + bg_intensity

    Parameters
    ----------
    guv_stack, mt_stack : np.ndarray, shape (N, H, W)
    guv_bg, mt_bg : np.ndarray, shape (H, W)
        Background images from compute_background_median().
    guv_bg_intensity, mt_bg_intensity : float
        1st percentile of each channel's full stack, from
        compute_background_intensity(). Added back after subtraction.
    pad : int
        Currently unused — reserved for future border padding.

    Returns
    -------
    guv_corrected, mt_corrected : np.ndarray, shape (N, H, W), float32
    """
    guv_corr = subtract_background(guv_stack, guv_bg, bg_intensity=guv_bg_intensity)
    mt_corr  = subtract_background(mt_stack,  mt_bg,  bg_intensity=mt_bg_intensity)

    # Replace NaN with bg_intensity so downstream (cv2, cellpose) don't break
    guv_corr = np.nan_to_num(guv_corr, nan=guv_bg_intensity)
    mt_corr  = np.nan_to_num(mt_corr,  nan=mt_bg_intensity)

    return guv_corr, mt_corr


# ============================================================
#               PERCENTILE STATISTICS (per stack)
# ============================================================

STANDARD_PERCENTILES = (0.001, 0.1, 0.2, 0.5, 1.0, 2.0, 50.0, 98.0, 99.0, 99.5, 99.8, 99.9, 99.95)

# Normalization percentiles matching notebook (0.001 low, 99.95 high)
NORM_P_LOW  = 0.001
NORM_P_HIGH = 99.95


def smooth_stack(
    stack: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply per-image Gaussian smoothing to a (N, H, W) stack.
    NaN pixels are zeroed before smoothing and restored after.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W), float32
    sigma : float
        Gaussian sigma (default 1.0, matches notebook).

    Returns
    -------
    smoothed : np.ndarray, shape (N, H, W), float32
    """
    from skimage.filters import gaussian

    smoothed = np.zeros_like(stack, dtype=np.float32)
    for i in range(stack.shape[0]):
        img = np.nan_to_num(stack[i], nan=0.0)
        smoothed[i] = gaussian(img, sigma=sigma, preserve_range=True).astype(np.float32)
    return smoothed


def compute_stack_percentiles(
    stack: np.ndarray,
    percentiles: tuple[float, ...] = STANDARD_PERCENTILES,
    nonzero_only: bool = True,
) -> dict[float, float]:
    """
    Compute percentiles across the entire stack (all pixels, all images).
    NaN values are always ignored. Zero pixels are optionally excluded
    (nonzero_only=True matches notebook behaviour).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
    percentiles : tuple of floats
    nonzero_only : bool
        If True, exclude zero pixels (matches notebook: img[img>0]).

    Returns
    -------
    dict mapping percentile -> value
    """
    flat = stack.ravel()
    flat = flat[~np.isnan(flat)]
    if nonzero_only:
        flat = flat[flat > 0]
    if flat.size == 0:
        return {p: 0.0 for p in percentiles}
    values = np.percentile(flat, list(percentiles))
    return {p: float(v) for p, v in zip(percentiles, values)}


def compute_channel_statistics(
    guv_stack: np.ndarray,
    mt_stack: np.ndarray,
    sigma: float = 1.0,
    percentiles: tuple[float, ...] = STANDARD_PERCENTILES,
) -> dict[str, dict]:
    """
    Smooth each image in the stack (Gaussian, sigma=1), then compute
    percentile statistics across all non-zero pixels of the smoothed stack.

    Matches notebook behaviour:
      - gaussian smooth per image
      - percentiles from smoothed stack, nonzero pixels only
      - low = 0.001, high = 99.95

    Parameters
    ----------
    guv_stack, mt_stack : np.ndarray, shape (N, H, W)
        Raw stacks (before background subtraction).
    sigma : float
        Gaussian sigma for smoothing (default 1.0).
    percentiles : tuple of floats

    Returns
    -------
    dict with keys 'guv' and 'mt', each containing:
        - 'bg_intensity' : float  — 1st percentile of raw stack
        - 'percentiles'  : dict {percentile: value} from smoothed stack
        - 'norm_low'     : float  — NORM_P_LOW  (0.001) percentile
        - 'norm_high'    : float  — NORM_P_HIGH (99.95) percentile

    Example
    -------
    stats = compute_channel_statistics(guv_stack, mt_stack)
    guv_1p    = stats['guv']['norm_low']
    guv_99p   = stats['guv']['norm_high']
    """
    guv_smooth = smooth_stack(guv_stack, sigma=sigma)
    mt_smooth  = smooth_stack(mt_stack,  sigma=sigma)

    guv_pcts = compute_stack_percentiles(guv_smooth, percentiles, nonzero_only=True)
    mt_pcts  = compute_stack_percentiles(mt_smooth,  percentiles, nonzero_only=True)

    return {
        "guv": {
            "bg_intensity": compute_background_intensity(guv_stack),
            "percentiles":  guv_pcts,
            "norm_low":     guv_pcts[NORM_P_LOW],
            "norm_high":    guv_pcts[NORM_P_HIGH],
        },
        "mt": {
            "bg_intensity": compute_background_intensity(mt_stack),
            "percentiles":  mt_pcts,
            "norm_low":     mt_pcts[NORM_P_LOW],
            "norm_high":    mt_pcts[NORM_P_HIGH],
        },
    }


# ============================================================
#               CIRCLE PATCH CROPPING WITH BG FILL
# ============================================================

def crop_circle_patch(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    bg_intensity: float,
    patch_size: int = 96,
) -> np.ndarray:
    """
    Crop a fixed patch_size x patch_size square centered on (cx, cy),
    filling pixels outside the circle boundary with bg_intensity.

    For large circles (radius > patch_size//2), only the central portion
    of the circle is visible — intentional and consistent with training data.

    Parameters
    ----------
    img : np.ndarray, shape (H, W), float32
    cx, cy : int
        Circle center in image coordinates.
    radius : int
        Circle radius in pixels. Used only for the bg fill mask.
    bg_intensity : float
        Fill value for pixels outside the circle mask.
    patch_size : int
        Side length of the output square patch (default 96).

    Returns
    -------
    patch : np.ndarray, shape (patch_size, patch_size), float32
    """
    h, w = img.shape[:2]
    half = patch_size // 2

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    # clamp to image bounds
    x0c = max(0, x0)
    x1c = min(w, x1)
    y0c = max(0, y0)
    y1c = min(h, y1)

    # initialize patch with background fill
    patch = np.full((patch_size, patch_size), fill_value=bg_intensity, dtype=np.float32)

    # copy valid region into patch canvas
    src = img[y0c:y1c, x0c:x1c]
    dst_y0 = y0c - y0
    dst_x0 = x0c - x0
    patch[dst_y0: dst_y0 + src.shape[0], dst_x0: dst_x0 + src.shape[1]] = src

    # build circle mask centered in patch and fill outside with bg
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(mask, (half, half), radius, 255, thickness=-1)
    patch[mask == 0] = bg_intensity

    return patch.astype(np.float32, copy=False)


def crop_mask_patch(
    img: np.ndarray,
    mask: np.ndarray,
    bg_intensity: float,
    patch_size: int = 96,
) -> np.ndarray:
    """
    Crop a fixed patch_size x patch_size square centered on the mask centroid,
    filling pixels outside the mask boundary with bg_intensity.

    Parameters
    ----------
    img : np.ndarray, shape (H, W), float32
        MT channel image.
    mask : np.ndarray, shape (H, W), bool or uint8
        Binary mask for a single object (True/255 = inside object).
    bg_intensity : float
        Fill value for pixels outside the mask.
    patch_size : int
        Side length of the output square patch (default 96).

    Returns
    -------
    patch : np.ndarray, shape (patch_size, patch_size), float32
    """
    h, w = img.shape[:2]
    half = patch_size // 2

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return np.full((patch_size, patch_size), fill_value=bg_intensity, dtype=np.float32)

    # centroid
    cy = round(float(ys.mean()))
    cx = round(float(xs.mean()))

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    # clamp to image bounds
    x0c = max(0, x0)
    x1c = min(w, x1)
    y0c = max(0, y0)
    y1c = min(h, y1)

    # initialize patch with background fill
    patch = np.full((patch_size, patch_size), fill_value=bg_intensity, dtype=np.float32)

    # copy valid image region into patch canvas
    src = img[y0c:y1c, x0c:x1c]
    dst_y0 = y0c - y0
    dst_x0 = x0c - x0
    patch[dst_y0: dst_y0 + src.shape[0], dst_x0: dst_x0 + src.shape[1]] = src

    # translate mask into patch coordinates and fill outside with bg
    mask_patch = np.zeros((patch_size, patch_size), dtype=np.uint8)
    src_mask = mask[y0c:y1c, x0c:x1c].astype(np.uint8)
    mask_patch[dst_y0: dst_y0 + src_mask.shape[0], dst_x0: dst_x0 + src_mask.shape[1]] = src_mask
    patch[mask_patch == 0] = bg_intensity

    return patch.astype(np.float32, copy=False)
