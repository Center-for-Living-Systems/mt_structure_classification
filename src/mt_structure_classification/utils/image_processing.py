from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from tifffile import imread
from skimage.filters import median
from skimage.morphology import disk


def load_tiff_2d_max(path: str | Path) -> np.ndarray:
    """
    Read a TIFF. If it is 3D (Z,Y,X), do max projection over axis=0.
    Returns float32 array.
    """
    img = imread(str(path))
    if img.ndim == 3:
        img = img.max(axis=0)
    return img.astype(np.float32, copy=False)


def pad_or_crop_to_shape(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Pads with zeros (bottom/right) or crops (top-left region) to target_shape.
    Mirrors your behavior where you placed images into [0:h,0:w] in a 512x512 canvas.
    """
    th, tw = target_shape
    out = np.zeros((th, tw), dtype=img.dtype)

    h = min(th, img.shape[0])
    w = min(tw, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def stack_pairs_to_arrays(
    df,
    target_shape: Tuple[int, int] = (512, 512),
    nan_for_zero: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (N,H,W) stacks for GUV and MT.
    """
    n = len(df)
    guv = np.zeros((n, *target_shape), dtype=np.float32)
    mt  = np.zeros((n, *target_shape), dtype=np.float32)

    for i in range(n):
        guv_path = Path(df.loc[i, "GUV_folder_path"]) / df.loc[i, "GUV_file_name"]
        mt_path  = Path(df.loc[i, "MT_folder_path"])  / df.loc[i, "MT_file_name"]

        guv_img = load_tiff_2d_max(guv_path)
        mt_img  = load_tiff_2d_max(mt_path)

        guv[i] = pad_or_crop_to_shape(guv_img, target_shape)
        mt[i]  = pad_or_crop_to_shape(mt_img,  target_shape)

    if nan_for_zero:
        guv[guv == 0] = np.nan
        mt[mt == 0] = np.nan

    return guv, mt


def compute_background_median(
    stack: np.ndarray,
    disk_radius: int = 5,
) -> np.ndarray:
    """
    Median background across stack, then median filter (like your code).
    Expects stack shape (N,H,W) with NaNs for missing pixels.
    """
    bg = np.nanmedian(stack, axis=0)
    bg_f = median(bg, footprint=disk(disk_radius))
    return bg_f.astype(np.float32, copy=False)

def load_pair_image_2d(guv_path, mt_path):        
    guv_img = imread(str(guv_path))
    mt_img  = imread(str(mt_path))
    return guv_img, mt_img 
