from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive — safe for scripts and notebooks
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tifffile

# ============================================================
#                        HELPERS
# ============================================================

def _norm(img: np.ndarray, p1: float = 1.0, p99: float = 99.0) -> np.ndarray:
    """Percentile-normalize to [0,1] for display. NaNs → 0."""
    x = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0)
    lo = np.percentile(x, p1)
    hi = np.percentile(x, p99)
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path}")


def _make_label_overlay(
    label_mask: np.ndarray,
    n_colors: int = 201,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Build an RGBA overlay from a label mask.
    Uses a fixed random colormap seeded at 42 (matches notebook newmap style).
    Returns (H, W, 4) float32.
    """
    rng = np.random.default_rng(seed=42)
    # +1 so index 0 = background (transparent)
    colors = rng.uniform(0.2, 1.0, size=(max(n_colors, int(label_mask.max()) + 1), 3))
    colors[0] = 0.0

    overlay = np.zeros((*label_mask.shape, 4), dtype=np.float32)
    for lab in np.unique(label_mask):
        if lab == 0:
            continue
        m = label_mask == lab
        overlay[m, :3] = colors[int(lab) % len(colors)]
        overlay[m,  3] = alpha
    return overlay


# ============================================================
#   PLOT A — Cellpose segmentation panel (matches notebook fig2)
#
#   Layout  (2 rows × 3 cols):
#   [0,0] GUV raw       [0,1] GUV + all masks   [0,2] GUV + filtered masks
#   [1,0] MT raw        [1,1] MT + all masks     [1,2] MT + circles (filtered)
# ============================================================

def plot_preprocessing_panel(
    guv_raw: np.ndarray,
    mt_raw: np.ndarray,
    guv_bg: np.ndarray,
    mt_bg: np.ndarray,
    guv_corr: np.ndarray,
    mt_corr: np.ndarray,
    guv_norm: np.ndarray,
    mt_norm: np.ndarray,
    image_index: int,
    title: str,
    out_path: str | Path,
    guv_vmax: float | None = None,
    mt_vmax: float | None = None,
) -> None:
    """
    4-row × 2-col preprocessing diagnostic panel saved per image.

    Row 0: raw GUV       | raw MT
    Row 1: background    | background
    Row 2: bg-subtracted | bg-subtracted
    Row 3: normalized    | normalized

    All rows 0-2 share the same display range per channel:
      vmin = 0, vmax = 99th percentile of the full stack (passed in).
    Row 3 (normalized) always uses clim=(0, 1).

    Parameters
    ----------
    guv_raw, mt_raw   : (H,W) float32 — raw (NaN→0) stacked image
    guv_bg, mt_bg     : (H,W) float32 — background images
    guv_corr, mt_corr : (H,W) float32 — after bg subtraction + clip
    guv_norm, mt_norm : (H,W) float32 — percentile-normalized [0,1]
    image_index       : int
    title             : suptitle string
    out_path          : full path to save PNG
    guv_vmax          : float — 99th percentile of full GUV stack (vmax for rows 0-2)
    mt_vmax           : float — 99th percentile of full MT stack (vmax for rows 0-2)
    """
    # fallback: compute from the raw image itself if not provided
    if guv_vmax is None:
        guv_vmax = float(np.nanpercentile(guv_raw, 99))
    if mt_vmax is None:
        mt_vmax = float(np.nanpercentile(mt_raw, 99))

    fig, axes = plt.subplots(4, 2, figsize=(10, 18))
    fig.suptitle(f"{title}\nImage {image_index:05d}", fontsize=10)

    pairs = [
        (guv_raw,   mt_raw,   "GUV raw",           "MT raw",           False),
        (guv_bg,    mt_bg,    "GUV background",     "MT background",    False),
        (guv_corr,  mt_corr,  "GUV bg-subtracted",  "MT bg-subtracted", False),
        (guv_norm,  mt_norm,  "GUV normalized",      "MT normalized",   True),
    ]

    for row_i, (limg, rimg, ltitle, rtitle, is_norm) in enumerate(pairs):
        for col_i, (img, t) in enumerate([(limg, ltitle), (rimg, rtitle)]):
            ax  = axes[row_i, col_i]
            arr = np.nan_to_num(np.asarray(img, dtype=np.float32), nan=0.0)

            if is_norm:
                # normalized row: always 0–1
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                # raw/bg/corrected rows: fixed range from full-stack percentiles
                vmax = guv_vmax if col_i == 0 else mt_vmax
                ax.imshow(arr, cmap="gray", vmin=0, vmax=vmax,
                          interpolation="nearest")

            ax.set_title(t, fontsize=9)
            nonzero = arr[arr > 0]
            stats_str = (
                f"min={arr.min():.1f}  max={arr.max():.1f}  "
                f"median={np.median(nonzero):.1f}"
                if nonzero.size > 0 else "all zeros"
            )
            ax.set_xlabel(stats_str, fontsize=7)
            ax.axis("off")

    fig.tight_layout()
    _save(fig, Path(out_path))


def plot_cellpose_panel(
    guv_norm: np.ndarray,
    mt_norm: np.ndarray,
    label_mask_all: np.ndarray,
    label_mask_filtered: np.ndarray,
    title: str,
    out_path: str | Path,
) -> None:
    """
    2×3 diagnostic panel matching the notebook fig2 layout.

    Parameters
    ----------
    guv_norm            : (H,W) float32, percentile-normalized GUV
    mt_norm             : (H,W) float32, percentile-normalized MT
    label_mask_all      : (H,W) int32, cellpose output before filtering
    label_mask_filtered : (H,W) int32, after eccentricity/area/MT filtering
    title               : figure suptitle (condition + date + filename)
    out_path            : full path to save PNG
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=11)

    overlay_all      = _make_label_overlay(label_mask_all)
    overlay_filtered = _make_label_overlay(label_mask_filtered)

    # row 0 — GUV
    axes[0, 0].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0, 0].set_title("GUV")

    axes[0, 1].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0, 1].imshow(overlay_all, interpolation="None")
    axes[0, 1].set_title("Segmentation")

    axes[0, 2].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0, 2].imshow(overlay_filtered, interpolation="None")
    axes[0, 2].set_title("Filtered")

    # row 1 — MT
    axes[1, 0].imshow(mt_norm, clim=(0, 1), cmap="gray")
    axes[1, 0].set_title("MT")

    axes[1, 1].imshow(mt_norm, clim=(0, 1), cmap="gray")
    axes[1, 1].imshow(overlay_all, interpolation="None")
    axes[1, 1].set_title("Segmentation")

    axes[1, 2].imshow(mt_norm, clim=(0, 1), cmap="gray")
    axes[1, 2].imshow(overlay_filtered, interpolation="None")
    axes[1, 2].set_title("Filtered")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, Path(out_path))


# ============================================================
#   PLOT B — Hough circle panel
#
#   Layout (1 row × 2 cols):
#   [0] GUV + all circles (color-coded by flag)
#   [1] GUV + good circles only
# ============================================================

FLAG_LABELS = {
    0: ("good",           "lime"),
    1: ("out of bounds",  "red"),
    2: ("radius range",   "orange"),
    3: ("no MT signal",   "magenta"),
    4: ("low MT std",     "cyan"),
    5: ("overlap small",  "yellow"),
    6: ("overlap med",    "white"),
}


def plot_hough_panel(
    guv_norm: np.ndarray,
    circles_all: np.ndarray,
    flags: np.ndarray,
    good_circles: np.ndarray,
    title: str,
    out_path: str | Path,
) -> None:
    """
    1×2 panel: all detected circles (color-coded by flag) and good only.

    Parameters
    ----------
    guv_norm     : (H,W) float32
    circles_all  : (N,3) float32 [x,y,r]
    flags        : (N,)  int32
    good_circles : (M,3) float32
    title        : figure suptitle
    out_path     : full path to save PNG
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=11)

    # left — all circles color-coded
    axes[0].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0].set_title(f"All Hough circles ({circles_all.shape[0]} total)")
    for k in range(circles_all.shape[0]):
        cx, cy, r = float(circles_all[k, 0]), float(circles_all[k, 1]), float(circles_all[k, 2])
        flag = int(flags[k]) if k < len(flags) else 0
        _, color = FLAG_LABELS.get(flag, ("unknown", "white"))
        axes[0].add_patch(
            mpatches.Circle((cx, cy), r, fill=False, edgecolor=color, linewidth=1.2)
        )
        axes[0].text(cx, cy, str(flag), color=color, fontsize=5,
                     ha="center", va="center")

    legend_handles = [
        mpatches.Patch(color=col, label=f"{code}: {lbl}")
        for code, (lbl, col) in FLAG_LABELS.items()
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=6, framealpha=0.5)

    # right — good circles only
    axes[1].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[1].set_title(f"Good circles only ({good_circles.shape[0]} accepted)")
    for k in range(good_circles.shape[0]):
        cx, cy, r = float(good_circles[k, 0]), float(good_circles[k, 1]), float(good_circles[k, 2])
        axes[1].add_patch(
            mpatches.Circle((cx, cy), r, fill=False, edgecolor="lime", linewidth=1.5)
        )
        axes[1].plot(cx, cy, "g+", markersize=5, markeredgewidth=1)

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, Path(out_path))


# ============================================================
#   PLOT C — Final accepted objects overlaid on MT
#   (matches notebook ax2[1,2] with circles drawn per object)
# ============================================================

def plot_final_objects(
    mt_norm: np.ndarray,
    objects: dict[str, Any],
    title: str,
    out_path: str | Path,
) -> None:
    """
    Show accepted objects overlaid on MT channel.
    Circles drawn for circle/combined; mask overlay for cellpose.

    Parameters
    ----------
    mt_norm  : (H,W) float32
    objects  : output of combine_segmentations()
    title    : figure suptitle
    out_path : full path to save PNG
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    fig.suptitle(title, fontsize=11)
    ax.imshow(mt_norm, clim=(0, 1), cmap="gray")
    ax.axis("off")

    method = objects.get("method", "unknown")
    rng = np.random.default_rng(seed=42)

    if "circles" in objects:
        circles = np.asarray(objects["circles"], dtype=np.float32)
        n = circles.shape[0]
        colors = rng.uniform(0.3, 1.0, size=(max(n, 1), 3))
        for k in range(n):
            cx, cy, r = float(circles[k, 0]), float(circles[k, 1]), float(circles[k, 2])
            color = tuple(colors[k])
            ax.add_patch(
                mpatches.Circle((cx, cy), r, fill=False,
                                edgecolor=color, linewidth=2.5)
            )
        ax.set_title(f"Final objects [{method}] — {n} circles")

    elif "masks" in objects:
        label_mask = np.asarray(objects["masks"], dtype=np.int32)
        overlay = _make_label_overlay(label_mask, alpha=0.5)
        ax.imshow(overlay, interpolation="None")

        try:
            from skimage.measure import regionprops
            for region in regionprops(label_mask):
                cy_r, cx_r = region.centroid
                r_est = region.equivalent_diameter_area / 2
                color = rng.uniform(0.3, 1.0, size=3)
                ax.add_patch(
                    mpatches.Circle((cx_r, cy_r), r_est, fill=False,
                                    edgecolor=tuple(color), linewidth=2.0)
                )
        except Exception:
            pass

        n = int(label_mask.max())
        ax.set_title(f"Final objects [{method}] — {n} masks")

    fig.tight_layout()
    _save(fig, Path(out_path))


# ============================================================
#   PLOT D — Per-object crop strip (matches notebook fig3)
#
#   One row of crops, one column per accepted object.
# ============================================================

def plot_object_crop_strip(
    patch_paths: list[Path],
    title: str,
    out_path: str | Path,
) -> None:
    """
    Horizontal strip of MT patch crops for one image (matches fig3).

    Parameters
    ----------
    patch_paths : list of Path to TIF files, one per accepted object
    title       : figure suptitle (condition + date + filename)
    out_path    : full path to save PNG
    """
    if not patch_paths:
        print(f"  [plot_object_crop_strip] no patches — skipping {out_path}")
        return

    n = len(patch_paths)
    fig, axes = plt.subplots(1, n, figsize=(max(12, n * 1.5), 4))
    fig.suptitle(title, fontsize=10)

    if n == 1:
        axes = [axes]

    for i, p in enumerate(patch_paths):
        patch = tifffile.imread(str(p)).astype(np.float32)
        axes[i].imshow(_norm(patch), cmap="gray", clim=(0, 1))
        axes[i].set_title(f"obj {i+1:02d}", fontsize=8)
        axes[i].axis("off")

    fig.tight_layout()
    _save(fig, Path(out_path))


# ============================================================
#   PLOT E — Per-object diagnostic panel (matches notebook fig4)
#
#   Layout (2 rows × 3 cols) saved per object:
#   [0,0] GUV raw        [0,1] GUV + filtered masks   [0,2] (cell index label)
#   [1,0] MT raw         [1,1] MT + circle overlay     [1,2] cropped patch
# ============================================================

def plot_per_object_panel(
    guv_norm: np.ndarray,
    mt_norm: np.ndarray,
    label_mask_filtered: np.ndarray,
    patch: np.ndarray,
    cx: int,
    cy: int,
    radius: float,
    cell_index: int,
    cell_id: int,
    title: str,
    out_path: str | Path,
) -> None:
    """
    Per-object 2×3 diagnostic panel matching notebook fig4.

    Parameters
    ----------
    guv_norm            : (H,W) float32 normalized GUV
    mt_norm             : (H,W) float32 normalized MT
    label_mask_filtered : (H,W) int32 filtered cellpose mask
    patch               : (96,96) float32 cropped MT patch
    cx, cy              : object centroid (image coords)
    radius              : estimated radius for circle overlay
    cell_index          : global cell counter
    cell_id             : label ID in this image
    title               : suptitle
    out_path            : full path to save PNG
    """
    rng = np.random.default_rng(seed=cell_id)
    color = tuple(rng.uniform(0.4, 1.0, size=3))

    overlay_filtered = _make_label_overlay(label_mask_filtered, alpha=0.7)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=10)

    # [0,0] GUV raw
    axes[0, 0].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0, 0].set_title("GUV")

    # [0,1] GUV + filtered masks
    axes[0, 1].imshow(guv_norm, clim=(0, 1), cmap="gray")
    axes[0, 1].imshow(overlay_filtered, interpolation="None")
    axes[0, 1].set_title("Filtered")

    # [0,2] cell index label only
    axes[0, 2].text(0.5, 0.5, f"All cell index\n{cell_index:04d}",
                    ha="center", va="center", fontsize=14,
                    transform=axes[0, 2].transAxes)
    axes[0, 2].set_title(f"All cell index={cell_index:04d}")

    # [1,0] MT raw
    axes[1, 0].imshow(mt_norm, clim=(0, 1), cmap="gray")
    axes[1, 0].set_title("MT")

    # [1,1] MT + circle overlay for this object
    axes[1, 1].imshow(mt_norm, clim=(0, 1), cmap="gray")
    axes[1, 1].add_patch(
        mpatches.Circle((cx, cy), radius, color=color, fill=False, linewidth=3)
    )
    axes[1, 1].set_title(f"In this image: Cell {cell_id:02d}")

    # [1,2] cropped patch
    axes[1, 2].imshow(_norm(patch), clim=(0, 1), cmap="gray")
    axes[1, 2].set_title("Cropped object")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    _save(fig, Path(out_path))
