#!/usr/bin/env python
"""
run_segmentation.py — Steps 3–4: Segment GUVs + crop 96×96 MT patches
========================================================================
  3. Segment GUVs (Hough circles and/or Cellpose)
  4. Crop 96×96 MT patches around each detected object

Requires output from run_preprocessing.py (steps 1–2): --preprocessed-dir must
contain metadata/pairs.csv, metadata/stats.json, metadata/guv_bg.npy, metadata/mt_bg.npy.

Usage (from repo root, with package installed: pip install -e .):
    python scripts/run_segmentation.py \\
        --preprocessed-dir results/preprocessed \\
        --output-dir results/segmentation \\
        --method combined

Output:
    results/segmentation/
    ├── metadata/
    │   └── objects_metadata.csv   # per-object metadata
    ├── crops/                     # 96×96 MT patches (TIFF)
    └── debug/                     # optional, with --save-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mt_structure_classification.core.GUV_mt_segmentation import (
    DEFAULT_HOUGH_SCALES,
    combine_segmentations,
    crop_objects_from_masks_or_circles,
    segment_guv_cellpose,
    segment_guv_hough_circles,
)
from mt_structure_classification.utils.image_processing import (
    load_tiff_2d_max,
    pad_or_crop_to_shape,
    subtract_background,
)

# Defaults matching pipeline
MT_STD_THRESHOLD = 15.0
MAX_ECCENTRICITY = 0.5
MIN_AREA = 1000
MAX_AREA = 40000
PATCH_SIZE = 96
TARGET_SHAPE = (512, 512)


def run_segmentation(
    preprocessed_dir: Path,
    output_dir: Path,
    method: str = "combined",
    cellpose_gpu: bool = False,
    cellpose_model: str = "cyto3",
    save_plots: bool = False,
    max_images: int | None = None,
) -> pd.DataFrame:
    """
    Run steps 3–4 using precomputed metadata from run_preprocessing.py.
    """
    print("\n" + "=" * 70)
    print("MT SEGMENTATION + CROPPING (Steps 3–4)")
    print("=" * 70)

    metadata_dir = preprocessed_dir / "metadata"
    if not metadata_dir.is_dir():
        raise SystemExit(
            f"Preprocessed metadata not found: {metadata_dir}\n"
            "Run run_preprocessing.py first with --output-dir pointing to the parent of metadata/."
        )

    pairs_csv = metadata_dir / "pairs.csv"
    stats_path = metadata_dir / "stats.json"
    guv_bg_path = metadata_dir / "guv_bg.npy"
    mt_bg_path = metadata_dir / "mt_bg.npy"
    for p, name in [
        (pairs_csv, "pairs.csv"),
        (stats_path, "stats.json"),
        (guv_bg_path, "guv_bg.npy"),
        (mt_bg_path, "mt_bg.npy"),
    ]:
        if not p.is_file():
            raise SystemExit(f"Missing {name} in {metadata_dir}")

    df = pd.read_csv(pairs_csv)
    with open(stats_path) as f:
        stats = json.load(f)
    guv_bg = np.load(guv_bg_path)
    mt_bg = np.load(mt_bg_path)

    guv_bg_int = float(stats["guv"]["bg_intensity"])
    mt_bg_int = float(stats["mt"]["bg_intensity"])
    guv_1p = stats["guv"]["norm_low"]
    guv_99p = stats["guv"]["norm_high"]
    mt_1p = stats["mt"]["norm_low"]
    mt_99p = stats["mt"]["norm_high"]

    if max_images is not None and len(df) > max_images:
        print(f"Limiting to first {max_images} images")
        df = df.iloc[:max_images].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_metadata = output_dir / "metadata"
    out_metadata.mkdir(exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    debug_dir = output_dir / "debug" if save_plots else None
    if debug_dir is not None:
        debug_dir.mkdir(exist_ok=True)

    print(f"\n[Step 3–4] Segmentation ({method}) + cropping ({len(df)} images)...")

    all_objects = []

    for i in tqdm(range(len(df)), desc="Processing images"):
        row = df.iloc[i]
        guv_path = Path(row["GUV_folder_path"]) / row["GUV_file_name"]
        mt_path = Path(row["MT_folder_path"]) / row["MT_file_name"]

        guv_img = load_tiff_2d_max(guv_path)
        mt_img = load_tiff_2d_max(mt_path)
        guv_img = pad_or_crop_to_shape(guv_img, TARGET_SHAPE)
        mt_img = pad_or_crop_to_shape(mt_img, TARGET_SHAPE)
        # Match stack behaviour: zeros -> nan before background subtract
        guv_img = guv_img.astype(np.float32)
        mt_img = mt_img.astype(np.float32)
        guv_img[guv_img == 0] = np.nan
        mt_img[mt_img == 0] = np.nan

        guv_corr = subtract_background(
            guv_img[np.newaxis, ...], guv_bg, bg_intensity=guv_bg_int
        )[0]
        mt_corr = subtract_background(
            mt_img[np.newaxis, ...], mt_bg, bg_intensity=mt_bg_int
        )[0]
        guv_corr = np.nan_to_num(guv_corr, nan=guv_bg_int)
        mt_corr = np.nan_to_num(mt_corr, nan=mt_bg_int)

        guv_norm = np.clip(
            (guv_corr - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)
        mt_norm = np.clip(
            (mt_corr - mt_1p) / max(mt_99p - mt_1p, 1e-6), 0, 1
        ).astype(np.float32)

        hough_result = None
        cellpose_masks = None

        if method in ("hough", "combined"):
            hough_result = segment_guv_hough_circles(
                guv_norm,
                mt_img=mt_corr,
                mt_bg_int=mt_bg_int,
                mt_std_threshold=MT_STD_THRESHOLD,
                hough_scales=DEFAULT_HOUGH_SCALES,
            )

        if method in ("cellpose", "combined"):
            cellpose_masks, _ = segment_guv_cellpose(
                guv_norm,
                mt_norm,
                mt_corr,
                model_type=cellpose_model,
                gpu=cellpose_gpu,
                diameter=None,
                channels=[1, 2],
                mt_bg_int=mt_bg_int,
                mt_std_threshold=MT_STD_THRESHOLD,
                max_eccentricity=MAX_ECCENTRICITY,
                min_area=MIN_AREA,
                max_area=MAX_AREA,
            )

        objects = combine_segmentations(
            masks=cellpose_masks,
            circles=hough_result,
            method=method,
        )

        obj_df = crop_objects_from_masks_or_circles(
            objects=objects,
            mt_img=mt_corr,
            mt_bg_intensity=mt_bg_int,
            crops_dir=crops_dir,
            source_row=row,
            image_index=i,
            patch_size=PATCH_SIZE,
        )
        all_objects.append(obj_df)

        if save_plots and len(obj_df) > 0 and debug_dir is not None:
            from mt_structure_classification.utils.plotting_functions import (
                plot_preprocessing_panel,
                plot_cellpose_panel,
                plot_hough_panel,
            )
            plot_preprocessing_panel(
                guv_img, mt_img,
                guv_bg, mt_bg,
                guv_corr, mt_corr,
                guv_norm, mt_norm,
                save_path=debug_dir / f"img{i:05d}_0_preprocessing.png",
            )
            if cellpose_masks is not None:
                plot_cellpose_panel(
                    guv_norm, mt_norm,
                    cellpose_masks, cellpose_masks,
                    save_path=debug_dir / f"img{i:05d}_A_cellpose.png",
                )
            if hough_result is not None:
                plot_hough_panel(
                    guv_norm,
                    hough_result["circles_all"],
                    hough_result["flags"],
                    save_path=debug_dir / f"img{i:05d}_B_hough.png",
                )

    objects_df = pd.concat([d for d in all_objects if not d.empty], ignore_index=True)
    objects_csv = out_metadata / "objects_metadata.csv"
    objects_df.to_csv(objects_csv, index=False)

    print(f"\n  Total images processed: {len(df)}")
    print(f"  Total objects detected: {len(objects_df)}")
    print(f"  Crops: {crops_dir}")
    print(f"  Metadata: {objects_csv}")
    if save_plots and debug_dir:
        print(f"  Debug plots: {debug_dir}")
    print("\n" + "=" * 70 + "\n")
    return objects_df


def main():
    parser = argparse.ArgumentParser(
        description="Steps 3–4: Segment GUVs and crop MT patches (requires run_preprocessing.py output)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        required=True,
        help="Directory containing metadata/ from run_preprocessing.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for crops and objects_metadata.csv",
    )
    parser.add_argument(
        "--method",
        choices=["hough", "cellpose", "combined"],
        default="combined",
        help="Segmentation method (default: combined)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images (default: all)",
    )
    parser.add_argument("--cellpose-gpu", action="store_true", help="Use GPU for Cellpose")
    parser.add_argument("--cellpose-model", default="cyto3", help="Cellpose model (default: cyto3)")
    parser.add_argument("--save-plots", action="store_true", help="Save debug panels to output debug/")
    args = parser.parse_args()

    run_segmentation(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        method=args.method,
        cellpose_gpu=args.cellpose_gpu,
        cellpose_model=args.cellpose_model,
        save_plots=args.save_plots,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
