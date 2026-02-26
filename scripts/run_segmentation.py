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

Quick test on first 10 images:
    python scripts/run_segmentation.py --preprocessed-dir ... --output-dir ... --max-images 10

GPU: Use --cellpose-gpu for faster Cellpose; requires a CUDA-enabled PyTorch (e.g. conda/pip env with CUDA).
Without CUDA, Cellpose falls back to CPU and is much slower (~70s vs ~1s per image).

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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from mt_structure_classification.core.GUV_mt_segmentation import (
    DEFAULT_CELLPOSE_DIAMETER,
    DEFAULT_HOUGH_SCALES,
    combine_segmentations,
    crop_objects_from_masks_or_circles,
    get_cellpose_model,
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
    cellpose_diameter: float | None = None,
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
    object_index = 0  # global counter for plot_per_object_panel

    # Timing accumulators (seconds)
    t_load_prep = 0.0
    t_hough = 0.0
    t_cellpose = 0.0
    t_combine_crop = 0.0
    t_plots = 0.0

    timing_log_path = out_metadata / "timing_log.txt"
    with open(timing_log_path, "w") as timing_f:
        timing_f.write(
            "image_index\tload_prep\hough\tcellpose\tcombine_crop\tplots\ttotal\tn_processed\tsec_per_image\tn_objects\n"
        )

    # Load Cellpose model once when needed (avoids ~100s load per image)
    cellpose_model_instance = None
    if method in ("cellpose", "combined"):
        print("Loading Cellpose model (once)...")
        cellpose_model_instance = get_cellpose_model(
            model_type=cellpose_model, gpu=cellpose_gpu
        )

    for i in tqdm(range(len(df)), desc="Processing images"):
        row = df.iloc[i]
        guv_path = Path(row["GUV_folder_path"]) / row["GUV_file_name"]
        mt_path = Path(row["MT_folder_path"]) / row["MT_file_name"]

        t0 = time.perf_counter()
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
        t_load_prep += time.perf_counter() - t0

        hough_result = None
        cellpose_masks = None

        if method in ("hough", "combined"):
            t0 = time.perf_counter()
            hough_result = segment_guv_hough_circles(
                guv_norm,
                mt_img=mt_corr,
                mt_bg_int=mt_bg_int,
                mt_std_threshold=MT_STD_THRESHOLD,
                hough_scales=DEFAULT_HOUGH_SCALES,
            )
            t_hough += time.perf_counter() - t0

        if method in ("cellpose", "combined"):
            t0 = time.perf_counter()
            cellpose_timing: dict[str, float] | None = {} if i == 0 else None
            cellpose_masks, _ = segment_guv_cellpose(
                guv_norm,
                mt_norm,
                mt_corr,
                model_type=cellpose_model,
                gpu=cellpose_gpu,
                diameter=cellpose_diameter,
                channels=[1, 2],
                mt_bg_int=mt_bg_int,
                mt_std_threshold=MT_STD_THRESHOLD,
                max_eccentricity=MAX_ECCENTRICITY,
                min_area=MIN_AREA,
                max_area=MAX_AREA,
                model=cellpose_model_instance,
                timing_out=cellpose_timing,
            )
            t_cellpose += time.perf_counter() - t0
            if cellpose_timing:
                with open(timing_log_path, "a") as timing_f:
                    timing_f.write(
                        f"  [Cellpose breakdown img0] eval={cellpose_timing.get('eval', 0):.2f}s "
                        f"filter+upsample={cellpose_timing.get('filter', 0):.2f}s\n"
                    )

        t0 = time.perf_counter()
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
        t_combine_crop += time.perf_counter() - t0
        all_objects.append(obj_df)

        n_processed = i + 1
        total_so_far = t_load_prep + t_hough + t_cellpose + t_combine_crop + t_plots
        with open(timing_log_path, "a") as timing_f:
            timing_f.write(
                f"{i}\t{t_load_prep:.2f}\t{t_hough:.2f}\t{t_cellpose:.2f}\t{t_combine_crop:.2f}\t{t_plots:.2f}\t"
                f"{total_so_far:.2f}\t{n_processed}\t{total_so_far / n_processed:.2f}\t{len(obj_df)}\n"
            )

        if save_plots and len(obj_df) > 0 and debug_dir is not None:
            t0 = time.perf_counter()
            from mt_structure_classification.utils.plotting_functions import (
                plot_cellpose_panel,
                plot_hough_panel,
                plot_final_objects,
                plot_object_crop_strip,
                plot_per_object_panel,
            )
            title = f"Image {i}"
            if cellpose_masks is not None:
                plot_cellpose_panel(
                    guv_norm, mt_norm,
                    cellpose_masks, cellpose_masks,
                    title=title,
                    out_path=debug_dir / f"img{i:05d}_A_cellpose.png",
                )
            if hough_result is not None:
                plot_hough_panel(
                    guv_norm,
                    hough_result["circles_all"],
                    hough_result["flags"],
                    hough_result["good_circles"],
                    title=title,
                    out_path=debug_dir / f"img{i:05d}_B_hough.png",
                )
            # Final accepted objects on MT channel
            plot_final_objects(
                mt_norm, objects, title=title,
                out_path=debug_dir / f"img{i:05d}_C_final_objects.png",
            )
            # Horizontal strip of crop patches for this image
            patch_paths = [crops_dir / fn for fn in obj_df["filename"]]
            plot_object_crop_strip(
                patch_paths, title=title,
                out_path=debug_dir / f"img{i:05d}_D_crop_strip.png",
            )
            # Per-object diagnostic panel
            label_mask_for_panel = (
                objects.get("masks")
                if objects.get("masks") is not None
                else np.zeros(guv_norm.shape, dtype=np.int32)
            )
            for _, row in obj_df.iterrows():
                patch = tifffile.imread(str(crops_dir / row["filename"])).astype(np.float32)
                plot_per_object_panel(
                    guv_norm,
                    mt_norm,
                    label_mask_for_panel,
                    patch,
                    cx=int(row["cx"]),
                    cy=int(row["cy"]),
                    radius=float(row["radius"]),
                    cell_index=object_index,
                    cell_id=int(row["object_index"]),
                    title=f"{title} Cell {int(row['object_index']):02d}",
                    out_path=debug_dir / f"img{i:05d}_obj{int(row['object_index']):03d}_E_panel.png",
                )
                object_index += 1
            t_plots += time.perf_counter() - t0
        else:
            object_index += len(obj_df)

    objects_df = pd.concat([d for d in all_objects if not d.empty], ignore_index=True)
    objects_csv = out_metadata / "objects_metadata.csv"
    objects_df.to_csv(objects_csv, index=False)

    n_img = len(df)
    total_wall = t_load_prep + t_hough + t_cellpose + t_combine_crop + t_plots

    print(f"\n  Total images processed: {len(df)}")
    print(f"  Total objects detected: {len(objects_df)}")
    print(f"  Crops: {crops_dir}")
    print(f"  Metadata: {objects_csv}")
    print(f"  Timing log: {timing_log_path}")
    if save_plots and debug_dir:
        print(f"  Debug plots: {debug_dir}")

    print("\n  --- Timing (total / per image) ---")
    print(f"  Load + prep:      {t_load_prep:7.2f}s  ({t_load_prep / n_img:.2f}s/img)")
    print(f"  Hough:            {t_hough:7.2f}s  ({t_hough / n_img:.2f}s/img)")
    print(f"  Cellpose:         {t_cellpose:7.2f}s  ({t_cellpose / n_img:.2f}s/img)")
    print(f"  Combine + crop:   {t_combine_crop:7.2f}s  ({t_combine_crop / n_img:.2f}s/img)")
    print(f"  Plotting:        {t_plots:7.2f}s  ({t_plots / n_img:.2f}s/img)")
    print(f"  Total (tracked):  {total_wall:7.2f}s  ({total_wall / n_img:.2f}s/img)")

    with open(timing_log_path, "a") as timing_f:
        timing_f.write("\n--- Final summary ---\n")
        timing_f.write(f"Load+prep:   {t_load_prep:.2f}s  ({t_load_prep / n_img:.2f}s/img)\n")
        timing_f.write(f"Hough:       {t_hough:.2f}s  ({t_hough / n_img:.2f}s/img)\n")
        timing_f.write(f"Cellpose:    {t_cellpose:.2f}s  ({t_cellpose / n_img:.2f}s/img)\n")
        timing_f.write(f"Combine+crop:{t_combine_crop:.2f}s  ({t_combine_crop / n_img:.2f}s/img)\n")
        timing_f.write(f"Plotting:    {t_plots:.2f}s  ({t_plots / n_img:.2f}s/img)\n")
        timing_f.write(f"Total:       {total_wall:.2f}s  ({total_wall / n_img:.2f}s/img)\n")

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
        help="Limit to first N images (e.g. 10 for a quick test; default: all)",
    )
    parser.add_argument("--cellpose-gpu", action="store_true", help="Use GPU for Cellpose")
    parser.add_argument("--cellpose-model", default="cyto3", help="Cellpose model (default: cyto3)")
    parser.add_argument(
        "--cellpose-diameter",
        type=float,
        default=None,
        metavar="PIX",
        help=f"Cell diameter in pixels (downsampled 256x256). Default {DEFAULT_CELLPOSE_DIAMETER}; set to avoid slow per-image estimation",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save debug panels to output debug/")
    args = parser.parse_args()

    diameter = args.cellpose_diameter if args.cellpose_diameter is not None else DEFAULT_CELLPOSE_DIAMETER

    run_segmentation(
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output_dir,
        method=args.method,
        cellpose_gpu=args.cellpose_gpu,
        cellpose_model=args.cellpose_model,
        cellpose_diameter=diameter,
        save_plots=args.save_plots,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
