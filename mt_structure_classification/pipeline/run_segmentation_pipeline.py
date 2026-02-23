#!/usr/bin/env python
"""
run_segmentation_pipeline.py
=============================
Production script for Steps 1-4 of the MT structure classification pipeline:
  1. Index GUV/MT image pairs
  2. Compute background and channel statistics
  3. Segment GUVs (Hough circles and/or Cellpose)
  4. Crop 96x96 MT patches around each detected object

Usage:
    python run_segmentation_pipeline.py \
        --data-root /path/to/Microtubule_GUV-Liu \
        --output-dir results/experiment1 \
        --method combined \
        --max-images 100

Output:
    results/experiment1/
    ├── metadata/
    │   ├── pairs.csv                  # indexed image pairs
    │   ├── stats.json                 # channel statistics
    │   └── objects_metadata.csv       # all detected objects
    ├── crops/                         # 96x96 MT patches (TIFF)
    │   ├── condition_date_image_cell00.tif
    │   └── ...
    └── debug/                         # diagnostic plots (optional)
        ├── img00000_0_preprocessing.png
        ├── img00000_A_cellpose.png
        └── ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
#                      CONFIGURATION
# ============================================================

# Segmentation parameters
MT_STD_THRESHOLD = 15.0
MAX_ECCENTRICITY = 0.5
MIN_AREA = 1000
MAX_AREA = 40000
PATCH_SIZE = 96

# Hough circle detection scales
from mt_structure_classification.utils.GUV_mt_segmentation import DEFAULT_HOUGH_SCALES


# ============================================================
#                      MAIN PIPELINE
# ============================================================

def run_pipeline(
    data_root: Path,
    output_dir: Path,
    method: str = "combined",  # "hough", "cellpose", or "combined"
    max_images: int | None = None,
    cellpose_gpu: bool = False,
    cellpose_model: str = "cyto3",
    save_plots: bool = False,
):
    """
    Run the complete segmentation pipeline (steps 1-4).
    
    Parameters
    ----------
    data_root : Path
        Root directory containing condition/date/GUV+MT folders
    output_dir : Path
        Output directory for results
    method : str
        Segmentation method: "hough", "cellpose", or "combined"
    max_images : int | None
        Limit number of images to process (None = all)
    cellpose_gpu : bool
        Use GPU for Cellpose
    cellpose_model : str
        Cellpose model type
    save_plots : bool
        Save diagnostic plots
    """
    
    print("\n" + "="*70)
    print("MT STRUCTURE SEGMENTATION PIPELINE (Steps 1-4)")
    print("="*70)
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    
    if save_plots:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
    
    # ================================================================
    # STEP 1: INDEX IMAGE PAIRS
    # ================================================================
    print("\n[Step 1] Indexing GUV/MT image pairs...")
    from mt_structure_classification.dataset.image_files_indexing import (
        build_pairs_dataframe_flexible,
    )
    
    df = build_pairs_dataframe_flexible(data_root, output_debug_missing=True)
    print(f"  Found {len(df)} image pairs")
    
    pairs_csv = metadata_dir / "pairs.csv"
    df.to_csv(pairs_csv, index=False)
    print(f"  Saved to {pairs_csv}")
    
    # Limit images if requested
    if max_images is not None and len(df) > max_images:
        print(f"  Limiting to first {max_images} images")
        df = df.iloc[:max_images].copy()
    
    # ================================================================
    # STEP 2: LOAD STACKS & COMPUTE BACKGROUND
    # ================================================================
    print("\n[Step 2] Loading images and computing background...")
    from mt_structure_classification.utils.image_processing import (
        stack_pairs_to_arrays,
        compute_background_median,
        compute_channel_statistics,
        remove_background_and_pad,
    )
    
    # Load all images into stacks
    guv_stack, mt_stack = stack_pairs_to_arrays(
        df, target_shape=(512, 512), nan_for_zero=True,
    )
    print(f"  Loaded stacks: {guv_stack.shape}")
    
    # Compute background from full dataset
    print("  Computing background median (disk radius=5)...")
    guv_bg = compute_background_median(guv_stack, disk_radius=5)
    mt_bg = compute_background_median(mt_stack, disk_radius=5)
    
    # Compute channel statistics (percentiles for normalization)
    print("  Computing channel statistics...")
    stats = compute_channel_statistics(guv_stack, mt_stack)
    
    print(f"    GUV: bg={stats['guv']['bg_intensity']:.1f}, "
          f"norm=[{stats['guv']['norm_low']:.1f}, {stats['guv']['norm_high']:.1f}]")
    print(f"    MT:  bg={stats['mt']['bg_intensity']:.1f}, "
          f"norm=[{stats['mt']['norm_low']:.1f}, {stats['mt']['norm_high']:.1f}]")
    
    # Save statistics
    stats_path = metadata_dir / "stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved statistics to {stats_path}")
    
    # Background subtraction: image - bg_image + bg_intensity
    print("  Subtracting background...")
    guv_corr, mt_corr = remove_background_and_pad(
        guv_stack, mt_stack,
        guv_bg=guv_bg, mt_bg=mt_bg,
        guv_bg_intensity=stats["guv"]["bg_intensity"],
        mt_bg_intensity=stats["mt"]["bg_intensity"],
    )
    
    # ================================================================
    # STEP 3 & 4: SEGMENTATION + CROPPING (per image)
    # ================================================================
    print(f"\n[Step 3-4] Segmentation ({method}) + Cropping...")
    
    from mt_structure_classification.utils.GUV_mt_segmentation import (
        segment_guv_hough_circles,
        segment_guv_cellpose,
        combine_segmentations,
        crop_objects_from_masks_or_circles,
    )
    
    guv_1p = stats["guv"]["norm_low"]
    guv_99p = stats["guv"]["norm_high"]
    mt_1p = stats["mt"]["norm_low"]
    mt_99p = stats["mt"]["norm_high"]
    mt_bg_int = stats["mt"]["bg_intensity"]
    
    all_objects = []
    
    for i in tqdm(range(len(df)), desc="Processing images"):
        # Normalize current image
        guv_norm = np.clip(
            (guv_corr[i] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)
        mt_norm = np.clip(
            (mt_corr[i] - mt_1p) / max(mt_99p - mt_1p, 1e-6), 0, 1
        ).astype(np.float32)
        
        # Run segmentation based on method
        hough_result = None
        cellpose_masks = None
        
        if method in ("hough", "combined"):
            hough_result = segment_guv_hough_circles(
                guv_norm,
                mt_img=mt_corr[i],
                mt_bg_int=mt_bg_int,
                mt_std_threshold=MT_STD_THRESHOLD,
                hough_scales=DEFAULT_HOUGH_SCALES,
            )
        
        if method in ("cellpose", "combined"):
            cellpose_masks, _bad_flags = segment_guv_cellpose(
                guv_norm, mt_norm, mt_corr[i],
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
        
        # Combine segmentation results
        objects = combine_segmentations(
            masks=cellpose_masks,
            circles=hough_result,
            method=method,
        )
        
        # Crop 96x96 patches
        obj_df = crop_objects_from_masks_or_circles(
            objects=objects,
            mt_img=mt_corr[i],
            mt_bg_intensity=mt_bg_int,
            crops_dir=crops_dir,
            source_row=df.iloc[i],
            image_index=i,
            patch_size=PATCH_SIZE,
        )
        
        all_objects.append(obj_df)
        
        # Optional: save diagnostic plots
        if save_plots and len(obj_df) > 0:
            from mt_structure_classification.utils.plotting_functions import (
                plot_preprocessing_panel,
                plot_cellpose_panel,
                plot_hough_panel,
            )
            
            # Preprocessing panel
            plot_preprocessing_panel(
                guv_stack[i], mt_stack[i],
                guv_bg, mt_bg,
                guv_corr[i], mt_corr[i],
                guv_norm, mt_norm,
                save_path=debug_dir / f"img{i:05d}_0_preprocessing.png"
            )
            
            # Segmentation panels
            if cellpose_masks is not None:
                plot_cellpose_panel(
                    guv_norm, mt_norm,
                    cellpose_masks, cellpose_masks,  # all vs filtered
                    save_path=debug_dir / f"img{i:05d}_A_cellpose.png"
                )
            
            if hough_result is not None:
                plot_hough_panel(
                    guv_norm,
                    hough_result["circles_all"],
                    hough_result["flags"],
                    save_path=debug_dir / f"img{i:05d}_B_hough.png"
                )
    
    # ================================================================
    # SAVE FINAL METADATA
    # ================================================================
    print("\n[Summary] Saving results...")
    
    # Combine all object metadata
    objects_df = pd.concat([d for d in all_objects if not d.empty], ignore_index=True)
    objects_csv = metadata_dir / "objects_metadata.csv"
    objects_df.to_csv(objects_csv, index=False)
    
    print(f"\n  Total images processed: {len(df)}")
    print(f"  Total objects detected: {len(objects_df)}")
    print(f"  Crops saved to: {crops_dir}")
    print(f"  Metadata saved to: {metadata_dir}")
    
    if save_plots:
        print(f"  Diagnostic plots: {debug_dir}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70 + "\n")
    
    return objects_df


# ============================================================
#                      CLI INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run MT structure segmentation pipeline (steps 1-4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing condition/date/GUV+MT folders"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--method",
        choices=["hough", "cellpose", "combined"],
        default="combined",
        help="Segmentation method (default: combined)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images to process (default: all)"
    )
    parser.add_argument(
        "--cellpose-gpu",
        action="store_true",
        help="Use GPU for Cellpose"
    )
    parser.add_argument(
        "--cellpose-model",
        default="cyto3",
        help="Cellpose model type (default: cyto3)"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save diagnostic plots to debug/"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        method=args.method,
        max_images=args.max_images,
        cellpose_gpu=args.cellpose_gpu,
        cellpose_model=args.cellpose_model,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()
