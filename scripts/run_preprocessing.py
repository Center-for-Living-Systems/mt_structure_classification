#!/usr/bin/env python
"""
run_preprocessing.py — Steps 1–2: Index pairs + background & channel statistics
================================================================================
  1. Index GUV/MT image pairs
  2. Load stacks, compute background median and channel statistics

With --save-plots, writes preprocessing diagnostic panels to debug/.

Output is written to --output-dir for use by run_segmentation.py (steps 3–4).

Usage (from repo root, with package installed: pip install -e .):
    python scripts/run_preprocessing.py \\
        --data-root /path/to/Microtubule_GUV-Liu \\
        --output-dir results/preprocessed

Output:
    results/preprocessed/
    ├── metadata/
    │   ├── pairs.csv      # indexed image pairs
    │   ├── stats.json     # channel statistics (norm percentiles, bg intensity)
    │   ├── guv_bg.npy     # GUV background image (512×512)
    │   └── mt_bg.npy      # MT background image (512×512)
    └── debug/             # optional, with --save-plots (preprocessing panels)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mt_structure_classification.dataset.image_files_indexing import (
    build_pairs_dataframe_flexible,
)
from mt_structure_classification.utils.image_processing import (
    compute_background_median,
    compute_channel_statistics,
    remove_background_and_pad,
    stack_pairs_to_arrays,
)


def run_preprocessing(
    data_root: Path,
    output_dir: Path,
    target_shape: tuple[int, int] = (512, 512),
    disk_radius: int = 5,
    max_images: int | None = None,
    save_plots: bool = False,
    max_plot_images: int | None = 10,
) -> Path:
    """
    Run steps 1–2: index pairs, load stacks, compute background and stats.
    Returns the metadata directory path.
    """
    print("\n" + "=" * 70)
    print("MT PREPROCESSING (Steps 1–2)")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    # Step 1: index pairs
    print("\n[Step 1] Indexing GUV/MT image pairs...")
    df = build_pairs_dataframe_flexible(data_root, output_debug_missing=True)
    print(f"  Found {len(df)} image pairs")

    if len(df) == 0:
        raise SystemExit("No image pairs found. Check --data-root and folder layout (GUV/ and Microtubule/ as siblings).")

    if max_images is not None and len(df) > max_images:
        print(f"  Limiting to first {max_images} images")
        df = df.iloc[:max_images].copy()

    pairs_csv = metadata_dir / "pairs.csv"
    df.to_csv(pairs_csv, index=False)
    print(f"  Saved {pairs_csv}")

    # Step 2: load stacks, background, statistics
    print("\n[Step 2] Loading images and computing background + statistics...")
    guv_stack, mt_stack = stack_pairs_to_arrays(
        df, target_shape=target_shape, nan_for_zero=True
    )
    print(f"  Loaded stacks: {guv_stack.shape}")

    guv_bg = compute_background_median(guv_stack, disk_radius=disk_radius)
    mt_bg = compute_background_median(mt_stack, disk_radius=disk_radius)
    stats = compute_channel_statistics(guv_stack, mt_stack)

    print(
        f"    GUV: bg={stats['guv']['bg_intensity']:.1f}, "
        f"norm=[{stats['guv']['norm_low']:.1f}, {stats['guv']['norm_high']:.1f}]"
    )
    print(
        f"    MT:  bg={stats['mt']['bg_intensity']:.1f}, "
        f"norm=[{stats['mt']['norm_low']:.1f}, {stats['mt']['norm_high']:.1f}]"
    )

    stats_path = metadata_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved {stats_path}")

    np.save(metadata_dir / "guv_bg.npy", guv_bg)
    np.save(metadata_dir / "mt_bg.npy", mt_bg)
    print(f"  Saved guv_bg.npy, mt_bg.npy")

    if save_plots:
        print("  Saving preprocessing panels to debug/...")
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        guv_corr, mt_corr = remove_background_and_pad(
            guv_stack, mt_stack,
            guv_bg=guv_bg, mt_bg=mt_bg,
            guv_bg_intensity=stats["guv"]["bg_intensity"],
            mt_bg_intensity=stats["mt"]["bg_intensity"],
        )
        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]
        mt_1p = stats["mt"]["norm_low"]
        mt_99p = stats["mt"]["norm_high"]
        n_plot = len(df) if max_plot_images is None else min(len(df), max_plot_images)
        from mt_structure_classification.utils.plotting_functions import plot_preprocessing_panel
        for i in range(n_plot):
            guv_norm = np.clip(
                (guv_corr[i] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
            ).astype(np.float32)
            mt_norm = np.clip(
                (mt_corr[i] - mt_1p) / max(mt_99p - mt_1p, 1e-6), 0, 1
            ).astype(np.float32)
            title = f"Image {i}"
            plot_preprocessing_panel(
                np.nan_to_num(guv_stack[i], nan=0.0),
                np.nan_to_num(mt_stack[i], nan=0.0),
                guv_bg, mt_bg,
                guv_corr[i], mt_corr[i],
                guv_norm, mt_norm,
                image_index=i,
                title=title,
                out_path=debug_dir / f"img{i:05d}_0_preprocessing.png",
            )
        print(f"  Saved {n_plot} preprocessing panels to {debug_dir}")

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE. Run run_segmentation.py with --preprocessed-dir")
    print("=" * 70 + "\n")
    return metadata_dir


def main():
    parser = argparse.ArgumentParser(
        description="Steps 1–2: Index pairs and compute background/stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing condition/date/GUV+MT folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (metadata/ will be created inside)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of image pairs (default: all)",
    )
    parser.add_argument(
        "--disk-radius",
        type=int,
        default=5,
        help="Disk radius for background median filter (default: 5)",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save preprocessing diagnostic panels to output debug/",
    )
    parser.add_argument(
        "--max-plot-images",
        type=int,
        default=10,
        metavar="N",
        help="Max number of preprocessing panels to save when --save-plots (default: 10)",
    )
    args = parser.parse_args()

    run_preprocessing(
        data_root=args.data_root,
        output_dir=args.output_dir,
        disk_radius=args.disk_radius,
        max_images=args.max_images,
        save_plots=args.save_plots,
        max_plot_images=args.max_plot_images,
    )


if __name__ == "__main__":
    main()
