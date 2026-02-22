from __future__ import annotations

from pathlib import Path

import numpy as np

from mt_structure_classification.dataset.image_files_indexing import build_pairs_dataframe_flexible
from mt_structure_classification.utils.filesystem import ensure_dir
from mt_structure_classification.utils.image_processing import (
    compute_background_median,
    stack_pairs_to_arrays,
)


def run_mt_guv_background_pipeline(
    root_folder: str | Path,
    output_folder: str | Path,
    dataset_name: str,
    target_shape: tuple[int, int] = (512, 512),
    disk_radius: int = 5,
    save_intermediates: bool = True,
) -> dict[str, object]:
    """
    High-level pipeline:
    1) index pairs -> df
    2) stack images
    3) compute background images
    4) save outputs
    """
    out_root = ensure_dir(output_folder)
    processed_folder = ensure_dir(out_root / dataset_name / "processed_MT")

    df = build_pairs_dataframe_flexible(root_folder, output_debug_missing=True)

    guv_stack, mt_stack = stack_pairs_to_arrays(df, target_shape=target_shape, nan_for_zero=True)

    guv_bg = compute_background_median(guv_stack, disk_radius=disk_radius)
    mt_bg  = compute_background_median(mt_stack,  disk_radius=disk_radius)

    if save_intermediates:
        df.to_csv(processed_folder / "pairs_index.csv", index=False)
        # save as numpy for exact reproducibility
        np.save(processed_folder / "guv_bg.npy", guv_bg)
        np.save(processed_folder / "mt_bg.npy", mt_bg)

    return {
        "df": df,
        "guv_bg": guv_bg,
        "mt_bg": mt_bg,
        "processed_folder": str(processed_folder),
    }




