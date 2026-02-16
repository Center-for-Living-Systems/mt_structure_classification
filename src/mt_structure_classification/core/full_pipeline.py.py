# src/mt_structure_classification/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

# ---------- local imports (package-internal) ----------
from mt_structure_classification.utils.filesystem import ensure_dir

# indexing / pairing + stacking + background
from mt_structure_classification.dataset.image_files_indexing import build_pairs_dataframe_flexible
from mt_structure_classification.utils.image_processing import (
    stack_pairs_to_arrays,
    compute_background_median,
    remove_background_and_pad,
)

# segmentation backends
from mt_structure_classification.utils.GUV_mt_segmentation import (
    segment_guv_cellpose,
    segment_guv_hough_circles,  # UPDATED usage (scales list)
    combine_segmentations,
    crop_objects_from_masks_or_circles,
)

# training + inference
from mt_structure_classification.core.train import train_classifier
from mt_structure_classification.core.predict import predict_on_folder


SegMethod = Literal["cellpose", "circle", "combined"]
Task = Literal["preprocess", "segment", "train", "predict", "all"]


# ======================================================
#                       CONFIGS
# ======================================================
@dataclass(frozen=True)
class PreprocessConfig:
    target_shape: tuple[int, int] = (512, 512)
    disk_radius: int = 5
    nan_for_zero: bool = True
    save_intermediates: bool = True

    # background-pad behavior
    pad: int = 64


@dataclass(frozen=True)
class HoughScale:
    """One HoughCircles pass (e.g. small/med/large)."""
    name: str
    minDist: int
    param1: int
    param2: int
    minRadius: int
    maxRadius: int


@dataclass(frozen=True)
class SegmentationConfig:
    method: SegMethod = "cellpose"

    # common
    sigma_smooth: float = 1.0
    save_debug_panels: bool = True

    # cellpose params (only used if method includes cellpose)
    cellpose_model: str = "cyto2"
    cellpose_diameter: Optional[float] = None
    cellpose_channels: tuple[int, int] = (0, 0)

    # circle/hough params (only used if method includes circle)
    hough_dp: float = 1.1
    hough_scales: tuple[HoughScale, ...] = (
        HoughScale("small", minDist=20, param1=30, param2=30, minRadius=10, maxRadius=30),
        HoughScale("med",   minDist=40, param1=25, param2=40, minRadius=30, maxRadius=60),
        HoughScale("large", minDist=60, param1=20, param2=60, minRadius=60, maxRadius=90),
    )

    # filtering rules you used (MT std checks etc.)
    mt_bg_int: float = 120.0
    mt_std_threshold: float = 15.0

    # optional extra filters you might implement later
    # e.g. reject circles too close to each other / duplicates
    dedup_center_dist_px: int = 15


@dataclass(frozen=True)
class TrainingConfig:
    labeled_csv: str | Path
    image_dir: str | Path

    model_name: str = "efficientnet_b0"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 30
    dropout: float = 0.3

    seed: int = 42
    val_split: float = 0.2


@dataclass(frozen=True)
class PredictConfig:
    model_ckpt: str | Path
    image_dir: str | Path
    output_csv: str | Path
    batch_size: int = 32


# ======================================================
#                PIPELINE: PREPROCESS
# ======================================================
def run_mt_guv_background_pipeline(
    root_folder: str | Path,
    output_folder: str | Path,
    dataset_name: str,
    preprocess: PreprocessConfig = PreprocessConfig(),
) -> dict[str, object]:
    """
    1) index pairs -> df
    2) stack images
    3) compute background images
    4) save outputs
    """
    out_root = ensure_dir(output_folder)
    processed_folder = ensure_dir(out_root / dataset_name / "processed_MT")

    df = build_pairs_dataframe_flexible(root_folder, output_debug_missing=True)

    guv_stack, mt_stack = stack_pairs_to_arrays(
        df,
        target_shape=preprocess.target_shape,
        nan_for_zero=preprocess.nan_for_zero,
    )

    guv_bg = compute_background_median(guv_stack, disk_radius=preprocess.disk_radius)
    mt_bg = compute_background_median(mt_stack, disk_radius=preprocess.disk_radius)

    if preprocess.save_intermediates:
        df.to_csv(processed_folder / "pairs_index.csv", index=False)
        np.save(processed_folder / "guv_bg.npy", guv_bg)
        np.save(processed_folder / "mt_bg.npy", mt_bg)

    return {
        "df": df,
        "guv_bg": guv_bg,
        "mt_bg": mt_bg,
        "processed_folder": str(processed_folder),
    }


# ======================================================
#                PIPELINE: SEGMENTATION
# ======================================================
def run_segmentation_pipeline(
    *,
    df: pd.DataFrame,
    output_folder: str | Path,
    dataset_name: str,
    seg: SegmentationConfig = SegmentationConfig(),
    preprocess: PreprocessConfig = PreprocessConfig(),
) -> dict[str, object]:
    """
    Produces:
      - per-object crops (png/tif)
      - per-object metadata CSV (centers/radius or mask regionprops)
      - optional debug panels
    """
    out_root = ensure_dir(output_folder)
    processed = ensure_dir(out_root / dataset_name / "processed_MT")

    seg_root = ensure_dir(processed / f"segmentation_{seg.method}")
    debug_dir = ensure_dir(seg_root / "debug") if seg.save_debug_panels else None
    crops_dir = ensure_dir(seg_root / "crops_png")
    crops_tif_dir = ensure_dir(seg_root / "crops_tif")
    meta_csv = seg_root / "objects.csv"

    # Load & stack images once
    guv_stack, mt_stack = stack_pairs_to_arrays(
        df,
        target_shape=preprocess.target_shape,
        nan_for_zero=preprocess.nan_for_zero,
    )

    # Backgrounds (compute here so segmentation is standalone)
    guv_bg = compute_background_median(guv_stack, disk_radius=preprocess.disk_radius)
    mt_bg = compute_background_median(mt_stack, disk_radius=preprocess.disk_radius)

    # Background removal / padding
    guv_pad, mt_pad = remove_background_and_pad(
        guv_stack,
        mt_stack,
        guv_bg=guv_bg,
        mt_bg=mt_bg,
        pad=preprocess.pad,
    )

    all_objects: list[pd.DataFrame] = []

    # --- per-image segmentation ---
    for i in range(len(df)):
        row = df.iloc[i]
        guv = guv_pad[i]
        mt = mt_pad[i]

        masks_cellpose = None
        circles = None

        if seg.method in ("cellpose", "combined"):
            masks_cellpose = segment_guv_cellpose(
                guv,
                model_type=seg.cellpose_model,
                diameter=seg.cellpose_diameter,
                channels=seg.cellpose_channels,
            )

        if seg.method in ("circle", "combined"):
            # UPDATED: pass the scale list instead of 3 separate parameter groups
            circles = segment_guv_hough_circles(
                guv,
                mt_img=mt,
                sigma_smooth=seg.sigma_smooth,
                dp=seg.hough_dp,
                scales=seg.hough_scales,  # <-- the big change
                mt_bg_int=seg.mt_bg_int,
                mt_std_threshold=seg.mt_std_threshold,
                dedup_center_dist_px=seg.dedup_center_dist_px,
            )

        objects = combine_segmentations(
            masks=masks_cellpose,
            circles=circles,
            method=seg.method,
        )

        obj_df = crop_objects_from_masks_or_circles(
            objects=objects,
            guv_img=guv,
            mt_img=mt,
            crops_dir=crops_dir,
            crops_tif_dir=crops_tif_dir,
            debug_dir=debug_dir,
            source_row=row,
            image_index=i,
        )
        all_objects.append(obj_df)

    objects_df = pd.concat(all_objects, ignore_index=True)
    objects_df.to_csv(meta_csv, index=False)

    return {
        "processed_folder": str(processed),
        "segmentation_folder": str(seg_root),
        "objects_df": objects_df,
        "objects_csv": str(meta_csv),
        "crops_dir": str(crops_dir),
    }


# ======================================================
#                PIPELINE: TRAINING
# ======================================================
def run_training_pipeline(
    *,
    training: TrainingConfig,
    output_folder: str | Path,
    experiment_name: str = "train_run",
) -> dict[str, object]:
    out_root = ensure_dir(output_folder)
    exp_dir = ensure_dir(out_root / experiment_name)

    result = train_classifier(
        labeled_csv=training.labeled_csv,
        image_dir=training.image_dir,
        out_dir=exp_dir,
        model_name=training.model_name,
        batch_size=training.batch_size,
        lr=training.lr,
        weight_decay=training.weight_decay,
        num_epochs=training.num_epochs,
        patience=training.patience,
        dropout=training.dropout,
        seed=training.seed,
        val_split=training.val_split,
    )
    return {"experiment_dir": str(exp_dir), **result}


# ======================================================
#                PIPELINE: PREDICTION
# ======================================================
def run_prediction_pipeline(
    *,
    pred: PredictConfig,
    output_folder: str | Path,
    run_name: str = "predict_run",
) -> dict[str, object]:
    out_root = ensure_dir(output_folder)
    run_dir = ensure_dir(out_root / run_name)

    out_csv = Path(pred.output_csv)
    if not out_csv.is_absolute():
        out_csv = run_dir / out_csv

    result = predict_on_folder(
        model_ckpt=pred.model_ckpt,
        image_dir=pred.image_dir,
        output_csv=out_csv,
        batch_size=pred.batch_size,
    )
    return {"run_dir": str(run_dir), "output_csv": str(out_csv), **result}


# ======================================================
#                  ONE-SHOT ORCHESTRATOR
# ======================================================
def run_full_pipeline(
    *,
    task: Task,
    root_folder: str | Path,
    output_folder: str | Path,
    dataset_name: str,
    seg: SegmentationConfig = SegmentationConfig(),
    preprocess: PreprocessConfig = PreprocessConfig(),
    training: Optional[TrainingConfig] = None,
    pred: Optional[PredictConfig] = None,
) -> dict[str, object]:
    results: dict[str, object] = {}

    if task in ("preprocess", "segment", "all"):
        preprocess_res = run_mt_guv_background_pipeline(
            root_folder=root_folder,
            output_folder=output_folder,
            dataset_name=dataset_name,
            preprocess=preprocess,
        )
        results["preprocess"] = preprocess_res

    if task in ("segment", "all"):
        df = results["preprocess"]["df"]  # type: ignore[index]
        seg_res = run_segmentation_pipeline(
            df=df,
            output_folder=output_folder,
            dataset_name=dataset_name,
            seg=seg,
            preprocess=preprocess,
        )
        results["segment"] = seg_res

    if task in ("train", "all"):
        if training is None:
            raise ValueError("training config is required for task='train' or 'all'.")
        train_res = run_training_pipeline(
            training=training,
            output_folder=output_folder,
            experiment_name=f"{dataset_name}_train",
        )
        results["train"] = train_res

    if task in ("predict", "all"):
        if pred is None:
            raise ValueError("pred config is required for task='predict' or 'all'.")
        pred_res = run_prediction_pipeline(
            pred=pred,
            output_folder=output_folder,
            run_name=f"{dataset_name}_predict",
        )
        results["predict"] = pred_res

    return results
