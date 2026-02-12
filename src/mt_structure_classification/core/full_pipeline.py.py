# src/mt_structure_classification/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------- local imports (package-internal) ----------
from mt_structure_classification.utils.filesystem import ensure_dir

# indexing / pairing + stacking + background
from mt_structure_classification.dataset.image_files_indexing import (
    build_pairs_dataframe_flexible,
)
from mt_structure_classification.utils.image_processing import (
    stack_pairs_to_arrays,
    compute_background_median,
    remove_background_and_pad,  # you likely want this helper (see note below)
)

# segmentation backends
from mt_structure_classification.utils.GUV_mt_segmentation import (
    segment_guv_cellpose,
    segment_guv_hough_circles,
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
    # these map well to what you had in the notebook (small/med/large groups)
    hough_dp: float = 1.1
    hough_minDist_small: int = 20
    hough_minDist_med: int = 40
    hough_minDist_large: int = 60

    hough_param1_small: int = 30
    hough_param2_small: int = 30
    hough_minRadius_small: int = 10
    hough_maxRadius_small: int = 30
    
    hough_param1_med: int = 25
    hough_param2_med: int = 40
    hough_minRadius_med: int = 30
    hough_maxRadius_med: int = 60
   
    hough_param1_large: int = 20
    hough_param2_large: int = 60
    hough_minRadius_large: int = 60
    hough_maxRadius_large: int = 90

    # filtering rules you used (MT std checks etc.)
    mt_bg_int: float = 120.0
    mt_std_threshold: float = 15.0


@dataclass(frozen=True)
class TrainingConfig:
    # dataset csv (filename,label)
    labeled_csv: str | Path

    # images root folder (the cropped object PNGs)
    image_dir: str | Path

    # core training choices
    model_name: str = "efficientnet_b0"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 30
    dropout: float = 0.3

    # misc
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

    Notes:
    - This pipeline assumes backgrounds were computed (or can be computed from df).
    - It supports 3 modes:
        seg.method="cellpose"  -> mask-based
        seg.method="circle"    -> circle-based (Hough)
        seg.method="combined"  -> merge cellpose mask + circles (you define strategy)
    """
    out_root = ensure_dir(output_folder)
    processed = ensure_dir(out_root / dataset_name / "processed_MT")

    # standard output locations (you can rename later)
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

    # Background removal / padding (your notebook did this; keep it as a single helper)
    # remove_background_and_pad should return padded arrays and bg-int estimates if needed.
    # If you don’t have it yet, implement it in utils/image_processing.py
    guv_pad, mt_pad = remove_background_and_pad(
        guv_stack,
        mt_stack,
        guv_bg=guv_bg,
        mt_bg=mt_bg,
        pad=64,
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
            circles = segment_guv_hough_circles(
                guv,
                sigma_smooth=seg.sigma_smooth,
                dp=seg.hough_dp,
                minDist_small=seg.hough_minDist_small,
                minDist_med=seg.hough_minDist_med,
                minDist_large=seg.hough_minDist_large,
                param1_small=seg.hough_param1_small,
                param2_small=seg.hough_param2_small,
                minRadius_small=seg.hough_minRadius_small,
                maxRadius_small=seg.hough_maxRadius_small,
                param1_med=seg.hough_param1_med,
                param2_med=seg.hough_param2_med,
                minRadius_med=seg.hough_minRadius_med,
                maxRadius_med=seg.hough_maxRadius_med,
                param1_large=seg.hough_param1_large,
                param2_large=seg.hough_param2_large,
                minRadius_large=seg.hough_minRadius_large,
                maxRadius_large=seg.hough_maxRadius_large,
                mt_img=mt,
                mt_bg_int=seg.mt_bg_int,
                mt_std_threshold=seg.mt_std_threshold,
            )

        # unify into a single representation used for cropping
        objects = combine_segmentations(
            masks=masks_cellpose,
            circles=circles,
            method=seg.method,
        )

        # crops + metadata rows
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
    """
    Thin wrapper around core.train.train_classifier().
    Keeps notebooks clean: you call this function and it handles outputs.
    """
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
    # train_classifier should return at least:
    # {"best_ckpt": "...pth", "label_map": {...}, "metrics": {...}, ...}
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
    """
    Convenience entry point for scripts/CLI.
    """
    results: dict[str, object] = {}

    if task in ("preprocess", "all", "segment", "train", "predict"):
        # indexing is needed for preprocess/segment (and sometimes for logging)
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
