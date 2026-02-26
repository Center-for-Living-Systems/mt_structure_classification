# Full pipeline orchestration (steps 1-6) with config dataclasses.
# For CLI entry points that reproduce paper results, use scripts in the repo:
#   scripts/run_preprocessing.py (steps 1-2), scripts/run_segmentation.py (steps 3-4)
#   scripts/run_training.py (step 5)
#   scripts/run_prediction.py (step 6)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mt_structure_classification.core.predict import predict_on_folder
from mt_structure_classification.core.train import TrainConfig, train_classifier
from mt_structure_classification.dataset.image_files_indexing import build_pairs_dataframe_flexible
from mt_structure_classification.utils.filesystem import ensure_dir
from mt_structure_classification.core.GUV_mt_segmentation import (
    DEFAULT_HOUGH_SCALES,
    combine_segmentations,
    crop_objects_from_masks_or_circles,
    get_cellpose_model,
    segment_guv_cellpose,
    segment_guv_hough_circles,
)
from mt_structure_classification.utils.image_processing import (
    compute_background_median,
    compute_channel_statistics,
    remove_background_and_pad,
    stack_pairs_to_arrays,
)

SegMethod = Literal["cellpose", "circle", "combined"]
Task = Literal["preprocess", "segment", "train", "predict", "all"]

PATCH_SIZE = 96


# ======================================================
#                       CONFIGS
# ======================================================
@dataclass(frozen=True)
class PreprocessConfig:
    target_shape: tuple[int, int] = (512, 512)
    disk_radius: int = 5
    nan_for_zero: bool = True
    save_intermediates: bool = True
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
    sigma_smooth: float = 1.0
    save_debug_panels: bool = True
    cellpose_model: str = "cyto2"
    cellpose_diameter: float | None = None
    cellpose_channels: tuple[int, int] = (0, 0)
    hough_dp: float = 1.1
    hough_scales: tuple[HoughScale, ...] = (
        HoughScale("small", minDist=20, param1=30, param2=30, minRadius=10, maxRadius=30),
        HoughScale("med",   minDist=40, param1=25, param2=40, minRadius=30, maxRadius=60),
        HoughScale("large", minDist=60, param1=20, param2=60, minRadius=60, maxRadius=90),
    )
    mt_bg_int: float = 120.0
    mt_std_threshold: float = 15.0
    dedup_center_dist_px: int = 15
    max_eccentricity: float = 0.5
    min_area: int = 1000
    max_area: int = 40000


@dataclass(frozen=True)
class TrainingConfig:
    labeled_csv: str | Path
    image_dir: str | Path
    model_name: str = "efficientnet"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    patience: int = 30
    num_workers: int = 4
    seed: int = 42


@dataclass(frozen=True)
class PredictConfig:
    model_ckpt: str | Path
    image_dir: str | Path
    output_csv: str | Path
    label_map_path: str | Path | None = None  # default: model_ckpt.parent / "label_map.json"
    batch_size: int = 32


_DEFAULT_PREPROCESS = PreprocessConfig()
_DEFAULT_SEGMENTATION = SegmentationConfig()


# ======================================================
#                PIPELINE: PREPROCESS
# ======================================================
def run_mt_guv_background_pipeline(
    root_folder: str | Path,
    output_folder: str | Path,
    dataset_name: str,
    preprocess: PreprocessConfig | None = None,
    target_shape: tuple[int, int] | None = None,
    disk_radius: int = 5,
    save_intermediates: bool = True,
) -> dict[str, object]:
    """
    Steps 1-2: Index pairs -> stack -> compute background images -> save.
    Accepts either preprocess config or legacy kwargs (target_shape, disk_radius, save_intermediates).
    """
    if preprocess is None:
        preprocess = PreprocessConfig(
            target_shape=target_shape if target_shape is not None else _DEFAULT_PREPROCESS.target_shape,
            disk_radius=disk_radius,
            save_intermediates=save_intermediates,
            nan_for_zero=_DEFAULT_PREPROCESS.nan_for_zero,
            pad=_DEFAULT_PREPROCESS.pad,
        )
    elif target_shape is not None:
        preprocess = PreprocessConfig(
            target_shape=target_shape,
            disk_radius=disk_radius,
            save_intermediates=save_intermediates,
            nan_for_zero=preprocess.nan_for_zero,
            pad=preprocess.pad,
        )
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
    seg: SegmentationConfig | None = None,
    preprocess: PreprocessConfig | None = None,
) -> dict[str, object]:
    """
    Steps 3-4: Segment GUVs (Hough and/or Cellpose) and crop 96x96 MT patches.
    Uses same APIs as scripts/run_preprocessing.py, scripts/run_segmentation.py and tests.
    """
    if seg is None:
        seg = _DEFAULT_SEGMENTATION
    if preprocess is None:
        preprocess = _DEFAULT_PREPROCESS
    out_root = ensure_dir(output_folder)
    processed = ensure_dir(out_root / dataset_name / "processed_MT")

    seg_root = ensure_dir(processed / f"segmentation_{seg.method}")
    crops_dir = ensure_dir(seg_root / "crops_tif")
    meta_csv = seg_root / "objects.csv"

    guv_stack, mt_stack = stack_pairs_to_arrays(
        df,
        target_shape=preprocess.target_shape,
        nan_for_zero=preprocess.nan_for_zero,
    )

    guv_bg = compute_background_median(guv_stack, disk_radius=preprocess.disk_radius)
    mt_bg = compute_background_median(mt_stack, disk_radius=preprocess.disk_radius)
    stats = compute_channel_statistics(guv_stack, mt_stack)
    mt_bg_intensity = float(stats["mt"]["bg_intensity"])

    guv_pad, mt_pad = remove_background_and_pad(
        guv_stack,
        mt_stack,
        guv_bg=guv_bg,
        mt_bg=mt_bg,
        guv_bg_intensity=stats["guv"]["bg_intensity"],
        mt_bg_intensity=mt_bg_intensity,
        pad=preprocess.pad,
    )

    all_objects: list[pd.DataFrame] = []
    guv_1p = stats["guv"]["norm_low"]
    guv_99p = stats["guv"]["norm_high"]
    mt_1p = stats["mt"]["norm_low"]
    mt_99p = stats["mt"]["norm_high"]

    cellpose_model_instance = None
    if seg.method in ("cellpose", "combined"):
        cellpose_model_instance = get_cellpose_model(
            model_type=seg.cellpose_model, gpu=False
        )

    for i in range(len(df)):
        row = df.iloc[i]
        guv = guv_pad[i]
        mt = mt_pad[i]

        guv_norm = np.clip(
            (guv - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)
        mt_norm = np.clip(
            (mt - mt_1p) / max(mt_99p - mt_1p, 1e-6), 0, 1
        ).astype(np.float32)

        masks_cellpose = None
        circles = None

        if seg.method in ("cellpose", "combined"):
            masks_cellpose, _ = segment_guv_cellpose(
                guv_norm,
                mt_norm,
                mt,
                model_type=seg.cellpose_model,
                gpu=False,
                diameter=seg.cellpose_diameter,
                channels=seg.cellpose_channels,
                mt_bg_int=seg.mt_bg_int,
                mt_std_threshold=seg.mt_std_threshold,
                max_eccentricity=seg.max_eccentricity,
                min_area=seg.min_area,
                max_area=seg.max_area,
                model=cellpose_model_instance,
            )

        if seg.method in ("circle", "combined"):
            circles = segment_guv_hough_circles(
                guv_norm,
                mt_img=mt,
                sigma_smooth=seg.sigma_smooth,
                dp=seg.hough_dp,
                hough_scales=DEFAULT_HOUGH_SCALES,
                mt_bg_int=seg.mt_bg_int,
                mt_std_threshold=seg.mt_std_threshold,
                overlap_dist_thresh=float(seg.dedup_center_dist_px),
            )

        objects = combine_segmentations(
            masks=masks_cellpose,
            circles=circles,
            method=seg.method,
        )

        obj_df = crop_objects_from_masks_or_circles(
            objects=objects,
            mt_img=mt,
            mt_bg_intensity=mt_bg_intensity,
            crops_dir=crops_dir,
            source_row=row,
            image_index=i,
            patch_size=PATCH_SIZE,
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
    """Step 5: Train classifier. Uses core.train.train_classifier API."""
    out_root = ensure_dir(output_folder)
    exp_dir = ensure_dir(out_root / experiment_name)

    train_cfg = TrainConfig(
        batch_size=training.batch_size,
        lr=training.lr,
        weight_decay=training.weight_decay,
        max_epochs=training.num_epochs,
        patience=training.patience,
        num_workers=training.num_workers,
        seed=training.seed,
    )
    ckpt = train_classifier(
        csv_path=training.labeled_csv,
        image_root=training.image_dir,
        out_dir=exp_dir,
        model_name=training.model_name,
        train_cfg=train_cfg,
    )
    return {"experiment_dir": str(exp_dir), "checkpoint": ckpt}


# ======================================================
#                PIPELINE: PREDICTION
# ======================================================
def run_prediction_pipeline(
    *,
    pred: PredictConfig,
    output_folder: str | Path,
    run_name: str = "predict_run",
) -> dict[str, object]:
    """Step 6: Predict on folder. Uses core.predict.predict_on_folder API."""
    out_root = ensure_dir(output_folder)
    run_dir = ensure_dir(out_root / run_name)

    out_csv = Path(pred.output_csv)
    if not out_csv.is_absolute():
        out_csv = run_dir / out_csv

    label_map = pred.label_map_path
    if label_map is None:
        label_map = Path(pred.model_ckpt).parent / "label_map.json"

    result = predict_on_folder(
        image_folder=pred.image_dir,
        model_path=pred.model_ckpt,
        label_map_path=label_map,
        out_csv=out_csv,
        batch_size=pred.batch_size,
    )
    return {"run_dir": str(run_dir), "output_csv": str(out_csv), "out_csv": result}


# ======================================================
#                  ONE-SHOT ORCHESTRATOR
# ======================================================
def run_full_pipeline(
    *,
    task: Task,
    root_folder: str | Path,
    output_folder: str | Path,
    dataset_name: str,
    seg: SegmentationConfig | None = None,
    preprocess: PreprocessConfig | None = None,
    training: TrainingConfig | None = None,
    pred: PredictConfig | None = None,
) -> dict[str, object]:
    if seg is None:
        seg = _DEFAULT_SEGMENTATION
    if preprocess is None:
        preprocess = _DEFAULT_PREPROCESS
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
