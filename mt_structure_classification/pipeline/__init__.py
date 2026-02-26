"""Pipeline orchestration: full_pipeline (config-based) and CLI scripts in scripts/."""

from mt_structure_classification.pipeline.full_pipeline import (
    PreprocessConfig,
    SegmentationConfig,
    TrainingConfig,
    PredictConfig,
    run_mt_guv_background_pipeline,
    run_segmentation_pipeline,
    run_training_pipeline,
    run_prediction_pipeline,
    run_full_pipeline,
)

__all__ = [
    "PreprocessConfig",
    "SegmentationConfig",
    "TrainingConfig",
    "PredictConfig",
    "run_mt_guv_background_pipeline",
    "run_segmentation_pipeline",
    "run_training_pipeline",
    "run_prediction_pipeline",
    "run_full_pipeline",
]
