"""
test_pipeline_steps5to6.py
==========================
End-to-end pytest tests for steps 5-6 of the pipeline, using the
bundled test data at test/classifier_test_data/.

Steps tested:
  5. Classifier training (EfficientNet-based, small run to verify plumbing)
  6. Prediction on folder of crops

Test data:
  - CSV:    test/classifier_test_data/test_annotation_50samples.csv
            columns: filename, label
  - Images: test/classifier_test_data/images/

Run:
    pytest test/ -v
    pytest test/ -v --runslow   # include longer training run
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ── test data root ──
TEST_DATA_ROOT = Path(__file__).parent / "classifier_test_data"
TEST_CSV       = TEST_DATA_ROOT / "test_annotation_50samples.csv"
TEST_IMAGE_DIR = TEST_DATA_ROOT / "images"


# ============================================================
#   SHARED FIXTURES
# ============================================================


@pytest.fixture(scope="module")
def labeled_df():
    """Load the 50-sample annotation CSV once."""
    return pd.read_csv(TEST_CSV)


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    """
    Train a minimal 2-epoch model once and share the checkpoint
    across all Step 6 prediction tests.
    """
    from mt_structure_classification.core.train import TrainConfig, train_classifier

    out_dir = tmp_path_factory.mktemp("train_shared")
    ckpt = train_classifier(
        csv_path=TEST_CSV,
        image_root=TEST_IMAGE_DIR,
        out_dir=out_dir,
        model_name = "efficientnet",
        train_cfg=TrainConfig(
            max_epochs=2,
            patience=10,
            batch_size=8,
            num_workers=0,   # 0 workers avoids multiprocessing issues in pytest
            seed=42,
        ),
    )
    return ckpt


# ============================================================
#   STEP 5 — TRAINING
# ============================================================


class TestStep5Training:
    def test_csv_has_required_columns(self, labeled_df):
        assert "filename" in labeled_df.columns, "CSV missing 'filename' column"
        assert "label" in labeled_df.columns,    "CSV missing 'label' column"

    def test_csv_row_count(self, labeled_df):
        assert len(labeled_df) == 50, f"Expected 50 samples, got {len(labeled_df)}"

    def test_images_exist(self, labeled_df):
        missing = [
            fname for fname in labeled_df["filename"]
            if not (TEST_IMAGE_DIR / fname).is_file()
        ]
        assert len(missing) == 0, f"{len(missing)} image files missing: {missing[:5]}"

    def test_labels_not_empty(self, labeled_df):
        assert labeled_df["label"].notna().all(), "Some labels are NaN"
        assert labeled_df["label"].nunique() >= 2, "Need at least 2 classes to train"

    def test_train_returns_checkpoint_path(self, trained_checkpoint):
        """train_classifier should return a Path pointing to best_model.pth."""
        assert isinstance(trained_checkpoint, Path), \
            "train_classifier should return a Path"
        assert trained_checkpoint.is_file(), \
            f"Checkpoint not saved: {trained_checkpoint}"
        assert trained_checkpoint.suffix == ".pth", \
            "Checkpoint should be a .pth file"

    def test_train_saves_label_map(self, trained_checkpoint):
        """label_map.json should be saved alongside the checkpoint."""
        label_map_path = trained_checkpoint.parent / "label_map.json"
        assert label_map_path.is_file(), "label_map.json not saved during training"

        import json
        label_map = json.loads(label_map_path.read_text())
        assert len(label_map) >= 2, "label_map should have at least 2 classes"

    def test_train_saves_split_csvs(self, trained_checkpoint):
        """train_split.csv and val_split.csv should be saved."""
        out_dir = trained_checkpoint.parent
        assert (out_dir / "train_split.csv").is_file(), "train_split.csv not saved"
        assert (out_dir / "val_split.csv").is_file(),   "val_split.csv not saved"


# ============================================================
#   STEP 6 — PREDICTION
# ============================================================


class TestStep6Prediction:
    def test_predict_on_folder_runs(self, trained_checkpoint, tmp_path):
        from mt_structure_classification.core.predict import predict_on_folder

        out_csv = tmp_path / "predictions.csv"

        # Predict on all images in a folder
        predict_on_folder(
            image_folder=TEST_IMAGE_DIR,
            model_path=trained_checkpoint,
            label_map_path="models/label_map.json",
            out_csv=out_csv,
            batch_size=64,
        )

        assert out_csv.is_file(), "predict_on_folder should save a CSV"

        def test_prediction_csv_columns(self, trained_checkpoint, tmp_path):
            from mt_structure_classification.core.predict import predict_on_folder

            out_csv = tmp_path / "predictions.csv"
            
            predict_on_folder(
                image_folder=TEST_IMAGE_DIR,                              # ✅ folder with images
                model_ckpt=trained_checkpoint,                            # ✅ trained model
                label_map_path=trained_checkpoint.parent / "label_map.json",  # ✅ label map
                out_csv=out_csv,                                          # ✅ output CSV
                batch_size=8,
            )

            pred_df = pd.read_csv(out_csv)
            assert "filename" in pred_df.columns
            assert "pred_label" in pred_df.columns    # ✅ match your actual column name
            assert "pred_conf" in pred_df.columns     # ✅ add confidence check
            assert len(pred_df) > 0

        def test_prediction_covers_all_images(self, trained_checkpoint, tmp_path):
            from mt_structure_classification.core.predict import predict_on_folder

            out_csv = tmp_path / "predictions.csv"
            
            predict_on_folder(
                image_folder=TEST_IMAGE_DIR,                              
                model_ckpt=trained_checkpoint,                            
                label_map_path=trained_checkpoint.parent / "label_map.json", 
                out_csv=out_csv,                                          
                batch_size=8,
            )

            pred_df = pd.read_csv(out_csv)
            
            # Count actual images in folder
            from pathlib import Path
            num_images = len(list(Path(TEST_IMAGE_DIR).glob("*.tif")))
            
            assert len(pred_df) == num_images, \
                f"Expected {num_images} predictions, got {len(pred_df)}"