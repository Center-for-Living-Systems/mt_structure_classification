"""
test_pipeline_steps1to4.py
==========================
End-to-end pytest tests for steps 1-4 of the pipeline, using the
bundled test data at test/Microtubule_GUV-Liu/.

Steps tested:
  1. Image pair indexing
  2. Background computation + channel statistics
  3. Segmentation (hough circles by default; cellpose with --runslow)
  4. Per-object 96x96 MT patch cropping

Run:
    pytest test/ -v
    pytest test/ -v --runslow   # include cellpose (needs model download)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── test data root (bundled in repo) ──
TEST_DATA_ROOT = Path(__file__).parent / "Microtubule_GUV-Liu"

# ── segmentation params ──
MT_STD_THRESHOLD = 15.0
MAX_ECCENTRICITY = 0.5
MIN_AREA = 1000
MAX_AREA = 40000
PATCH_SIZE = 96


# ============================================================
#   STEP 1 — IMAGE PAIR INDEXING
# ============================================================


class TestStep1Indexing:
    def test_finds_pairs(self):
        from mt_structure_classification.dataset.image_files_indexing import (
            build_pairs_dataframe_flexible,
        )

        df = build_pairs_dataframe_flexible(TEST_DATA_ROOT, output_debug_missing=True)

        assert len(df) == 5, f"Expected 5 image pairs, got {len(df)}"
        assert "GUV_folder_path" in df.columns
        assert "MT_folder_path" in df.columns
        assert "GUV_file_name" in df.columns
        assert "MT_file_name" in df.columns

        # every MT file should exist
        for _, row in df.iterrows():
            mt_path = Path(row["MT_folder_path"]) / row["MT_file_name"]
            assert mt_path.is_file(), f"MT file missing: {mt_path}"

    def test_condition_and_date_extracted(self):
        from mt_structure_classification.dataset.image_files_indexing import (
            build_pairs_dataframe_flexible,
        )

        df = build_pairs_dataframe_flexible(TEST_DATA_ROOT)

        # structure: Microtubule_GUV-Liu / 1_10 Tau_Tubulin / Date 1 / GUV/
        assert "condition" in df.columns
        assert "date" in df.columns
        assert df["date"].iloc[0] == "Date 1"
        assert df["condition"].iloc[0] == "1_10 Tau_Tubulin"


# ============================================================
#   STEP 2 — BACKGROUND + STATISTICS
# ============================================================


@pytest.fixture(scope="module")
def indexed_pairs():
    """Load the pairs dataframe once for all step 2+ tests."""
    from mt_structure_classification.dataset.image_files_indexing import (
        build_pairs_dataframe_flexible,
    )
    return build_pairs_dataframe_flexible(TEST_DATA_ROOT, output_debug_missing=False)


@pytest.fixture(scope="module")
def image_stacks(indexed_pairs):
    """Load GUV and MT stacks once for all step 2+ tests."""
    from mt_structure_classification.utils.image_processing import stack_pairs_to_arrays
    guv_stack, mt_stack = stack_pairs_to_arrays(
        indexed_pairs, target_shape=(512, 512), nan_for_zero=True,
    )
    return guv_stack, mt_stack


@pytest.fixture(scope="module")
def background_and_stats(image_stacks):
    """Compute backgrounds and channel statistics once."""
    from mt_structure_classification.utils.image_processing import (
        compute_background_median,
        compute_channel_statistics,
        remove_background_and_pad,
    )

    guv_stack, mt_stack = image_stacks

    guv_bg = compute_background_median(guv_stack, disk_radius=5)
    mt_bg = compute_background_median(mt_stack, disk_radius=5)
    stats = compute_channel_statistics(guv_stack, mt_stack)

    guv_corr, mt_corr = remove_background_and_pad(
        guv_stack, mt_stack,
        guv_bg=guv_bg, mt_bg=mt_bg,
        guv_bg_intensity=stats["guv"]["bg_intensity"],
        mt_bg_intensity=stats["mt"]["bg_intensity"],
    )

    return {
        "guv_bg": guv_bg,
        "mt_bg": mt_bg,
        "stats": stats,
        "guv_corr": guv_corr,
        "mt_corr": mt_corr,
    }


class TestStep2Background:
    def test_stacks_shape(self, image_stacks):
        guv_stack, mt_stack = image_stacks
        assert guv_stack.shape == (5, 512, 512)
        assert mt_stack.shape == (5, 512, 512)
        assert guv_stack.dtype == np.float32

    def test_background_shape(self, background_and_stats):
        guv_bg = background_and_stats["guv_bg"]
        mt_bg = background_and_stats["mt_bg"]
        assert guv_bg.shape == (512, 512)
        assert mt_bg.shape == (512, 512)

    def test_stats_keys(self, background_and_stats):
        stats = background_and_stats["stats"]
        for channel in ("guv", "mt"):
            assert channel in stats
            assert "bg_intensity" in stats[channel]
            assert "norm_low" in stats[channel]
            assert "norm_high" in stats[channel]
            assert "percentiles" in stats[channel]
            # bg_intensity should be a reasonable positive number
            assert stats[channel]["bg_intensity"] >= 0

    def test_background_subtracted_no_nans(self, background_and_stats):
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]
        assert not np.any(np.isnan(guv_corr)), "guv_corr has NaNs after bg subtraction"
        assert not np.any(np.isnan(mt_corr)), "mt_corr has NaNs after bg subtraction"

    def test_corrected_shape(self, background_and_stats):
        assert background_and_stats["guv_corr"].shape == (5, 512, 512)
        assert background_and_stats["mt_corr"].shape == (5, 512, 512)


# ============================================================
#   STEP 3 — SEGMENTATION (Hough circles — no GPU needed)
# ============================================================


class TestStep3HoughSegmentation:
    def test_hough_circles_runs(self, background_and_stats):
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            segment_guv_hough_circles,
            DEFAULT_HOUGH_SCALES,
        )

        stats = background_and_stats["stats"]
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]

        # normalize one image
        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]
        guv_norm = np.clip(
            (guv_corr[0] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)

        result = segment_guv_hough_circles(
            guv_norm,
            mt_img=mt_corr[0],
            mt_bg_int=stats["mt"]["bg_intensity"],
            mt_std_threshold=MT_STD_THRESHOLD,
            hough_scales=DEFAULT_HOUGH_SCALES,
        )

        assert "circles_all" in result
        assert "good_circles" in result
        assert "flags" in result
        assert result["circles_all"].ndim == 2
        if result["circles_all"].shape[0] > 0:
            assert result["circles_all"].shape[1] == 3  # x, y, r

    def test_combine_circle_only(self, background_and_stats):
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            segment_guv_hough_circles,
            combine_segmentations,
            DEFAULT_HOUGH_SCALES,
        )

        stats = background_and_stats["stats"]
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]

        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]
        guv_norm = np.clip(
            (guv_corr[0] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)

        hough_result = segment_guv_hough_circles(
            guv_norm,
            mt_img=mt_corr[0],
            mt_bg_int=stats["mt"]["bg_intensity"],
            mt_std_threshold=MT_STD_THRESHOLD,
            hough_scales=DEFAULT_HOUGH_SCALES,
        )

        objects = combine_segmentations(
            masks=None,
            circles=hough_result,
            method="circle",
        )

        assert objects["method"] == "circle"
        if hough_result["good_circles"].shape[0] > 0:
            assert "circles" in objects


# ============================================================
#   STEP 3b — SEGMENTATION (Cellpose — requires model download)
# ============================================================


@pytest.mark.slow
class TestStep3CellposeSegmentation:
    def test_cellpose_runs(self, background_and_stats):
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            segment_guv_cellpose,
        )

        stats = background_and_stats["stats"]
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]

        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]
        mt_1p = stats["mt"]["norm_low"]
        mt_99p = stats["mt"]["norm_high"]

        guv_norm = np.clip(
            (guv_corr[0] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)
        mt_norm = np.clip(
            (mt_corr[0] - mt_1p) / max(mt_99p - mt_1p, 1e-6), 0, 1
        ).astype(np.float32)

        label_mask_filtered, bad_flags = segment_guv_cellpose(
            guv_norm, mt_norm, mt_corr[0],
            model_type="cyto3",
            gpu=False,  # CPU for CI
            diameter=None,
            channels=[1, 2],
            mt_bg_int=stats["mt"]["bg_intensity"],
            mt_std_threshold=MT_STD_THRESHOLD,
            max_eccentricity=MAX_ECCENTRICITY,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
        )

        assert label_mask_filtered.shape == (512, 512)
        assert label_mask_filtered.dtype == np.int32


# ============================================================
#   STEP 4 — CROPPING
# ============================================================


class TestStep4Cropping:
    def test_crop_from_hough(self, background_and_stats, tmp_path, indexed_pairs):
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            segment_guv_hough_circles,
            combine_segmentations,
            crop_objects_from_masks_or_circles,
            DEFAULT_HOUGH_SCALES,
        )

        stats = background_and_stats["stats"]
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]

        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]
        guv_norm = np.clip(
            (guv_corr[0] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
        ).astype(np.float32)

        hough_result = segment_guv_hough_circles(
            guv_norm,
            mt_img=mt_corr[0],
            mt_bg_int=stats["mt"]["bg_intensity"],
            mt_std_threshold=MT_STD_THRESHOLD,
            hough_scales=DEFAULT_HOUGH_SCALES,
        )

        objects = combine_segmentations(
            masks=None,
            circles=hough_result,
            method="circle",
        )

        crops_dir = tmp_path / "crops"
        obj_df = crop_objects_from_masks_or_circles(
            objects=objects,
            mt_img=mt_corr[0],
            mt_bg_intensity=stats["mt"]["bg_intensity"],
            crops_dir=crops_dir,
            source_row=indexed_pairs.iloc[0],
            image_index=0,
            patch_size=PATCH_SIZE,
        )

        assert isinstance(obj_df, pd.DataFrame)

        if len(obj_df) > 0:
            # check patches were saved
            for fname in obj_df["filename"]:
                patch_path = crops_dir / fname
                assert patch_path.is_file(), f"Patch not saved: {patch_path}"

            # check patch dimensions
            import tifffile
            sample = tifffile.imread(str(crops_dir / obj_df["filename"].iloc[0]))
            assert sample.shape == (PATCH_SIZE, PATCH_SIZE)
            assert sample.dtype == np.float32

    def test_all_images_process(self, background_and_stats, tmp_path, indexed_pairs):
        """Run the full pipeline loop over all 5 test images."""
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            segment_guv_hough_circles,
            combine_segmentations,
            crop_objects_from_masks_or_circles,
            DEFAULT_HOUGH_SCALES,
        )

        stats = background_and_stats["stats"]
        guv_corr = background_and_stats["guv_corr"]
        mt_corr = background_and_stats["mt_corr"]
        crops_dir = tmp_path / "crops_all"

        guv_1p = stats["guv"]["norm_low"]
        guv_99p = stats["guv"]["norm_high"]

        all_dfs = []
        for i in range(len(indexed_pairs)):
            guv_norm = np.clip(
                (guv_corr[i] - guv_1p) / max(guv_99p - guv_1p, 1e-6), 0, 1
            ).astype(np.float32)

            hough_result = segment_guv_hough_circles(
                guv_norm,
                mt_img=mt_corr[i],
                mt_bg_int=stats["mt"]["bg_intensity"],
                mt_std_threshold=MT_STD_THRESHOLD,
                hough_scales=DEFAULT_HOUGH_SCALES,
            )

            objects = combine_segmentations(
                masks=None, circles=hough_result, method="circle",
            )

            obj_df = crop_objects_from_masks_or_circles(
                objects=objects,
                mt_img=mt_corr[i],
                mt_bg_intensity=stats["mt"]["bg_intensity"],
                crops_dir=crops_dir,
                source_row=indexed_pairs.iloc[i],
                image_index=i,
                patch_size=PATCH_SIZE,
            )
            all_dfs.append(obj_df)

        # should process all 5 images without error
        assert len(all_dfs) == 5

        # concatenate and check metadata
        combined = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)
        if len(combined) > 0:
            assert "filename" in combined.columns
            assert "cx" in combined.columns
            assert "cy" in combined.columns
            assert "image_index" in combined.columns
