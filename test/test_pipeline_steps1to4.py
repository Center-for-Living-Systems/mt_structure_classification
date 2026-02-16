"""
test_pipeline_steps1to4.py
==========================
End-to-end test for steps 1-4:
  1. Image pair indexing
  2. Background computation + channel statistics
  3. Segmentation (cellpose / circle / combined)
  4. Per-object 96x96 MT patch cropping → TIF

Diagnostic plots saved to <OUTPUT_FOLDER>/<DATASET_NAME>/debug/

Usage
-----
Edit the CONFIG section, then run:
    python test_pipeline_steps1to4.py
Or in a notebook:
    %run test_pipeline_steps1to4.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
#                   CONFIG — edit these
# ============================================================

ROOT_FOLDER   = Path("/mnt/d/lding/CLS/mousumiLiuDinner/raw_data/Microtubule_GUV-Liu-20250106T211105Z-001")
OUTPUT_FOLDER = Path("/mnt/d/lding/CLS/mousumiLiuDinner/set1to5_processed_results/Microtubule_GUV-Liu-20250106T211105Z-001/test20260214")
DATASET_NAME  = "test_step1_4_combined_run"

# Segmentation method: "cellpose", "circle", or "combined"
SEG_METHOD = "combined"

# Cellpose
CELLPOSE_MODEL    = "cyto3"
CELLPOSE_GPU      = True
CELLPOSE_DIAMETER = None   # None = auto-estimate

# Shared MT filter thresholds
# MT_BG_INT is computed from data (1st percentile of raw MT stack) — see Step 2
MT_STD_THRESHOLD = 15.0

# Cellpose-specific filters (regionprops)
MAX_ECCENTRICITY = 0.5
MIN_AREA         = 1000
MAX_AREA         = 40000

# Hough
HOUGH_IOT_THRESHOLD = 0.5  # IoU threshold for combined matching

# Patch
PATCH_SIZE = 96

# Limit to first N images for quick test (None = run all)
MAX_IMAGES = 385

# ============================================================
#                     IMPORTS
# ============================================================

from mt_structure_classification.dataset.image_files_indexing import (
    build_pairs_dataframe_flexible,
)
from mt_structure_classification.utils.image_processing import (
    stack_pairs_to_arrays,
    compute_background_median,
    compute_background_intensity,
    compute_channel_statistics,
    remove_background_and_pad,
)
from mt_structure_classification.utils.GUV_mt_segmentation import (
    segment_guv_cellpose,
    segment_guv_hough_circles,
    combine_segmentations,
    crop_objects_from_masks_or_circles,
    DEFAULT_HOUGH_SCALES,
    percentile_normalize,
)
from mt_structure_classification.utils.plotting_functions import (
    plot_cellpose_panel,
    plot_hough_panel,
    plot_final_objects,
    plot_object_crop_strip,
    plot_per_object_panel,
    plot_preprocessing_panel,
)
from mt_structure_classification.utils.filesystem import ensure_dir


# ============================================================
#                   SETUP DIRECTORIES
# ============================================================

out_root  = ensure_dir(Path(OUTPUT_FOLDER) / DATASET_NAME)
crops_dir = ensure_dir(out_root / "crops_tif")
debug_dir = ensure_dir(out_root / "debug")
meta_csv  = out_root / "objects.csv"

print(f"\nOutput root : {out_root}")
print(f"Crops dir   : {crops_dir}")
print(f"Debug dir   : {debug_dir}")


# ============================================================
#   STEP 1 — IMAGE PAIR INDEXING
# ============================================================

print("\n" + "="*60)
print("Step 1: Image pair indexing")
print("="*60)

df = build_pairs_dataframe_flexible(ROOT_FOLDER, output_debug_missing=True)
print(f"Found {len(df)} image pairs")
print(df.head())

if len(df) == 0:
    raise RuntimeError(
        f"No image pairs found under {ROOT_FOLDER}.\n"
        "Check GUV/ and Microtubule/ subdirectories exist "
        "and filenames match PairingRules."
    )

# NOTE: do NOT limit df here — background + stats must use the full dataset


# ============================================================
#   STEP 2 — BACKGROUND + STATISTICS (full dataset)
# ============================================================

print("\n" + "="*60)
print("Step 2: Background computation + channel statistics")
print("="*60)

guv_stack, mt_stack = stack_pairs_to_arrays(
    df, target_shape=(512, 512), nan_for_zero=True,
)
print(f"Stacks loaded: guv={guv_stack.shape}, mt={mt_stack.shape}")

guv_bg = compute_background_median(guv_stack, disk_radius=5)
mt_bg  = compute_background_median(mt_stack,  disk_radius=5)
print("Background images computed")

stats = compute_channel_statistics(guv_stack, mt_stack)
print("\nChannel statistics:")
print(f"  GUV bg_intensity : {stats['guv']['bg_intensity']:.2f}")
print(f"  MT  bg_intensity : {stats['mt']['bg_intensity']:.2f}")
print("  MT percentiles:")
for p, v in stats["mt"]["percentiles"].items():
    print(f"    p{p:5.1f} = {v:.2f}")

# save stats for reference
import json
(out_root / "channel_stats.json").write_text(
    json.dumps({k: {kk: float(vv) if not isinstance(vv, dict)
                    else {str(kkk): float(vvv) for kkk, vvv in vv.items()}
                    for kk, vv in v.items()}
                for k, v in stats.items()}, indent=2)
)

# background-subtracted stacks — formula: image - bg_image + bg_intensity
# bg_offset = 1st percentile of raw stack (used as additive offset + crop fill value)
guv_bg_offset = stats["guv"]["bg_intensity"]   # 1st percentile of GUV stack
mt_bg_offset  = stats["mt"]["bg_intensity"]    # 1st percentile of MT stack

guv_corr, mt_corr = remove_background_and_pad(
    guv_stack, mt_stack,
    guv_bg=guv_bg, mt_bg=mt_bg,
    guv_bg_intensity=guv_bg_offset,
    mt_bg_intensity=mt_bg_offset,
)
print(f"Background subtracted  (guv_bg_offset={guv_bg_offset:.1f}, mt_bg_offset={mt_bg_offset:.1f})")

# normalization range from smoothed stack (0.001 / 99.95 percentile, nonzero only)
# matches notebook: guv_1p / guv_99p computed on gaussian-smoothed images
guv_1p  = stats["guv"]["norm_low"]    # 0.001 percentile of smoothed GUV
guv_99p = stats["guv"]["norm_high"]   # 99.95 percentile of smoothed GUV
mt_1p   = stats["mt"]["norm_low"]     # 0.001 percentile of smoothed MT
mt_99p  = stats["mt"]["norm_high"]    # 99.95 percentile of smoothed MT
print(f"  Normalization range — GUV: [{guv_1p:.1f}, {guv_99p:.1f}]  MT: [{mt_1p:.1f}, {mt_99p:.1f}]")
# guv_bg_offset / mt_bg_offset = 1st percentile of raw stack (set above)

# NOW limit to MAX_IMAGES for the segmentation/cropping loop
if MAX_IMAGES is not None:
    df_loop = df.head(MAX_IMAGES).reset_index(drop=True)
    guv_corr_loop = guv_corr[:MAX_IMAGES]
    mt_corr_loop  = mt_corr[:MAX_IMAGES]
    guv_stack_loop = guv_stack[:MAX_IMAGES]
    mt_stack_loop  = mt_stack[:MAX_IMAGES]
    print(f"\n[test] Segmentation limited to first {MAX_IMAGES} images")
else:
    df_loop        = df
    guv_corr_loop  = guv_corr
    mt_corr_loop   = mt_corr
    guv_stack_loop = guv_stack
    mt_stack_loop  = mt_stack
# mt_bg_offset (1st percentile) is used for both filtering and crop fill


# ============================================================
#   STEPS 3 + 4 — SEGMENTATION + CROPPING (per image)
# ============================================================

print("\n" + "="*60)
print(f"Steps 3+4: Segmentation [{SEG_METHOD}] + cropping")
print("="*60)

all_objects: list[pd.DataFrame] = []
cell_index = 0  # global object counter across all images

for i in range(len(df_loop)):
    row       = df_loop.iloc[i]
    title_str = f"{row.get('condition','?')} | {row.get('date','?')} | {row.get('GUV_file_name','?')}"
    print(f"\n--- Image {i:05d}: {title_str}")

    guv_raw = guv_corr_loop[i]
    mt_raw  = mt_corr_loop[i]

    # ---- normalize for cellpose + display (matches notebook) ----
    guv_norm = (guv_raw - guv_1p) / max(guv_99p - guv_1p, 1e-6)
    guv_norm = np.clip(guv_norm, 0, 1).astype(np.float32)

    mt_norm  = (mt_raw - mt_1p) / max(mt_99p - mt_1p, 1e-6)
    mt_norm  = np.clip(mt_norm, 0, 1).astype(np.float32)

    # ---- Plot 0 — preprocessing diagnostic (raw → bg → subtracted → norm) ----
    plot_preprocessing_panel(
        guv_raw=guv_stack_loop[i],
        mt_raw=mt_stack_loop[i],
        guv_bg=guv_bg,
        mt_bg=mt_bg,
        guv_corr=guv_raw,
        mt_corr=mt_raw,
        guv_norm=guv_norm,
        mt_norm=mt_norm,
        image_index=i,
        title=title_str,
        out_path=debug_dir / f"img{i:05d}_0_preprocessing.png",
        guv_vmax=guv_99p,   # 99th percentile of full GUV stack
        mt_vmax=mt_99p,     # 99th percentile of full MT stack
    )

    # ---- cellpose ----
    label_mask_all      = None
    label_mask_filtered = None
    bad_flags_cp        = None

    if SEG_METHOD in ("cellpose", "combined"):
        print("  Running Cellpose...")
        label_mask_filtered, bad_flags_cp = segment_guv_cellpose(
            guv_norm, mt_norm, mt_raw,  # norm for cellpose, raw for filtering
            model_type=CELLPOSE_MODEL,
            gpu=CELLPOSE_GPU,
            diameter=CELLPOSE_DIAMETER,
            channels=[1, 2],
            mt_bg_int=mt_bg_offset,  # 1st percentile of raw MT stack
            mt_std_threshold=MT_STD_THRESHOLD,
            max_eccentricity=MAX_ECCENTRICITY,
            min_area=MIN_AREA,
            max_area=MAX_AREA,
        )
        # re-run without filtering to get "all masks" for the panel
        # (we want to show before/after filtering)
        from mt_structure_classification.utils.GUV_mt_segmentation import (
            get_cellpose_model, _upsample_labels,
        )
        import numpy as _np
        _h, _w = guv_norm.shape
        _img = _np.zeros((2, _h, _w), dtype=_np.float32)
        _img[0], _img[1] = guv_norm, mt_norm
        _model = get_cellpose_model(CELLPOSE_MODEL, CELLPOSE_GPU)
        _masks_small, _, _, _ = _model.eval(
            _img[:, ::2, ::2], diameter=CELLPOSE_DIAMETER, channels=[1, 2],
        )
        label_mask_all = _upsample_labels(_masks_small, _h, _w)

        n_good = int(label_mask_filtered.max())
        print(f"  Cellpose: {int(label_mask_all.max())} detected → {n_good} after filtering")

        # Plot A — cellpose panel
        plot_cellpose_panel(
            guv_norm, mt_norm,
            label_mask_all, label_mask_filtered,
            title=title_str,
            out_path=debug_dir / f"img{i:05d}_A_cellpose.png",
        )

    # ---- hough circles ----
    hough_result = None

    if SEG_METHOD in ("circle", "combined"):
        print("  Running Hough circles...")
        hough_result = segment_guv_hough_circles(
            guv_norm,           # normalized GUV for Hough circle detection
            mt_img=mt_raw,      # raw MT for threshold filtering (bg_int/std in raw units)
            mt_bg_int=mt_bg_offset,  # 1st percentile of raw MT stack
            mt_std_threshold=MT_STD_THRESHOLD,
            hough_scales=DEFAULT_HOUGH_SCALES,
        )
        n_all  = hough_result["circles_all"].shape[0]
        n_good = hough_result["good_circles"].shape[0]
        print(f"  Hough: {n_all} detected → {n_good} good")

        # Plot B — hough panel
        plot_hough_panel(
            hough_result["guv_norm"],
            hough_result["circles_all"],
            hough_result["flags"],
            hough_result["good_circles"],
            title=title_str,
            out_path=debug_dir / f"img{i:05d}_B_hough.png",
        )

    # ---- combine ----
    objects = combine_segmentations(
        masks=label_mask_filtered,
        circles=hough_result,
        method=SEG_METHOD,
        iou_threshold=HOUGH_IOT_THRESHOLD,
    )

    # Plot C — final objects on MT
    plot_final_objects(
        mt_norm, objects,
        title=title_str,
        out_path=debug_dir / f"img{i:05d}_C_final_objects.png",
    )

    # ---- crop patches ----
    obj_df = crop_objects_from_masks_or_circles(
        objects=objects,
        mt_img=mt_raw,           # raw float32 for TIF saving
        mt_bg_intensity=mt_bg_offset,  # 1st percentile offset for crop fill
        crops_dir=crops_dir,
        source_row=row,
        image_index=i,
        patch_size=PATCH_SIZE,
    )

    if obj_df.empty:
        print(f"  No objects — skipping plots for image {i:05d}")
        all_objects.append(obj_df)
        continue

    print(f"  Cropped {len(obj_df)} patches")

    # Plot D — crop strip (fig3 equivalent)
    patch_paths = [crops_dir / fn for fn in obj_df["filename"].tolist()]
    plot_object_crop_strip(
        patch_paths,
        title=title_str,
        out_path=debug_dir / f"img{i:05d}_D_crop_strip.png",
    )

    # Plot E — per-object diagnostic (fig4 equivalent)
    from skimage.measure import regionprops as _regionprops
    props_lookup: dict[int, Any] = {}
    if "masks" in objects:
        for region in _regionprops(np.asarray(objects["masks"], dtype=np.int32)):
            props_lookup[region.label] = region

    for _, obj_row in obj_df.iterrows():
        cell_index += 1
        cx     = int(obj_row["cx"])
        cy     = int(obj_row["cy"])
        radius = float(obj_row.get("radius", 30))
        lab    = int(obj_row.get("mask_label", obj_row.get("object_index", 1)))

        patch_path = crops_dir / obj_row["filename"]
        import tifffile as _tif
        patch = _tif.imread(str(patch_path)).astype(np.float32)

        plot_per_object_panel(
            guv_norm, mt_norm,
            label_mask_filtered if label_mask_filtered is not None
                else np.zeros_like(mt_norm, dtype=np.int32),
            patch=patch,
            cx=cx, cy=cy, radius=radius,
            cell_index=cell_index,
            cell_id=lab,
            title=title_str,
            out_path=debug_dir / f"img{i:05d}_E_obj{lab:03d}_cellindex{cell_index:04d}.png",
        )

    all_objects.append(obj_df)


# ============================================================
#   SAVE METADATA CSV
# ============================================================

if all_objects:
    objects_df = pd.concat(
        [o for o in all_objects if not o.empty], ignore_index=True
    )
    objects_df.to_csv(meta_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Done. {len(objects_df)} total objects saved.")
    print(f"  Metadata CSV : {meta_csv}")
    print(f"  Crops TIF    : {crops_dir}")
    print(f"  Debug plots  : {debug_dir}")
else:
    print("\nNo objects found across all images.")
