# Data layout and pipeline file structure

This document describes (1) the raw data layout expected by the indexer, and (2) the **finalized pipeline file structure** and command-line usage for preprocessing, segmentation, training, and prediction.

---

## 1. Raw data layout expected by the repo

The pipeline expects microscopy data in a folder structure where each acquisition has two **sibling** folders:

- **GUV/** — GUV channel images (`.tif` / `.tiff`)
- **Microtubule/** — microtubule channel images

The indexer finds every directory named `GUV` under the root and expects a sibling directory named `Microtubule` at the same level.

### Example layout

```
<DATA_ROOT>/
  <experiment_or_batch>/
    <condition>/
      <date_or_run_id>/
        GUV/
          *.tif
        Microtubule/
          *.tif
```

Any nesting is fine as long as `GUV/` and `Microtubule/` are siblings:

```
<DATA_ROOT>/**/<something>/<run_id>/
  GUV/
  Microtubule/
```

### File pairing (GUV → Microtubule)

Pairs are inferred from filenames via `PairingRules` in `image_files_indexing.py`:

1. **Primary:** If the GUV file ends with `GUV.TIF`, the MT file is the same name with `GUV.TIF` replaced by `MT.TIF`.
2. **Otherwise:** `_w1561.TIF` is replaced by `_w2640.TIF` to get the MT filename.
3. **Fallback:** If that file is missing, `_w2561.TIF` is replaced by `_w1640.TIF`.

If your naming differs, either rename files to match these patterns or pass a custom `PairingRules` instance to `build_pairs_dataframe_flexible(rules=...)`.

### Condition and date

When `infer_condition_date_from_parents=True` (the default), the indexer sets:

- **date** = immediate parent folder of `GUV` (e.g. `Date 1`, `run_001`)
- **condition** = parent of that folder (e.g. `1_10 Tau_Tubulin`)

So for `.../<condition>/<date>/GUV/`, the DataFrame columns `condition` and `date` are filled from the path. For flatter layouts, these columns are `None`.

### Quick start: check pairing

From Python:

```python
from mt_structure_classification.dataset.image_files_indexing import build_pairs_dataframe_flexible

df = build_pairs_dataframe_flexible("<DATA_ROOT>", output_debug_missing=True)
# df columns: GUV_folder_path, GUV_file_name, MT_folder_path, MT_file_name, condition, date
```

---

## 2. Pipeline file structure and CLI (finalized)

All commands are run from the **repo root** with the package installed (`pip install -e .`).

### Overview: output locations

| Step | Script | Main output directory / files |
|------|--------|------------------------------|
| 1–2  | `run_preprocessing.py`  | `--output-dir` (e.g. `results/preprocessed/`) |
| 3–4  | `run_segmentation.py`  | `--output-dir` (e.g. `results/segmentation/`) |
| 5    | `run_training.py`      | `--output-dir` (e.g. `models/exp001/`) |
| 6    | `run_prediction.py`    | `--output-csv` (single CSV file) |

### Part 1: Preprocessing (steps 1–2)

**Input:** Raw data root with `GUV/` and `Microtubule/` sibling folders.

**Command:**

```bash
python scripts/run_preprocessing.py \
    --data-root /path/to/Microtubule_GUV-Liu \
    --output-dir results/preprocessed
```

Optional: `--max-images N`, `--save-plots` (writes debug panels to `debug/`).

**Output (under `--output-dir`):**

```
results/preprocessed/
├── metadata/
│   ├── pairs.csv       # indexed GUV/MT pairs (paths, condition, date)
│   ├── stats.json      # channel statistics (norm percentiles, bg intensity)
│   ├── guv_bg.npy      # GUV background image (512×512)
│   └── mt_bg.npy       # MT background image (512×512)
└── debug/              # optional; with --save-plots (preprocessing panels)
```

---

### Part 2: Segmentation (steps 3–4)

**Input:** Output of preprocessing; `--preprocessed-dir` must contain `metadata/pairs.csv`, `metadata/stats.json`, `metadata/guv_bg.npy`, `metadata/mt_bg.npy`.

**Command:**

```bash
python scripts/run_segmentation.py \
    --preprocessed-dir results/preprocessed \
    --output-dir results/segmentation \
    --method combined
```

Optional: `--max-images N`, `--cellpose-gpu`, `--cellpose-model`, `--cellpose-diameter`, `--save-plots`.

**Output (under `--output-dir`):**

```
results/segmentation/
├── metadata/
│   ├── objects_metadata.csv   # per-object metadata (source image, cx, cy, radius, filename, etc.)
│   └── timing_log.txt         # per-image timing (load, Hough, Cellpose, etc.)
├── crops/                     # 96×96 MT patches (TIFF), one per detected object
└── debug/                     # optional; with --save-plots (segmentation panels)
```

---

### Part 3: Training (step 5)

**Input:** Annotation CSV with columns `filename`, `label` (paths relative to `--image-dir`); directory of crop images (e.g. `results/segmentation/crops/`).

**Command:**

```bash
python scripts/run_training.py \
    --csv path/to/annotations.csv \
    --image-dir results/segmentation/crops \
    --output-dir models/exp001 \
    --epochs 200 \
    --patience 30 \
    --batch-size 64 \
    --device cuda
```

Optional: `--model`, `--lr`, `--seed`, `--num-workers`; `--device` can be `auto`, `cuda`, or `cpu`.

**Output (under `--output-dir`):**

```
models/exp001/
├── best_model.pth        # best checkpoint by validation accuracy
├── label_map.json        # class name → index (for prediction)
├── train_split.csv       # train subset of annotations
├── val_split.csv         # validation subset
├── loss_curves.png       # train/val loss and accuracy vs epoch
└── confusion_matrix.png  # validation-set confusion matrix
```

---

### Part 4: Prediction (step 6)

**Input:** Folder of 96×96 crop images; trained checkpoint and label map (from training output). `--label-map` defaults to `label_map.json` next to `--model-path` if omitted.

**Command:**

```bash
python scripts/run_prediction.py \
    --image-dir path/to/crops \
    --model-path models/exp001/best_model.pth \
    --label-map models/exp001/label_map.json \
    --output-csv results/predictions.csv
```

Optional: `--batch-size`, `--model`, `--num-workers`.

**Output:**

- Single CSV at `--output-csv` with columns: `filename`, `abs_path`, `pred_label`, `pred_conf`.

---

## 3. Storing data

Do not put large image datasets in the repo. Use shared storage, institutional drives, or object storage. Git LFS is possible but not recommended for multi-GB data.
