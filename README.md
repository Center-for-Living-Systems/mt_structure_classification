### Image Classification of Microtubule-GUV Structures

This repository contains a complete workflow for performing image classification on grayscale microscopy images of microtubule-GUV (Giant Unilamellar Vesicle) structures. The project includes model training, evaluation, visualization, and inference on new test datasets.

#### What This Project Does

The pipeline takes paired fluorescence microscopy TIFFs (GUV channel + microtubule channel) and:

1. **Indexes image pairs** — matches GUV and MT files by naming conventions
2. **Preprocesses** — 2D max-projection, pad/crop to 512x512, background subtraction (median filter), percentile normalization
3. **Segments GUVs** — detects individual vesicles using Cellpose (deep learning instance segmentation) and/or Hough circle detection, with filtering by eccentricity, area, and MT signal
4. **Crops MT patches** — extracts 96x96 pixel patches from each detected GUV
5. **Classifies** — fine-tuned EfficientNet-B0 (also supports ResNet18, ConvNeXt-Tiny) classifies patches into 5 MT structure classes using focal loss

#### Pipeline overview (paper results)

| Step | Description | Script / test |
|------|-------------|----------------|
| **1** | Index GUV/MT image pairs | `scripts/run_preprocessing.py` / `test_pipeline_steps1to4.py` |
| **2** | Background + channel statistics | `scripts/run_preprocessing.py` |
| **3** | Segment GUVs (Hough and/or Cellpose) | `scripts/run_segmentation.py` |
| **4** | Crop 96×96 MT patches per object | `scripts/run_segmentation.py` |
| **5** | Train classifier (EfficientNet, focal loss) | `scripts/run_training.py` / `test_pipeline_steps5to6.py` |
| **6** | Predict on folder of crops | `scripts/run_prediction.py` / same test |

**Reproducing results**

From repo root with the package installed (`pip install -e .`).

**Option A — All steps 1–4 in one command:**
   ```bash
   python scripts/run_segmentation_pipeline.py \
       --data-root /path/to/Microtubule_GUV-Liu \
       --output-dir results/experiment1 \
       --method combined \
       --save-plots
   ```
   Output: `results/experiment1/metadata/`, `crops/`, optional `debug/`.

**Option B — Preprocessing (1–2) then segmentation (3–4) separately:**
   ```bash
   python scripts/run_preprocessing.py --data-root /path/to/data --output-dir results/preprocessed
   python scripts/run_segmentation.py --preprocessed-dir results/preprocessed --output-dir results/segmentation --method combined
   ```

2. **Training (step 5)** — annotation CSV columns: `filename`, `label`; images under `--image-dir`:
   ```bash
   python scripts/run_training.py \
       --csv path/to/annotations.csv \
       --image-dir results/experiment1/crops \
       --output-dir results/train_run \
       --epochs 200 --patience 30
   ```
   Output: `results/train_run/best_model.pth`, `label_map.json`, `train_split.csv`, `val_split.csv`.

3. **Prediction (step 6)**:
   ```bash
   python scripts/run_prediction.py \
       --image-dir path/to/crops \
       --model-path results/train_run/best_model.pth \
       --output-csv results/predictions.csv
   ```
   Output: CSV with `filename`, `pred_label`, `pred_conf`.

Programmatic use: `mt_structure_classification.pipeline` (or `pipeline.full_pipeline`) provides config dataclasses and `run_*_pipeline` functions.

#### Directory layout

```
mt_structure_classification/       Python package
  core/        Training, prediction, model
  dataset/     Data loading, image pair indexing
  pipeline/    full_pipeline.py (orchestration + configs)
  utils/       Image processing, segmentation, plotting
scripts/       run_preprocessing (1–2), run_segmentation (3–4), run_segmentation_pipeline (1–4),
               run_training, run_prediction; cellpose2D_segmentation_MT_5sets_GUV.py
test/          test_pipeline_steps1to4.py, test_pipeline_steps5to6.py
```

#### Status

The workflow is implemented as a Python package with CLI scripts in `scripts/` for preprocessing (steps 1–4), training (step 5), and prediction (step 6). The earlier development notebooks have been retired in favor of these pipeline and run scripts.

<img src="https://github.com/user-attachments/assets/0ba464ca-b8cd-4d16-a89e-af54d19984f7" style="width:70%;"/>

#### Installation

**macOS (Apple Silicon) or CPU-only Linux:**
```bash
conda env create -f environment.yml
conda activate mt_structure_classification
```

**Linux with NVIDIA CUDA GPU:**
```bash
conda env create -f environment-cuda.yml
conda activate mt_structure_classification
```

**Pure pip (no conda, CPU only):**
```bash
pip install -e ".[torch-cpu,dev]"
```

**Development extras** (pytest, ruff, etc.):
```bash
pip install -e ".[dev]"
```

#### How pyproject.toml and conda interact

- `pyproject.toml` is the single source of truth for package dependencies
- `environment.yml` installs only what pip can't handle well (platform-specific PyTorch builds), then runs `pip install -e .` to pull everything else from pyproject.toml
- `torch` and `torchvision` are NOT in pyproject.toml's main deps — conda provides the correct platform build (CUDA / MPS / CPU)

#### Acknowledgments

This pipeline was developed to support the classification of cytoskeletal structures in microscopy images from microtubule experiments involving GUVs. Inspired by efforts at the Center for Living Systems, University of Chicago.
