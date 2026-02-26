### Image Classification of Microtubule-GUV Structures

This repository contains a complete workflow for performing image classification on grayscale microscopy images of microtubule-GUV (Giant Unilamellar Vesicle) structures. The project includes model training, evaluation, visualization, and inference on new test datasets.

#### What This Project Does

The pipeline takes paired fluorescence microscopy TIFFs (GUV channel + microtubule channel) and:

1. **Preprocesses** — matches GUV and MT files, background subtraction (median filter), percentile normalization
2. **Segments GUVs and MT** — detects individual vesicles using Cellpose (deep learning instance segmentation) and/or Hough circle detection, with filtering by eccentricity, area, and MT signal. And crops MT patches — extracts 96x96 pixel patches from each detected GUV
3. **Classifies** — fine-tuned EfficientNet-B0 (also supports ResNet18, ConvNeXt-Tiny) classifies patches into 5 MT structure classes using focal loss

A **trained model** is provided in the `models/` folder. This is the model used for the results presented in the paper (trained on the paper’s data). Use it for inference with `run_prediction.py` by passing `--model-path models/trained_model_20250618.pth` (and `--label-map models/label_map.json` if needed).

Figure 1: Example GUV segmentation.

<img src="https://github.com/user-attachments/assets/7ce6fcab-f337-4aa0-b69f-9f815f02f8f3" style="width:60%;"/>

Figure 2: Classifier training.

<img src="https://github.com/user-attachments/assets/0ba464ca-b8cd-4d16-a89e-af54d19984f7" style="width:60%;"/>


#### Pipeline overview (paper results)

| Step | Description | CLI / Package |
|------|-------------|----------------|
| **1** | Index GUV/MT image pairs | `scripts/run_preprocessing.py` (with step 2) |
| **2** | Background + channel statistics | `scripts/run_preprocessing.py` |
| **3** | Segment GUVs (Hough and/or Cellpose) | `scripts/run_segmentation.py` (with step 4) |
| **4** | Crop 96×96 MT patches per object | `scripts/run_segmentation.py` |
| **5** | Train classifier (EfficientNet, focal loss) | `scripts/run_training.py` |
| **6** | Predict on folder of crops | `scripts/run_prediction.py` |

Steps 1–4 can be run via CLI scripts (see below) or programmatically with `mt_structure_classification.pipeline.full_pipeline`. The tests in `test_pipeline_steps1to4.py` show the full flow.

**Reproducing results (CLI)**

From repo root with the package installed (`pip install -e .`). See `mt_structure_classification/dataset/README.md` for full file-structure and output descriptions.

1. **Preprocessing (steps 1–2)** — index pairs, compute background and channel statistics:
   ```bash
   python scripts/run_preprocessing.py \
       --data-root /path/to/Microtubule_GUV-Liu \
       --output-dir results/preprocessed
   ```
   Output: `results/preprocessed/metadata/` (pairs.csv, stats.json, guv_bg.npy, mt_bg.npy); optional `debug/` with `--save-plots`.

2. **Segmentation (steps 3–4)** — segment GUVs and crop 96×96 MT patches:
   ```bash
   python scripts/run_segmentation.py \
       --preprocessed-dir results/preprocessed \
       --output-dir results/segmentation \
       --method combined
   ```
   Output: `results/segmentation/metadata/objects_metadata.csv`, `results/segmentation/crops/` (96×96 TIFFs), `metadata/timing_log.txt`; optional `debug/` with `--save-plots`. Use `--cellpose-gpu` for faster Cellpose.

3. **Training (step 5)** — CSV with columns `filename`, `label`; crop images under `--image-dir`:
   ```bash
   python scripts/run_training.py \
       --csv path/to/annotations.csv \
       --image-dir results/segmentation/crops \
       --output-dir models/exp001 \
       --epochs 200 --patience 30 --device cuda
   ```
   Output: `models/exp001/best_model.pth`, `label_map.json`, `train_split.csv`, `val_split.csv`, `loss_curves.png`, `confusion_matrix.png`.

4. **Prediction (step 6)**:
   ```bash
   python scripts/run_prediction.py \
       --image-dir path/to/crops \
       --model-path models/exp001/best_model.pth \
       --output-csv results/predictions.csv
   ```
   Output: CSV with columns `filename`, `abs_path`, `pred_label`, `pred_conf`. `--label-map` defaults to `label_map.json` beside the checkpoint.

Programmatic use: `mt_structure_classification.pipeline` (or `pipeline.full_pipeline`) provides config dataclasses and `run_*_pipeline` functions.

#### Project structure

```
mt_structure_classification/
├── mt_structure_classification/      # Main package
│   ├── core/                         # Model training, prediction, and segmentation
│   │   ├── model.py                  # Neural network architectures
│   │   ├── train.py                  # Training loop with focal loss
│   │   ├── predict.py                # Inference functions
│   │   └── GUV_mt_segmentation.py   # Hough/Cellpose segmentation and cropping
│   ├── dataset/                      # Data loading and indexing
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   ├── image_files_indexing.py  # GUV/MT file pair matching
│   │   └── README.md                # Data layout and pipeline CLI
│   ├── pipeline/                     # High-level orchestration
│   │   └── full_pipeline.py         # Config and pipeline runners
│   └── utils/                        # Image processing utilities
│       ├── image_processing.py      # Background correction, transforms
│       ├── plotting_functions.py    # Visualization tools
│       ├── device.py                # CPU/CUDA device selection
│       └── colormap_definition.py   # Colormap definitions
├── scripts/                          # Command-line interfaces
│   ├── run_preprocessing.py         # Steps 1–2: index pairs, background & stats
│   ├── run_segmentation.py          # Steps 3–4: segment GUVs, crop MT patches
│   ├── run_training.py              # Step 5: classifier training
│   └── run_prediction.py            # Step 6: batch inference
├── test/                             # Test suite
│   ├── conftest.py                  # Pytest config and fixtures
│   ├── test_pipeline_steps1to4.py  # Preprocessing and segmentation tests
│   └── test_pipeline_steps5to6.py  # Training and prediction tests
├── models/                           # Trained model (paper)
│   ├── trained_model_20250618.pth    # Model weights trained and used for paper results
│   └── label_map.json               # Class name → index
├── environment.yml                   # Conda environment (CPU/macOS)
├── environment-cuda.yml             # Conda environment (CUDA GPU)
├── pyproject.toml                   # Package configuration
├── Makefile                         # Development commands
└── README.md                        # This file
```

#### Status

The workflow is implemented as a Python package. All steps have CLI scripts in `scripts/`; steps 1–4 can also be run via the package API (`pipeline.full_pipeline`, `core.GUV_mt_segmentation`, `dataset`, `utils`).


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

#### Advanced configuration

**Preprocessing** (`run_preprocessing.py`): `--max-images N`, `--save-plots`, `--disk-radius`, `--max-plot-images`.

**Segmentation** (`run_segmentation.py`): `--method` (hough | cellpose | combined), `--max-images N`, `--cellpose-gpu`, `--cellpose-model` (default cyto3), `--cellpose-diameter` (default 40; avoids slow per-image estimation), `--save-plots`.

**Training** (`run_training.py`): `--model` (e.g. efficientnet), `--epochs`, `--patience`, `--batch-size`, `--lr`, `--seed`, `--num-workers`, `--device` (auto | cuda | cpu).

**Prediction** (`run_prediction.py`): `--batch-size`, `--model` (must match training), `--num-workers`. `--label-map` defaults to `label_map.json` next to `--model-path`.

**Programmatic config:** `mt_structure_classification.pipeline.full_pipeline` exposes `PreprocessConfig`, `SegmentationConfig`, `TrainingConfig`, `PredictConfig` for running steps 1–6 from Python with custom parameters.


## About
This computational pipeline was developed by Liya Ding at the Center for Living Systems, University of Chicago, as a contribution to the quantitative image analysis for a collaborative study on tau-mediated cytoskeletal crosstalk in reconstituted systems.
