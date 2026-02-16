### Image Classification of Microtubule-GUV Structures

This repository contains a complete workflow for performing image classification on grayscale microscopy images of microtubule-GUV (Giant Unilamellar Vesicle) structures. The project includes model training, evaluation, visualization, and inference on new test datasets.

#### What This Project Does

The pipeline takes paired fluorescence microscopy TIFFs (GUV channel + microtubule channel) and:

1. **Indexes image pairs** — matches GUV and MT files by naming conventions
2. **Preprocesses** — 2D max-projection, pad/crop to 512x512, background subtraction (median filter), percentile normalization
3. **Segments GUVs** — detects individual vesicles using Cellpose (deep learning instance segmentation) and/or Hough circle detection, with filtering by eccentricity, area, and MT signal
4. **Crops MT patches** — extracts 96x96 pixel patches from each detected GUV
5. **Classifies** — fine-tuned EfficientNet-B0 (also supports ResNet18, ConvNeXt-Tiny) classifies patches into 5 MT structure classes using focal loss

#### Directory Layout

```
src/mt_structure_classification/   Python package (being built from notebooks)
  core/        Pipeline orchestration, training, prediction, model definitions
  dataset/     Data loading, image pair indexing
  utils/       Image processing, segmentation, plotting
notebooks/                         5 active notebooks (in-progress conversion)
original_notebooks/                48+ archived development notebooks
scripts/                           Converted scripts (incomplete)
test/                              Integration tests
```

#### Status

This code is being converted from Jupyter notebooks into a proper Python package. The pipeline structure exists in `core/pipeline.py` but some modules have unresolved import mismatches (e.g., `build_efficientnet_b0` vs `initialize_model`). The notebook-to-package conversion is roughly 30-40% complete.

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
