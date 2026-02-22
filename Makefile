CONDA_ENV = mt_structure_classification
RUN = conda run -n $(CONDA_ENV)

.PHONY: env env-cuda install install-dev test test-cellpose test-cuda test-cuda-cellpose lint format clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

env:  ## Create conda env (macOS / CPU)
	conda env create -f environment.yml

env-cuda:  ## Create conda env (Linux + NVIDIA CUDA)
	conda env create -f environment-cuda.yml

env-update:  ## Update existing conda env
	conda env update -f environment.yml --prune

install:  ## pip install package in editable mode
	$(RUN) pip install -e .

install-dev:  ## pip install with dev extras
	$(RUN) pip install -e ".[dev]"

install-cpu:  ## pip install with torch CPU (no conda needed)
	pip install -e ".[torch-cpu,dev]"



# ── per group tests ─────────────────────────────────────────────────────────────────

test-segmentation:  ## Run segmentation tests only (steps 1-4)
	$(RUN) pytest test/test_pipeline_steps1to4.py -v --device cpu

test-training:  ## Run training/prediction tests only (steps 5-6)
	$(RUN) pytest test/test_pipeline_steps5to6.py -v --device cpu


test-training:  ## Run training/prediction tests only (steps 5-6)
	$(RUN) pytest test/test_pipeline_steps5to6.py -v --device cuda


# ── CPU tests ─────────────────────────────────────────────────────────────────

test:  ## Run fast tests on CPU (hough circles + classifier smoke test)
	$(RUN) pytest test/ -v \
		-m "not cellpose" \
		--device cpu

test-cellpose:  ## Run all CPU tests including cellpose (needs model download)
	$(RUN) pytest test/ -v \
		-m cellpose \
		--device cpu

test-all:  ## Run all tests on CPU
	$(RUN) pytest test/ -v --device cpu

# ── CUDA tests ────────────────────────────────────────────────────────────────

test-cuda:  ## Run fast tests on CUDA GPU
	$(RUN) pytest test/ -v \
		-m "not cellpose" \
		--device cuda

test-cuda-cellpose:  ## Run all tests on CUDA GPU including cellpose
	$(RUN) pytest test/ -v \
		-m cellpose \
		--device cuda

test-cuda-all:  ## Run all tests on CUDA GPU
	$(RUN) pytest test/ -v --device cuda

# ── Code quality ──────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	$(RUN) ruff check mt_structure_classification/ test/ scripts/

format:  ## Auto-format with ruff + black
	$(RUN) ruff check --fix mt_structure_classification/ test/ scripts/
	$(RUN) black mt_structure_classification/ test/ scripts/

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

check-device:  ## Print which accelerator PyTorch sees
	$(RUN) python -c "from mt_structure_classification.utils.device import get_device; print('Device:', get_device())"
