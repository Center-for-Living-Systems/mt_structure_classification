CONDA_ENV = mt_structure_classification
RUN = conda run -n $(CONDA_ENV)

.PHONY: env env-cuda install install-dev test test-slow lint format clean help

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

test:  ## Run tests (fast — hough circles only)
	$(RUN) pytest test/ -v

test-slow:  ## Run all tests including cellpose (needs model download)
	$(RUN) pytest test/ -v --runslow

lint:  ## Run ruff linter
	$(RUN) ruff check src/ test/ scripts/

format:  ## Auto-format with ruff + black
	$(RUN) ruff check --fix src/ test/ scripts/
	$(RUN) black src/ test/ scripts/

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

check-device:  ## Print which accelerator PyTorch sees
	$(RUN) python -c "from mt_structure_classification.utils.device import get_device; print('Device:', get_device())"
