"""
conftest.py
===========
Pytest configuration for mt_structure_classification tests.

Adds custom markers and command-line options:
  --device cuda|cpu  : device for tests (default: cpu). Use cuda for Cellpose GPU.
  -m cellpose        : mark cellpose tests (requires model download)

Examples:
  pytest test/ -v                    # CPU (default)
  pytest test/ -v -m cellpose        # Cellpose tests on CPU
  pytest test/ -v -m cellpose --device cuda   # Cellpose tests on GPU
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device for tests: cpu or cuda (default: cpu)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cellpose: tests that require cellpose model (slow, needs download)",
    )


@pytest.fixture(scope="session")
def device(request):
    """Provide device for tests from command line."""
    return request.config.getoption("--device")


@pytest.fixture(scope="session")
def cellpose_gpu(request):
    """Use GPU for Cellpose when --device cuda; otherwise CPU (e.g. for CI)."""
    return request.config.getoption("--device") == "cuda"
