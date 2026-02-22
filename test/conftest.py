"""
conftest.py
===========
Pytest configuration for mt_structure_classification tests.

Adds custom markers and command-line options:
  --device cuda|cpu  : which device to use for tests
  -m cellpose        : mark cellpose tests (requires model download)
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
