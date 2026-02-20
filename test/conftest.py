import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runcellpose",
        action="store_true",
        default=False,
        help="Include cellpose tests (requires model download)",
    )
    parser.addoption(
        "--device",
        default="cpu",
        help="Device to run torch on: cpu or cuda",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "cellpose: mark test as requiring cellpose model download"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runcellpose"):
        skip_cellpose = pytest.mark.skip(reason="pass --runcellpose to run cellpose tests")
        for item in items:
            if "cellpose" in item.keywords or "cellpose" in item.name.lower():
                item.add_marker(skip_cellpose)


@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device")
