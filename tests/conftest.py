import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run integration tests that hit live RCSB/EBI APIs",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_net = pytest.mark.skip(reason="needs --run-network to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_net)
