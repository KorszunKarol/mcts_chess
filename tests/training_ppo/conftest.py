import os
import sys

import pytest

# Ensure project root is importable so `src` can be resolved in tests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    """Register common markers used across Coach Tal tests."""
    config.addinivalue_line("markers", "gpu: requires GPU execution")
    config.addinivalue_line("markers", "slow: marks resource-intensive tests")


@pytest.fixture
def require_gpu():
    """Skip the test if no GPU is available to JAX."""
    import jax

    if not jax.devices("gpu"):
        pytest.skip("GPU is required for this test")

    return jax.devices("gpu")[0]


