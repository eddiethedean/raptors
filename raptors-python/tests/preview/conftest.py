"""The 0.2 numeric foundation has one pinned NumPy oracle and no skips."""
import numpy as np
import pytest

if np.__version__ != "2.5.3":
    raise RuntimeError(f"preview tests require NumPy 2.5.3, found {np.__version__}")

@pytest.fixture
def numpy_module():
    return np
