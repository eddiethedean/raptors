"""Pytest configuration and fixtures for Raptors Python tests"""

import pytest
import sys
import os
import warnings

# Filter out pytest deprecation warnings about async fixtures
pytest_removed_in_9_warning = getattr(pytest, "PytestRemovedIn9Warning", None)
if pytest_removed_in_9_warning is not None:
    warnings.filterwarnings("ignore", category=pytest_removed_in_9_warning)

# Add the parent directory to the path so we can import raptors
# This ensures we can import the module even if it's not installed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import raptors - if it fails, provide helpful error
try:
    import raptors
except ImportError as e:
    pytest.skip(
        f"raptors module not available: {e}. "
        "Please build the module first with 'maturin develop'",
        allow_module_level=True
    )


@pytest.fixture
def float64_array():
    """Fixture for creating a float64 array"""
    import raptors
    return raptors.zeros([3, 4], dtype=raptors.float64)


@pytest.fixture
def float32_array():
    """Fixture for creating a float32 array"""
    import raptors
    return raptors.zeros([3, 4], dtype=raptors.float32)


@pytest.fixture
def ones_array():
    """Fixture for creating an array of ones"""
    import raptors
    return raptors.ones([3, 4], dtype=raptors.float64)
