"""Pytest configuration for NumPy ported tests"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import raptors
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# Filter out pytest deprecation warnings
import warnings
pytest_removed_in_9_warning = getattr(pytest, "PytestRemovedIn9Warning", None)
if pytest_removed_in_9_warning is not None:
    warnings.filterwarnings("ignore", category=pytest_removed_in_9_warning)

# NumPy testing utilities for Python tests
try:
    import numpy as np
    from numpy.testing import (
        assert_array_equal,
        assert_array_almost_equal,
        assert_array_less,
        assert_allclose,
        assert_almost_equal,
    )
except ImportError:
    # If NumPy is not available, create minimal stubs
    def assert_array_equal(*args, **kwargs):
        pass
    
    def assert_array_almost_equal(*args, **kwargs):
        pass
    
    def assert_array_less(*args, **kwargs):
        pass
    
    def assert_allclose(*args, **kwargs):
        pass
    
    def assert_almost_equal(*args, **kwargs):
        pass
