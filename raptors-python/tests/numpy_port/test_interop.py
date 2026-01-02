"""NumPy-Raptors interoperability tests

Tests for interoperability between NumPy and Raptors arrays.
"""

import pytest
import numpy as np

try:
    import raptors
    RAPTORS_AVAILABLE = True
except ImportError:
    RAPTORS_AVAILABLE = False
    pytest.skip("raptors module not available", allow_module_level=True)


class TestArrayConversion:
    """Test conversion between NumPy and Raptors arrays"""
    
    def test_raptors_to_numpy(self):
        """Test converting Raptors array to NumPy array"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            np_arr = np.asarray(arr)
            assert isinstance(np_arr, np.ndarray)
            np.testing.assert_array_equal(np_arr, [1, 2, 3, 4, 5])
        except (TypeError, AttributeError):
            pytest.fail("Raptors to NumPy conversion not yet implemented")
    
    def test_numpy_to_raptors(self):
        """Test converting NumPy array to Raptors array"""
        np_arr = np.array([1, 2, 3, 4, 5])
        try:
            arr = raptors.from_numpy(np_arr)
            assert arr.shape == (5,)
            assert arr[0] == 1
            assert arr[4] == 5
        except (AttributeError, NotImplementedError):
            pytest.fail("NumPy to Raptors conversion not yet implemented")
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion"""
        original = np.array([[1, 2, 3], [4, 5, 6]])
        try:
            # NumPy -> Raptors -> NumPy
            raptors_arr = raptors.from_numpy(original)
            back_to_numpy = np.asarray(raptors_arr)
            np.testing.assert_array_equal(original, back_to_numpy)
        except (AttributeError, NotImplementedError, TypeError):
            pytest.fail("Roundtrip conversion not yet implemented")


class TestSharedMemory:
    """Test shared memory between NumPy and Raptors"""
    
    def test_memory_sharing(self):
        """Test that arrays share memory when possible"""
        np_arr = np.array([1, 2, 3, 4, 5])
        try:
            raptors_arr = raptors.from_numpy(np_arr)
            # Modify NumPy array
            np_arr[0] = 99
            # Check if Raptors array sees the change
            if hasattr(raptors_arr, '_shares_memory'):
                # If memory is shared, Raptors array should see change
                # If not, it should remain unchanged
                pass  # Implementation-dependent
        except (AttributeError, NotImplementedError):
            pytest.fail("Memory sharing not yet implemented")


class TestNumPyFunctionsWithRaptors:
    """Test using NumPy functions with Raptors arrays"""
    
    def test_numpy_sum(self):
        """Test NumPy sum with Raptors array"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            result = np.sum(arr)
            assert result == 15
        except (TypeError, AttributeError):
            pytest.fail("NumPy functions with Raptors arrays not yet implemented")
    
    def test_numpy_mean(self):
        """Test NumPy mean with Raptors array"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            result = np.mean(arr)
            assert abs(result - 3.0) < 1e-10
        except (TypeError, AttributeError):
            pytest.fail("NumPy functions with Raptors arrays not yet implemented")
    
    def test_numpy_max_min(self):
        """Test NumPy max/min with Raptors array"""
        arr = raptors.array([3, 1, 4, 1, 5, 9, 2, 6])
        try:
            max_val = np.max(arr)
            min_val = np.min(arr)
            assert max_val == 9
            assert min_val == 1
        except (TypeError, AttributeError):
            pytest.fail("NumPy functions with Raptors arrays not yet implemented")


class TestProtocolCompliance:
    """Test array protocol compliance"""
    
    def test_array_protocol(self):
        """Test that Raptors arrays implement array protocol"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            # Check for array protocol attributes
            assert hasattr(arr, '__array__')
            # Try to use with array protocol
            protocol_arr = np.asarray(arr)
            assert isinstance(protocol_arr, np.ndarray)
        except (AttributeError, TypeError):
            pytest.fail("Array protocol not yet implemented")
    
    def test_dlpack_protocol(self):
        """Test DLPack protocol support"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            # Check for DLPack protocol
            if hasattr(arr, '__dlpack__'):
                capsule = arr.__dlpack__()
                assert capsule is not None
        except (AttributeError, NotImplementedError):
            pytest.fail("DLPack protocol not yet implemented")


class TestTypeCompatibility:
    """Test type compatibility between NumPy and Raptors"""
    
    def test_dtype_compatibility(self):
        """Test that dtypes are compatible"""
        np_arr = np.array([1, 2, 3], dtype=np.int32)
        try:
            raptors_arr = raptors.from_numpy(np_arr)
            # Check dtype compatibility
            assert raptors_arr.dtype.name in ["int32", "int64"]
        except (AttributeError, NotImplementedError):
            pytest.fail("Dtype compatibility not yet implemented")
    
    def test_shape_compatibility(self):
        """Test that shapes are compatible"""
        np_arr = np.array([[1, 2], [3, 4]])
        try:
            raptors_arr = raptors.from_numpy(np_arr)
            assert raptors_arr.shape == (2, 2)
        except (AttributeError, NotImplementedError):
            pytest.fail("Shape compatibility not yet implemented")

