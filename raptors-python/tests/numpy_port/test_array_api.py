"""NumPy-compatible array API tests

Ported from NumPy's test suite to verify Python API compatibility.
These tests focus on the public Python API rather than internal implementation.
"""

import pytest
import numpy as np

try:
    import raptors
    RAPTORS_AVAILABLE = True
except ImportError:
    RAPTORS_AVAILABLE = False
    pytest.skip("raptors module not available", allow_module_level=True)


class TestArrayCreation:
    """Test array creation functions"""
    
    def test_array_from_list(self):
        """Test creating array from Python list"""
        arr = raptors.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert arr.dtype.name == "int64" or arr.dtype.name == "int32"
    
    def test_array_from_nested_list(self):
        """Test creating 2D array from nested list"""
        arr = raptors.array([[1, 2], [3, 4]])
        assert arr.shape == (2, 2)
    
    def test_zeros(self):
        """Test zeros function"""
        arr = raptors.zeros((3, 4))
        assert arr.shape == (3, 4)
        # Verify all elements are zero
        assert arr[0, 0] == 0.0
    
    def test_ones(self):
        """Test ones function"""
        arr = raptors.ones((2, 3))
        assert arr.shape == (2, 3)
        # Verify all elements are one
        assert arr[0, 0] == 1.0
    
    def test_empty(self):
        """Test empty function"""
        arr = raptors.empty((3, 4))
        assert arr.shape == (3, 4)
        # Note: empty() doesn't initialize, so we can't verify values
    
    def test_array_dtype_specification(self):
        """Test creating array with specific dtype"""
        arr = raptors.array([1, 2, 3], dtype='float64')
        assert arr.dtype.name == "float64"


class TestArrayProperties:
    """Test array properties"""
    
    def test_shape_property(self):
        """Test shape property"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)
    
    def test_ndim_property(self):
        """Test ndim property"""
        arr1d = raptors.array([1, 2, 3])
        assert arr1d.ndim == 1
        
        arr2d = raptors.array([[1, 2], [3, 4]])
        assert arr2d.ndim == 2
    
    def test_size_property(self):
        """Test size property"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        assert arr.size == 6
    
    def test_dtype_property(self):
        """Test dtype property"""
        arr = raptors.array([1, 2, 3])
        assert hasattr(arr, 'dtype')
        assert hasattr(arr.dtype, 'name')


class TestArrayIndexing:
    """Test array indexing operations"""
    
    def test_basic_indexing_1d(self):
        """Test basic 1D indexing"""
        arr = raptors.array([10, 20, 30, 40, 50])
        assert arr[0] == 10
        assert arr[2] == 30
        assert arr[4] == 50
    
    def test_basic_indexing_2d(self):
        """Test basic 2D indexing"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        assert arr[0, 0] == 1
        assert arr[1, 2] == 6
    
    def test_slicing_1d(self):
        """Test 1D slicing"""
        arr = raptors.array([0, 1, 2, 3, 4, 5])
        # Test basic slice
        # Note: Slicing may not be fully implemented yet
        try:
            sliced = arr[1:4]
            assert sliced.shape == (3,)
        except (NotImplementedError, AttributeError):
            pytest.fail("Slicing not yet implemented")
    
    def test_negative_indexing(self):
        """Test negative indexing"""
        arr = raptors.array([10, 20, 30, 40, 50])
        # Note: Negative indexing may not be supported yet
        try:
            assert arr[-1] == 50
        except (IndexError, NotImplementedError):
            pytest.fail("Negative indexing not yet implemented")


class TestArrayOperations:
    """Test array operations"""
    
    def test_add_arrays(self):
        """Test array addition"""
        a = raptors.array([1, 2, 3])
        b = raptors.array([4, 5, 6])
        try:
            result = a + b
            assert result.shape == (3,)
            # Verify result values
            assert result[0] == 5
            assert result[1] == 7
            assert result[2] == 9
        except (TypeError, NotImplementedError):
            pytest.fail("Array addition not yet implemented")
    
    def test_multiply_arrays(self):
        """Test array multiplication"""
        a = raptors.array([2, 3, 4])
        b = raptors.array([5, 6, 7])
        try:
            result = a * b
            assert result.shape == (3,)
            assert result[0] == 10
            assert result[1] == 18
            assert result[2] == 28
        except (TypeError, NotImplementedError):
            pytest.fail("Array multiplication not yet implemented")
    
    def test_scalar_operations(self):
        """Test scalar operations"""
        arr = raptors.array([1, 2, 3])
        try:
            result = arr * 2
            assert result.shape == (3,)
            assert result[0] == 2
            assert result[1] == 4
            assert result[2] == 6
        except (TypeError, NotImplementedError):
            pytest.fail("Scalar operations not yet implemented")


class TestArrayMethods:
    """Test array methods"""
    
    def test_reshape(self):
        """Test reshape method"""
        arr = raptors.array([1, 2, 3, 4, 5, 6])
        try:
            reshaped = arr.reshape((2, 3))
            assert reshaped.shape == (2, 3)
        except (AttributeError, NotImplementedError):
            pytest.fail("Reshape not yet implemented")
    
    def test_flatten(self):
        """Test flatten method"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        try:
            flattened = arr.flatten()
            assert flattened.shape == (6,)
        except (AttributeError, NotImplementedError):
            pytest.fail("Flatten not yet implemented")
    
    def test_sum(self):
        """Test sum method"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            total = arr.sum()
            assert total == 15
        except (AttributeError, NotImplementedError):
            pytest.fail("Sum not yet implemented")
    
    def test_max_min(self):
        """Test max and min methods"""
        arr = raptors.array([3, 1, 4, 1, 5, 9, 2, 6])
        try:
            max_val = arr.max()
            min_val = arr.min()
            assert max_val == 9
            assert min_val == 1
        except (AttributeError, NotImplementedError):
            pytest.fail("Max/min not yet implemented")


class TestArrayBroadcasting:
    """Test broadcasting behavior"""
    
    def test_broadcast_scalar(self):
        """Test broadcasting with scalar"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        try:
            result = arr + 10
            assert result.shape == (2, 3)
            assert result[0, 0] == 11
        except (TypeError, NotImplementedError):
            pytest.fail("Broadcasting not yet implemented")
    
    def test_broadcast_shapes(self):
        """Test broadcasting with compatible shapes"""
        a = raptors.array([[1], [2], [3]])  # Shape (3, 1)
        b = raptors.array([10, 20, 30])     # Shape (3,)
        try:
            result = a + b
            assert result.shape == (3, 3)
        except (TypeError, NotImplementedError):
            pytest.fail("Broadcasting not yet implemented")


class TestArrayConversion:
    """Test array conversion methods"""
    
    def test_to_list(self):
        """Test converting array to Python list"""
        arr = raptors.array([1, 2, 3])
        try:
            lst = arr.tolist()
            assert lst == [1, 2, 3]
        except (AttributeError, NotImplementedError):
            pytest.fail("tolist not yet implemented")
    
    def test_to_numpy(self):
        """Test converting to NumPy array"""
        arr = raptors.array([1, 2, 3])
        try:
            np_arr = arr.to_numpy()
            assert isinstance(np_arr, np.ndarray)
            np.testing.assert_array_equal(np_arr, [1, 2, 3])
        except (AttributeError, NotImplementedError):
            pytest.fail("to_numpy not yet implemented")


class TestArrayInteroperability:
    """Test interoperability with NumPy"""
    
    def test_from_numpy(self):
        """Test creating Raptors array from NumPy array"""
        np_arr = np.array([1, 2, 3, 4, 5])
        try:
            arr = raptors.from_numpy(np_arr)
            assert arr.shape == (5,)
        except (AttributeError, NotImplementedError):
            pytest.fail("from_numpy not yet implemented")
    
    def test_numpy_compatibility(self):
        """Test that Raptors arrays work with NumPy functions"""
        arr = raptors.array([1, 2, 3, 4, 5])
        try:
            # Test if NumPy functions accept Raptors arrays
            result = np.sum(arr)
            assert result == 15
        except (TypeError, AttributeError):
            pytest.fail("NumPy compatibility not yet implemented")


class TestArrayEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_array(self):
        """Test empty array creation"""
        arr = raptors.array([])
        assert arr.shape == (0,)
        assert arr.size == 0
    
    def test_zero_dimensional_array(self):
        """Test 0-dimensional array (scalar)"""
        arr = raptors.array(42)
        assert arr.ndim == 0
        assert arr.size == 1
    
    def test_large_array(self):
        """Test creating large arrays"""
        # Test with reasonably large size
        arr = raptors.zeros((1000,))
        assert arr.shape == (1000,)
        assert arr.size == 1000
    
    def test_index_out_of_bounds(self):
        """Test indexing out of bounds raises error"""
        arr = raptors.array([1, 2, 3])
        with pytest.raises((IndexError, ValueError)):
            _ = arr[10]
    
    def test_shape_mismatch_error(self):
        """Test that shape mismatches raise errors"""
        a = raptors.array([1, 2, 3])
        b = raptors.array([1, 2])
        try:
            result = a + b
            # If broadcasting is not supported, this should fail
            if result.shape != a.shape:
                pytest.fail("Shape mismatch should raise error")
        except (ValueError, TypeError):
            pass  # Expected error


class TestArrayDtypeOperations:
    """Test dtype-related operations"""
    
    def test_dtype_conversion(self):
        """Test converting array dtype"""
        arr = raptors.array([1, 2, 3])
        try:
            float_arr = arr.astype('float64')
            assert float_arr.dtype.name == "float64"
        except (AttributeError, NotImplementedError):
            pytest.fail("astype not yet implemented")
    
    def test_dtype_preservation(self):
        """Test that dtype is preserved in operations"""
        arr = raptors.array([1, 2, 3], dtype='int32')
        try:
            result = arr * 2
            # Result dtype may be promoted or preserved
            assert hasattr(result, 'dtype')
        except (TypeError, NotImplementedError):
            pytest.fail("Dtype operations not yet implemented")

