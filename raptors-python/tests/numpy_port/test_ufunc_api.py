"""NumPy-compatible ufunc API tests

Ported from NumPy's test_umath.py and test_ufunc.py
Tests focus on the Python API for universal functions.
"""

import pytest
import numpy as np

try:
    import raptors
    RAPTORS_AVAILABLE = True
except ImportError:
    RAPTORS_AVAILABLE = False
    pytest.skip("raptors module not available", allow_module_level=True)


class TestUfuncBasic:
    """Test basic ufunc operations"""
    
    def test_add_ufunc(self):
        """Test add ufunc"""
        a = raptors.array([1, 2, 3])
        b = raptors.array([4, 5, 6])
        try:
            result = raptors.add(a, b)
            assert result.shape == (3,)
            assert result[0] == 5
            assert result[1] == 7
            assert result[2] == 9
        except (AttributeError, NotImplementedError):
            pytest.fail("add ufunc not yet implemented")
    
    def test_subtract_ufunc(self):
        """Test subtract ufunc"""
        a = raptors.array([5, 7, 9])
        b = raptors.array([1, 2, 3])
        try:
            result = raptors.subtract(a, b)
            assert result.shape == (3,)
            assert result[0] == 4
            assert result[1] == 5
            assert result[2] == 6
        except (AttributeError, NotImplementedError):
            pytest.fail("subtract ufunc not yet implemented")
    
    def test_multiply_ufunc(self):
        """Test multiply ufunc"""
        a = raptors.array([2, 3, 4])
        b = raptors.array([5, 6, 7])
        try:
            result = raptors.multiply(a, b)
            assert result.shape == (3,)
            assert result[0] == 10
            assert result[1] == 18
            assert result[2] == 28
        except (AttributeError, NotImplementedError):
            pytest.fail("multiply ufunc not yet implemented")
    
    def test_divide_ufunc(self):
        """Test divide ufunc"""
        a = raptors.array([10, 18, 28])
        b = raptors.array([2, 3, 4])
        try:
            result = raptors.divide(a, b)
            assert result.shape == (3,)
            assert abs(result[0] - 5.0) < 1e-10
            assert abs(result[1] - 6.0) < 1e-10
            assert abs(result[2] - 7.0) < 1e-10
        except (AttributeError, NotImplementedError):
            pytest.fail("divide ufunc not yet implemented")


class TestUfuncBroadcasting:
    """Test ufunc broadcasting"""
    
    def test_add_broadcast_scalar(self):
        """Test add with scalar broadcasting"""
        arr = raptors.array([[1, 2, 3], [4, 5, 6]])
        try:
            result = raptors.add(arr, 10)
            assert result.shape == (2, 3)
            assert result[0, 0] == 11
        except (AttributeError, NotImplementedError):
            pytest.fail("Broadcasting in ufuncs not yet implemented")
    
    def test_multiply_broadcast_shapes(self):
        """Test multiply with shape broadcasting"""
        a = raptors.array([[1], [2], [3]])  # (3, 1)
        b = raptors.array([10, 20, 30])     # (3,)
        try:
            result = raptors.multiply(a, b)
            assert result.shape == (3, 3)
        except (AttributeError, NotImplementedError):
            pytest.fail("Broadcasting in ufuncs not yet implemented")


class TestUfuncTypePromotion:
    """Test type promotion in ufuncs"""
    
    def test_int_float_promotion(self):
        """Test that int + float promotes to float"""
        a = raptors.array([1, 2, 3], dtype='int32')
        b = raptors.array([1.5, 2.5, 3.5], dtype='float64')
        try:
            result = raptors.add(a, b)
            assert result.dtype.name == "float64"
        except (AttributeError, NotImplementedError, TypeError):
            pytest.fail("Type promotion not yet implemented")
    
    def test_same_type_preservation(self):
        """Test that same types don't promote unnecessarily"""
        a = raptors.array([1, 2, 3], dtype='int32')
        b = raptors.array([4, 5, 6], dtype='int32')
        try:
            result = raptors.add(a, b)
            # Result may be int32 or int64 depending on implementation
            assert 'int' in result.dtype.name
        except (AttributeError, NotImplementedError):
            pytest.skip("Type promotion not yet implemented")


class TestUfuncEdgeCases:
    """Test edge cases in ufuncs"""
    
    def test_empty_array_ufunc(self):
        """Test ufunc with empty array"""
        a = raptors.array([])
        b = raptors.array([])
        try:
            result = raptors.add(a, b)
            assert result.shape == (0,)
        except (AttributeError, NotImplementedError):
            pytest.fail("Empty array ufuncs not yet implemented")
    
    def test_zero_dimensional_ufunc(self):
        """Test ufunc with 0-d arrays"""
        a = raptors.array(5)
        b = raptors.array(3)
        try:
            result = raptors.add(a, b)
            assert result.ndim == 0
            assert result.item() == 8
        except (AttributeError, NotImplementedError):
            pytest.fail("0-d array ufuncs not yet implemented")
    
    def test_nan_handling(self):
        """Test NaN handling in ufuncs"""
        a = raptors.array([1.0, float('nan'), 3.0])
        b = raptors.array([1.0, 2.0, 3.0])
        try:
            result = raptors.add(a, b)
            assert result[0] == 2.0
            assert np.isnan(result[1])
            assert result[2] == 6.0
        except (AttributeError, NotImplementedError):
            pytest.fail("NaN handling not yet implemented")
    
    def test_infinity_handling(self):
        """Test infinity handling in ufuncs"""
        a = raptors.array([1.0, float('inf'), 3.0])
        b = raptors.array([1.0, 2.0, 3.0])
        try:
            result = raptors.add(a, b)
            assert result[0] == 2.0
            assert np.isinf(result[1])
            assert result[2] == 6.0
        except (AttributeError, NotImplementedError):
            pytest.fail("Infinity handling not yet implemented")


class TestUfuncComparison:
    """Test comparison ufuncs"""
    
    def test_equal_ufunc(self):
        """Test equal ufunc"""
        a = raptors.array([1, 2, 3])
        b = raptors.array([1, 5, 3])
        try:
            result = raptors.equal(a, b)
            assert result.dtype.name == "bool"
            assert result[0] == True
            assert result[1] == False
            assert result[2] == True
        except (AttributeError, NotImplementedError):
            pytest.fail("equal ufunc not yet implemented")
    
    def test_less_ufunc(self):
        """Test less than ufunc"""
        a = raptors.array([1, 2, 3])
        b = raptors.array([2, 1, 4])
        try:
            result = raptors.less(a, b)
            assert result.dtype.name == "bool"
            assert result[0] == True
            assert result[1] == False
            assert result[2] == True
        except (AttributeError, NotImplementedError):
            pytest.fail("less ufunc not yet implemented")
    
    def test_greater_ufunc(self):
        """Test greater than ufunc"""
        a = raptors.array([3, 2, 1])
        b = raptors.array([2, 3, 0])
        try:
            result = raptors.greater(a, b)
            assert result.dtype.name == "bool"
            assert result[0] == True
            assert result[1] == False
            assert result[2] == True
        except (AttributeError, NotImplementedError):
            pytest.fail("greater ufunc not yet implemented")

