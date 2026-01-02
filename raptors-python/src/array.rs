//! Array Python bindings
//!
//! This module provides Python bindings for the Array type.

#![allow(clippy::arc_with_non_send_sync)] // Arc used for Python reference counting, not thread safety

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyTuple};
use raptors_core::{Array, empty, zeros, ones};
use raptors_core::types::{DType, NpyType};
use raptors_core::indexing::index_array;
use raptors_core::operations::{add, subtract, multiply, divide};
use raptors_core::operations::{equal, not_equal, less, greater, less_equal, greater_equal};
use std::sync::Arc;
use crate::dtype::PyDType;
use crate::iterators;

/// Python Array class
#[pyclass]
pub struct PyArray {
    #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting, not thread safety
    pub(crate) inner: Arc<Array>,
}

// SAFETY: Array contains raw pointers but they are managed safely through Arc
// The data is owned by the Array struct, and Arc provides thread-safe reference counting
unsafe impl Send for PyArray {}
unsafe impl Sync for PyArray {}

impl PyArray {
    /// Get reference to inner array (for internal use)
    pub(crate) fn get_inner(&self) -> &Arc<Array> {
        &self.inner
    }
}

impl PyArray {
    /// Get reference to inner array from Bound (for internal use)
    pub(crate) fn get_inner_from_bound<'a>(bound: &Bound<'a, PyArray>) -> &'a Arc<Array> {
        // SAFETY: Bound ensures the reference is valid for its lifetime
        unsafe { &*(&bound.borrow().inner as *const Arc<Array>) }
    }
}

#[pymethods]
impl PyArray {
    /// Create an empty array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn empty(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = empty(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Create a zero-filled array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn zeros(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = zeros(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Create a one-filled array
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None))]
    fn ones(shape: Vec<i64>, dtype: Option<&PyDType>) -> PyResult<Self> {
        let dtype = match dtype {
            Some(dt) => dt.get_inner().clone(),
            None => DType::new(NpyType::Double),
        };
        let array = ones(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(array),
        })
    }
    
    /// Get the shape of the array
    #[getter]
    fn shape(&self, py: Python) -> PyResult<Py<PyTuple>> {
        // Return as tuple for NumPy compatibility
        let shape_vec = self.get_inner().shape().to_vec();
        // Convert to Vec<i64> for PyTuple::new (i64 implements ToPyObject)
        let items: Vec<i64> = shape_vec.iter().map(|&x| x as i64).collect();
        let tuple = PyTuple::new(py, items)?;
        Ok(tuple.into())
    }
    
    /// Get the dtype
    #[getter]
    fn dtype(&self) -> PyDType {
        PyDType {
            inner: self.inner.dtype().clone(),
        }
    }
    
    /// Create an iterator over array elements
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<iterators::PyArrayIterator>> {
        // Create iterator directly
        let inner_ref = slf.get_inner();
        let iter = iterators::PyArrayIterator {
            array: inner_ref.clone(),
            index: 0,
            size: inner_ref.size(),
        };
        Py::new(slf.py(), iter)
    }
    
    /// Get the size (total number of elements)
    #[getter]
    fn size(&self) -> usize {
        self.get_inner().size()
    }
    
    /// Get the number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.get_inner().ndim()
    }
    
    /// Get the itemsize (size of each element in bytes)
    #[getter]
    fn itemsize(&self) -> usize {
        self.get_inner().itemsize()
    }
    
    /// Get the strides
    #[getter]
    fn strides(&self) -> Vec<i64> {
        self.get_inner().strides().to_vec()
    }
    
    /// Check if array is C-contiguous
    #[getter]
    fn is_c_contiguous(&self) -> bool {
        self.get_inner().is_c_contiguous()
    }
    
    /// Check if array is Fortran-contiguous
    #[getter]
    fn is_f_contiguous(&self) -> bool {
        self.get_inner().is_f_contiguous()
    }
    
    /// Check if array is writeable
    #[getter]
    fn is_writeable(&self) -> bool {
        self.get_inner().is_writeable()
    }
    
    /// Create a copy of the array
    fn copy(&self) -> PyResult<Self> {
        let copied = self.inner.copy();
        Ok(PyArray {
            inner: Arc::new(copied),
        })
    }
    
    /// Create a view of the array
    fn view(&self) -> PyResult<Self> {
        // Create a view with the same shape and strides
        let view = self.inner.view(
            self.get_inner().shape().to_vec(),
            self.get_inner().strides().to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(view),
        })
    }
    
    /// Reshape the array
    fn reshape(&self, shape: Vec<i64>) -> PyResult<Self> {
        use raptors_core::shape::shape::validate_reshape_shape;
        validate_reshape_shape(self.get_inner().shape(), &shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Create new array with new shape using view
        let itemsize = self.get_inner().itemsize();
        let new_strides = raptors_core::shape::shape::compute_reshape_strides(&shape, itemsize);
        
        let reshaped = self.get_inner().view(shape, new_strides)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyArray {
            inner: Arc::new(reshaped),
        })
    }
    
    /// Transpose the array
    fn transpose(&self) -> PyResult<Self> {
        use raptors_core::shape::shape::transpose_dimensions;
        let (new_shape, axes) = transpose_dimensions(self.get_inner().shape(), None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Create view with transposed shape and strides
        let mut new_strides = vec![0; new_shape.len()];
        for (i, &axis) in axes.iter().enumerate() {
            new_strides[i] = self.get_inner().strides()[axis as usize];
        }
        
        let transposed = self.get_inner().view(new_shape, new_strides)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyArray {
            inner: Arc::new(transposed),
        })
    }
    
    /// Flatten the array to 1D
    fn flatten(&self) -> PyResult<Self> {
        use raptors_core::shape::shape::flatten_shape;
        let flat_shape = flatten_shape(self.get_inner().shape());
        let itemsize = self.get_inner().itemsize();
        let new_strides = raptors_core::shape::shape::compute_reshape_strides(&flat_shape, itemsize);
        
        let flattened = self.get_inner().view(flat_shape, new_strides)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        Ok(PyArray {
            inner: Arc::new(flattened),
        })
    }
    
    /// Sum array elements along an axis
    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<usize>) -> PyResult<Self> {
        use raptors_core::ufunc::reduction::sum_along_axis;
        let result = sum_along_axis(self.get_inner(), axis)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Maximum of array elements along an axis
    #[pyo3(signature = (axis=None))]
    fn max(&self, axis: Option<usize>) -> PyResult<Self> {
        use raptors_core::ufunc::reduction::max_along_axis;
        let result = max_along_axis(self.get_inner(), axis)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Minimum of array elements along an axis
    #[pyo3(signature = (axis=None))]
    fn min(&self, axis: Option<usize>) -> PyResult<Self> {
        use raptors_core::ufunc::reduction::min_along_axis;
        let result = min_along_axis(self.get_inner(), axis)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        Ok(PyArray {
            inner: Arc::new(result),
        })
    }
    
    /// Convert array to Python list
    fn tolist(&self, py: Python) -> PyResult<Py<PyAny>> {
        use raptors_core::types::NpyType;
        
        let inner = self.get_inner();
        let shape = inner.shape();
        
        // Helper function to convert a value to Python object
        fn value_to_python(py: Python, ptr: *const u8, dtype: &raptors_core::types::DType, offset: usize) -> PyResult<Py<PyAny>> {
            use raptors_core::types::NpyType;
            use NpyType::*;
            let val_ptr = unsafe { ptr.add(offset * dtype.itemsize()) };
            match dtype.type_() {
                Bool => {
                    let val = unsafe { *(val_ptr as *const bool) };
                    let py_obj = unsafe { pyo3::ffi::PyBool_FromLong(val as i64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Byte => {
                    let val = unsafe { *(val_ptr as *const i8) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                UByte => {
                    let val = unsafe { *val_ptr };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Short => {
                    let val = unsafe { *(val_ptr as *const i16) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                UShort => {
                    let val = unsafe { *(val_ptr as *const u16) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Int => {
                    let val = unsafe { *(val_ptr as *const i32) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                UInt => {
                    let val = unsafe { *(val_ptr as *const u32) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Long | LongLong => {
                    let val = unsafe { *(val_ptr as *const i64) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromLongLong(val) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                ULong | ULongLong => {
                    let val = unsafe { *(val_ptr as *const u64) };
                    let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLongLong(val) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Float | Half => {
                    let val = unsafe { *(val_ptr as *const f32) };
                    let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val as f64) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                Double | LongDouble => {
                    let val = unsafe { *(val_ptr as *const f64) };
                    let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val) };
                    Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                    "Dtype not supported for tolist"
                ))
            }
        }
        
        // Recursive function to build nested lists
        fn build_list(
            py: Python,
            ptr: *const u8,
            dtype: &raptors_core::types::DType,
            shape: &[i64],
            strides: &[i64],
            base_offset: usize,
        ) -> PyResult<Py<PyAny>> {
            if shape.is_empty() {
                // Scalar - return the value
                return value_to_python(py, ptr, dtype, base_offset);
            }
            
            if shape.len() == 1 {
                // 1D - return a list of values
                let mut list = Vec::new();
                let dim = shape[0] as usize;
                for i in 0..dim {
                    let offset = base_offset + i * strides[0] as usize / dtype.itemsize();
                    let val = value_to_python(py, ptr, dtype, offset)?;
                    list.push(val);
                }
                // Create Python list from Vec
                let py_list = PyList::empty(py);
                for item in list {
                    py_list.append(item)?;
                }
                Ok(py_list.into())
            } else {
                // Multi-dimensional - recursively build nested lists
                let py_list = PyList::empty(py);
                let dim = shape[0] as usize;
                for i in 0..dim {
                    let sub_shape = &shape[1..];
                    let sub_strides = &strides[1..];
                    let offset = base_offset + i * strides[0] as usize / dtype.itemsize();
                    let sub_list = build_list(py, ptr, dtype, sub_shape, sub_strides, offset)?;
                    py_list.append(sub_list)?;
                }
                Ok(py_list.into())
            }
        }
        
        let strides = inner.strides();
        build_list(py, inner.data_ptr(), inner.dtype(), shape, strides, 0)
    }
    
    /// Convert array to a different dtype
    fn astype(&self, dtype: &PyDType) -> PyResult<Self> {
        // For now, create a new array with the target dtype and copy data
        // This is a simplified implementation - full version would handle type conversions
        let target_dtype = dtype.get_inner().clone();
        let shape = self.get_inner().shape().to_vec();
        let mut new_array = empty(shape, target_dtype.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Copy data (simplified - assumes compatible types)
        // Full implementation would use type conversion
        let src = self.get_inner();
        let src_size = src.size() * src.itemsize();
        let dst_size = new_array.size() * new_array.itemsize();
        let copy_size = src_size.min(dst_size);
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.data_ptr(),
                new_array.data_ptr_mut(),
                copy_size,
            );
        }
        
        Ok(PyArray {
            inner: Arc::new(new_array),
        })
    }
    
    /// Get item at index
    fn __getitem__(&self, py: Python, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Try tuple indexing (multi-dimensional: arr[0, 0])
        // In Python, arr[0, 0] is passed as a tuple to __getitem__
        // For PyO3 0.27, use Bound::cast instead of downcast
        if let Ok(tuple) = index.cast::<PyTuple>() {
            let ndim = self.inner.ndim();
            if tuple.len() == ndim {
                let mut indices = Vec::new();
                let shape = self.inner.shape();
                for i in 0..tuple.len() {
                    let item = tuple.get_item(i)?;
                    let mut idx: i64 = item.extract()?;
                    
                    // Normalize negative index
                    if idx < 0 {
                        idx += shape[i] as i64;
                    }
                    
                    // Bounds check
                    if idx < 0 || idx >= shape[i] as i64 {
                        return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            format!("Index {} out of bounds for dimension {} of size {}", idx, i, shape[i])
                        ));
                    }
                    
                    indices.push(idx);
                }
                
                let ptr = index_array(&self.inner, &indices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{}", e)))?;
                
                return self.extract_value(py, ptr);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Expected {} indices, got {}", ndim, tuple.len())
                ));
            }
        }
        
        // Try integer indexing (including negative indices)
        if let Ok(mut idx) = index.extract::<i64>() {
            if self.inner.ndim() == 1 {
                let shape = self.inner.shape();
                let dim_size = shape[0] as i64;
                
                // Normalize negative index
                if idx < 0 {
                    idx += dim_size;
                }
                
                // Bounds check
                if idx < 0 || idx >= dim_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Index {} out of bounds for dimension of size {}", idx, dim_size)
                    ));
                }
                
                let indices = vec![idx];
                let ptr = index_array(&self.inner, &indices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{}", e)))?;
                
                // Extract value based on dtype
                return self.extract_value(py, ptr);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    format!("Single index provided but array has {} dimensions", self.inner.ndim())
                ));
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Advanced indexing not yet implemented"
        ))
    }
    
    /// Set item at index
    fn __setitem__(&mut self, py: Python, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try integer indexing (including negative indices)
        if let Ok(mut idx) = index.extract::<i64>() {
            if self.inner.ndim() == 1 {
                let shape = self.inner.shape();
                let dim_size = shape[0] as i64;
                
                // Normalize negative index
                if idx < 0 {
                    idx += dim_size;
                }
                
                // Bounds check
                if idx < 0 || idx >= dim_size {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        format!("Index {} out of bounds for dimension of size {}", idx, dim_size)
                    ));
                }
                
                let indices = vec![idx];
                let ptr = index_array(&self.inner, &indices)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!("{}", e)))?;
                
                // Set value based on dtype (cast to mutable)
                return Self::set_value_static(py, ptr as *mut u8, value, self.get_inner().dtype());
            }
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Advanced indexing not yet implemented"
        ))
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("Array(shape={:?}, dtype={})", self.get_inner().shape(), self.get_inner().dtype().name())
    }
    
    /// String representation
    fn __str__(&self) -> String {
        self.__repr__()
    }
    
    /// Check if this is an instance of a type (isinstance equivalent)
    fn isinstance(&self, type_name: String) -> bool {
        // For now, always return true for "Array" or "PyArray"
        // Full implementation would check actual type hierarchy
        type_name == "Array" || type_name == "PyArray" || type_name == "raptors.Array"
    }
    
    /// Get the class name
    fn __class__(&self) -> String {
        "PyArray".to_string()
    }
    
    /// Array protocol - return self for NumPy compatibility
    fn __array__(&self) -> PyResult<Self> {
        // Return self - PyArray implements the array protocol
        // NumPy will call this method to convert array-like objects
        Ok(PyArray {
            inner: self.inner.clone(),
        })
    }
    
    /// Convert to NumPy array (convenience method)
    fn to_numpy(&self, py: Python) -> PyResult<Py<PyAny>> {
        crate::numpy_interop::to_numpy(py, self)
    }
    
    /// Create from NumPy array (class method)
    #[staticmethod]
    fn from_numpy(py: Python, np_array: &Bound<'_, PyAny>) -> PyResult<Self> {
        crate::numpy_interop::from_numpy(py, np_array)
    }
    
    /// Addition operator
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try to extract as PyArray first
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = add(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            // Scalar addition: array + scalar
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = add(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for addition"
            ))
        }
    }
    
    /// Subtraction operator
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = subtract(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = subtract(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for subtraction"
            ))
        }
    }
    
    /// Multiplication operator
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = multiply(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = multiply(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for multiplication"
            ))
        }
    }
    
    /// True division operator
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = divide(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = divide(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for division"
            ))
        }
    }
    
    /// Equality operator
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = equal(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            // For non-array types, return False array
            let shape = self.get_inner().shape().to_vec();
            let bool_dtype = DType::new(NpyType::Bool);
            let result = zeros(shape, bool_dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        }
    }
    
    /// Less than operator
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = less(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = less(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for comparison"
            ))
        }
    }
    
    /// Greater than operator
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = greater(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = greater(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for comparison"
            ))
        }
    }
    
    /// Not equal operator
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = not_equal(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            // For non-array types, return True array
            let shape = self.get_inner().shape().to_vec();
            let bool_dtype = DType::new(NpyType::Bool);
            let mut result = ones(shape, bool_dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        }
    }
    
    /// Less than or equal operator
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = less_equal(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = less_equal(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for comparison"
            ))
        }
    }
    
    /// Greater than or equal operator
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(other_array) = other.downcast::<PyArray>() {
            let result = greater_equal(self.get_inner(), PyArray::get_inner_from_bound(&other_array))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = greater_equal(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for comparison"
            ))
        }
    }
    
    /// In-place addition operator
    fn __iadd__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(arr) = other.downcast::<PyArray>() {
            add(self.get_inner(), PyArray::get_inner_from_bound(&arr))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else if let Ok(val) = other.extract::<f64>() {
            // Create scalar array
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            add(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for in-place addition"
            ));
        };
        // Try to modify in-place if we have unique ownership
        if let Some(inner_mut) = Arc::get_mut(&mut self.inner) {
            // Copy result data into existing array
            if inner_mut.shape() == result.shape() && inner_mut.dtype().type_() == result.dtype().type_() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.data_ptr(),
                        inner_mut.data_ptr_mut(),
                        inner_mut.size() * inner_mut.itemsize(),
                    );
                }
                Ok(())
            } else {
                // Shape or dtype mismatch - replace the array
                self.inner = Arc::new(result);
                Ok(())
            }
        } else {
            // No unique ownership - replace the array
            self.inner = Arc::new(result);
            Ok(())
        }
    }
    
    /// In-place subtraction operator
    fn __isub__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(arr) = other.downcast::<PyArray>() {
            subtract(self.get_inner(), PyArray::get_inner_from_bound(&arr))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            subtract(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for in-place subtraction"
            ));
        };
        if let Some(inner_mut) = Arc::get_mut(&mut self.inner) {
            if inner_mut.shape() == result.shape() && inner_mut.dtype().type_() == result.dtype().type_() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.data_ptr(),
                        inner_mut.data_ptr_mut(),
                        inner_mut.size() * inner_mut.itemsize(),
                    );
                }
                Ok(())
            } else {
                self.inner = Arc::new(result);
                Ok(())
            }
        } else {
            self.inner = Arc::new(result);
            Ok(())
        }
    }
    
    /// In-place multiplication operator
    fn __imul__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(arr) = other.downcast::<PyArray>() {
            multiply(self.get_inner(), PyArray::get_inner_from_bound(&arr))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            multiply(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for in-place multiplication"
            ));
        };
        if let Some(inner_mut) = Arc::get_mut(&mut self.inner) {
            if inner_mut.shape() == result.shape() && inner_mut.dtype().type_() == result.dtype().type_() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.data_ptr(),
                        inner_mut.data_ptr_mut(),
                        inner_mut.size() * inner_mut.itemsize(),
                    );
                }
                Ok(())
            } else {
                self.inner = Arc::new(result);
                Ok(())
            }
        } else {
            self.inner = Arc::new(result);
            Ok(())
        }
    }
    
    /// In-place true division operator
    fn __itruediv__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = if let Ok(arr) = other.downcast::<PyArray>() {
            divide(self.get_inner(), PyArray::get_inner_from_bound(&arr))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            divide(self.get_inner(), &scalar_array)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for in-place division"
            ));
        };
        if let Some(inner_mut) = Arc::get_mut(&mut self.inner) {
            if inner_mut.shape() == result.shape() && inner_mut.dtype().type_() == result.dtype().type_() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.data_ptr(),
                        inner_mut.data_ptr_mut(),
                        inner_mut.size() * inner_mut.itemsize(),
                    );
                }
                Ok(())
            } else {
                self.inner = Arc::new(result);
                Ok(())
            }
        } else {
            self.inner = Arc::new(result);
            Ok(())
        }
    }
    
    /// Right-hand addition (scalar + array)
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Try to extract scalar value
        if let Ok(val) = other.extract::<f64>() {
            // Create scalar array with same shape
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            // Fill with scalar value
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = add(&scalar_array, self.get_inner())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for right-hand addition"
            ))
        }
    }
    
    /// Right-hand subtraction (scalar - array)
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = subtract(&scalar_array, self.get_inner())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for right-hand subtraction"
            ))
        }
    }
    
    /// Right-hand multiplication (scalar * array)
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(val) = other.extract::<f64>() {
            let shape = self.get_inner().shape().to_vec();
            let dtype = self.get_inner().dtype().clone();
            let mut scalar_array = empty(shape, dtype)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            unsafe {
                let ptr = scalar_array.data_ptr_mut() as *mut f64;
                for i in 0..scalar_array.size() {
                    *ptr.add(i) = val;
                }
            }
            let result = multiply(&scalar_array, self.get_inner())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
            Ok(PyArray {
                inner: Arc::new(result),
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported type for right-hand multiplication"
            ))
        }
    }
}

impl PyArray {
    /// Extract value from pointer based on dtype
    fn extract_value(&self, py: Python, ptr: *const u8) -> PyResult<Py<PyAny>> {
        use raptors_core::types::NpyType;
        use NpyType::*;
        match self.get_inner().dtype().type_() {
            Bool => {
                let val = unsafe { *(ptr as *const bool) };
                let py_obj = unsafe { pyo3::ffi::PyBool_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Byte => {
                let val = unsafe { *(ptr as *const i8) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UByte => {
                let val = unsafe { *ptr };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Short => {
                let val = unsafe { *(ptr as *const i16) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UShort => {
                let val = unsafe { *(ptr as *const u16) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Int => {
                let val = unsafe { *(ptr as *const i32) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLong(val as i64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            UInt => {
                let val = unsafe { *(ptr as *const u32) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLong(val as u64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Long | LongLong => {
                let val = unsafe { *(ptr as *const i64) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromLongLong(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            ULong | ULongLong => {
                let val = unsafe { *(ptr as *const u64) };
                let py_obj = unsafe { pyo3::ffi::PyLong_FromUnsignedLongLong(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Float | Half => {
                let val = unsafe { *(ptr as *const f32) };
                let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val as f64) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            Double | LongDouble => {
                let val = unsafe { *(ptr as *const f64) };
                let py_obj = unsafe { pyo3::ffi::PyFloat_FromDouble(val) };
                Ok(unsafe { Py::from_owned_ptr_or_err(py, py_obj)? })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Dtype not supported for indexing"
            ))
        }
    }
    
    /// Set value at pointer based on dtype (static method)
    fn set_value_static(_py: Python, ptr: *mut u8, value: &Bound<'_, PyAny>, dtype: &raptors_core::types::DType) -> PyResult<()> {
        use raptors_core::types::NpyType;
        use NpyType::*;
        match dtype.type_() {
            Bool => {
                let val: bool = value.extract()?;
                unsafe { *(ptr as *mut bool) = val; }
            }
            Byte => {
                let val: i8 = value.extract()?;
                unsafe { *(ptr as *mut i8) = val; }
            }
            UByte => {
                let val: u8 = value.extract()?;
                unsafe { *ptr = val; }
            }
            Short => {
                let val: i16 = value.extract()?;
                unsafe { *(ptr as *mut i16) = val; }
            }
            UShort => {
                let val: u16 = value.extract()?;
                unsafe { *(ptr as *mut u16) = val; }
            }
            Int => {
                let val: i32 = value.extract()?;
                unsafe { *(ptr as *mut i32) = val; }
            }
            UInt => {
                let val: u32 = value.extract()?;
                unsafe { *(ptr as *mut u32) = val; }
            }
            Long | LongLong => {
                let val: i64 = value.extract()?;
                unsafe { *(ptr as *mut i64) = val; }
            }
            ULong | ULongLong => {
                let val: u64 = value.extract()?;
                unsafe { *(ptr as *mut u64) = val; }
            }
            Float | Half => {
                let val: f32 = value.extract()?;
                unsafe { *(ptr as *mut f32) = val; }
            }
            Double | LongDouble => {
                let val: f64 = value.extract()?;
                unsafe { *(ptr as *mut f64) = val; }
            }
            _ => return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "Dtype not supported for indexing"
            ))
        }
        Ok(())
    }
}

