//! Iterator Python bindings
//!
//! This module provides Python bindings for array iterators.

use pyo3::prelude::*;
use pyo3::exceptions::PyStopIteration;
use pyo3::types::PyAny;
use std::sync::Arc;
use crate::array::PyArray;

/// Python iterator for arrays
#[pyclass]
pub struct PyArrayIterator {
    // Store array reference to keep data alive
    pub(crate) array: Arc<raptors_core::Array>,
    // Current index for iteration
    pub(crate) index: usize,
    // Size of array
    pub(crate) size: usize,
}

// Legacy unsafe assertions requiring review: Arc keeps storage alive but does not
// prevent another alias from mutating the data while this iterator reads it.
unsafe impl Send for PyArrayIterator {}
unsafe impl Sync for PyArrayIterator {}

#[pymethods]
impl PyArrayIterator {
    /// Create a new iterator
    #[new]
    fn new(array: &PyArray) -> PyResult<Self> {
        // Get inner array from PyArray
        let inner_ref = array.get_inner();
        Ok(PyArrayIterator {
            array: inner_ref.clone(),
            index: 0,
            size: inner_ref.size(),
        })
    }
    
    /// Get next element
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    /// Get next element
    fn __next__(&mut self, py: Python) -> PyResult<Py<PyAny>> {
        if self.index >= self.size {
            return Err(PyStopIteration::new_err("Iterator exhausted"));
        }
        
        // Calculate pointer to current element
        let ptr = self.get_element_ptr(self.index);
        self.index += 1;
        
        // Extract value based on dtype
        self.extract_value(py, ptr)
    }
}

impl PyArrayIterator {
    /// Get pointer to element at index
    fn get_element_ptr(&self, index: usize) -> *const u8 {
        use raptors_core::indexing::index_array;
        // For 1D arrays, use simple indexing
        if self.array.ndim() == 1 {
            if let Ok(ptr) = index_array(&self.array, &[index as i64]) {
                return ptr;
            }
        }
        // Fallback: calculate offset manually for 1D
        let offset = index * self.array.itemsize();
        unsafe {
            self.array.data_ptr().add(offset)
        }
    }
    
    /// Extract value from pointer based on dtype
    fn extract_value(&self, py: Python, ptr: *const u8) -> PyResult<Py<PyAny>> {
        use raptors_core::types::NpyType;
        use NpyType::*;
        match self.array.dtype().type_() {
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
                "Dtype not supported for iteration"
            ))
        }
    }
}
