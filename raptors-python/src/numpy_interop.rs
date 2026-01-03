//! NumPy interoperability
//!
//! This module provides seamless interoperability with NumPy arrays.

use pyo3::prelude::*;
use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use raptors_core::{Array, DType};
use raptors_core::types::NpyType;
use std::sync::Arc;
use crate::array::PyArray;

/// Convert NumPy array to Raptors array
#[pyfunction]
pub fn from_numpy(_py: Python, np_array: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    // Try to get as PyArrayDyn first (most general)
    #[allow(deprecated)] // TODO: Migrate to Bound::cast when numpy crate supports it
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<f64>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Double);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        // Copy data
        unsafe {
            let dst = array.data_ptr_mut() as *mut f64;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
            inner: Arc::new(array),
        });
    }
    
    // Try float32
    #[allow(deprecated)] // TODO: Migrate to Bound::cast when numpy crate supports it
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<f32>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Float);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
            inner: Arc::new(array),
        });
    }
    
    // Try int32
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<i32>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Int);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut i32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try int64
    #[allow(deprecated)] // TODO: Migrate to Bound::cast when numpy crate supports it
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<i64>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::LongLong);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut i64;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
            inner: Arc::new(array),
        });
    }
    
    // Try int16
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<i16>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Short);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut i16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try int8
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<i8>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Byte);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut i8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try uint64
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<u64>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::ULongLong);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut u64;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try uint32
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<u32>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::UInt);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut u32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try uint16
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<u16>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::UShort);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut u16;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try uint8
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<u8>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::UByte);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    // Try bool
    #[allow(deprecated)]
    if let Ok(np_arr) = np_array.downcast::<PyArrayDyn<bool>>() {
        let shape = np_arr.shape().to_vec();
        let readonly = np_arr.readonly();
        let data = readonly.as_slice()?;
        
        let dtype = DType::new(NpyType::Bool);
        let mut array = Array::new(shape.iter().map(|&x| x as i64).collect(), dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        
        unsafe {
            let dst = array.data_ptr_mut() as *mut bool;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        
        return Ok(PyArray {
            #[allow(clippy::arc_with_non_send_sync)]
            inner: Arc::new(array),
        });
    }
    
    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "Unsupported NumPy array type"
    ))
}

/// Convert Raptors array to NumPy array
#[pyfunction]
pub fn to_numpy(py: Python, array: &PyArray) -> PyResult<Py<PyAny>> {
    let inner = array.get_inner();
    let shape: Vec<usize> = inner.shape().iter().map(|&x| x as usize).collect();
    
    match inner.dtype().type_() {
        NpyType::Double => {
            // Create NumPy array with correct shape using unsafe new
            let np_array = unsafe { PyArrayDyn::<f64>::new(py, shape.as_slice(), false) };
            // Copy data
            unsafe {
                let src = inner.data_ptr() as *const f64;
                let dst = np_array.as_slice_mut()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?
                    .as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, inner.size());
            }
            Ok(np_array.into())
        }
        NpyType::Float => {
            // Create NumPy array with correct shape using unsafe new
            let np_array = unsafe { PyArrayDyn::<f32>::new(py, shape.as_slice(), false) };
            // Copy data
            unsafe {
                let src = inner.data_ptr() as *const f32;
                let dst = np_array.as_slice_mut()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?
                    .as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, inner.size());
            }
            Ok(np_array.into())
        }
        NpyType::Int => {
            let np_array = unsafe { PyArrayDyn::<i32>::new(py, shape.as_slice(), false) };
            unsafe {
                let src = inner.data_ptr() as *const i32;
                let dst = np_array.as_slice_mut()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?
                    .as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, inner.size());
            }
            Ok(np_array.into())
        }
        NpyType::UInt => {
            let np_array = unsafe { PyArrayDyn::<u32>::new(py, shape.as_slice(), false) };
            unsafe {
                let src = inner.data_ptr() as *const u32;
                let dst = np_array.as_slice_mut()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?
                    .as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, inner.size());
            }
            Ok(np_array.into())
        }
        NpyType::Bool => {
            let np_array = unsafe { PyArrayDyn::<bool>::new(py, shape.as_slice(), false) };
            unsafe {
                let src = inner.data_ptr() as *const bool;
                let dst = np_array.as_slice_mut()
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to get mutable slice: {}", e)))?
                    .as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, inner.size());
            }
            Ok(np_array.into())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Conversion to NumPy not yet implemented for this dtype"
        ))
    }
}

