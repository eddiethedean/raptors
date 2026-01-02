//! Raptors Python bindings
//!
//! This module provides Python bindings for Raptors Core using PyO3.

pub mod array;
pub mod dtype;
mod ufunc;
pub mod iterators;
mod numpy_interop;

use pyo3::prelude::*;

/// Python module definition
#[pymodule]
fn raptors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add classes
    m.add_class::<array::PyArray>()?;
    m.add_class::<dtype::PyDType>()?;
    m.add_class::<iterators::PyArrayIterator>()?;
    
    // Add Array as alias for PyArray (NumPy-compatible naming)
    m.add("Array", m.getattr("PyArray")?)?;
    
    // Add module-level convenience functions
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(empty, m)?)?;
    // Register array function with name "array" in Python
    // Explicitly register with name "array" instead of relying on name attribute
    let array_func = wrap_pyfunction!(array_from_list, m)?;
    m.add("array", array_func)?;
    
    // Add DType class alias
    m.add("DType", m.getattr("PyDType")?)?;
    
    // Add ufunc functions
    ufunc::add_ufuncs(m)?;
    
    // Add NumPy interop functions
    m.add_function(wrap_pyfunction!(numpy_interop::from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_interop::to_numpy, m)?)?;
    
    // Add custom dtype functions
    m.add_function(wrap_pyfunction!(dtype::register_custom_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(dtype::get_custom_dtype_id, m)?)?;
    
    // Add module-level constants
    dtype::add_dtype_constants(m)?;
    
    Ok(())
}

/// Create a zero-filled array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn zeros(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{zeros as core_zeros, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_zeros(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

/// Create a one-filled array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn ones(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{ones as core_ones, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_ones(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

/// Create an empty array (module-level function)
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn empty(shape: Vec<i64>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{empty as core_empty, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    let array = core_empty(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)] // Arc needed for Python reference counting
        inner: Arc::new(array),
    })
}

/// Create an array from a Python list (module-level function)
#[pyfunction]
#[pyo3(signature = (data, dtype=None))]
pub fn array_from_list(py: Python, data: &Bound<'_, PyAny>, dtype: Option<&dtype::PyDType>) -> PyResult<array::PyArray> {
    use raptors_core::{Array, DType};
    use raptors_core::types::NpyType;
    use std::sync::Arc;
    
    // Helper function to determine shape and flatten data
    fn extract_shape_and_data(py: Python, obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<(Vec<i64>, Vec<f64>)> {
        if depth > 32 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Array nesting too deep"
            ));
        }
        
        // Try to extract as list
        if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
            if list.is_empty() {
                return Ok((vec![0], vec![]));
            }
            
            // Check if first element is also a list (multi-dimensional)
            let first = list.get_item(0)?;
            if first.downcast::<pyo3::types::PyList>().is_ok() {
                // Multi-dimensional
                let mut shape = Vec::new();
                let mut flat_data = Vec::new();
                let mut first_shape = None;
                
                for item in list.iter() {
                    let (sub_shape, sub_data) = extract_shape_and_data(py, &item, depth + 1)?;
                    if let Some(ref fs) = first_shape {
                        if sub_shape != *fs {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Inconsistent array dimensions"
                            ));
                        }
                    } else {
                        first_shape = Some(sub_shape.clone());
                        shape = sub_shape;
                    }
                    flat_data.extend(sub_data);
                }
                
                shape.insert(0, list.len() as i64);
                return Ok((shape, flat_data));
            } else {
                // 1D list
                let mut flat_data = Vec::new();
                for item in list.iter() {
                    let val: f64 = item.extract()?;
                    flat_data.push(val);
                }
                return Ok((vec![list.len() as i64], flat_data));
            }
        }
        
        // Try to extract as scalar
        if let Ok(val) = obj.extract::<f64>() {
            return Ok((vec![], vec![val]));
        }
        
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot convert to array"
        ))
    }
    
    let (shape, flat_data) = extract_shape_and_data(py, data, 0)?;
    
    let dtype_val = match dtype {
        Some(dt) => dt.get_inner().clone(),
        None => DType::new(NpyType::Double),
    };
    
    let mut array = Array::new(shape, dtype_val)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    
    // Copy data
    unsafe {
        let dst = array.data_ptr_mut() as *mut f64;
        std::ptr::copy_nonoverlapping(flat_data.as_ptr(), dst, flat_data.len());
    }
    
    Ok(array::PyArray {
        #[allow(clippy::arc_with_non_send_sync)]
        inner: Arc::new(array),
    })
}

