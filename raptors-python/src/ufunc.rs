//! Ufunc Python bindings
//!
//! This module provides Python bindings for universal functions.

#![allow(clippy::arc_with_non_send_sync)] // Arc used for Python reference counting, not thread safety

use pyo3::prelude::*;
use raptors_core::{empty, operations};
use raptors_core::ufunc::reduction::{sum_along_axis, mean_along_axis, min_along_axis, max_along_axis};
use raptors_core::ufunc::loop_exec::create_unary_ufunc_loop;
use raptors_core::ufunc::advanced::*;
use std::sync::Arc;

use crate::array::PyArray;

/// Add ufunc functions to module
pub fn add_ufuncs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Arithmetic ufuncs (NumPy-named)
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    
    // Comparison ufuncs (NumPy-named)
    m.add_function(wrap_pyfunction!(equal, m)?)?;
    m.add_function(wrap_pyfunction!(less, m)?)?;
    m.add_function(wrap_pyfunction!(greater, m)?)?;
    m.add_function(wrap_pyfunction!(not_equal, m)?)?;
    m.add_function(wrap_pyfunction!(less_equal, m)?)?;
    m.add_function(wrap_pyfunction!(greater_equal, m)?)?;
    
    // Keep old names as aliases for backward compatibility
    m.add_function(wrap_pyfunction!(add_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(divide_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(equal_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(less_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(greater_arrays, m)?)?;
    
    // Math ufuncs
    m.add_function(wrap_pyfunction!(sin, m)?)?;
    m.add_function(wrap_pyfunction!(cos, m)?)?;
    m.add_function(wrap_pyfunction!(tan, m)?)?;
    m.add_function(wrap_pyfunction!(exp, m)?)?;
    m.add_function(wrap_pyfunction!(log, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(abs, m)?)?;
    
    // Reductions
    m.add_function(wrap_pyfunction!(sum, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(min, m)?)?;
    m.add_function(wrap_pyfunction!(max, m)?)?;
    
    Ok(())
}

/// Add two arrays (NumPy-named)
#[pyfunction]
fn add(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    add_arrays(a, b)
}

/// Add two arrays (legacy name)
#[pyfunction]
fn add_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::add(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Subtract two arrays (NumPy-named)
#[pyfunction]
fn subtract(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    subtract_arrays(a, b)
}

/// Subtract two arrays (legacy name)
#[pyfunction]
fn subtract_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::subtract(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Multiply two arrays (NumPy-named)
#[pyfunction]
fn multiply(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    multiply_arrays(a, b)
}

/// Multiply two arrays (legacy name)
#[pyfunction]
fn multiply_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::multiply(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Divide two arrays (NumPy-named)
#[pyfunction]
fn divide(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    divide_arrays(a, b)
}

/// Divide two arrays (legacy name)
#[pyfunction]
fn divide_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::divide(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if arrays are equal (NumPy-named)
#[pyfunction]
fn equal(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    equal_arrays(a, b)
}

/// Check if arrays are equal (legacy name)
#[pyfunction]
fn equal_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::equal(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a < b (NumPy-named)
#[pyfunction]
fn less(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    less_arrays(a, b)
}

/// Check if a < b (legacy name)
#[pyfunction]
fn less_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    let result = operations::less(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a > b (NumPy-named)
#[pyfunction]
fn greater(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    greater_arrays(a, b)
}

/// Check if a > b (legacy name)
#[pyfunction]
fn greater_arrays(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    // Use less with swapped arguments
    less_arrays(b, a)
}

/// Check if arrays are not equal (NumPy-named)
#[pyfunction]
fn not_equal(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    use raptors_core::operations::not_equal as core_not_equal;
    let result = core_not_equal(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a <= b (NumPy-named)
#[pyfunction]
fn less_equal(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    use raptors_core::operations::less_equal as core_less_equal;
    let result = core_less_equal(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Check if a >= b (NumPy-named)
#[pyfunction]
fn greater_equal(a: &PyArray, b: &PyArray) -> PyResult<PyArray> {
    use raptors_core::operations::greater_equal as core_greater_equal;
    let result = core_greater_equal(a.get_inner(), b.get_inner())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Compute sine
#[pyfunction]
fn sin(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_sin_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute cosine
#[pyfunction]
fn cos(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_cos_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute tangent
#[pyfunction]
fn tan(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_tan_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute exponential
#[pyfunction]
fn exp(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_exp_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute natural logarithm
#[pyfunction]
fn log(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_log_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute square root
#[pyfunction]
fn sqrt(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_sqrt_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Compute absolute value
#[pyfunction]
fn abs(a: &PyArray) -> PyResult<PyArray> {
    let ufunc = create_abs_ufunc();
    let inner = a.get_inner();
    let output_dtype = inner.dtype().clone();
    let mut output = empty(inner.shape().to_vec(), output_dtype)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    create_unary_ufunc_loop(&ufunc, a.get_inner(), &mut output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(output),
    })
}

/// Sum array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn sum(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = sum_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Mean of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn mean(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = mean_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Minimum of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn min(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = min_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

/// Maximum of array elements
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
fn max(a: &PyArray, axis: Option<usize>) -> PyResult<PyArray> {
    let result = max_along_axis(a.get_inner(), axis)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
    Ok(PyArray {
        inner: Arc::new(result),
    })
}

