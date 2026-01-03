//! NumPy testing utilities
//!
//! This module provides utilities matching numpy.testing functionality
//! for porting NumPy tests to Rust.

use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::shape::shape::squeeze_dims;
use std::sync::Arc;

/// Assert that two arrays are equal element-wise
///
/// # Panics
/// Panics if arrays have different shapes, dtypes, or element values
pub fn assert_array_equal(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape(), "Arrays must have the same shape");
    assert_eq!(a.dtype().type_(), b.dtype().type_(), "Arrays must have the same dtype");
    
    let size = a.size();
    assert_eq!(size, b.size(), "Arrays must have the same size");
    
    match a.dtype().type_() {
        NpyType::Bool => {
            unsafe {
                let a_ptr = a.data_ptr() as *const bool;
                let b_ptr = b.data_ptr() as *const bool;
                for i in 0..size {
                    assert_eq!(
                        *a_ptr.add(i),
                        *b_ptr.add(i),
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
        NpyType::Double => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                for i in 0..size {
                    assert_eq!(
                        *a_ptr.add(i),
                        *b_ptr.add(i),
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
        NpyType::Float => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f32;
                let b_ptr = b.data_ptr() as *const f32;
                for i in 0..size {
                    assert_eq!(
                        *a_ptr.add(i),
                        *b_ptr.add(i),
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
        NpyType::Int => {
            unsafe {
                let a_ptr = a.data_ptr() as *const i32;
                let b_ptr = b.data_ptr() as *const i32;
                for i in 0..size {
                    assert_eq!(
                        *a_ptr.add(i),
                        *b_ptr.add(i),
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
        NpyType::LongLong => {
            unsafe {
                let a_ptr = a.data_ptr() as *const i64;
                let b_ptr = b.data_ptr() as *const i64;
                for i in 0..size {
                    assert_eq!(
                        *a_ptr.add(i),
                        *b_ptr.add(i),
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
        _ => {
            // For other types, do byte-wise comparison
            unsafe {
                let a_ptr = a.data_ptr();
                let b_ptr = b.data_ptr();
                let itemsize = a.itemsize();
                for i in 0..size {
                    let a_bytes = std::slice::from_raw_parts(a_ptr.add(i * itemsize), itemsize);
                    let b_bytes = std::slice::from_raw_parts(b_ptr.add(i * itemsize), itemsize);
                    assert_eq!(
                        a_bytes,
                        b_bytes,
                        "Arrays differ at index {}",
                        i
                    );
                }
            }
        }
    }
}

/// Assert that two floating-point arrays are almost equal
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Panics
/// Panics if arrays are not close within tolerance
pub fn assert_array_almost_equal(a: &Array, b: &Array, rtol: f64, atol: f64) {
    assert_eq!(a.shape(), b.shape(), "Arrays must have the same shape");
    assert_eq!(a.size(), b.size(), "Arrays must have the same size");
    
    let size = a.size();
    
    match (a.dtype().type_(), b.dtype().type_()) {
        (NpyType::Double, NpyType::Double) => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                for i in 0..size {
                    let a_val = *a_ptr.add(i);
                    let b_val = *b_ptr.add(i);
                    let diff = (a_val - b_val).abs();
                    let tol = atol + rtol * b_val.abs();
                    assert!(
                        diff <= tol || (a_val.is_nan() && b_val.is_nan()) || (a_val.is_infinite() && b_val.is_infinite() && a_val.signum() == b_val.signum()),
                        "Arrays differ at index {}: {} vs {} (diff: {}, tol: {})",
                        i, a_val, b_val, diff, tol
                    );
                }
            }
        }
        (NpyType::Float, NpyType::Float) => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f32;
                let b_ptr = b.data_ptr() as *const f32;
                let rtol_f32 = rtol as f32;
                let atol_f32 = atol as f32;
                for i in 0..size {
                    let a_val = *a_ptr.add(i);
                    let b_val = *b_ptr.add(i);
                    let diff = (a_val - b_val).abs();
                    let tol = atol_f32 + rtol_f32 * b_val.abs();
                    assert!(
                        diff <= tol || (a_val.is_nan() && b_val.is_nan()) || (a_val.is_infinite() && b_val.is_infinite() && a_val.signum() == b_val.signum()),
                        "Arrays differ at index {}: {} vs {} (diff: {}, tol: {})",
                        i, a_val, b_val, diff, tol
                    );
                }
            }
        }
        _ => {
            panic!("assert_array_almost_equal only supports floating-point types");
        }
    }
}

/// Assert that all elements of array a are less than corresponding elements of array b
///
/// # Panics
/// Panics if any element of a is not less than the corresponding element of b
pub fn assert_array_less(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape(), "Arrays must have the same shape");
    assert_eq!(a.size(), b.size(), "Arrays must have the same size");
    
    let size = a.size();
    
    match (a.dtype().type_(), b.dtype().type_()) {
        (NpyType::Double, NpyType::Double) => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f64;
                let b_ptr = b.data_ptr() as *const f64;
                for i in 0..size {
                    assert!(
                        *a_ptr.add(i) < *b_ptr.add(i),
                        "Array a[{}] = {} is not less than b[{}] = {}",
                        i, *a_ptr.add(i), i, *b_ptr.add(i)
                    );
                }
            }
        }
        (NpyType::Float, NpyType::Float) => {
            unsafe {
                let a_ptr = a.data_ptr() as *const f32;
                let b_ptr = b.data_ptr() as *const f32;
                for i in 0..size {
                    assert!(
                        *a_ptr.add(i) < *b_ptr.add(i),
                        "Array a[{}] = {} is not less than b[{}] = {}",
                        i, *a_ptr.add(i), i, *b_ptr.add(i)
                    );
                }
            }
        }
        (NpyType::Int, NpyType::Int) => {
            unsafe {
                let a_ptr = a.data_ptr() as *const i32;
                let b_ptr = b.data_ptr() as *const i32;
                for i in 0..size {
                    assert!(
                        *a_ptr.add(i) < *b_ptr.add(i),
                        "Array a[{}] = {} is not less than b[{}] = {}",
                        i, *a_ptr.add(i), i, *b_ptr.add(i)
                    );
                }
            }
        }
        _ => {
            panic!("assert_array_less not implemented for this dtype combination");
        }
    }
}

/// Assert that all elements of two arrays are close (allclose)
///
/// This is similar to assert_array_almost_equal but uses NumPy's allclose semantics:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Panics
/// Panics if arrays are not all close
pub fn assert_allclose(a: &Array, b: &Array, rtol: f64, atol: f64) {
    assert_array_almost_equal(a, b, rtol, atol);
}

/// Assert that a scalar value is almost equal to another
///
/// # Arguments
/// * `a` - First value
/// * `b` - Second value
/// * `decimal` - Number of decimal places to compare (default: 7)
///
/// # Panics
/// Panics if values are not almost equal
pub fn assert_almost_equal_f64(a: f64, b: f64, decimal: i32) {
    let tol = 10.0_f64.powi(-decimal);
    let diff = (a - b).abs();
    assert!(
        diff <= tol || (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum()),
        "Values {} and {} differ by {} (tolerance: {})",
        a, b, diff, tol
    );
}

/// Assert that a scalar value is almost equal to another (f32 version)
pub fn assert_almost_equal_f32(a: f32, b: f32, decimal: i32) {
    let tol = 10.0_f32.powi(-decimal);
    let diff = (a - b).abs();
    assert!(
        diff <= tol || (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite() && a.signum() == b.signum()),
        "Values {} and {} differ by {} (tolerance: {})",
        a, b, diff, tol
    );
}

/// Test data generators

/// Generate a random array with specified shape and dtype
pub fn random_array(shape: Vec<i64>, dtype: DType) -> Array {
    use raptors_core::array::Array;
    use raptors_core::types::NpyType;
    
    let mut arr = Array::new(shape, dtype).unwrap();
    let size = arr.size();
    
    match arr.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f64;
                for i in 0..size {
                    // Simple pseudo-random: use index as seed
                    let val = ((i * 1103515245 + 12345) % 2147483647) as f64 / 2147483647.0;
                    *ptr.add(i) = val;
                }
            }
        }
        NpyType::Float => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f32;
                for i in 0..size {
                    let val = ((i * 1103515245 + 12345) % 2147483647) as f32 / 2147483647.0;
                    *ptr.add(i) = val;
                }
            }
        }
        _ => {
            // For other types, fill with zeros for now
            unsafe {
                let itemsize = arr.itemsize();
                let ptr = arr.data_ptr_mut();
                std::ptr::write_bytes(ptr, 0, size * itemsize);
            }
        }
    }
    
    arr
}

/// Generate an array filled with NaN values
pub fn nan_array(shape: Vec<i64>, dtype: DType) -> Array {
    use raptors_core::array::Array;
    use raptors_core::types::NpyType;
    
    let mut arr = Array::new(shape, dtype).unwrap();
    let size = arr.size();
    
    match arr.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f64;
                for i in 0..size {
                    *ptr.add(i) = f64::NAN;
                }
            }
        }
        NpyType::Float => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f32;
                for i in 0..size {
                    *ptr.add(i) = f32::NAN;
                }
            }
        }
        _ => {
            // NaN not applicable for non-float types
        }
    }
    
    arr
}

/// Generate an array filled with Infinity values
pub fn inf_array(shape: Vec<i64>, dtype: DType) -> Array {
    use raptors_core::array::Array;
    use raptors_core::types::NpyType;
    
    let mut arr = Array::new(shape, dtype).unwrap();
    let size = arr.size();
    
    match arr.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f64;
                for i in 0..size {
                    *ptr.add(i) = f64::INFINITY;
                }
            }
        }
        NpyType::Float => {
            unsafe {
                let ptr = arr.data_ptr_mut() as *mut f32;
                for i in 0..size {
                    *ptr.add(i) = f32::INFINITY;
                }
            }
        }
        _ => {
            // Infinity not applicable for non-float types
        }
    }
    
    arr
}

/// Helper to create test arrays with specific patterns
pub mod test_data {
    use raptors_core::array::Array;
    use raptors_core::types::{DType, NpyType};
    
    /// Create an array with sequential values [0, 1, 2, ...]
    pub fn sequential(shape: Vec<i64>, dtype: DType) -> Array {
        let mut arr = Array::new(shape, dtype).unwrap();
        let size = arr.size();
        
        match arr.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let ptr = arr.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *ptr.add(i) = i as f64;
                    }
                }
            }
            NpyType::Float => {
                unsafe {
                    let ptr = arr.data_ptr_mut() as *mut f32;
                    for i in 0..size {
                        *ptr.add(i) = i as f32;
                    }
                }
            }
            NpyType::Int => {
                unsafe {
                    let ptr = arr.data_ptr_mut() as *mut i32;
                    for i in 0..size {
                        *ptr.add(i) = i as i32;
                    }
                }
            }
            _ => {
                // For other types, fill with zeros
                unsafe {
                    let itemsize = arr.itemsize();
                    let ptr = arr.data_ptr_mut();
                    std::ptr::write_bytes(ptr, 0, size * itemsize);
                }
            }
        }
        
        arr
    }
}

/// Squeeze an array along a specific axis
///
/// Removes the specified axis if it has size 1, creating a zero-copy view.
///
/// # Arguments
/// * `array` - The array to squeeze
/// * `axis` - The axis to remove (must have size 1)
///
/// # Returns
/// A view of the array with the specified axis removed.
///
/// # Errors
/// Returns an error if the axis is out of bounds or doesn't have size 1.
pub fn squeeze_axis(array: &Array, axis: usize) -> Result<Array, String> {
    let shape = array.shape();
    let strides = array.strides();
    
    // Check if axis is valid and has size 1
    if axis >= shape.len() {
        return Err(format!("Axis {} is out of bounds for array with {} dimensions", axis, shape.len()));
    }
    
    if shape[axis] != 1 {
        return Err(format!("Cannot squeeze axis {}: dimension has size {}, not 1", axis, shape[axis]));
    }
    
    // Compute new shape with axis removed
    let new_shape = squeeze_dims(shape, Some(axis));
    
    // Compute new strides by removing the stride for the squeezed axis
    // This preserves the memory layout of the view
    let mut new_strides = Vec::new();
    for (i, &stride) in strides.iter().enumerate() {
        if i != axis {
            new_strides.push(stride);
        }
    }
    
    // Wrap array in Arc to create a view
    let base_arc = Arc::new(array.clone());
    
    // Create view with squeezed shape and adjusted strides
    Array::view_from_arc(&base_arc, new_shape, new_strides)
        .map_err(|e| format!("Failed to create squeezed view: {:?}", e))
}

