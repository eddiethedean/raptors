//! Comparison operations
//!
//! High-level comparison operations built on ufuncs

use crate::array::{Array, ArrayError};
use crate::broadcasting::broadcast_shapes;
use crate::types::DType;

/// Equal comparison
pub fn equal(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut output = Array::new(broadcast_shape, bool_dtype)?;
    
    // Simplified implementation
    if a1.shape() == a2.shape() {
        let size = a1.size();
        match a1.dtype().type_() {
            crate::types::NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut bool;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) == *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

/// Not equal comparison
pub fn not_equal(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let mut result = equal(a1, a2)?;
    // Negate the result
    unsafe {
        let out_ptr = result.data_ptr_mut() as *mut bool;
        let size = result.size();
        for i in 0..size {
            *out_ptr.add(i) = !*out_ptr.add(i);
        }
    }
    Ok(result)
}

/// Less than comparison
pub fn less(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    let bool_dtype = DType::new(crate::types::NpyType::Bool);
    let mut output = Array::new(broadcast_shape, bool_dtype)?;
    
    if a1.shape() == a2.shape() {
        let size = a1.size();
        match a1.dtype().type_() {
            crate::types::NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut bool;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) < *in2_ptr.add(i);
                    }
                }
            }
            _ => return Err(ArrayError::TypeMismatch),
        }
    } else {
        return Err(ArrayError::TypeMismatch);
    }
    
    Ok(output)
}

/// Greater than comparison
pub fn greater(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Greater is less with swapped arguments
    less(a2, a1)
}

/// Less than or equal comparison
pub fn less_equal(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // less_equal = not greater
    let mut greater_result = greater(a1, a2)?;
    // Negate the result
    unsafe {
        let out_ptr = greater_result.data_ptr_mut() as *mut bool;
        let size = greater_result.size();
        for i in 0..size {
            *out_ptr.add(i) = !*out_ptr.add(i);
        }
    }
    Ok(greater_result)
}

/// Greater than or equal comparison
pub fn greater_equal(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // greater_equal = not less
    let mut less_result = less(a1, a2)?;
    // Negate the result
    unsafe {
        let out_ptr = less_result.data_ptr_mut() as *mut bool;
        let size = less_result.size();
        for i in 0..size {
            *out_ptr.add(i) = !*out_ptr.add(i);
        }
    }
    Ok(less_result)
}

