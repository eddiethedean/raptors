//! Arithmetic operations
//!
//! High-level arithmetic operations built on ufuncs

use crate::array::{Array, ArrayError};
use crate::broadcasting::broadcast_shapes;
use crate::types::NpyType;
use crate::conversion::{promote_dtypes, convert_array, ConversionError};
use crate::ufunc::{create_add_ufunc, create_subtract_ufunc, create_multiply_ufunc, create_divide_ufunc, create_ufunc_loop};

/// Add two arrays
///
/// Returns a new array with the result of element-wise addition
/// Supports broadcasting and type promotion
pub fn add(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    // Promote types
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    // Convert inputs to promoted type if needed
    let a1_converted = if a1.dtype().type_() != output_dtype.type_() {
        convert_array(a1, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a1.copy()
    };
    
    let a2_converted = if a2.dtype().type_() != output_dtype.type_() {
        convert_array(a2, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a2.copy()
    };
    
    // Create output array
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    // Use ufunc system for broadcasting support
    let add_ufunc = create_add_ufunc();
    let inputs = vec![&a1_converted, &a2_converted];
    
    // Try using ufunc system first (handles broadcasting)
    if let Ok(()) = create_ufunc_loop(&add_ufunc, &inputs, &mut output) {
        return Ok(output);
    }
    
    // Fallback to simple same-shape, same-type implementation
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) + *in2_ptr.add(i);
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

/// Subtract two arrays
pub fn subtract(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    // Promote types
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    // Convert inputs to promoted type if needed
    let a1_converted = if a1.dtype().type_() != output_dtype.type_() {
        convert_array(a1, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a1.copy()
    };
    
    let a2_converted = if a2.dtype().type_() != output_dtype.type_() {
        convert_array(a2, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a2.copy()
    };
    
    // Create output array
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    // Use ufunc system for broadcasting support
    let subtract_ufunc = create_subtract_ufunc();
    let inputs = vec![&a1_converted, &a2_converted];
    
    // Try using ufunc system first (handles broadcasting)
    if let Ok(()) = create_ufunc_loop(&subtract_ufunc, &inputs, &mut output) {
        return Ok(output);
    }
    
    // Fallback to simple same-shape, same-type implementation
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) - *in2_ptr.add(i);
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

/// Multiply two arrays
pub fn multiply(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    // Promote types
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    // Convert inputs to promoted type if needed
    let a1_converted = if a1.dtype().type_() != output_dtype.type_() {
        convert_array(a1, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a1.copy()
    };
    
    let a2_converted = if a2.dtype().type_() != output_dtype.type_() {
        convert_array(a2, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a2.copy()
    };
    
    // Create output array
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    // Use ufunc system for broadcasting support
    let multiply_ufunc = create_multiply_ufunc();
    let inputs = vec![&a1_converted, &a2_converted];
    
    // Try using ufunc system first (handles broadcasting)
    if let Ok(()) = create_ufunc_loop(&multiply_ufunc, &inputs, &mut output) {
        return Ok(output);
    }
    
    // Fallback to simple same-shape, same-type implementation
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) * *in2_ptr.add(i);
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

/// Divide two arrays
pub fn divide(a1: &Array, a2: &Array) -> Result<Array, ArrayError> {
    // Compute broadcast shape
    let broadcast_shape = broadcast_shapes(a1.shape(), a2.shape())
        .map_err(|_| ArrayError::InvalidShape)?;
    
    // Promote types
    let output_dtype = promote_dtypes(a1.dtype(), a2.dtype())
        .map_err(|_| ArrayError::TypeMismatch)?;
    
    // Convert inputs to promoted type if needed
    let a1_converted = if a1.dtype().type_() != output_dtype.type_() {
        convert_array(a1, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a1.copy()
    };
    
    let a2_converted = if a2.dtype().type_() != output_dtype.type_() {
        convert_array(a2, output_dtype.clone())
            .map_err(|e| match e {
                ConversionError::ArrayError(ae) => ae,
                ConversionError::UnsupportedConversion => ArrayError::TypeMismatch,
            })?
    } else {
        a2.copy()
    };
    
    // Create output array
    let mut output = Array::new(broadcast_shape, output_dtype)?;
    
    // Use ufunc system for broadcasting support
    let divide_ufunc = create_divide_ufunc();
    let inputs = vec![&a1_converted, &a2_converted];
    
    // Try using ufunc system first (handles broadcasting)
    if let Ok(()) = create_ufunc_loop(&divide_ufunc, &inputs, &mut output) {
        return Ok(output);
    }
    
    // Fallback to simple same-shape, same-type implementation
    if a1.shape() == a2.shape() && a1.dtype().type_() == a2.dtype().type_() {
        let size = a1.size();
        match a1.dtype().type_() {
            NpyType::Double => {
                unsafe {
                    let in1_ptr = a1.data_ptr() as *const f64;
                    let in2_ptr = a2.data_ptr() as *const f64;
                    let out_ptr = output.data_ptr_mut() as *mut f64;
                    for i in 0..size {
                        *out_ptr.add(i) = *in1_ptr.add(i) / *in2_ptr.add(i);
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

