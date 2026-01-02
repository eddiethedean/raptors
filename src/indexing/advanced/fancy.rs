//! Fancy indexing implementation
//!
//! This module provides fancy indexing (integer array indexing),
//! equivalent to NumPy's fancy indexing functionality

use crate::array::Array;
use crate::types::NpyType;
use crate::indexing::IndexError;

/// Fancy index an array using an index array
///
/// Returns a new array with elements selected by the index array.
/// The index array must be 1D and contain integer indices.
pub fn fancy_index_array(array: &Array, indices: &Array) -> Result<Array, IndexError> {
    // Validate index array
    if indices.ndim() != 1 {
        return Err(IndexError::DimensionMismatch);
    }
    
    // Index array must be integer type
    match indices.dtype().type_() {
        NpyType::Int => {}
        _ => return Err(IndexError::InvalidIndex),
    }
    
    let array_shape = array.shape();
    let index_size = indices.size();
    
    // Handle negative indices and validate bounds
    let normalized_indices: Vec<i64> = unsafe {
        let index_ptr = indices.data_ptr() as *const i32;
        (0..index_size)
            .map(|i| {
                let mut idx = *index_ptr.add(i) as i64;
                // Normalize negative indices
                if idx < 0 {
                    idx += array_shape[0];
                }
                // Validate bounds
                if idx < 0 || idx >= array_shape[0] {
                    return Err(IndexError::OutOfBounds);
                }
                Ok(idx)
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    
    // Create output array with shape from indices
    let output_shape = vec![index_size as i64];
    let output_dtype = array.dtype().clone();
    let mut output = Array::new(output_shape, output_dtype)?;
    
    // Copy selected elements
    let itemsize = array.itemsize();
    unsafe {
        let array_data = array.data_ptr();
        let output_data = output.data_ptr_mut();
        let array_strides = array.strides();
        
        for (i, &idx) in normalized_indices.iter().enumerate() {
            let src_offset = (idx * array_strides[0]) as usize;
            let dst_offset = i * itemsize;
            
            std::ptr::copy_nonoverlapping(
                array_data.add(src_offset),
                output_data.add(dst_offset),
                itemsize,
            );
        }
    }
    
    Ok(output)
}

/// Fancy index a multi-dimensional array
///
/// For now, simplified implementation - treats array as 1D for indexing
/// Full implementation would handle multi-dimensional fancy indexing
pub fn fancy_index_ndarray(array: &Array, indices: &Array, axis: usize) -> Result<Array, IndexError> {
    if axis >= array.ndim() {
        return Err(IndexError::DimensionMismatch);
    }
    
    // For now, simplified: just use 1D fancy indexing on first axis
    // Full implementation would handle arbitrary axis and multi-dimensional indexing
    fancy_index_array(array, indices)
}

