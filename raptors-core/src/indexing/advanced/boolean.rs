//! Boolean indexing implementation
//!
//! This module provides boolean indexing (mask indexing),
//! equivalent to NumPy's boolean indexing functionality

use crate::array::Array;
use crate::types::NpyType;
use crate::indexing::IndexError;
use crate::broadcasting::BroadcastError;

impl From<BroadcastError> for IndexError {
    fn from(_: BroadcastError) -> Self {
        IndexError::DimensionMismatch
    }
}

/// Boolean index an array using a boolean mask
///
/// Returns a new array with elements where the mask is True.
/// The mask array must be boolean type and broadcastable to the array shape.
pub fn boolean_index_array(array: &Array, mask: &Array) -> Result<Array, IndexError> {
    // Validate mask is boolean type
    if mask.dtype().type_() != NpyType::Bool {
        return Err(IndexError::InvalidIndex);
    }
    
    // Check if shapes are compatible (mask must be broadcastable to array shape)
    let array_shape = array.shape();
    let mask_shape = mask.shape();
    
    // For now, require exact shape match or 1D mask
    // Full implementation would handle full broadcasting
    let use_mask_shape = if mask_shape.len() == 1 && mask_shape[0] == array_shape[0] {
        // 1D mask matching first dimension - can broadcast
        true
    } else if mask_shape == array_shape {
        // Exact shape match
        true
    } else {
        return Err(IndexError::DimensionMismatch);
    };
    
    // Count True values to determine output size
    let (true_count, output_shape) = unsafe {
        let mask_ptr = mask.data_ptr() as *const bool;
        let mut count = 0;
        
        if use_mask_shape && mask_shape == array_shape {
            // Exact shape match - count all True values
            let mask_size = mask.size();
            for i in 0..mask_size {
                if *mask_ptr.add(i) {
                    count += 1;
                }
            }
            (count, vec![count as i64])
        } else if mask_shape.len() == 1 && mask_shape[0] == array_shape[0] {
            // 1D mask on multi-D array
            // Count True values in mask
            let mask_size = mask_shape[0] as usize;
            for i in 0..mask_size {
                if *mask_ptr.add(i) {
                    count += 1;
                }
            }
            // Calculate elements per row (for multi-D arrays)
            let first_dim_size = array_shape[0] as usize;
            let elements_per_row = array.size() / first_dim_size;
            // Output shape is [true_count * elements_per_row] - one element per selected row element
            let output_size = count * elements_per_row;
            (count, vec![output_size as i64])
        } else {
            return Err(IndexError::DimensionMismatch);
        }
    };
    
    // Create output array with correct shape
    let output_dtype = array.dtype().clone();
    let mut output = Array::new(output_shape, output_dtype)?;
    
    // Copy selected elements
    let itemsize = array.itemsize();
    unsafe {
        let array_data = array.data_ptr();
        let output_data = output.data_ptr_mut();
        let mask_ptr = mask.data_ptr() as *const bool;
        
        let mut output_idx = 0;
        
        if mask_shape == array_shape {
            // Exact shape match - iterate through all elements
            let size = array.size();
            for i in 0..size {
                if *mask_ptr.add(i) {
                    std::ptr::copy_nonoverlapping(
                        array_data.add(i * itemsize),
                        output_data.add(output_idx * itemsize),
                        itemsize,
                    );
                    output_idx += 1;
                }
            }
        } else if mask_shape.len() == 1 && mask_shape[0] == array_shape[0] {
            // 1D mask on first dimension
            // For now, simplified: flatten array and index by mask
            // Full implementation would handle proper multi-dimensional indexing
            let first_dim_size = array_shape[0] as usize;
            let elements_per_row = array.size() / first_dim_size;
            
            for i in 0..first_dim_size {
                if *mask_ptr.add(i) {
                    // Copy elements for this row
                    for j in 0..elements_per_row {
                        let src_offset = (i * elements_per_row + j) * itemsize;
                        let dst_offset = output_idx * itemsize;
                        std::ptr::copy_nonoverlapping(
                            array_data.add(src_offset),
                            output_data.add(dst_offset),
                            itemsize,
                        );
                        output_idx += 1;
                    }
                }
            }
        }
    }
    
    Ok(output)
}

