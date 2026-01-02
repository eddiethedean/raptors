//! Array tiling operations

use crate::array::Array;

use super::ManipulationError;

/// Tile array to create larger array
///
/// # Arguments
/// * `array` - Array to tile
/// * `reps` - Number of repetitions along each axis
///
/// # Returns
/// * `Ok(Array)` - Tiled array
/// * `Err(ManipulationError)` if tile fails
pub fn tile(array: &Array, reps: &[usize]) -> Result<Array, ManipulationError> {
    let ndim = array.ndim();
    let reps_len = reps.len();
    
    // Extend reps to match array dimensions if needed
    let mut full_reps = vec![1; ndim.max(reps_len)];
    if reps_len < ndim {
        // Prepend 1s if reps is shorter
        full_reps[0..(ndim - reps_len)].fill(1);
        full_reps[(ndim - reps_len)..].copy_from_slice(&reps[0..reps_len]);
    } else {
        // Use reps as-is if longer or equal
        for (i, &rep) in reps.iter().enumerate() {
            full_reps[i] = rep;
        }
    }
    
    // Compute output shape
    let shape = array.shape();
    let mut output_shape = Vec::new();
    
    // Handle case where reps has more dimensions than array
    let extra_dims = full_reps.len().saturating_sub(ndim);
    for _ in 0..extra_dims {
        output_shape.push(full_reps[output_shape.len()] as i64);
    }
    
    // Multiply each dimension by its repeat count
    for (i, &dim_size) in shape.iter().enumerate().take(ndim) {
        let rep_idx = extra_dims + i;
        if rep_idx < full_reps.len() {
            output_shape.push(dim_size * full_reps[rep_idx] as i64);
        } else {
            output_shape.push(dim_size);
        }
    }
    
    let dtype = array.dtype().clone();
    let mut output = Array::new(output_shape, dtype)?;
    
    copy_tiled(array, &mut output, &full_reps)?;
    
    Ok(output)
}

/// Copy array with tiling
fn copy_tiled(
    src: &Array,
    dst: &mut Array,
    _reps: &[usize],
) -> Result<(), ManipulationError> {
    let src_shape = src.shape();
    let dst_shape = dst.shape().to_vec();
    let itemsize = src.itemsize();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
    
    // Iterate through destination and compute source coordinates
    let mut dst_coords = vec![0; dst_shape.len()];
    let mut src_coords = vec![0; src_shape.len()];
    
    for flat_idx in 0..dst.size() {
        index_to_coords(flat_idx, &dst_shape, &mut dst_coords);
        
        // Map destination coordinates to source coordinates
        let src_ndim = src_shape.len();
        let dst_ndim = dst_shape.len();
        let offset = dst_ndim.saturating_sub(src_ndim);
        
        for i in 0..src_ndim {
            let dst_idx = offset + i;
            if dst_idx < dst_ndim {
                src_coords[i] = dst_coords[dst_idx] % src_shape[i];
            } else {
                src_coords[i] = 0;
            }
        }
        
        let src_offset = coords_to_offset(&src_coords, src_strides.as_slice());
        let dst_offset = coords_to_offset(&dst_coords, dst_strides.as_slice());
        
        unsafe {
            let src_ptr = src.data_ptr().add(src_offset);
            let dst_ptr = dst.data_ptr_mut().add(dst_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
        }
    }
    
    Ok(())
}

/// Convert flat index to coordinates
fn index_to_coords(index: usize, shape: &[i64], coords: &mut [i64]) {
    let mut idx = index;
    for i in (0..shape.len()).rev() {
        coords[i] = (idx % shape[i] as usize) as i64;
        idx /= shape[i] as usize;
    }
}

/// Convert coordinates to byte offset
fn coords_to_offset(coords: &[i64], strides: &[i64]) -> usize {
    let mut offset = 0;
    for (i, &coord) in coords.iter().enumerate() {
        offset += (coord * strides[i]) as usize;
    }
    offset
}

