//! Slice indexing implementation
//!
//! This module provides slice indexing functionality,
//! equivalent to NumPy's slice indexing from mapping.c

use crate::indexing::IndexError;
use crate::array::Array;

/// Represents a slice with start, stop, and step
///
/// This is equivalent to Python's slice object (start:stop:step)
#[derive(Debug, Clone, Copy)]
pub struct Slice {
    /// Start index (None means from beginning)
    pub start: Option<i64>,
    /// Stop index (None means to end)
    pub stop: Option<i64>,
    /// Step size (None means step of 1)
    pub step: Option<i64>,
}

impl Slice {
    /// Create a new slice
    pub fn new(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Self {
        Slice { start, stop, step }
    }
    
    /// Create a slice with default step of 1
    pub fn range(start: Option<i64>, stop: Option<i64>) -> Self {
        Slice {
            start,
            stop,
            step: Some(1),
        }
    }
    
    /// Create a full slice (::)
    pub fn full() -> Self {
        Slice {
            start: None,
            stop: None,
            step: Some(1),
        }
    }
}

/// Normalize a slice for a given dimension size
///
/// Converts a slice with optional values to a normalized slice with
/// explicit start, stop, and step values.
pub fn normalize_slice(slice: &Slice, dim_size: i64) -> Result<(i64, i64, i64), IndexError> {
    let step = slice.step.unwrap_or(1);
    
    if step == 0 {
        return Err(IndexError::InvalidIndex);
    }
    
    let (start, stop) = if step > 0 {
        // Positive step
        let start = slice.start.unwrap_or(0);
        let stop = slice.stop.unwrap_or(dim_size);
        
        // Clamp to valid range
        let start = if start < 0 {
            (dim_size + start).max(0)
        } else {
            start.min(dim_size)
        };
        
        let stop = if stop < 0 {
            (dim_size + stop).max(0)
        } else {
            stop.min(dim_size)
        };
        
        (start, stop)
    } else {
        // Negative step
        let start = slice.start.unwrap_or(dim_size - 1);
        let stop = slice.stop.unwrap_or(-1);
        
        // Clamp to valid range
        let start = if start < 0 {
            (dim_size + start).max(-1)
        } else {
            start.min(dim_size - 1)
        };
        
        let stop = if stop < 0 {
            (dim_size + stop).max(-1)
        } else {
            stop.min(dim_size - 1)
        };
        
        (start, stop)
    };
    
    Ok((start, stop, step))
}

/// Compute the resulting length from a normalized slice
pub fn slice_length(start: i64, stop: i64, step: i64) -> i64 {
    if step == 0 {
        return 0;
    }
    
    if (step > 0 && start >= stop) || (step < 0 && start <= stop) {
        return 0;
    }
    
    let diff = (stop - start).abs();
    (diff + step.abs() - 1) / step.abs()
}

/// Compute slice shape from a list of slices applied to an array shape
pub fn compute_slice_shape(
    slices: &[Slice],
    array_shape: &[i64],
) -> Result<Vec<i64>, IndexError> {
    if slices.len() > array_shape.len() {
        return Err(IndexError::DimensionMismatch);
    }
    
    let mut result_shape = Vec::new();
    
    for (i, slice) in slices.iter().enumerate() {
        if i >= array_shape.len() {
            break;
        }
        
        let dim_size = array_shape[i];
        let (start, stop, step) = normalize_slice(slice, dim_size)?;
        let length = slice_length(start, stop, step);
        
        if length < 0 {
            return Err(IndexError::InvalidIndex);
        }
        
        result_shape.push(length);
    }
    
    // Add remaining dimensions unchanged
    result_shape.extend_from_slice(&array_shape[slices.len()..]);
    
    Ok(result_shape)
}

/// Compute slice strides from a normalized slice and original stride
pub fn compute_slice_stride(original_stride: i64, step: i64) -> i64 {
    original_stride * step
}

/// Apply slices to an array and return a new array with the sliced data
///
/// This function creates a new array by applying the given slices to the input array.
/// Each slice in the `slices` vector corresponds to a dimension of the array.
/// If fewer slices are provided than dimensions, the remaining dimensions are copied as-is.
///
/// # Arguments
/// * `array` - The input array to slice
/// * `slices` - Vector of slices to apply, one per dimension (can be fewer than array dimensions)
///
/// # Returns
/// A new array containing the sliced data
pub fn slice_array(array: &Array, slices: &[Slice]) -> Result<Array, IndexError> {
    // Compute output shape
    let output_shape = compute_slice_shape(slices, array.shape())?;
    
    // Create output array
    let mut output = Array::new(output_shape.clone(), array.dtype().clone())
        .map_err(|_| IndexError::InvalidIndex)?;
    
    // Normalize all slices and store normalized parameters
    let array_shape = array.shape();
    let mut normalized_slices = Vec::new();
    for (i, slice) in slices.iter().enumerate() {
        if i >= array_shape.len() {
            break;
        }
        let (start, stop, step) = normalize_slice(slice, array_shape[i])?;
        normalized_slices.push((start, stop, step));
    }
    
    // Copy data by iterating through output coordinates and mapping to input coordinates
    let itemsize = array.itemsize();
    let src_strides = array.strides().to_vec();
    let dst_strides = output.strides().to_vec();
    
    // Helper to convert flat index to coordinates
    fn index_to_coords(index: usize, shape: &[i64], coords: &mut [i64]) {
        let mut idx = index;
        for i in (0..shape.len()).rev() {
            coords[i] = (idx % shape[i] as usize) as i64;
            idx /= shape[i] as usize;
        }
    }
    
    // Helper to convert coordinates to byte offset
    fn coords_to_offset(coords: &[i64], strides: &[i64]) -> usize {
        let mut offset = 0;
        for (i, &coord) in coords.iter().enumerate() {
            offset += (coord * strides[i]) as usize;
        }
        offset
    }
    
    let mut dst_coords = vec![0; output_shape.len()];
    let mut src_coords = vec![0; array_shape.len()];
    
    // Iterate through all output elements
    for flat_idx in 0..output.size() {
        index_to_coords(flat_idx, &output_shape, &mut dst_coords);
        
        // Map destination coordinates to source coordinates using slice parameters
        for i in 0..normalized_slices.len() {
            let (start, _stop, step) = normalized_slices[i];
            // Map dst_coord to src_coord: src = start + dst * step
            src_coords[i] = start + dst_coords[i] * step;
        }
        
        // Copy remaining dimensions as-is (output_shape extends with remaining array_shape dimensions)
        // Since compute_slice_shape extends output_shape with remaining dimensions unchanged,
        // output_shape.len() == array_shape.len(), so we can directly map coordinates
        for i in normalized_slices.len()..array_shape.len() {
            src_coords[i] = dst_coords[i];
        }
        
        // Compute offsets and copy element
        let src_offset = coords_to_offset(&src_coords, &src_strides);
        let dst_offset = coords_to_offset(&dst_coords, &dst_strides);
        
        unsafe {
            let src_ptr = array.data_ptr().add(src_offset);
            let dst_ptr = output.data_ptr_mut().add(dst_offset);
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
        }
    }
    
    Ok(output)
}

