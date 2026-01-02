//! Array concatenation operations
//!
//! This module provides concatenate, stack, and split operations

use crate::array::{Array, ArrayError};

/// Concatenation error
#[derive(Debug, Clone)]
pub enum ConcatenationError {
    /// Array error
    ArrayError(ArrayError),
    /// Shape mismatch
    ShapeMismatch,
    /// Invalid axis
    InvalidAxis,
    /// Empty array list
    EmptyArrayList,
}

impl std::fmt::Display for ConcatenationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConcatenationError::ArrayError(e) => write!(f, "Array error: {}", e),
            ConcatenationError::ShapeMismatch => write!(f, "Shape mismatch"),
            ConcatenationError::InvalidAxis => write!(f, "Invalid axis"),
            ConcatenationError::EmptyArrayList => write!(f, "Empty array list"),
        }
    }
}

impl std::error::Error for ConcatenationError {}

impl From<ArrayError> for ConcatenationError {
    fn from(err: ArrayError) -> Self {
        ConcatenationError::ArrayError(err)
    }
}

/// Concatenate arrays along an axis
///
/// All arrays must have compatible shapes (same shape except on the concatenation axis).
/// If axis is None, arrays are flattened before concatenation.
pub fn concatenate(arrays: &[&Array], axis: Option<usize>) -> Result<Array, ConcatenationError> {
    if arrays.is_empty() {
        return Err(ConcatenationError::EmptyArrayList);
    }
    
    if arrays.len() == 1 {
        // Single array - just return a copy (for now, simplified)
        let arr = arrays[0];
        let shape = arr.shape().to_vec();
        let dtype = arr.dtype().clone();
        return Array::new(shape, dtype).map_err(ConcatenationError::from);
    }
    
    let first_array = arrays[0];
    let dtype = first_array.dtype().clone();
    let ndim = first_array.ndim();
    
    // Validate all arrays have same dtype and compatible shapes
    for arr in arrays.iter().skip(1) {
        if arr.dtype().type_() != dtype.type_() {
            return Err(ConcatenationError::ShapeMismatch);
        }
        if arr.ndim() != ndim {
            return Err(ConcatenationError::ShapeMismatch);
        }
    }
    
    if let Some(ax) = axis {
        if ax >= ndim {
            return Err(ConcatenationError::InvalidAxis);
        }
        
        // Validate shapes are compatible (same except on axis)
        let first_shape = first_array.shape();
        for arr in arrays.iter().skip(1) {
            let arr_shape = arr.shape();
            for i in 0..ndim {
                if i != ax && arr_shape[i] != first_shape[i] {
                    return Err(ConcatenationError::ShapeMismatch);
                }
            }
        }
        
        // Calculate output shape
        let mut output_shape = first_shape.to_vec();
        let axis_size: i64 = arrays.iter().map(|arr| arr.shape()[ax]).sum();
        output_shape[ax] = axis_size;
        
        // Create output array
        let mut output = Array::new(output_shape, dtype)?;
        
        // Copy data from each array
        let itemsize = output.itemsize();
        unsafe {
            let output_data = output.data_ptr_mut();
            let mut output_offset = 0;
            
            for arr in arrays {
                let arr_size = arr.size();
                let arr_data = arr.data_ptr();
                
                // Calculate stride for copying along the axis
                // For now, simplified: assume arrays are contiguous
                std::ptr::copy_nonoverlapping(
                    arr_data,
                    output_data.add(output_offset * itemsize),
                    arr_size * itemsize,
                );
                
                output_offset += arr_size;
            }
        }
        
        Ok(output)
    } else {
        // axis=None: flatten all arrays and concatenate
        let total_size: usize = arrays.iter().map(|arr| arr.size()).sum();
        let output_shape = vec![total_size as i64];
        
        let mut output = Array::new(output_shape, dtype)?;
        
        let itemsize = output.itemsize();
        unsafe {
            let output_data = output.data_ptr_mut();
            let mut output_offset = 0;
            
            for arr in arrays {
                let arr_size = arr.size();
                let arr_data = arr.data_ptr();
                
                std::ptr::copy_nonoverlapping(
                    arr_data,
                    output_data.add(output_offset * itemsize),
                    arr_size * itemsize,
                );
                
                output_offset += arr_size;
            }
        }
        
        Ok(output)
    }
}

/// Stack arrays along a new axis
///
/// All arrays must have the same shape.
pub fn stack(arrays: &[&Array], axis: usize) -> Result<Array, ConcatenationError> {
    if arrays.is_empty() {
        return Err(ConcatenationError::EmptyArrayList);
    }
    
    let first_array = arrays[0];
    let shape = first_array.shape();
    let dtype = first_array.dtype().clone();
    let ndim = first_array.ndim();
    
    // Validate all arrays have same shape
    for arr in arrays.iter().skip(1) {
        if arr.shape() != shape {
            return Err(ConcatenationError::ShapeMismatch);
        }
        if arr.dtype().type_() != dtype.type_() {
            return Err(ConcatenationError::ShapeMismatch);
        }
    }
    
    if axis > ndim {
        return Err(ConcatenationError::InvalidAxis);
    }
    
    // Create output shape with new dimension inserted
    let mut output_shape = Vec::with_capacity(ndim + 1);
    output_shape.extend_from_slice(&shape[0..axis]);
    output_shape.push(arrays.len() as i64);
    output_shape.extend_from_slice(&shape[axis..ndim]);
    
    let mut output = Array::new(output_shape, dtype)?;
    
    // Copy data from each array
    let itemsize = output.itemsize();
    let array_size = first_array.size();
    
    unsafe {
        let output_data = output.data_ptr_mut();
        
        for (i, arr) in arrays.iter().enumerate() {
            let arr_data = arr.data_ptr();
            let output_offset = i * array_size;
            
            std::ptr::copy_nonoverlapping(
                arr_data,
                output_data.add(output_offset * itemsize),
                array_size * itemsize,
            );
        }
    }
    
    Ok(output)
}

/// Split specification
pub enum SplitSpec {
    /// Split into N equal sections
    Sections(usize),
    /// Split at specific indices
    Indices(Vec<usize>),
}

/// Split an array along an axis
///
/// Returns a vector of arrays split according to the specification.
pub fn split(array: &Array, spec: SplitSpec, axis: usize) -> Result<Vec<Array>, ConcatenationError> {
    if axis >= array.ndim() {
        return Err(ConcatenationError::InvalidAxis);
    }
    
    let shape = array.shape();
    let axis_size = shape[axis] as usize;
    let dtype = array.dtype().clone();
    
    let split_points = match spec {
        SplitSpec::Sections(n) => {
            if !axis_size.is_multiple_of(n) {
                return Err(ConcatenationError::ShapeMismatch);
            }
            let section_size = axis_size / n;
            (1..n).map(|i| i * section_size).collect()
        }
        SplitSpec::Indices(indices) => {
            for &idx in &indices {
                if idx >= axis_size {
                    return Err(ConcatenationError::InvalidAxis);
                }
            }
            indices
        }
    };
    
    // Build list of start indices (add 0 at beginning and axis_size at end)
    let mut start_indices = vec![0];
    start_indices.extend_from_slice(&split_points);
    start_indices.push(axis_size);
    
    // Create output arrays
    let mut results = Vec::new();
    
    // For now, simplified implementation: assume 1D arrays
    // Full implementation would handle multi-dimensional splitting
    if array.ndim() == 1 {
        let itemsize = array.itemsize();
        unsafe {
            let array_data = array.data_ptr();
            
            for i in 0..(start_indices.len() - 1) {
                let start = start_indices[i];
                let end = start_indices[i + 1];
                let size = end - start;
                
                let output_shape = vec![size as i64];
                let mut output = Array::new(output_shape, dtype.clone())?;
                let output_data = output.data_ptr_mut();
                
                std::ptr::copy_nonoverlapping(
                    array_data.add(start * itemsize),
                    output_data,
                    size * itemsize,
                );
                
                results.push(output);
            }
        }
    } else {
        // Multi-dimensional split not yet implemented
        return Err(ConcatenationError::ShapeMismatch);
    }
    
    Ok(results)
}

