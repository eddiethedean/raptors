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
        // Single array - create a copy with the same shape and data
        let arr = arrays[0];
        let shape = arr.shape().to_vec();
        let dtype = arr.dtype().clone();
        let mut output = Array::new(shape, dtype)?;
        
        // Copy data from input array to output array
        let itemsize = output.itemsize();
        let arr_size = arr.size();
        unsafe {
            std::ptr::copy_nonoverlapping(
                arr.data_ptr(),
                output.data_ptr_mut(),
                arr_size * itemsize,
            );
        }
        
        return Ok(output);
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

/// Convert flat index to coordinates in dimensions other than the split axis
fn index_to_coords_other_dims(
    index: usize,
    other_dims_shape: &[i64],
    coords: &mut [i64],
) {
    let mut idx = index;
    for i in (0..other_dims_shape.len()).rev() {
        coords[i] = (idx % other_dims_shape[i] as usize) as i64;
        idx /= other_dims_shape[i] as usize;
    }
}

/// Calculate byte offset from coordinates and strides
fn coords_to_offset(coords: &[i64], strides: &[i64]) -> usize {
    let mut offset = 0usize;
    for (i, &coord) in coords.iter().enumerate() {
        // Use wrapping arithmetic to avoid overflow panics in debug mode
        // The result will be correct for valid coordinates and strides
        let coord_usize = coord as usize;
        let stride_usize = strides[i] as usize;
        offset = offset.wrapping_add(coord_usize.wrapping_mul(stride_usize));
    }
    offset
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
            // Handle uneven splits by distributing remainder
            // First (n - remainder) sections get base_size
            // Last remainder sections get base_size + 1
            let base_size = axis_size / n;
            let remainder = axis_size % n;
            let mut points = Vec::new();
            let mut current = 0;
            for i in 0..n {
                if i < n - remainder {
                    current += base_size;
                } else {
                    current += base_size + 1;
                }
                if current < axis_size {
                    points.push(current);
                }
            }
            points
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
    
    let itemsize = array.itemsize();
    let array_strides = array.strides();
    let axis_stride = array_strides[axis];
    
    // Calculate output shape for each split
    // Remove the axis dimension and replace with split size
    let mut base_output_shape = shape.to_vec();
    base_output_shape.remove(axis);
    
    unsafe {
        let array_data = array.data_ptr();
        
        for i in 0..(start_indices.len() - 1) {
            let start = start_indices[i];
            let end = start_indices[i + 1];
            let split_size = end - start;
            
            // Create output shape: insert split_size at axis position
            let mut output_shape = base_output_shape.clone();
            output_shape.insert(axis, split_size as i64);
            
            let mut output = Array::new(output_shape.clone(), dtype.clone())?;
            let output_data = output.data_ptr_mut();
            let output_strides = output.strides();
            
            // Copy data along the axis
            // For each position in other dimensions, copy the split section along axis
            if array.ndim() == 1 {
                // Simple 1D case
                std::ptr::copy_nonoverlapping(
                    array_data.add(start * itemsize),
                    output_data,
                    split_size * itemsize,
                );
            } else if array.ndim() == 2 && axis == 0 {
                // Special case: 2D array split along axis 0 (rows)
                // We can copy row-by-row more efficiently
                let row_size = output_shape[1] as usize * itemsize;
                let num_rows = split_size;
                // Strides are already in bytes, so use them directly
                let input_row_stride = array_strides[0] as usize;
                let output_row_stride = output_strides[0] as usize;
                
                for row in 0..num_rows {
                    let input_row_start = start + row;
                    // Strides are in bytes, so no need to multiply by itemsize
                    let input_offset = input_row_start * input_row_stride;
                    let output_offset = row * output_row_stride;
                    
                    // Note: Bounds checking is handled by the array size calculations
                    // Offsets are guaranteed to be within bounds for valid splits
                    
                    std::ptr::copy_nonoverlapping(
                        array_data.add(input_offset),
                        output_data.add(output_offset),
                        row_size,
                    );
                }
            } else if array.ndim() == 2 && axis == 1 {
                // Special case: 2D array split along axis 1 (columns)
                // Copy column-by-column (each column is a single element per row)
                let num_rows = output_shape[0] as usize;
                // Strides are already in bytes, so use them directly
                let input_col_stride = array_strides[1] as usize;
                let output_col_stride = output_strides[1] as usize;
                let input_row_stride = array_strides[0] as usize;
                let output_row_stride = output_strides[0] as usize;
                
                for row in 0..num_rows {
                    for col in 0..split_size {
                        let input_col_idx = start + col;
                        // Strides are in bytes, so no need to multiply by itemsize
                        let input_offset = row * input_row_stride + input_col_idx * input_col_stride;
                        let output_offset = row * output_row_stride + col * output_col_stride;
                        
                        // Note: Bounds checking is handled by the array size calculations
                        // Offsets are guaranteed to be within bounds for valid splits
                        
                        std::ptr::copy_nonoverlapping(
                            array_data.add(input_offset),
                            output_data.add(output_offset),
                            itemsize,
                        );
                    }
                }
            } else {
                // Multi-dimensional case: use coordinate-based approach
                // Build shape of other dimensions (excluding the split axis)
                let mut other_dims_shape = Vec::new();
                let mut other_dims_strides = Vec::new();
                for (i, &dim) in shape.iter().enumerate() {
                    if i != axis {
                        other_dims_shape.push(dim);
                        other_dims_strides.push(array_strides[i]);
                    }
                }
                
                // Calculate number of combinations in other dimensions
                let num_other_combinations: usize = if other_dims_shape.is_empty() {
                    1
                } else {
                    other_dims_shape.iter().product::<i64>() as usize
                };
                
                // Allocate buffer for coordinates in other dimensions
                let mut other_coords = vec![0i64; other_dims_shape.len()];
                
                // Iterate over each combination of other dimensions
                for other_idx in 0..num_other_combinations {
                    // Convert flat index to coordinates in other dimensions
                    index_to_coords_other_dims(other_idx, &other_dims_shape, &mut other_coords);
                    
                    // Calculate base offset for this combination (excluding axis dimension)
                    let mut base_offset = 0;
                    let mut coord_idx = 0;
                    for (i, &stride) in array_strides.iter().enumerate() {
                        if i != axis {
                            base_offset += (other_coords[coord_idx] * stride) as usize;
                            coord_idx += 1;
                        }
                    }
                    
                    // Copy split_size elements along the axis
                    for axis_pos in 0..split_size {
                        let i = start + axis_pos;
                        
                        // Calculate input offset: base_offset + axis position offset
                        // Strides are already in bytes, so use axis_stride directly
                        let input_offset = base_offset + ((i as i64) * axis_stride) as usize;
                        
                        // Calculate output offset: need to map to output coordinates
                        // Output coordinates: insert axis_pos at axis position
                        let mut output_coords = Vec::new();
                        coord_idx = 0;
                        for j in 0..output_shape.len() {
                            if j == axis {
                                output_coords.push(axis_pos as i64);
                            } else {
                                output_coords.push(other_coords[coord_idx]);
                                coord_idx += 1;
                            }
                        }
                        let output_offset = coords_to_offset(&output_coords, &output_strides);
                        
                        // Copy one element
                        // Strides are already in bytes, so use offsets directly
                        std::ptr::copy_nonoverlapping(
                            array_data.add(input_offset),
                            output_data.add(output_offset),
                            itemsize,
                        );
                    }
                }
            }
            
            results.push(output);
        }
    }
    
    Ok(results)
}

