//! Slice indexing implementation
//!
//! This module provides slice indexing functionality,
//! equivalent to NumPy's slice indexing from mapping.c

use crate::indexing::IndexError;

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

