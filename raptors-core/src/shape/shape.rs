//! Shape manipulation operations

/// Shape operation error
#[derive(Debug, Clone)]
pub enum ShapeError {
    /// Invalid shape for operation
    InvalidShape,
    /// Total size mismatch
    SizeMismatch,
    /// Invalid dimension index
    InvalidDimension,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::InvalidShape => write!(f, "Invalid shape"),
            ShapeError::SizeMismatch => write!(f, "Size mismatch"),
            ShapeError::InvalidDimension => write!(f, "Invalid dimension"),
        }
    }
}

impl std::error::Error for ShapeError {}

/// Compute total size from shape
pub fn compute_size(shape: &[i64]) -> i64 {
    shape.iter().product()
}

/// Resolve -1 dimension in reshape shape by auto-calculating the size
///
/// In NumPy, -1 means "calculate this dimension automatically" such that
/// the total size matches the original array size.
/// Returns a new shape with -1 replaced by the calculated dimension.
pub fn resolve_reshape_shape(old_shape: &[i64], new_shape: &[i64]) -> Result<Vec<i64>, ShapeError> {
    let old_size = compute_size(old_shape);
    
    // Count how many -1 dimensions we have (should be at most 1)
    let minus_one_count = new_shape.iter().filter(|&&x| x == -1).count();
    if minus_one_count > 1 {
        return Err(ShapeError::InvalidShape);
    }
    
    if minus_one_count == 0 {
        // No -1 dimension, just validate
        let new_size = compute_size(new_shape);
        if old_size != new_size {
            return Err(ShapeError::SizeMismatch);
        }
        return Ok(new_shape.to_vec());
    }
    
    // We have exactly one -1 dimension, calculate it
    let mut resolved_shape = new_shape.to_vec();
    let known_size: i64 = new_shape.iter()
        .filter(|&&x| x != -1 && x > 0)
        .product();
    
    if known_size == 0 {
        return Err(ShapeError::InvalidShape);
    }
    
    if old_size % known_size != 0 {
        return Err(ShapeError::SizeMismatch);
    }
    
    let calculated_dim = old_size / known_size;
    
    // Replace -1 with calculated dimension
    for dim in &mut resolved_shape {
        if *dim == -1 {
            *dim = calculated_dim;
            break;
        }
    }
    
    Ok(resolved_shape)
}

/// Validate that a reshape shape matches the total size
pub fn validate_reshape_shape(old_shape: &[i64], new_shape: &[i64]) -> Result<(), ShapeError> {
    let resolved_shape = resolve_reshape_shape(old_shape, new_shape)?;
    let old_size = compute_size(old_shape);
    let new_size = compute_size(&resolved_shape);
    
    if old_size != new_size {
        return Err(ShapeError::SizeMismatch);
    }
    
    Ok(())
}

/// Compute strides for a reshape operation (C-order)
pub fn compute_reshape_strides(shape: &[i64], itemsize: usize) -> Vec<i64> {
    let mut strides = vec![0; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = itemsize as i64;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Transpose dimensions according to axes
///
/// Returns the new shape and a mapping from old dimension to new dimension
pub fn transpose_dimensions(
    shape: &[i64],
    axes: Option<&[usize]>,
) -> Result<(Vec<i64>, Vec<i64>), ShapeError> {
    let ndim = shape.len();
    
    let axes: Vec<i64> = match axes {
        Some(ax) => {
            if ax.len() != ndim {
                return Err(ShapeError::InvalidDimension);
            }
            ax.iter().map(|&x| x as i64).collect()
        }
        None => (0..ndim).rev().map(|x| x as i64).collect(), // Reverse all axes by default
    };
    
    // Validate axes
    let mut seen = vec![false; ndim];
    for &axis in &axes {
        let axis_usize = axis as usize;
        if axis_usize >= ndim {
            return Err(ShapeError::InvalidDimension);
        }
        if seen[axis_usize] {
            return Err(ShapeError::InvalidDimension);
        }
        seen[axis_usize] = true;
    }
    
    // Compute new shape
    let new_shape: Vec<i64> = axes.iter().map(|&i| shape[i as usize]).collect();
    
    Ok((new_shape, axes))
}

/// Squeeze dimensions of size 1
///
/// Removes dimensions of size 1 from the shape
pub fn squeeze_dims(shape: &[i64], axis: Option<usize>) -> Vec<i64> {
    match axis {
        Some(ax) => {
            // Remove specific axis if it has size 1
            if ax < shape.len() && shape[ax] == 1 {
                let mut new_shape = shape.to_vec();
                new_shape.remove(ax);
                new_shape
            } else {
                shape.to_vec()
            }
        }
        None => {
            // Remove all dimensions of size 1
            shape.iter().copied().filter(|&s| s != 1).collect()
        }
    }
}

/// Expand dimensions by inserting axis of size 1
pub fn expand_dims(shape: &[i64], axis: usize) -> Result<Vec<i64>, ShapeError> {
    if axis > shape.len() {
        return Err(ShapeError::InvalidDimension);
    }
    
    let mut new_shape = shape.to_vec();
    new_shape.insert(axis, 1);
    Ok(new_shape)
}

/// Flatten shape to 1D
pub fn flatten_shape(shape: &[i64]) -> Vec<i64> {
    if shape.is_empty() {
        vec![1]
    } else {
        vec![compute_size(shape)]
    }
}

