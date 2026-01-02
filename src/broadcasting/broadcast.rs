//! Broadcasting implementation
//!
//! Broadcasting allows arrays with different shapes to be used together
//! in operations, following NumPy's broadcasting rules


/// Broadcasting error
#[derive(Debug, Clone)]
pub enum BroadcastError {
    /// Shapes are not compatible for broadcasting
    IncompatibleShapes,
    /// Invalid dimension
    InvalidDimension,
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BroadcastError::IncompatibleShapes => write!(f, "Shapes are not compatible for broadcasting"),
            BroadcastError::InvalidDimension => write!(f, "Invalid dimension"),
        }
    }
}

impl std::error::Error for BroadcastError {}

/// Check if two shapes can be broadcast together
///
/// Returns true if the shapes are compatible for broadcasting
pub fn can_broadcast(shape1: &[i64], shape2: &[i64]) -> bool {
    broadcast_shapes(shape1, shape2).is_ok()
}

/// Compute the broadcast shape from two input shapes
///
/// Returns the resulting shape after broadcasting, or an error if
/// the shapes are incompatible.
pub fn broadcast_shapes(shape1: &[i64], shape2: &[i64]) -> Result<Vec<i64>, BroadcastError> {
    let len1 = shape1.len();
    let len2 = shape2.len();
    let max_len = len1.max(len2);
    
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let idx1 = len1.wrapping_sub(max_len - i);
        let idx2 = len2.wrapping_sub(max_len - i);
        
        let dim1 = if idx1 < len1 {
            shape1[idx1]
        } else {
            1
        };
        
        let dim2 = if idx2 < len2 {
            shape2[idx2]
        } else {
            1
        };
        
        // Broadcasting rules:
        // 1. If dimensions are equal, use that dimension
        // 2. If one dimension is 1, use the other
        // 3. Otherwise, incompatible
        if dim1 == dim2 {
            result.push(dim1);
        } else if dim1 == 1 {
            result.push(dim2);
        } else if dim2 == 1 {
            result.push(dim1);
        } else {
            return Err(BroadcastError::IncompatibleShapes);
        }
    }
    
    Ok(result)
}

/// Compute broadcast shapes for multiple input shapes
pub fn broadcast_shapes_multi(shapes: &[&[i64]]) -> Result<Vec<i64>, BroadcastError> {
    if shapes.is_empty() {
        return Ok(Vec::new());
    }
    
    let mut result = shapes[0].to_vec();
    
    for shape in shapes.iter().skip(1) {
        result = broadcast_shapes(&result, shape)?;
    }
    
    Ok(result)
}

/// Compute broadcast strides for a shape when broadcasting to a target shape
///
/// Returns strides that can be used to iterate over the array as if it
/// had the target shape (with dimensions of size 1 having stride 0).
pub fn broadcast_strides(
    original_shape: &[i64],
    original_strides: &[i64],
    target_shape: &[i64],
) -> Result<Vec<i64>, BroadcastError> {
    let orig_len = original_shape.len();
    let target_len = target_shape.len();
    
    if orig_len > target_len {
        return Err(BroadcastError::InvalidDimension);
    }
    
    let mut result = vec![0; target_len];
    let offset = target_len - orig_len;
    
    for i in 0..orig_len {
        let target_idx = offset + i;
        if original_shape[i] == 1 && target_shape[target_idx] != 1 {
            // Dimension of size 1 is broadcast - stride is 0
            result[target_idx] = 0;
        } else {
            result[target_idx] = original_strides[i];
        }
    }
    
    Ok(result)
}

/// Validate that a shape can be broadcast to a target shape
pub fn validate_broadcast(shape: &[i64], target_shape: &[i64]) -> Result<(), BroadcastError> {
    let len = shape.len();
    let target_len = target_shape.len();
    
    if len > target_len {
        return Err(BroadcastError::InvalidDimension);
    }
    
    let offset = target_len - len;
    
    for (i, &dim) in shape.iter().enumerate().take(len) {
        let target_idx = offset + i;
        let target_dim = target_shape[target_idx];
        
        if dim != target_dim && dim != 1 && target_dim != 1 {
            return Err(BroadcastError::IncompatibleShapes);
        }
    }
    
    Ok(())
}

