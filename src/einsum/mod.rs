//! Einstein Summation (einsum) implementation
//!
//! This module provides Einstein summation notation functionality,
//! equivalent to NumPy's einsum implementation

mod parser;
mod contract;
mod path;

pub use parser::*;
pub use contract::*;
pub use path::*;

use crate::array::{Array, ArrayError};

/// Einsum error
#[derive(Debug, Clone)]
pub enum EinsumError {
    /// Array error
    ArrayError(ArrayError),
    /// Parsing error
    ParseError(String),
    /// Shape mismatch error
    ShapeMismatch(String),
    /// Invalid einsum notation
    InvalidNotation(String),
}

impl std::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EinsumError::ArrayError(e) => write!(f, "Array error: {}", e),
            EinsumError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            EinsumError::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
            EinsumError::InvalidNotation(msg) => write!(f, "Invalid notation: {}", msg),
        }
    }
}

impl std::error::Error for EinsumError {}

impl From<ArrayError> for EinsumError {
    fn from(err: ArrayError) -> Self {
        EinsumError::ArrayError(err)
    }
}

/// Compute Einstein summation
///
/// Performs tensor operations according to Einstein summation notation.
///
/// # Arguments
/// * `subscripts` - Einstein summation notation string (e.g., "ij,jk->ik")
/// * `arrays` - Input arrays to contract
///
/// # Returns
/// * `Ok(Array)` - Result array
/// * `Err(EinsumError)` - If operation fails
pub fn einsum(subscripts: &str, arrays: &[&Array]) -> Result<Array, EinsumError> {
    if arrays.is_empty() {
        return Err(EinsumError::InvalidNotation("No input arrays provided".to_string()));
    }
    
    // Parse the einsum notation
    let parsed = parse_einsum(subscripts, arrays.len())?;
    
    // Perform the contraction
    execute_einsum(&parsed, arrays)
}

