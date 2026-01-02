//! Memory-mapped array creation functions

use crate::types::DType;
use std::path::Path;

use super::{MemMapArray, MapMode, MemMapError};

/// Create memory-mapped array from file
///
/// # Arguments
/// * `file_path` - Path to file
/// * `dtype` - Data type
/// * `shape` - Shape of array
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if creation fails
pub fn memmap_array(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    MemMapArray::new(file_path, dtype, shape, MapMode::ReadOnly)
}

/// Create memory-mapped array for writing
/// 
/// Creates a new file if it doesn't exist, or opens an existing file.
pub fn memmap_array_writable(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    let itemsize = dtype.itemsize();
    let total_elements: usize = shape.iter().product::<i64>() as usize;
    let file_size = total_elements * itemsize;
    
    // Create file if it doesn't exist or extend/truncate to required size
    if !file_path.exists() {
        // Create new file with required size
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(file_path)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
        file.set_len(file_size as u64)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
    } else {
        // Ensure file is at least the required size
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(file_path)
            .map_err(|e| MemMapError::IoError(e.to_string()))?;
        let current_size = file.metadata()
            .map_err(|e| MemMapError::IoError(e.to_string()))?
            .len();
        if current_size < file_size as u64 {
            file.set_len(file_size as u64)
                .map_err(|e| MemMapError::IoError(e.to_string()))?;
        }
    }
    
    MemMapArray::new(file_path, dtype, shape, MapMode::ReadWrite)
}

