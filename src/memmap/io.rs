//! Memory-mapped array I/O operations

use crate::array::Array;
use crate::types::DType;
use std::path::Path;

use super::{MemMapArray, MemMapError};

/// Save array to memory-mapped file
///
/// # Arguments
/// * `array` - Array to save
/// * `file_path` - Path to save file
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if save fails
pub fn save_memmap(array: &Array, file_path: &Path) -> Result<MemMapArray, MemMapError> {
    // First, create a writable memory-mapped array
    let mut mmap_array = super::creation::memmap_array_writable(
        file_path,
        array.dtype().clone(),
        array.shape().to_vec(),
    )?;
    
    // Copy data from array to memory-mapped array
    let data_size = array.size() * array.itemsize();
    unsafe {
        std::ptr::copy_nonoverlapping(
            array.data_ptr(),
            mmap_array.array_mut().data_ptr_mut(),
            data_size,
        );
    }
    
    // Flush to ensure data is written
    mmap_array.flush()?;
    
    Ok(mmap_array)
}

/// Load array from memory-mapped file
///
/// # Arguments
/// * `file_path` - Path to file
/// * `dtype` - Expected data type
/// * `shape` - Expected shape
///
/// # Returns
/// * `Ok(MemMapArray)` - Memory-mapped array
/// * `Err(MemMapError)` if load fails
pub fn load_memmap(file_path: &Path, dtype: DType, shape: Vec<i64>) -> Result<MemMapArray, MemMapError> {
    MemMapArray::new(file_path, dtype, shape, super::MapMode::ReadOnly)
}

