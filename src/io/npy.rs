//! NPY file format implementation
//!
//! This module provides save/load functionality for arrays in NPY format,
//! equivalent to NumPy's .npy file format

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// I/O error
#[derive(Debug, Clone)]
pub enum IoError {
    /// Array error
    ArrayError(ArrayError),
    /// File I/O error
    FileError(String),
    /// Invalid format
    InvalidFormat,
    /// Unsupported dtype
    UnsupportedDtype,
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::ArrayError(e) => write!(f, "Array error: {}", e),
            IoError::FileError(msg) => write!(f, "File error: {}", msg),
            IoError::InvalidFormat => write!(f, "Invalid format"),
            IoError::UnsupportedDtype => write!(f, "Unsupported dtype"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<ArrayError> for IoError {
    fn from(err: ArrayError) -> Self {
        IoError::ArrayError(err)
    }
}

/// NPY magic number
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
const NPY_VERSION: u8 = 1;

/// Convert NpyType to NumPy dtype string
fn dtype_to_string(ty: NpyType) -> Result<String, IoError> {
    match ty {
        NpyType::Bool => Ok("|b1".to_string()),
        NpyType::Int => Ok("<i4".to_string()),
        NpyType::Float => Ok("<f4".to_string()),
        NpyType::Double => Ok("<f8".to_string()),
        _ => Err(IoError::UnsupportedDtype),
    }
}

/// Save array to NPY file format
pub fn save_npy(path: impl AsRef<Path>, array: &Array) -> Result<(), IoError> {
    let mut file = File::create(path).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Write magic number
    file.write_all(NPY_MAGIC).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Write version
    file.write_all(&[NPY_VERSION, 0]).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Build header dict
    let shape_str = format!("({})", array.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
    let dtype_str = dtype_to_string(array.dtype().type_())?;
    let fortran_order = "False";
    
    let header_dict = format!("{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}", dtype_str, fortran_order, shape_str);
    let header_str = format!("{}\n", header_dict);
    
    // Calculate header length (must be divisible by 64 for alignment)
    let header_len = header_str.len();
    let padded_header_len = header_len.div_ceil(64) * 64;
    let padding_len = padded_header_len - header_len;
    
    // Write header length (little-endian u16)
    let header_len_bytes = (padded_header_len as u16).to_le_bytes();
    file.write_all(&header_len_bytes).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Write header
    file.write_all(header_str.as_bytes()).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Write padding
    let padding = vec![b' '; padding_len];
    file.write_all(&padding).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Write array data
    let data_size = array.size() * array.itemsize();
    unsafe {
        let data_ptr = array.data_ptr();
        let data_slice = std::slice::from_raw_parts(data_ptr, data_size);
        file.write_all(data_slice).map_err(|e| IoError::FileError(e.to_string()))?;
    }
    
    Ok(())
}

/// Load array from NPY file format
pub fn load_npy(path: impl AsRef<Path>) -> Result<Array, IoError> {
    let mut file = File::open(path).map_err(|e| IoError::FileError(e.to_string()))?;
    
    // Read and validate magic number
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic).map_err(|e| IoError::FileError(e.to_string()))?;
    if magic != NPY_MAGIC {
        return Err(IoError::InvalidFormat);
    }
    
    // Read version
    let mut version = [0u8; 2];
    file.read_exact(&mut version).map_err(|e| IoError::FileError(e.to_string()))?;
    if version[0] != NPY_VERSION {
        return Err(IoError::InvalidFormat);
    }
    
    // Read header length
    let mut header_len_bytes = [0u8; 2];
    file.read_exact(&mut header_len_bytes).map_err(|e| IoError::FileError(e.to_string()))?;
    let header_len = u16::from_le_bytes(header_len_bytes) as usize;
    
    // Read header
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(|e| IoError::FileError(e.to_string()))?;
    let header_str = String::from_utf8_lossy(&header);
    
    // Parse header (simplified - full implementation would parse dict properly)
    // For now, assume format: {'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }
    // Extract shape and dtype (simplified parsing)
    let shape_start = header_str.find("'shape': ").ok_or(IoError::InvalidFormat)? + 9;
    let shape_end = header_str[shape_start..].find(')').ok_or(IoError::InvalidFormat)? + shape_start + 1;
    let shape_str = &header_str[shape_start..shape_end];
    
    // Parse shape (simplified - assumes format (n, m, ...))
    let shape: Vec<i64> = shape_str
        .trim_matches(|c| c == '(' || c == ')' || c == ' ')
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    
    // Extract dtype (simplified)
    let descr_start = header_str.find("'descr': '").ok_or(IoError::InvalidFormat)? + 10;
    let descr_end = header_str[descr_start..].find('\'').ok_or(IoError::InvalidFormat)? + descr_start;
    let dtype_str = &header_str[descr_start..descr_end];
    
    // Parse dtype (simplified - only handles basic types)
    let dtype = match dtype_str {
        "<f8" | ">f8" | "=f8" => DType::new(NpyType::Double),
        "<f4" | ">f4" | "=f4" => DType::new(NpyType::Float),
        "<i4" | ">i4" | "=i4" => DType::new(NpyType::Int),
        _ => return Err(IoError::UnsupportedDtype),
    };
    
    // Create array
    let mut array = Array::new(shape, dtype)?;
    
    // Read array data
    let data_size = array.size() * array.itemsize();
    unsafe {
        let data_ptr = array.data_ptr_mut();
        let data_slice = std::slice::from_raw_parts_mut(data_ptr, data_size);
        file.read_exact(data_slice).map_err(|e| IoError::FileError(e.to_string()))?;
    }
    
    Ok(array)
}

