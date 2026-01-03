//! Structured array creation

use crate::array::Array;
use crate::types::DType;

use super::{StructuredDType, StructuredError};

/// Create structured array from field definitions and data
///
/// # Arguments
/// * `fields` - Vector of (name, dtype) pairs
/// * `data` - Raw byte data
/// * `shape` - Shape of the array
///
/// # Returns
/// * `Ok(Array)` - Structured array
/// * `Err(StructuredError)` if creation fails
pub fn structured_array(
    fields: Vec<(String, DType)>,
    data: &[u8],
    shape: Vec<i64>,
) -> Result<Array, StructuredError> {
    // Create structured dtype
    let structured_dtype = StructuredDType::new(fields)?;
    let itemsize = structured_dtype.itemsize();
    let total_elements: usize = shape.iter().product::<i64>() as usize;
    let required_size = total_elements * itemsize;
    
    if data.len() < required_size {
        return Err(StructuredError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    // Create array with structured dtype
    // For now, use a placeholder dtype - full implementation would
    // extend DType to support structured types
    let dtype = DType::new(crate::types::NpyType::Void); // Placeholder
    let mut array = Array::new(shape, dtype)?;
    
    // Copy data (skip if size is 0, as copy_nonoverlapping with size 0 is safe but unnecessary)
    if required_size > 0 {
        unsafe {
            let dst = array.data_ptr_mut();
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, required_size);
        }
    }
    
    Ok(array)
}

/// Create structured array from record data
///
/// Records are represented as vectors of field values
pub fn structured_array_from_records(
    fields: Vec<(String, DType)>,
    records: Vec<Vec<u8>>,
    shape: Vec<i64>,
) -> Result<Array, StructuredError> {
    let structured_dtype = StructuredDType::new(fields)?;
    let itemsize = structured_dtype.itemsize();
    let total_elements: usize = shape.iter().product::<i64>() as usize;
    
    if records.len() != total_elements {
        return Err(StructuredError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    // Allocate data buffer
    let mut data = vec![0u8; total_elements * itemsize];
    
    // Copy record data
    for (i, record) in records.iter().enumerate() {
        let offset = i * itemsize;
        let copy_size = record.len().min(itemsize);
        data[offset..offset + copy_size].copy_from_slice(&record[..copy_size]);
    }
    
    let field_defs: Vec<(String, DType)> = structured_dtype.fields().iter()
        .map(|f| (f.name.clone(), f.dtype.clone()))
        .collect();
    structured_array(field_defs, &data, shape)
}

