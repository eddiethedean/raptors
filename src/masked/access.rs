//! Masked array access and manipulation

use crate::array::Array;

use super::{MaskedArray, MaskedError};

/// Fill masked values with a specific value
///
/// # Arguments
/// * `array` - Masked array to fill
/// * `fill_value` - Value to use for masked elements
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(MaskedError)` if fill fails
pub fn fill_masked(array: &mut MaskedArray, fill_value: f64) -> Result<(), MaskedError> {
    let size = array.size();
    // Get mask data before mutable borrow
    let mask_data: Vec<bool> = {
        let mask = array.mask();
        let mask_ptr = mask.data_ptr() as *const bool;
        unsafe {
            (0..size).map(|i| *mask_ptr.add(i)).collect()
        }
    };
    
    let data = array.data_mut();
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let data_ptr = data.data_ptr_mut() as *mut f64;
                
                for (i, &masked) in mask_data.iter().enumerate().take(size) {
                    if masked {
                        *data_ptr.add(i) = fill_value;
                    }
                }
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let data_ptr = data.data_ptr_mut() as *mut f32;
                
                for (i, &masked) in mask_data.iter().enumerate().take(size) {
                    if masked {
                        *data_ptr.add(i) = fill_value as f32;
                    }
                }
            }
        }
        _ => return Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
    
    Ok(())
}

/// Get array of valid (unmasked) values
///
/// Returns a new array containing only the unmasked values
pub fn get_valid_values(array: &MaskedArray) -> Result<Array, MaskedError> {
    let data = array.data();
    let mask = array.mask();
    let size = array.size();
    let valid_count = array.count_valid();
    
    if valid_count == 0 {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    let dtype = data.dtype().clone();
    let mut result = Array::new(vec![valid_count as i64], dtype)?;
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let data_ptr = data.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr() as *const bool;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let mut out_idx = 0;
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        *result_ptr.add(out_idx) = *data_ptr.add(i);
                        out_idx += 1;
                    }
                }
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let data_ptr = data.data_ptr() as *const f32;
                let mask_ptr = mask.data_ptr() as *const bool;
                let result_ptr = result.data_ptr_mut() as *mut f32;
                
                let mut out_idx = 0;
                for i in 0..size {
                    if !*mask_ptr.add(i) {
                        *result_ptr.add(out_idx) = *data_ptr.add(i);
                        out_idx += 1;
                    }
                }
            }
        }
        _ => return Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
    
    Ok(result)
}

/// Get array of masked values
///
/// Returns a new array containing only the masked values
pub fn get_masked_values(array: &MaskedArray) -> Result<Array, MaskedError> {
    let data = array.data();
    let mask = array.mask();
    let size = array.size();
    let masked_count = array.count_masked();
    
    if masked_count == 0 {
        return Err(MaskedError::ArrayError(crate::array::ArrayError::InvalidShape));
    }
    
    let dtype = data.dtype().clone();
    let mut result = Array::new(vec![masked_count as i64], dtype)?;
    
    match data.dtype().type_() {
        crate::types::NpyType::Double => {
            unsafe {
                let data_ptr = data.data_ptr() as *const f64;
                let mask_ptr = mask.data_ptr() as *const bool;
                let result_ptr = result.data_ptr_mut() as *mut f64;
                
                let mut out_idx = 0;
                for i in 0..size {
                    if *mask_ptr.add(i) {
                        *result_ptr.add(out_idx) = *data_ptr.add(i);
                        out_idx += 1;
                    }
                }
            }
        }
        crate::types::NpyType::Float => {
            unsafe {
                let data_ptr = data.data_ptr() as *const f32;
                let mask_ptr = mask.data_ptr() as *const bool;
                let result_ptr = result.data_ptr_mut() as *mut f32;
                
                let mut out_idx = 0;
                for i in 0..size {
                    if *mask_ptr.add(i) {
                        *result_ptr.add(out_idx) = *data_ptr.add(i);
                        out_idx += 1;
                    }
                }
            }
        }
        _ => return Err(MaskedError::ArrayError(crate::array::ArrayError::TypeMismatch)),
    }
    
    Ok(result)
}

