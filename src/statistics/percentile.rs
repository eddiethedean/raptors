//! Percentile calculations

use crate::array::{Array, ArrayError};

use super::StatisticsError;

/// Compute percentile of array
///
/// # Arguments
/// * `array` - Array to compute percentile for
/// * `q` - Percentile value (0-100)
/// * `axis` - Axis along which to compute (None means all elements)
///
/// # Returns
/// * `Ok(Array)` - Percentile value(s)
/// * `Err(StatisticsError)` if computation fails
pub fn percentile(array: &Array, q: f64, axis: Option<usize>) -> Result<Array, StatisticsError> {
    if !(0.0..=100.0).contains(&q) {
        return Err(StatisticsError::InvalidPercentile);
    }
    
    // For now, simplified implementation - sort and pick element
    // Full implementation would handle multiple methods
    match array.dtype().type_() {
        crate::types::NpyType::Double => percentile_double(array, q, axis),
        crate::types::NpyType::Float => percentile_float(array, q, axis),
        _ => Err(StatisticsError::UnsupportedType),
    }
}

/// Percentile for double
fn percentile_double(array: &Array, q: f64, _axis: Option<usize>) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f64;
        let mut values: Vec<f64> = (0..size).map(|i| *ptr.add(i)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (q / 100.0 * (size - 1) as f64) as usize;
        let percentile_val = values[index.min(size - 1)];
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f64;
        *out_ptr = percentile_val;
        
        Ok(output)
    }
}

/// Percentile for float
fn percentile_float(array: &Array, q: f64, _axis: Option<usize>) -> Result<Array, StatisticsError> {
    let size = array.size();
    if size == 0 {
        return Err(StatisticsError::ArrayError(ArrayError::InvalidShape));
    }
    
    unsafe {
        let ptr = array.data_ptr() as *const f32;
        let mut values: Vec<f32> = (0..size).map(|i| *ptr.add(i)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (q / 100.0 * (size - 1) as f64) as usize;
        let percentile_val = values[index.min(size - 1)];
        
        let dtype = array.dtype().clone();
        let mut output = Array::new(vec![1], dtype)?;
        let out_ptr = output.data_ptr_mut() as *mut f32;
        *out_ptr = percentile_val;
        
        Ok(output)
    }
}

