//! Array rotation operations

use crate::array::{Array, ArrayError};

use super::ManipulationError;

/// Rotate array by 90-degree increments
///
/// # Arguments
/// * `array` - Array to rotate (must be 2D)
/// * `k` - Number of 90-degree rotations (can be negative)
///
/// # Returns
/// * `Ok(Array)` - Rotated array
/// * `Err(ManipulationError)` if rotation fails
pub fn rotate90(array: &Array, k: i32) -> Result<Array, ManipulationError> {
    if array.ndim() != 2 {
        return Err(ManipulationError::ArrayError(ArrayError::InvalidShape));
    }
    
    let shape = array.shape();
    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    
    // Normalize k to 0-3 range
    let k_normalized = k.rem_euclid(4);
    
    let (new_rows, new_cols) = match k_normalized {
        0 => (rows, cols),
        1 | 3 => (cols, rows),
        2 => (rows, cols),
        _ => unreachable!(),
    };
    
    let dtype = array.dtype().clone();
    let mut output = Array::new(vec![new_rows as i64, new_cols as i64], dtype)?;
    
    copy_rotated(array, &mut output, k_normalized as usize, rows, cols)?;
    
    Ok(output)
}

/// Copy array with rotation
fn copy_rotated(
    src: &Array,
    dst: &mut Array,
    k: usize,
    rows: usize,
    cols: usize,
) -> Result<(), ManipulationError> {
    let itemsize = src.itemsize();
    let src_strides = src.strides().to_vec();
    let dst_strides = dst.strides().to_vec();
    
    for i in 0..rows {
        for j in 0..cols {
            let (src_i, src_j) = (i, j);
            let (dst_i, dst_j) = match k {
                0 => (i, j),
                1 => (j, rows - 1 - i), // 90 degrees clockwise
                2 => (rows - 1 - i, cols - 1 - j), // 180 degrees
                3 => (cols - 1 - j, i), // 270 degrees clockwise (90 counter-clockwise)
                _ => unreachable!(),
            };
            
            let src_offset = (src_i as i64 * src_strides[0] + src_j as i64 * src_strides[1]) as usize;
            let dst_offset = (dst_i as i64 * dst_strides[0] + dst_j as i64 * dst_strides[1]) as usize;
            
            unsafe {
                let src_ptr = src.data_ptr().add(src_offset);
                let dst_ptr = dst.data_ptr_mut().add(dst_offset);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, itemsize);
            }
        }
    }
    
    Ok(())
}

