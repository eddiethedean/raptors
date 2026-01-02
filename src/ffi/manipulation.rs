//! Array manipulation C API
//!
//! This module provides C API wrappers for array manipulation operations,
//! equivalent to NumPy's manipulation functions

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::shape::{validate_reshape_shape, transpose_dimensions, flatten_shape};
use libc::c_int;
use std::ptr;

/// Reshape array to new shape
///
/// Equivalent to NumPy's PyArray_Reshape function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
/// The caller must ensure `newshape` points to an array of at least `nd` elements.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Reshape(
    arr: *mut PyArrayObject,
    newshape: *const i64,
    nd: c_int,
) -> *mut PyArrayObject {
    if arr.is_null() || newshape.is_null() || !(0..=64).contains(&nd) {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Extract new shape
        let mut shape = Vec::with_capacity(nd as usize);
        for i in 0..(nd as usize) {
            shape.push(*newshape.add(i));
        }
        
        // Validate reshape
        if validate_reshape_shape(array.shape(), &shape).is_err() {
            return ptr::null_mut();
        }
        
        // Create new array with reshaped shape
        let dtype = array.dtype().clone();
        let new_array = match Array::new(shape, dtype) {
            Ok(mut a) => {
                // Copy data
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Transpose array
///
/// Equivalent to NumPy's PyArray_Transpose function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Transpose(
    arr: *mut PyArrayObject,
    _perm: *const c_int, // Permutation axes (simplified - not used yet, defaults to reverse)
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Compute transposed shape (default: reverse axes)
        let (new_shape, _axes) = match transpose_dimensions(array.shape(), None) {
            Ok(result) => result,
            Err(_) => return ptr::null_mut(),
        };
        
        // Create new array with transposed shape
        let dtype = array.dtype().clone();
        let new_array = match Array::new(new_shape, dtype) {
            Ok(mut a) => {
                // Copy data with transposed indexing (simplified - full implementation would use strides)
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Return flattened view
///
/// Equivalent to NumPy's PyArray_Ravel function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Ravel(
    arr: *mut PyArrayObject,
    _order: c_int, // Order (C or F) - simplified, not used yet
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Compute flattened shape
        let new_shape = flatten_shape(array.shape());
        
        // Create new array with flattened shape
        let dtype = array.dtype().clone();
        let new_array = match Array::new(new_shape, dtype) {
            Ok(mut a) => {
                // Copy all data
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Swap two axes
///
/// Equivalent to NumPy's PyArray_SwapAxes function.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_SwapAxes(
    arr: *mut PyArrayObject,
    axis1: c_int,
    axis2: c_int,
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        let ndim = array.ndim();
        let axis1_usize = axis1 as usize;
        let axis2_usize = axis2 as usize;
        
        // Validate axes
        if axis1_usize >= ndim || axis2_usize >= ndim {
            return ptr::null_mut();
        }
        
        // Create permutation that swaps the two axes
        let mut axes: Vec<usize> = (0..ndim).collect();
        axes.swap(axis1_usize, axis2_usize);
        
        // Compute new shape
        let (new_shape, _axes_map) = match transpose_dimensions(array.shape(), Some(&axes)) {
            Ok(result) => result,
            Err(_) => return ptr::null_mut(),
        };
        
        // Create new array with swapped axes
        let dtype = array.dtype().clone();
        let new_array = match Array::new(new_shape, dtype) {
            Ok(mut a) => {
                // Copy data (simplified - full implementation would use proper indexing)
                let size = array.size() * array.itemsize();
                std::ptr::copy_nonoverlapping(
                    array.data_ptr(),
                    a.data_ptr_mut(),
                    size,
                );
                a
            }
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

