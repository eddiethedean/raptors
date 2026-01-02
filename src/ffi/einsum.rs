//! Einsum C API
//!
//! This module provides C API wrappers for einsum operations

use crate::ffi::{PyArrayObject, conversion};
use crate::einsum::einsum;
use libc::c_char;
use std::ffi::CStr;
use std::ptr;

/// Compute Einstein summation (C API)
///
/// Equivalent to NumPy's einsum functionality.
///
/// # Safety
/// The caller must ensure `subscripts` is a valid null-terminated C string.
/// The caller must ensure `arrays` is a valid array of at least `num_arrays` PyArrayObject pointers.
/// The caller must ensure all array pointers are valid.
#[no_mangle]
pub unsafe extern "C" fn PyArray_Einsum(
    subscripts: *const c_char,
    arrays: *const *mut PyArrayObject,
    num_arrays: usize,
) -> *mut PyArrayObject {
    if subscripts.is_null() || arrays.is_null() || num_arrays == 0 {
        return ptr::null_mut();
    }
    
    // Convert subscripts string
    let subscripts_str = match CStr::from_ptr(subscripts).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    
    // Convert arrays
    let mut rust_arrays = Vec::with_capacity(num_arrays);
    for i in 0..num_arrays {
        let arr_ptr = *arrays.add(i);
        if arr_ptr.is_null() {
            return ptr::null_mut();
        }
        
        match conversion::pyarray_to_array_view(arr_ptr) {
            Ok(arr) => rust_arrays.push(arr),
            Err(_) => return ptr::null_mut(),
        }
    }
    
    // Create slice of references
    let array_refs: Vec<&crate::array::Array> = rust_arrays.iter().collect();
    
    // Call einsum
    match einsum(subscripts_str, &array_refs) {
        Ok(result) => conversion::array_to_pyarray_ptr(&result),
        Err(_) => ptr::null_mut(),
    }
}

