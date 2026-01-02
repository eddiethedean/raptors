//! Array views and copies C API
//!
//! This module provides C API wrappers for array view operations,
//! equivalent to NumPy's view and copy functions
#![allow(clippy::arc_with_non_send_sync)]

use crate::array::Array;
use crate::ffi::{PyArrayObject, conversion};
use crate::shape::{squeeze_dims, flatten_shape};
use libc::c_int;
use std::ptr;
use std::sync::Arc;

/// Create array view with new dtype
///
/// Equivalent to NumPy's PyArray_View function.
/// Creates a new view of the array with a different dtype.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
/// The caller is responsible for freeing the returned pointer using appropriate memory management.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_View(
    arr: *mut PyArrayObject,
    _descr: *mut libc::c_void, // Descriptor for new dtype (simplified - not used yet)
    _type: *mut libc::c_void,  // Type object (not used in pure C)
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array (this creates a view that doesn't own data)
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Wrap in Arc for proper reference counting
        let base_arc = Arc::new(array);
        
        // Create a zero-copy view with the same shape and dtype
        // For now, use the same shape (full implementation would allow dtype change via descriptor)
        let shape = base_arc.shape().to_vec();
        let strides = base_arc.strides().to_vec();
        let view = match Array::view_from_arc(&base_arc, shape, strides) {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&view)
    }
}

/// Create new view with different shape/strides
///
/// Equivalent to NumPy's PyArray_NewView function.
/// Creates a new view with different shape and strides.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_NewView(
    arr: *mut PyArrayObject,
    _type: *mut libc::c_void, // Type object (not used in pure C)
    _descr: *mut libc::c_void, // Descriptor (not used yet)
) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array view
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Wrap in Arc for reference counting
        let base_arc = Arc::new(array);
        
        // Create a zero-copy view with same shape and dtype
        // Full implementation would allow changing shape/strides via parameters
        let shape = base_arc.shape().to_vec();
        let strides = base_arc.strides().to_vec();
        let view = match Array::view_from_arc(&base_arc, shape, strides) {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&view)
    }
}

/// Remove dimensions of size 1
///
/// Equivalent to NumPy's PyArray_Squeeze function.
/// Removes dimensions of size 1 from the array.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Squeeze(arr: *mut PyArrayObject) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Wrap in Arc for reference counting
        let base_arc = Arc::new(array);
        
        // Compute squeezed shape
        let new_shape = squeeze_dims(base_arc.shape(), None);
        
        // Calculate new strides for squeezed shape
        let itemsize = base_arc.itemsize();
        let new_strides = {
            let mut strides = vec![0; new_shape.len()];
            if !new_shape.is_empty() {
                strides[new_shape.len() - 1] = itemsize as i64;
                for i in (0..new_shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * new_shape[i + 1];
                }
            }
            strides
        };
        
        // Create a zero-copy view with squeezed shape
        let view = match Array::view_from_arc(&base_arc, new_shape, new_strides) {
            Ok(v) => v,
            Err(_) => return ptr::null_mut(),
        };
        
        let new_array = view;
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

/// Flatten array to 1D
///
/// Equivalent to NumPy's PyArray_Flatten function.
/// Returns a flattened copy of the array.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_Flatten(arr: *mut PyArrayObject, _order: c_int) -> *mut PyArrayObject {
    if arr.is_null() {
        return ptr::null_mut();
    }
    
    unsafe {
        // Convert PyArrayObject to Array
        let array = match conversion::pyarray_to_array_view(arr) {
            Ok(a) => a,
            Err(_) => return ptr::null_mut(),
        };
        
        // Wrap in Arc for reference counting
        let base_arc = Arc::new(array);
        
        // Compute flattened shape
        let new_shape = flatten_shape(base_arc.shape());
        
        // For contiguous arrays, we can create a view
        // For non-contiguous arrays, we need to copy (NumPy behavior)
        let new_array = if base_arc.is_c_contiguous() || base_arc.is_f_contiguous() {
            // Create a zero-copy view
            let itemsize = base_arc.itemsize();
            let new_strides = vec![itemsize as i64]; // 1D array
            match Array::view_from_arc(&base_arc, new_shape, new_strides) {
                Ok(v) => v,
                Err(_) => return ptr::null_mut(),
            }
        } else {
            // Non-contiguous: create a copy (NumPy's flatten creates a copy)
            let dtype = base_arc.dtype().clone();
            let mut new_arr = match Array::new(new_shape, dtype) {
                Ok(a) => a,
                Err(_) => return ptr::null_mut(),
            };
            // Copy data
            let size = base_arc.size() * base_arc.itemsize();
            std::ptr::copy_nonoverlapping(
                base_arc.data_ptr(),
                new_arr.data_ptr_mut(),
                size,
            );
            new_arr
        };
        
        // Convert back to PyArrayObject
        conversion::array_to_pyarray_ptr(&new_array)
    }
}

