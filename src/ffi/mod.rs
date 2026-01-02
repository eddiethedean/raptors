//! Foreign Function Interface (FFI) module
//!
//! This module provides the C API compatibility layer,
//! exposing NumPy-compatible C functions for use as a drop-in replacement.

mod array_api;
mod conversion;
mod views;
mod manipulation;
mod indexing;
mod concatenation;
mod sorting;
mod linalg;
mod io;
mod operations;
mod einsum;

pub use array_api::*;
pub use conversion::*;
pub use views::*;
pub use manipulation::*;
pub use indexing::*;
pub use concatenation::*;
pub use sorting::*;
pub use linalg::*;
pub use io::*;
pub use operations::*;
pub use einsum::*;

use libc::{c_int, c_void, size_t};

/// C-compatible array object structure
///
/// This matches NumPy's PyArrayObject structure for C API compatibility.
/// Fields are public for C API compatibility.
#[repr(C)]
pub struct PyArrayObject {
    /// Object header (for Python compatibility, will be NULL in pure C usage)
    pub ob_base: *mut c_void,
    /// Data pointer
    pub data: *mut u8,
    /// Number of dimensions
    pub nd: c_int,
    /// Type descriptor (simplified for now)
    pub descr: *mut c_void,
    /// Flags
    pub flags: u32,
    /// Shape array (MAXDIMS elements)
    pub dimensions: [i64; 64],
    /// Strides array (MAXDIMS elements)
    pub strides: [i64; 64],
    /// Base object
    pub base: *mut PyArrayObject,
    /// Descriptor for the array element type
    pub _descr: *mut c_void,
    /// Weak references (for Python compatibility)
    pub weakreflist: *mut c_void,
}

/// Array creation function
///
/// Creates a new array with the specified parameters.
///
/// # Safety
/// The caller must ensure `_dimensions` points to an array of at least `_nd` elements if not null.
/// The caller must ensure `_strides` points to an array of at least `_nd` elements if not null.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_New(
    _subtype: *mut c_void,
    _nd: c_int,
    _dimensions: *const i64,
    _type_num: c_int,
    _strides: *const i64,
    _data: *mut c_void,
    _itemsize: c_int,
    _flags: c_int,
    _obj: *mut c_void,
) -> *mut PyArrayObject {
    use crate::{empty, types::DType};
    
    // Ignore unused parameters for now
    let _ = (_subtype, _strides, _data, _flags, _obj);
    
    if _dimensions.is_null() || _nd <= 0 || _nd > 64 {
        return std::ptr::null_mut();
    }
    
    // Convert type_num to NpyType
    let npy_type = match conversion::type_num_to_npytype(_type_num) {
        Some(t) => t,
        None => return std::ptr::null_mut(),
    };
    
    let dtype = DType::new(npy_type);
    
    // Build shape vector
    let mut shape = Vec::with_capacity(_nd as usize);
    unsafe {
        for i in 0..(_nd as usize) {
            shape.push(*_dimensions.add(i));
        }
    }
    
    // Create array
    match empty(shape, dtype) {
        Ok(array) => {
            // Convert to PyArrayObject
            unsafe {
                conversion::array_to_pyarray_ptr(&array)
            }
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Get array size
///
/// Returns the total number of elements in the array.
///
/// # Safety
/// The caller must ensure `arr` is a valid pointer to a PyArrayObject
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn PyArray_SIZE(arr: *mut PyArrayObject) -> size_t {
    if arr.is_null() {
        return 0;
    }
    
    unsafe {
        let arr_ref = &*arr;
        let mut size: size_t = 1;
        for i in 0..(arr_ref.nd as usize) {
            if arr_ref.dimensions[i] > 0 {
                size *= arr_ref.dimensions[i] as size_t;
            }
        }
        size
    }
}

/// Check if object is an array
///
/// Equivalent to NumPy's PyArray_Check function.
/// Checks if the object is a PyArrayObject (or subclass).
///
/// # Safety
/// The caller must ensure `op` is a valid pointer if not null.
#[no_mangle]
pub extern "C" fn PyArray_Check(op: *mut c_void) -> c_int {
    // Simplified check - in real implementation would check object type
    // For now, check if pointer is not null and could be a PyArrayObject
    if op.is_null() {
        return 0;
    }
    // TODO: Implement proper type checking by examining object structure
    // For now, assume non-null pointer could be an array
    // In full implementation, would check ob_type or similar
    1 // Assume it's an array if not null (simplified)
}

