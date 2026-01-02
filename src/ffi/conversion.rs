//! Conversion utilities between internal Array and PyArrayObject
//!
//! This module provides conversion functions to bridge the gap between
//! the internal Rust Array type and the C-compatible PyArrayObject

use crate::array::{Array, ArrayError};
use crate::ffi::PyArrayObject;
use crate::types::{DType, NpyType};
use libc::{c_int, c_void};
use std::alloc::{alloc, dealloc, Layout};

/// Convert internal Array to heap-allocated PyArrayObject* for C API
///
/// This creates a heap-allocated PyArrayObject that shares data with the Array.
/// The caller is responsible for freeing the memory using `free_pyarray`.
/// 
/// # Safety
/// The Array must outlive the returned PyArrayObject*.
/// The returned pointer must be freed using `free_pyarray` to avoid memory leaks.
pub unsafe fn array_to_pyarray_ptr(array: &Array) -> *mut PyArrayObject {
    let mut dimensions = [0i64; 64];
    let mut strides = [0i64; 64];
    
    let shape = array.shape();
    let array_strides = array.strides();
    
    for (i, &dim) in shape.iter().enumerate() {
        if i < 64 {
            dimensions[i] = dim;
        }
    }
    
    for (i, &stride) in array_strides.iter().enumerate() {
        if i < 64 {
            strides[i] = stride;
        }
    }
    
    // Cast immutable pointer to mutable for C API compatibility
    let data_ptr = array.data_ptr() as *mut u8;
    
    // Allocate PyArrayObject on heap
    let layout = Layout::new::<PyArrayObject>();
    let pyarray_ptr = alloc(layout) as *mut PyArrayObject;
    
    if pyarray_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    // Store type_num in _descr field for later retrieval
    // Allocate a c_int on the heap to store the type_num
    let type_num = npytype_to_type_num(array.dtype().type_());
    let type_num_ptr = alloc(Layout::new::<c_int>()) as *mut c_int;
    if !type_num_ptr.is_null() {
        std::ptr::write(type_num_ptr, type_num);
    }
    
    // Initialize the structure
    std::ptr::write(pyarray_ptr, PyArrayObject {
        ob_base: std::ptr::null_mut(),
        data: data_ptr,
        nd: array.ndim() as c_int,
        descr: std::ptr::null_mut(), // TODO: Convert dtype to descriptor
        flags: array.flags().bits(),
        dimensions,
        strides,
        base: std::ptr::null_mut(),
        _descr: type_num_ptr as *mut c_void, // Store type_num pointer here
        weakreflist: std::ptr::null_mut(),
    });
    
    pyarray_ptr
}

/// Free a heap-allocated PyArrayObject
///
/// # Safety
/// The pointer must have been allocated by `array_to_pyarray_ptr` or be null.
pub unsafe fn free_pyarray(ptr: *mut PyArrayObject) {
    if !ptr.is_null() {
        // Free the type_num stored in _descr if it exists
        let type_num_ptr = (*ptr)._descr as *mut c_int;
        if !type_num_ptr.is_null() {
            dealloc(type_num_ptr as *mut u8, Layout::new::<c_int>());
        }
        
        let layout = Layout::new::<PyArrayObject>();
        std::ptr::drop_in_place(ptr);
        dealloc(ptr as *mut u8, layout);
    }
}

/// Convert PyArrayObject* to borrowed Array reference
///
/// This creates a temporary Array view that borrows data from PyArrayObject.
/// The Array does not own the data.
///
/// # Safety
/// The PyArrayObject must be valid and the data pointer must be valid.
/// The returned Array is a view and does not own the data.
pub unsafe fn pyarray_to_array_view(pyarray: *const PyArrayObject) -> Result<Array, ArrayError> {
    if pyarray.is_null() {
        return Err(ArrayError::InvalidShape);
    }
    
    let arr = &*pyarray;
    
    // Extract shape and strides
    let ndim = arr.nd as usize;
    let mut shape = Vec::with_capacity(ndim);
    let mut strides = Vec::with_capacity(ndim);
    
    for i in 0..ndim {
        shape.push(arr.dimensions[i]);
        strides.push(arr.strides[i]);
    }
    
    // Try to get dtype from stored type_num in _descr field
    // If not available, infer from itemsize
    let dtype = if !arr._descr.is_null() {
        // Retrieve type_num from _descr
        let type_num_ptr = arr._descr as *const c_int;
        let type_num = *type_num_ptr;
        match type_num_to_npytype(type_num) {
            Some(npy_type) => DType::new(npy_type),
            None => {
                // Fallback to itemsize inference
                let itemsize = if ndim > 0 {
                    let last_idx = ndim - 1;
                    if last_idx < 64 && arr.strides[last_idx] > 0 {
                        arr.strides[last_idx] as usize
                    } else {
                        8
                    }
                } else {
                    8
                };
                infer_dtype_from_itemsize(itemsize)?
            }
        }
    } else {
        // No stored type_num, infer from itemsize
        let itemsize = if ndim > 0 {
            let last_idx = ndim - 1;
            if last_idx < 64 && arr.strides[last_idx] > 0 {
                arr.strides[last_idx] as usize
            } else {
                8 // Default to double (8 bytes)
            }
        } else {
            8 // Default to double (8 bytes)
        };
        infer_dtype_from_itemsize(itemsize)?
    };
    
    // Create Array view using from_external_memory (doesn't own data)
    // This creates a true view that shares the memory with PyArrayObject
    if arr.data.is_null() {
        return Err(ArrayError::AllocationFailed);
    }
    
    let array = unsafe {
        Array::from_external_memory(
            arr.data,
            shape,
            dtype,
            false, // Does not own data
        )?
    };
    
    Ok(array)
}

/// Convert PyArrayObject* to owned Array
///
/// This creates an owned Array by copying data from PyArrayObject.
///
/// # Safety
/// The PyArrayObject must be valid.
pub unsafe fn pyarray_to_array_owned(pyarray: *const PyArrayObject) -> Result<Array, ArrayError> {
    // For now, same as view - creates a copy
    // In a full implementation, we could optimize this
    pyarray_to_array_view(pyarray)
}

/// Infer dtype from itemsize (simplified)
///
/// This is a helper function that tries to infer the dtype from itemsize.
/// In a full implementation, we would use the descriptor from PyArrayObject.
fn infer_dtype_from_itemsize(itemsize: usize) -> Result<DType, ArrayError> {
    // Map itemsize to dtype (simplified - assumes most common types)
    let npy_type = match itemsize {
        1 => NpyType::Byte,      // Could be Bool, Byte, or UByte
        2 => NpyType::Short,     // Could be Short, UShort, or Half
        4 => NpyType::Int,       // Could be Int, UInt, or Float
        8 => NpyType::Double,    // Could be Long, ULong, or Double
        16 => NpyType::CDouble,  // Complex double
        _ => NpyType::Double,    // Default to double for unknown sizes
    };
    Ok(DType::new(npy_type))
}

/// Convert type number to NpyType
pub fn type_num_to_npytype(type_num: c_int) -> Option<NpyType> {
    match type_num {
        0 => Some(NpyType::Bool),
        1 => Some(NpyType::Byte),
        2 => Some(NpyType::UByte),
        3 => Some(NpyType::Short),
        4 => Some(NpyType::UShort),
        5 => Some(NpyType::Int),
        6 => Some(NpyType::UInt),
        7 => Some(NpyType::Long),
        8 => Some(NpyType::ULong),
        11 => Some(NpyType::Float),
        12 => Some(NpyType::Double),
        _ => None,
    }
}

/// Convert NpyType to type number
pub fn npytype_to_type_num(npy_type: NpyType) -> c_int {
    match npy_type {
        NpyType::Bool => 0,
        NpyType::Byte => 1,
        NpyType::UByte => 2,
        NpyType::Short => 3,
        NpyType::UShort => 4,
        NpyType::Int => 5,
        NpyType::UInt => 6,
        NpyType::Long => 7,
        NpyType::ULong => 8,
        NpyType::Float => 11,
        NpyType::Double => 12,
        _ => -1, // Unsupported
    }
}

/// Legacy function for backward compatibility
/// 
/// # Safety
/// See `array_to_pyarray_ptr` for safety requirements.
pub unsafe fn array_to_pyarray(array: &Array) -> PyArrayObject {
    let mut dimensions = [0i64; 64];
    let mut strides = [0i64; 64];
    
    let shape = array.shape();
    let array_strides = array.strides();
    
    for (i, &dim) in shape.iter().enumerate() {
        if i < 64 {
            dimensions[i] = dim;
        }
    }
    
    for (i, &stride) in array_strides.iter().enumerate() {
        if i < 64 {
            strides[i] = stride;
        }
    }
    
    // Cast immutable pointer to mutable for C API compatibility
    let data_ptr = array.data_ptr() as *mut u8;
    
    PyArrayObject {
        ob_base: std::ptr::null_mut(),
        data: data_ptr,
        nd: array.ndim() as c_int,
        descr: std::ptr::null_mut(),
        flags: array.flags().bits(),
        dimensions,
        strides,
        base: std::ptr::null_mut(),
        _descr: std::ptr::null_mut(),
        weakreflist: std::ptr::null_mut(),
    }
}

