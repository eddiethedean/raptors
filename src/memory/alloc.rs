//! Memory allocation functions
//!
//! This module provides memory allocation with proper alignment,
//! equivalent to NumPy's alloc.c

use std::alloc::{Layout, alloc, dealloc};

/// Allocate aligned memory
///
/// Allocates memory with the specified size and alignment.
/// Returns a pointer to the allocated memory, or null if allocation fails.
pub fn allocate_aligned(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        return align as *mut u8; // Return aligned pointer for zero-size allocation
    }
    
    let layout = match Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return std::ptr::null_mut(),
    };
    
    unsafe { alloc(layout) }
}

/// Deallocate aligned memory
///
/// Deallocates memory that was previously allocated with allocate_aligned.
///
/// # Safety
/// 
/// * `ptr` must be a pointer returned by `allocate_aligned` or be null
/// * `size` and `align` must match the values used when allocating
/// * The memory must not be used after this call
pub unsafe fn deallocate_aligned(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() || size == 0 {
        return;
    }
    
    let layout = match Layout::from_size_align(size, align) {
        Ok(layout) => layout,
        Err(_) => return,
    };
    
    dealloc(ptr, layout);
}

/// Reallocate aligned memory
///
/// Reallocates memory with a new size, preserving alignment.
///
/// # Safety
///
/// * `ptr` must be a pointer returned by `allocate_aligned` or be null
/// * `old_size` and `align` must match the values used when allocating
/// * The returned pointer must be deallocated with `deallocate_aligned` using `new_size` and `align`
pub unsafe fn reallocate_aligned(ptr: *mut u8, old_size: usize, new_size: usize, align: usize) -> *mut u8 {
    if new_size == 0 {
        if !ptr.is_null() {
            deallocate_aligned(ptr, old_size, align);
        }
        return std::ptr::null_mut();
    }
    
    // For simplicity, allocate new memory and copy data
    // A production implementation might use more sophisticated strategies
    let new_ptr = allocate_aligned(new_size, align);
    if new_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    if !ptr.is_null() {
        let copy_size = old_size.min(new_size);
        std::ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
        deallocate_aligned(ptr, old_size, align);
    }
    
    new_ptr
}

