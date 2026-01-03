//! Memory allocation functions
//!
//! This module provides memory allocation with proper alignment,
//! equivalent to NumPy's alloc.c

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// Allocate aligned memory
///
/// Allocates memory with the specified size and alignment.
/// Returns a pointer to the allocated memory, or null if allocation fails.
pub fn allocate_aligned(size: usize, align: usize) -> *mut u8 {
    if size == 0 {
        // Return a properly aligned dangling pointer for zero-size allocations
        // This pointer must be non-null and aligned, but should never be dereferenced
        // We use NonNull::dangling() and align it to the requested alignment
        let dangling = NonNull::<u8>::dangling().as_ptr();
        let addr = dangling as usize;
        // Calculate the next aligned address
        let aligned_addr = (addr + align - 1) & !(align - 1);
        return aligned_addr as *mut u8;
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

/// Get recommended SIMD alignment for the platform
///
/// Returns the alignment requirement for SIMD operations on this platform.
/// Typically 16 bytes for SSE, 32 bytes for AVX, 64 bytes for AVX-512.
/// Get the optimal SIMD alignment for the current platform
///
/// Returns the alignment requirement in bytes for SIMD operations.
/// This is platform-specific:
/// - x86_64: 64 bytes for AVX-512, 32 bytes for AVX2, 16 bytes for SSE (default)
/// - aarch64: 16 bytes for ARM NEON
/// - Other platforms: 16 bytes (default)
#[cfg(target_arch = "x86_64")]
pub fn simd_alignment() -> usize {
    // Check for AVX-512 support (64 bytes)
    #[cfg(target_feature = "avx512f")]
    {
        return 64;
    }
    // Check for AVX2 support (32 bytes)
    #[cfg(target_feature = "avx2")]
    {
        return 32;
    }
    // Default to SSE alignment (16 bytes)
    16
}

/// Get the optimal SIMD alignment for ARM platforms
///
/// Returns 16 bytes for ARM NEON instructions.
#[cfg(target_arch = "aarch64")]
pub fn simd_alignment() -> usize {
    // ARM NEON typically uses 16-byte alignment
    16
}

/// Get the optimal SIMD alignment for other platforms
///
/// Returns a default alignment of 16 bytes.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn simd_alignment() -> usize {
    // Default alignment for other platforms
    16
}

/// Allocate memory with SIMD-friendly alignment
///
/// Allocates memory aligned for optimal SIMD operations on the current platform.
pub fn allocate_simd_aligned(size: usize) -> *mut u8 {
    let align = simd_alignment();
    allocate_aligned(size, align)
}

/// Verify memory alignment
///
/// Checks if a pointer is aligned to the specified alignment requirement.
pub fn verify_alignment(ptr: *const u8, align: usize) -> bool {
    if align == 0 {
        return true;
    }
    (ptr as usize).is_multiple_of(align)
}

/// Verify SIMD alignment
///
/// Checks if a pointer is aligned for SIMD operations on the current platform.
pub fn verify_simd_alignment(ptr: *const u8) -> bool {
    let align = simd_alignment();
    verify_alignment(ptr, align)
}

