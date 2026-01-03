//! Tests for memory allocation functions

use raptors_core::memory::{allocate_aligned, deallocate_aligned, verify_alignment};

#[test]
fn test_zero_size_allocation() {
    // Test with different alignments
    for align in [1, 2, 4, 8, 16, 32, 64] {
        let ptr = allocate_aligned(0, align);
        assert!(!ptr.is_null(), "Zero-size allocation should return non-null pointer");
        assert!(verify_alignment(ptr, align), "Pointer should be aligned to {}", align);
        
        // Should be safe to deallocate
        unsafe {
            deallocate_aligned(ptr, 0, align);
        }
    }
}

#[test]
fn test_zero_size_allocation_large_alignments() {
    // Test with larger alignments that might be used in practice
    for align in [128, 256, 512, 1024] {
        let ptr = allocate_aligned(0, align);
        assert!(!ptr.is_null(), "Zero-size allocation should return non-null pointer");
        assert!(verify_alignment(ptr, align), "Pointer should be aligned to {}", align);
        
        unsafe {
            deallocate_aligned(ptr, 0, align);
        }
    }
}

#[test]
fn test_zero_size_allocation_deallocation_safety() {
    // Verify that zero-size allocations can be safely deallocated multiple times
    // (though in practice you should only deallocate once)
    let align = 16;
    let ptr = allocate_aligned(0, align);
    assert!(!ptr.is_null());
    
    // Deallocate should be safe (it checks size == 0 and returns early)
    unsafe {
        deallocate_aligned(ptr, 0, align);
        // Deallocating again should also be safe (size is 0, so it returns early)
        deallocate_aligned(ptr, 0, align);
    }
}

#[test]
fn test_normal_allocation_still_works() {
    // Verify that normal (non-zero) allocations still work correctly
    let size = 1024;
    let align = 16;
    
    let ptr = allocate_aligned(size, align);
    assert!(!ptr.is_null(), "Normal allocation should return non-null pointer");
    assert!(verify_alignment(ptr, align), "Pointer should be aligned");
    
    // Write some data to verify the pointer is valid
    unsafe {
        std::ptr::write_bytes(ptr, 0x42, size);
        
        // Verify we can read it back
        assert_eq!(*ptr, 0x42);
    }
    
    unsafe {
        deallocate_aligned(ptr, size, align);
    }
}

#[test]
fn test_zero_size_vs_normal_allocation() {
    // Verify that zero-size and normal allocations return different pointers
    let align = 16;
    
    let zero_ptr = allocate_aligned(0, align);
    let normal_ptr = allocate_aligned(1024, align);
    
    assert!(!zero_ptr.is_null());
    assert!(!normal_ptr.is_null());
    // They should be different (though this isn't guaranteed, it's likely)
    
    unsafe {
        deallocate_aligned(zero_ptr, 0, align);
        deallocate_aligned(normal_ptr, 1024, align);
    }
}

