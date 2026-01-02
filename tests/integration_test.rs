//! Integration tests
#![allow(unused_unsafe)]
//!
//! These tests verify C API compatibility and basic functionality

#[cfg(test)]
mod tests {
    use raptors_core::ffi::*;
    use std::ptr;

    #[test]
    fn test_pyarray_size_null() {
        // Test that PyArray_SIZE returns 0 for null pointer
        let result = unsafe { PyArray_SIZE(ptr::null_mut()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_pyarray_check_null() {
        // Test that PyArray_Check returns 0 for null pointer
        let result = unsafe { PyArray_Check(ptr::null_mut()) };
        assert_eq!(result, 0);
    }
}

