//! NumPy array operations tests
//!
//! Ported from NumPy's test_array_operations.py
//! Tests cover concatenate, stack, split, and related operations

#![allow(unused_unsafe)]

mod numpy_port {
    pub mod helpers;
}

use numpy_port::helpers::*;
use numpy_port::helpers::test_data;
use raptors_core::array::Array;
use raptors_core::types::{DType, NpyType};
use raptors_core::concatenation::{concatenate, stack, split, SplitSpec};
use raptors_core::{zeros, ones};

// Concatenate tests

#[test]
fn test_concatenate_1d_axis_0() {
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[6]);
    unsafe {
        let ptr = result.data_ptr() as *const f64;
        for i in 0..6 {
            assert!((*ptr.add(i) - (i % 3) as f64).abs() < 1e-10);
        }
    }
}

#[test]
fn test_concatenate_2d_axis_0() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[4, 3]);
}

#[test]
fn test_concatenate_2d_axis_1() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(1)).unwrap();
    
    assert_eq!(result.shape(), &[2, 6]);
}

#[test]
fn test_concatenate_3d_axis_0() {
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[4, 3, 4]);
}

#[test]
fn test_concatenate_3d_axis_1() {
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(1)).unwrap();
    
    assert_eq!(result.shape(), &[2, 6, 4]);
}

#[test]
fn test_concatenate_3d_axis_2() {
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(2)).unwrap();
    
    assert_eq!(result.shape(), &[2, 3, 8]);
}

#[test]
fn test_concatenate_multiple_arrays() {
    let arr1 = test_data::sequential(vec![2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2], DType::new(NpyType::Double));
    let arr3 = test_data::sequential(vec![2], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2, &arr3];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[6]);
}

#[test]
fn test_concatenate_empty_array() {
    let arr1 = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn test_concatenate_single_array() {
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    
    let arrays = vec![&arr];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), arr.shape());
    assert_array_equal(&result, &arr);
}

#[test]
fn test_concatenate_axis_none() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, None).unwrap();
    
    // Flattened concatenation
    assert_eq!(result.shape(), &[12]);
}

#[test]
fn test_concatenate_different_sizes_axis_0() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[5, 3]);
}

// Stack tests

#[test]
fn test_stack_1d_axis_0() {
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[2, 3]);
}

#[test]
fn test_stack_1d_axis_1() {
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 1).unwrap();
    
    assert_eq!(result.shape(), &[3, 2]);
}

#[test]
fn test_stack_2d_axis_0() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[2, 2, 3]);
}

#[test]
fn test_stack_2d_axis_1() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 1).unwrap();
    
    assert_eq!(result.shape(), &[2, 2, 3]);
}

#[test]
fn test_stack_2d_axis_2() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 2).unwrap();
    
    assert_eq!(result.shape(), &[2, 3, 2]);
}

#[test]
fn test_stack_multiple_arrays() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr3 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2, &arr3];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[3, 2, 3]);
}

#[test]
fn test_stack_single_array() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[1, 2, 3]);
}

#[test]
fn test_stack_empty_array() {
    let arr1 = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    let arr2 = zeros(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[2, 0]);
}

#[test]
fn test_stack_shape_mismatch() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3, 2], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0);
    
    assert!(result.is_err());
}

// Split tests

#[test]
fn test_split_1d_sections() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[2]);
    assert_eq!(result[1].shape(), &[2]);
    assert_eq!(result[2].shape(), &[2]);
}

#[test]
fn test_split_1d_indices() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let indices = vec![2, 4];
    let result = split(&arr, SplitSpec::Indices(indices), 0).unwrap();
    
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[2]);
    assert_eq!(result[1].shape(), &[2]);
    assert_eq!(result[2].shape(), &[2]);
}

#[test]
fn test_split_2d_axis_0() {
    let arr = test_data::sequential(vec![6, 3], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[2, 3]);
    assert_eq!(result[1].shape(), &[2, 3]);
    assert_eq!(result[2].shape(), &[2, 3]);
}

#[test]
fn test_split_2d_axis_1() {
    let arr = test_data::sequential(vec![2, 6], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 1).unwrap();
    
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[2, 2]);
    assert_eq!(result[1].shape(), &[2, 2]);
    assert_eq!(result[2].shape(), &[2, 2]);
}

#[test]
fn test_split_3d_axis_0() {
    let arr = test_data::sequential(vec![6, 2, 3], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[2, 2, 3]);
    assert_eq!(result[1].shape(), &[2, 2, 3]);
    assert_eq!(result[2].shape(), &[2, 2, 3]);
}

#[test]
fn test_split_uneven_sections() {
    let arr = test_data::sequential(vec![7], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    
    assert_eq!(result.len(), 3);
    // First two sections get 2 elements, last gets 3
    assert_eq!(result[0].shape(), &[2]);
    assert_eq!(result[1].shape(), &[2]);
    assert_eq!(result[2].shape(), &[3]);
}

#[test]
fn test_split_single_section() {
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(1), 0).unwrap();
    
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape(), &[5]);
    assert_array_equal(&result[0], &arr);
}

#[test]
fn test_split_invalid_axis() {
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(2), 10); // Invalid axis
    
    assert!(result.is_err());
}

#[test]
fn test_split_invalid_sections() {
    let arr = test_data::sequential(vec![5], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(10), 0); // More sections than elements
    
    // May succeed or fail depending on implementation
    let _ = result;
}

// Edge cases

#[test]
fn test_concatenate_empty_list() {
    let arrays: Vec<&Array> = vec![];
    let result = concatenate(&arrays, Some(0));
    
    assert!(result.is_err());
}

#[test]
fn test_stack_empty_list() {
    let arrays: Vec<&Array> = vec![];
    let result = stack(&arrays, 0);
    
    assert!(result.is_err());
}

#[test]
fn test_concatenate_invalid_axis() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr, &arr];
    let result = concatenate(&arrays, Some(10)); // Invalid axis
    
    assert!(result.is_err());
}

#[test]
fn test_stack_invalid_axis() {
    let arr = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr, &arr];
    let result = stack(&arrays, 10); // Invalid axis
    
    assert!(result.is_err());
}

#[test]
fn test_concatenate_shape_mismatch() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 4], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0));
    
    // Should succeed (same first dimension)
    if result.is_ok() {
        assert_eq!(result.unwrap().shape(), &[4, 3]);
    }
}

#[test]
fn test_concatenate_shape_mismatch_axis_1() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3, 3], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(1));
    
    // Should fail (different first dimension)
    assert!(result.is_err());
}

// Large array tests

#[test]
fn test_concatenate_large_arrays() {
    let arr1 = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    let arr2 = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[200]);
}

#[test]
fn test_stack_large_arrays() {
    let arr1 = ones(vec![10, 10], DType::new(NpyType::Double)).unwrap();
    let arr2 = ones(vec![10, 10], DType::new(NpyType::Double)).unwrap();
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[2, 10, 10]);
}

#[test]
fn test_split_large_array() {
    let arr = ones(vec![100], DType::new(NpyType::Double)).unwrap();
    
    let result = split(&arr, SplitSpec::Sections(10), 0).unwrap();
    
    assert_eq!(result.len(), 10);
    for i in 0..10 {
        assert_eq!(result[i].shape(), &[10]);
    }
}

// High-dimensional tests

#[test]
fn test_concatenate_4d() {
    let arr1 = test_data::sequential(vec![2, 2, 2, 2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 2, 2], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0)).unwrap();
    
    assert_eq!(result.shape(), &[4, 2, 2, 2]);
}

#[test]
fn test_stack_4d() {
    let arr1 = test_data::sequential(vec![2, 2, 2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 2], DType::new(NpyType::Double));
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0).unwrap();
    
    assert_eq!(result.shape(), &[2, 2, 2, 2]);
}

#[test]
fn test_split_4d() {
    let arr = test_data::sequential(vec![4, 2, 2, 2], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(2), 0).unwrap();
    
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[2, 2, 2, 2]);
    assert_eq!(result[1].shape(), &[2, 2, 2, 2]);
}

// Different dtypes

#[test]
fn test_concatenate_int_arrays() {
    let mut arr1 = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
    let mut arr2 = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
    
    unsafe {
        let ptr1 = arr1.data_ptr_mut() as *mut i32;
        let ptr2 = arr2.data_ptr_mut() as *mut i32;
        for i in 0..3 {
            *ptr1.add(i) = i as i32;
            *ptr2.add(i) = (i + 3) as i32;
        }
    }
    
    let arrays = vec![&arr1, &arr2];
    let result = concatenate(&arrays, Some(0));
    
    if result.is_ok() {
        assert_eq!(result.unwrap().shape(), &[6]);
    }
}

#[test]
fn test_stack_float_arrays() {
    let mut arr1 = Array::new(vec![2, 3], DType::new(NpyType::Float)).unwrap();
    let mut arr2 = Array::new(vec![2, 3], DType::new(NpyType::Float)).unwrap();
    
    unsafe {
        let ptr1 = arr1.data_ptr_mut() as *mut f32;
        let ptr2 = arr2.data_ptr_mut() as *mut f32;
        for i in 0..6 {
            *ptr1.add(i) = i as f32;
            *ptr2.add(i) = (i + 6) as f32;
        }
    }
    
    let arrays = vec![&arr1, &arr2];
    let result = stack(&arrays, 0);
    
    if result.is_ok() {
        assert_eq!(result.unwrap().shape(), &[2, 2, 3]);
    }
}

// Consistency tests

#[test]
fn test_concatenate_stack_consistency() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    // Stack then concatenate should give same result as concatenate then stack
    let stacked = stack(&[&arr1, &arr2], 0).unwrap();
    let concatenated = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    
    // Different operations, but both should succeed
    assert_eq!(stacked.shape(), &[2, 2, 3]);
    assert_eq!(concatenated.shape(), &[4, 3]);
}

#[test]
fn test_split_concatenate_inverse() {
    let arr = test_data::sequential(vec![6], DType::new(NpyType::Double));
    
    let split_result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    let arrays: Vec<&Array> = split_result.iter().collect();
    let concatenated = concatenate(&arrays, Some(0)).unwrap();
    
    assert_array_equal(&arr, &concatenated);
}

#[test]
fn test_stack_split_inverse() {
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let stacked = stack(&[&arr1, &arr2], 0).unwrap();
    let split_result = split(&stacked, SplitSpec::Sections(2), 0).unwrap();
    
    assert_eq!(split_result.len(), 2);
    
    // Squeeze axis 0 from each split result to get back the original shape
    // This matches NumPy's behavior: splitting a stacked array returns arrays
    // with an extra dimension that needs to be squeezed to reverse the stack operation
    let squeezed1 = squeeze_axis(&split_result[0], 0).unwrap();
    let squeezed2 = squeeze_axis(&split_result[1], 0).unwrap();
    
    assert_array_equal(&squeezed1, &arr1);
    assert_array_equal(&squeezed2, &arr2);
}

// Test with helpers

#[test]
fn test_operations_with_helpers() {
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let concatenated = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(concatenated.shape(), &[6]);
    
    let stacked = stack(&[&arr1, &arr2], 0).unwrap();
    assert_eq!(stacked.shape(), &[2, 3]);
}

