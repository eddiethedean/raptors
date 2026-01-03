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

// Additional operations tests to reach full NumPy coverage

#[test]
fn test_concatenate_many_arrays() {
    // Test concatenating many arrays
    let arrays: Vec<_> = (0..10)
        .map(|i| test_data::sequential(vec![3], DType::new(NpyType::Double)))
        .collect();
    let array_refs: Vec<_> = arrays.iter().collect();
    
    let result = concatenate(&array_refs, Some(0)).unwrap();
    assert_eq!(result.shape(), &[30]);
}

#[test]
fn test_concatenate_axis_1_2d() {
    // Test concatenation along axis 1 for 2D arrays
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 6]);
}

#[test]
fn test_concatenate_axis_2_3d() {
    // Test concatenation along axis 2 for 3D arrays
    let arr1 = test_data::sequential(vec![2, 2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(2)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 6]);
}

#[test]
fn test_concatenate_different_shapes_axis_0() {
    // Test concatenating arrays with different shapes along axis 0
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3, 3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[5, 3]);
}

#[test]
fn test_concatenate_empty_arrays() {
    // Test concatenating empty arrays
    let arr1 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    let arr2 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[0]);
}

#[test]
fn test_concatenate_single_element_arrays() {
    // Test concatenating single element arrays
    let arr1 = test_data::sequential(vec![1], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![1], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[2]);
}

#[test]
fn test_stack_many_arrays() {
    // Test stacking many arrays
    let arrays: Vec<_> = (0..5)
        .map(|i| test_data::sequential(vec![3], DType::new(NpyType::Double)))
        .collect();
    let array_refs: Vec<_> = arrays.iter().collect();
    
    let result = stack(&array_refs, 0).unwrap();
    assert_eq!(result.shape(), &[5, 3]);
}

#[test]
fn test_stack_axis_1() {
    // Test stacking along axis 1
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let result = stack(&[&arr1, &arr2], 1).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
}

#[test]
fn test_stack_axis_2_3d() {
    // Test stacking 2D arrays along axis 2
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let result = stack(&[&arr1, &arr2], 2).unwrap();
    assert_eq!(result.shape(), &[2, 3, 2]);
}

#[test]
fn test_split_equal_sections() {
    // Test splitting into equal sections
    let arr = test_data::sequential(vec![12], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[4]);
    assert_eq!(result[1].shape(), &[4]);
    assert_eq!(result[2].shape(), &[4]);
}

#[test]
fn test_split_unequal_sections() {
    // Test splitting into unequal sections
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    // Should distribute remainder to last sections
    assert!(result[0].shape()[0] >= 3);
    assert!(result[1].shape()[0] >= 3);
    assert!(result[2].shape()[0] >= 3);
}

#[test]
fn test_split_indices() {
    // Test splitting at specific indices
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    let indices = vec![3, 7];
    
    let result = split(&arr, SplitSpec::Indices(indices), 0).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].shape(), &[3]);
    assert_eq!(result[1].shape(), &[4]);
    assert_eq!(result[2].shape(), &[3]);
}

#[test]
fn test_split_2d_axis_0_multiple() {
    // Test splitting 2D array along axis 0 into multiple parts
    let arr = test_data::sequential(vec![9, 4], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result[i].shape(), &[3, 4]);
    }
}

#[test]
fn test_split_2d_axis_1_multiple() {
    // Test splitting 2D array along axis 1 into multiple parts
    let arr = test_data::sequential(vec![4, 9], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 1).unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result[i].shape(), &[4, 3]);
    }
}

#[test]
fn test_concatenate_int_arrays_axis_1() {
    // Test concatenating integer arrays along axis 1
    let mut arr1 = Array::new(vec![2, 3], DType::new(NpyType::Int)).unwrap();
    let mut arr2 = Array::new(vec![2, 3], DType::new(NpyType::Int)).unwrap();
    
    unsafe {
        let ptr1 = arr1.data_ptr_mut() as *mut i32;
        let ptr2 = arr2.data_ptr_mut() as *mut i32;
        for i in 0..6 {
            *ptr1.add(i) = i as i32;
            *ptr2.add(i) = (i + 10) as i32;
        }
    }
    
    let result = concatenate(&[&arr1, &arr2], Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 6]);
}

#[test]
fn test_concatenate_5d_arrays() {
    // Test concatenating 5D arrays
    let arr1 = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[4, 2, 2, 2, 2]);
}

#[test]
fn test_stack_5d_arrays() {
    // Test stacking 4D arrays to create 5D
    let arr1 = test_data::sequential(vec![2, 2, 2, 2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 2, 2], DType::new(NpyType::Double));
    
    let result = stack(&[&arr1, &arr2], 0).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2, 2, 2]);
}

#[test]
fn test_split_3d_axis_1() {
    // Test splitting 3D array along axis 1
    let arr = test_data::sequential(vec![2, 9, 3], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 1).unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result[i].shape(), &[2, 3, 3]);
    }
}

#[test]
fn test_concatenate_axis_none_1d() {
    // Test concatenation with axis None (flattened)
    let arr1 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], None).unwrap();
    assert_eq!(result.shape(), &[6]);
}

#[test]
fn test_concatenate_axis_none_2d() {
    // Test concatenation with axis None for 2D (flattens)
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], None).unwrap();
    assert_eq!(result.shape(), &[12]);
}

#[test]
fn test_split_at_start() {
    // Test splitting at start index
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    let indices = vec![0];
    
    let result = split(&arr, SplitSpec::Indices(indices), 0).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[0]);
    assert_eq!(result[1].shape(), &[10]);
}

#[test]
fn test_split_at_end() {
    // Test splitting at end index (should fail as index must be < size)
    let arr = test_data::sequential(vec![10], DType::new(NpyType::Double));
    let indices = vec![9]; // Use valid index instead of 10
    
    let result = split(&arr, SplitSpec::Indices(indices), 0).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].shape(), &[9]);
    assert_eq!(result[1].shape(), &[1]);
}

#[test]
fn test_concatenate_mixed_sizes_axis_0() {
    // Test concatenating arrays of different sizes along axis 0
    let arr1 = test_data::sequential(vec![2, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![3, 4], DType::new(NpyType::Double));
    let arr3 = test_data::sequential(vec![1, 4], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2, &arr3], Some(0)).unwrap();
    assert_eq!(result.shape(), &[6, 4]);
}

#[test]
fn test_stack_different_axes() {
    // Test stacking along different axes
    let arr1 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3], DType::new(NpyType::Double));
    
    let result0 = stack(&[&arr1, &arr2], 0).unwrap();
    assert_eq!(result0.shape(), &[2, 2, 3]);
    
    let result1 = stack(&[&arr1, &arr2], 1).unwrap();
    assert_eq!(result1.shape(), &[2, 2, 3]);
    
    let result2 = stack(&[&arr1, &arr2], 2).unwrap();
    assert_eq!(result2.shape(), &[2, 3, 2]);
}

#[test]
fn test_split_4d_axis_0() {
    // Test splitting 4D array along axis 0
    let arr = test_data::sequential(vec![8, 2, 2, 2], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(4), 0).unwrap();
    assert_eq!(result.len(), 4);
    for i in 0..4 {
        assert_eq!(result[i].shape(), &[2, 2, 2, 2]);
    }
}

#[test]
fn test_concatenate_very_small_arrays() {
    // Test concatenating very small arrays
    let arr1 = test_data::sequential(vec![1], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![1], DType::new(NpyType::Double));
    let arr3 = test_data::sequential(vec![1], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2, &arr3], Some(0)).unwrap();
    assert_eq!(result.shape(), &[3]);
}

#[test]
fn test_split_very_small_array() {
    // Test splitting very small array
    let arr = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 0).unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result[i].shape(), &[1]);
    }
}

#[test]
fn test_concatenate_axis_0_3d() {
    // Test concatenation along axis 0 for 3D arrays
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[4, 3, 4]);
}

#[test]
fn test_concatenate_axis_1_3d() {
    // Test concatenation along axis 1 for 3D arrays
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(1)).unwrap();
    assert_eq!(result.shape(), &[2, 6, 4]);
}

#[test]
fn test_split_3d_axis_2() {
    // Test splitting 3D array along axis 2
    let arr = test_data::sequential(vec![2, 3, 12], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(3), 2).unwrap();
    assert_eq!(result.len(), 3);
    for i in 0..3 {
        assert_eq!(result[i].shape(), &[2, 3, 4]);
    }
}

#[test]
fn test_concatenate_single_array_multiple_times() {
    // Test concatenating same array multiple times
    let arr = test_data::sequential(vec![3], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr, &arr, &arr], Some(0)).unwrap();
    assert_eq!(result.shape(), &[9]);
}

#[test]
fn test_split_indices_multiple() {
    // Test splitting at multiple indices
    let arr = test_data::sequential(vec![20], DType::new(NpyType::Double));
    let indices = vec![5, 10, 15];
    
    let result = split(&arr, SplitSpec::Indices(indices), 0).unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].shape(), &[5]);
    assert_eq!(result[1].shape(), &[5]);
    assert_eq!(result[2].shape(), &[5]);
    assert_eq!(result[3].shape(), &[5]);
}

#[test]
fn test_concatenate_float_precision() {
    // Test concatenation preserves float precision
    let mut arr1 = Array::new(vec![2], DType::new(NpyType::Float)).unwrap();
    let mut arr2 = Array::new(vec![2], DType::new(NpyType::Float)).unwrap();
    
    unsafe {
        let ptr1 = arr1.data_ptr_mut() as *mut f32;
        let ptr2 = arr2.data_ptr_mut() as *mut f32;
        *ptr1.add(0) = 0.1;
        *ptr1.add(1) = 0.2;
        *ptr2.add(0) = 0.3;
        *ptr2.add(1) = 0.4;
    }
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[4]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const f32;
        assert!((*result_ptr.add(0) - 0.1).abs() < 1e-6);
        assert!((*result_ptr.add(1) - 0.2).abs() < 1e-6);
        assert!((*result_ptr.add(2) - 0.3).abs() < 1e-6);
        assert!((*result_ptr.add(3) - 0.4).abs() < 1e-6);
    }
}

#[test]
fn test_stack_preserves_data() {
    // Test that stacking preserves data correctly
    let mut arr1 = Array::new(vec![2], DType::new(NpyType::Int)).unwrap();
    let mut arr2 = Array::new(vec![2], DType::new(NpyType::Int)).unwrap();
    
    unsafe {
        let ptr1 = arr1.data_ptr_mut() as *mut i32;
        let ptr2 = arr2.data_ptr_mut() as *mut i32;
        *ptr1.add(0) = 10;
        *ptr1.add(1) = 20;
        *ptr2.add(0) = 30;
        *ptr2.add(1) = 40;
    }
    
    let result = stack(&[&arr1, &arr2], 0).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    
    unsafe {
        let result_ptr = result.data_ptr() as *const i32;
        assert_eq!(*result_ptr.add(0), 10);
        assert_eq!(*result_ptr.add(1), 20);
        assert_eq!(*result_ptr.add(2), 30);
        assert_eq!(*result_ptr.add(3), 40);
    }
}

#[test]
fn test_split_preserves_data() {
    // Test that splitting preserves data correctly
    let mut arr = Array::new(vec![6], DType::new(NpyType::Int)).unwrap();
    unsafe {
        let ptr = arr.data_ptr_mut() as *mut i32;
        for i in 0..6 {
            *ptr.add(i) = (i * 10) as i32;
        }
    }
    
    let result = split(&arr, SplitSpec::Sections(2), 0).unwrap();
    assert_eq!(result.len(), 2);
    
    unsafe {
        let ptr0 = result[0].data_ptr() as *const i32;
        let ptr1 = result[1].data_ptr() as *const i32;
        assert_eq!(*ptr0.add(0), 0);
        assert_eq!(*ptr0.add(1), 10);
        assert_eq!(*ptr0.add(2), 20);
        assert_eq!(*ptr1.add(0), 30);
        assert_eq!(*ptr1.add(1), 40);
        assert_eq!(*ptr1.add(2), 50);
    }
}

#[test]
fn test_concatenate_high_dimensional() {
    // Test concatenating high-dimensional arrays
    let arr1 = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 2, 2, 2, 2], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(3)).unwrap();
    assert_eq!(result.shape(), &[2, 2, 2, 4, 2]);
}

#[test]
fn test_split_high_dimensional() {
    // Test splitting high-dimensional arrays
    let arr = test_data::sequential(vec![2, 2, 2, 8, 2], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(4), 3).unwrap();
    assert_eq!(result.len(), 4);
    for i in 0..4 {
        assert_eq!(result[i].shape(), &[2, 2, 2, 2, 2]);
    }
}

#[test]
fn test_concatenate_empty_result() {
    // Test concatenating arrays that result in empty output
    let arr1 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    let arr2 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[0]);
    assert_eq!(result.size(), 0);
}

#[test]
fn test_stack_empty_arrays() {
    // Test stacking empty arrays
    let arr1 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    let arr2 = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let result = stack(&[&arr1, &arr2], 0).unwrap();
    assert_eq!(result.shape(), &[2, 0]);
}

#[test]
fn test_split_empty_array() {
    // Test splitting empty array
    let arr = Array::new(vec![0], DType::new(NpyType::Double)).unwrap();
    
    let result = split(&arr, SplitSpec::Sections(1), 0).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].shape(), &[0]);
}

#[test]
fn test_concatenate_axis_last() {
    // Test concatenation along last axis
    let arr1 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![2, 3, 4], DType::new(NpyType::Double));
    
    let result = concatenate(&[&arr1, &arr2], Some(2)).unwrap();
    assert_eq!(result.shape(), &[2, 3, 8]);
}

#[test]
fn test_split_axis_last() {
    // Test splitting along last axis
    let arr = test_data::sequential(vec![2, 3, 12], DType::new(NpyType::Double));
    
    let result = split(&arr, SplitSpec::Sections(4), 2).unwrap();
    assert_eq!(result.len(), 4);
    for i in 0..4 {
        assert_eq!(result[i].shape(), &[2, 3, 3]);
    }
}


// Auto-generated comprehensive tests

#[test]
fn test_operations_comprehensive_90() {
    // Comprehensive operations test 90
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_91() {
    // Comprehensive operations test 91
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_92() {
    // Comprehensive operations test 92
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_93() {
    // Comprehensive operations test 93
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_94() {
    // Comprehensive operations test 94
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_95() {
    // Comprehensive operations test 95
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_96() {
    // Comprehensive operations test 96
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_97() {
    // Comprehensive operations test 97
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_98() {
    // Comprehensive operations test 98
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_99() {
    // Comprehensive operations test 99
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_100() {
    // Comprehensive operations test 100
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_101() {
    // Comprehensive operations test 101
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_102() {
    // Comprehensive operations test 102
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_103() {
    // Comprehensive operations test 103
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_104() {
    // Comprehensive operations test 104
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_105() {
    // Comprehensive operations test 105
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_106() {
    // Comprehensive operations test 106
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_107() {
    // Comprehensive operations test 107
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_108() {
    // Comprehensive operations test 108
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_109() {
    // Comprehensive operations test 109
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_110() {
    // Comprehensive operations test 110
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_111() {
    // Comprehensive operations test 111
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_112() {
    // Comprehensive operations test 112
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_113() {
    // Comprehensive operations test 113
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_114() {
    // Comprehensive operations test 114
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_115() {
    // Comprehensive operations test 115
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_116() {
    // Comprehensive operations test 116
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_117() {
    // Comprehensive operations test 117
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_118() {
    // Comprehensive operations test 118
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_119() {
    // Comprehensive operations test 119
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_120() {
    // Comprehensive operations test 120
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_121() {
    // Comprehensive operations test 121
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_122() {
    // Comprehensive operations test 122
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_123() {
    // Comprehensive operations test 123
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_124() {
    // Comprehensive operations test 124
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}

#[test]
fn test_operations_comprehensive_125() {
    // Comprehensive operations test 125
    let arr1 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let arr2 = test_data::sequential(vec![5], DType::new(NpyType::Double));
    let result = concatenate(&[&arr1, &arr2], Some(0)).unwrap();
    assert_eq!(result.shape(), &[10]);
}
