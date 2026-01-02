//! Tests for array view functionality
#![allow(clippy::arc_with_non_send_sync)]

#[cfg(test)]
mod tests {
    use raptors_core::array::Array;
    use raptors_core::types::{DType, NpyType};
    use std::sync::Arc;

    #[test]
    fn test_view_creation() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let array = Array::new(shape.clone(), dtype.clone()).unwrap();
        
        // Create a view
        let view = array.view(shape, vec![32, 8]).unwrap();
        
        assert!(view.is_view());
        assert!(!view.owns_data());
        assert_eq!(view.shape(), array.shape());
    }

    #[test]
    fn test_view_from_arc() {
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Int);
        let array = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        // Create view from Arc
        let view = Array::view_from_arc(&array, shape, vec![12, 4]).unwrap();
        
        assert!(view.is_view());
        assert!(view.base_array().is_some());
        assert_eq!(Arc::strong_count(&array), 2); // array + view
    }

    #[test]
    fn test_view_memory_sharing() {
        let shape = vec![3, 3];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        // Fill base array with values
        unsafe {
            let data_ptr = base.data_ptr() as *mut i32;
            for i in 0..9 {
                *data_ptr.add(i) = i as i32;
            }
        }
        
        // Create view
        let view = Array::view_from_arc(&base, shape, vec![12, 4]).unwrap();
        
        // Verify view sees the same data
        unsafe {
            let view_ptr = view.data_ptr() as *const i32;
            assert_eq!(*view_ptr, 0);
            assert_eq!(*view_ptr.add(1), 1);
        }
    }

    #[test]
    fn test_view_reference_counting() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Float);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        assert_eq!(Arc::strong_count(&base), 1);
        
        let view1 = Array::view_from_arc(&base, shape.clone(), vec![8, 4]).unwrap();
        assert_eq!(view1.base_reference_count(), Some(2)); // base + view1
        
        let view2 = Array::view_from_arc(&base, shape, vec![8, 4]).unwrap();
        assert_eq!(view2.base_reference_count(), Some(3)); // base + view1 + view2
        
        // Drop views
        drop(view1);
        drop(view2);
        
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_view_is_view() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let array = Array::new(shape, dtype).unwrap();
        
        assert!(!array.is_view());
        
        let base = Arc::new(array);
        let view = Array::view_from_arc(&base, vec![3], vec![8]).unwrap();
        
        assert!(view.is_view());
    }

    #[test]
    fn test_view_base_array() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        let view = Array::view_from_arc(&base, shape, vec![8, 4]).unwrap();
        
        let base_ref = view.base_array().unwrap();
        assert_eq!(base_ref.shape(), base.shape());
    }

    #[test]
    fn test_view_copy() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        // Fill with data
        unsafe {
            let data_ptr = base.data_ptr() as *mut i32;
            *data_ptr = 42;
        }
        
        let view = Array::view_from_arc(&base, shape, vec![8, 4]).unwrap();
        let copy = view.copy();
        
        // Copy should own data, not be a view
        assert!(!copy.is_view());
        assert!(copy.owns_data());
        
        // Verify data was copied
        unsafe {
            let copy_ptr = copy.data_ptr() as *const i32;
            assert_eq!(*copy_ptr, 42);
        }
    }

    #[test]
    fn test_view_atleast_1d() {
        // Test with 1D array
        let array = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
        let result = array.atleast_1d().unwrap();
        assert_eq!(result.ndim(), 1);
        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_view_atleast_2d() {
        // Test with 1D array
        let array = Array::new(vec![3], DType::new(NpyType::Int)).unwrap();
        let result = array.atleast_2d().unwrap();
        assert_eq!(result.ndim(), 2);
        assert_eq!(result.shape(), &[1, 3]);
    }

    #[test]
    fn test_moveaxis() {
        let shape = vec![2, 3, 4];
        let dtype = DType::new(NpyType::Int);
        let array = Array::new(shape, dtype).unwrap();
        
        // Move axis 0 to position 2
        let result = array.moveaxis(&[0], &[2]).unwrap();
        assert_eq!(result.shape(), &[3, 4, 2]);
    }

    // NumPy-style tests for view behavior

    #[test]
    fn test_view_modifies_base() {
        // NumPy test: modifying a view should modify the base
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        // Initialize base
        unsafe {
            let ptr = base.data_ptr() as *mut i32;
            for i in 0..5 {
                *ptr.add(i) = i as i32;
            }
        }
        
        let view = Array::view_from_arc(&base, shape, vec![4]).unwrap();
        
        // Modify through view
        unsafe {
            let view_ptr = view.data_ptr() as *mut i32;
            *view_ptr.add(2) = 999;
        }
        
        // Base should see the change
        unsafe {
            let base_ptr = base.data_ptr() as *const i32;
            assert_eq!(*base_ptr.add(2), 999);
        }
    }

    #[test]
    fn test_view_nested() {
        // NumPy test: view of a view
        let shape = vec![4, 4];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Create first view
        let view1 = Array::view_from_arc(&base, vec![4, 4], vec![32, 8]).unwrap();
        
        // Create view of view - should track original base
        let base2 = Arc::new(view1);
        let view2 = Array::view_from_arc(&base2, vec![2, 2], vec![32, 8]).unwrap();
        
        assert!(view2.is_view());
        // The second view should have base2 as its base
        assert!(view2.base_array().is_some());
    }

    #[test]
    fn test_view_owns_data_flag() {
        // NumPy test: views should have OWNDATA flag false
        let shape = vec![3, 3];
        let dtype = DType::new(NpyType::Float);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view = Array::view_from_arc(&base, shape, vec![12, 4]).unwrap();
        
        // View should not own data
        assert!(!view.owns_data());
        assert!(!view.flags().contains(raptors_core::array::ArrayFlags::OWNDATA));
        
        // Base should own data
        assert!(base.owns_data());
    }

    #[test]
    fn test_view_different_shape() {
        // NumPy test: view with different shape but same data
        let shape = vec![12];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Fill with data
        unsafe {
            let ptr = base.data_ptr() as *mut i32;
            for i in 0..12 {
                *ptr.add(i) = i as i32;
            }
        }
        
        // Create view with shape [3, 4]
        let view = Array::view_from_arc(&base, vec![3, 4], vec![16, 4]).unwrap();
        
        assert_eq!(view.shape(), &[3, 4]);
        assert_eq!(view.ndim(), 2);
        
        // Data should be the same (view shares memory)
        unsafe {
            let view_ptr = view.data_ptr() as *const i32;
            assert_eq!(*view_ptr, 0);
            assert_eq!(*view_ptr.add(1), 1);
        }
    }

    #[test]
    fn test_view_transpose() {
        // NumPy test: transpose as a view
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Fill with sequential data
        unsafe {
            let ptr = base.data_ptr() as *mut f64;
            for i in 0..6 {
                *ptr.add(i) = i as f64;
            }
        }
        
        // Create transposed view: shape [2, 3] -> [3, 2] with swapped strides
        let view = Array::view_from_arc(&base, vec![3, 2], vec![8, 24]).unwrap();
        
        assert_eq!(view.shape(), &[3, 2]);
        // Strides swapped: original [24, 8] -> [8, 24]
    }

    #[test]
    fn test_view_squeeze_returns_view() {
        // NumPy test: squeeze should return a view when possible
        let shape = vec![1, 5, 1];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Create squeezed view - this is tested through the C API
        // Here we test that views can have squeezed shapes
        let squeezed_shape = vec![5];
        let view = Array::view_from_arc(&base, squeezed_shape, vec![4]).unwrap();
        
        assert_eq!(view.shape(), &[5]);
        assert!(view.is_view());
    }

    #[test]
    fn test_view_flatten_contiguous() {
        // NumPy test: flatten of contiguous array can be a view
        let shape = vec![2, 3];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // C-contiguous arrays can have flattened views
        assert!(base.is_c_contiguous());
        
        let flattened = Array::view_from_arc(&base, vec![6], vec![4]).unwrap();
        assert_eq!(flattened.shape(), &[6]);
        assert!(flattened.is_view());
    }

    #[test]
    fn test_view_with_offset() {
        // NumPy test: view starting at an offset (slice)
        let shape = vec![10];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Fill with data
        unsafe {
            let ptr = base.data_ptr() as *mut i32;
            for i in 0..10 {
                *ptr.add(i) = i as i32;
            }
        }
        
        // Create view starting at index 3 (offset by 3 * 4 bytes = 12)
        // This would be a slice [3:7] in NumPy
        let view_data = unsafe { base.data_ptr().add(12) as *mut u8 };
        let view_array = unsafe {
            Array::from_external_memory(view_data, vec![4], dtype.clone(), false).unwrap()
        };
        
        // Verify view sees correct data
        unsafe {
            let view_ptr = view_array.data_ptr() as *const i32;
            assert_eq!(*view_ptr, 3);
            assert_eq!(*view_ptr.add(1), 4);
            assert_eq!(*view_ptr.add(2), 5);
            assert_eq!(*view_ptr.add(3), 6);
        }
    }

    #[test]
    fn test_multiple_views_same_base() {
        // NumPy test: multiple views of same base
        let shape = vec![10];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        let view1 = Array::view_from_arc(&base, vec![5], vec![8]).unwrap();
        let view2 = Array::view_from_arc(&base, vec![5], vec![8]).unwrap();
        
        assert_eq!(Arc::strong_count(&base), 3); // base + view1 + view2
        
        // Modify through view1
        unsafe {
            let ptr = view1.data_ptr() as *mut f64;
            *ptr = 42.0;
        }
        
        // view2 should see the change (they share the same base data)
        unsafe {
            let ptr = view2.data_ptr() as *const f64;
            assert_eq!(*ptr, 42.0);
        }
    }

    #[test]
    fn test_view_writeable_flag() {
        // NumPy test: views inherit writeable flag from base
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let mut base = Array::new(shape.clone(), dtype.clone()).unwrap();
        
        // Make base non-writeable
        base.setflags(raptors_core::array::ArrayFlags::WRITEABLE, false);
        assert!(!base.is_writeable());
        
        let base_arc = Arc::new(base);
        let view = Array::view_from_arc(&base_arc, shape, vec![4]).unwrap();
        
        // View should also be non-writeable
        assert!(!view.is_writeable());
    }

    #[test]
    fn test_view_contiguity() {
        // NumPy test: view contiguity depends on strides
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Base is C-contiguous
        assert!(base.is_c_contiguous());
        
        // Create transposed view (non-contiguous)
        let view = Array::view_from_arc(&base, vec![4, 3], vec![8, 32]).unwrap();
        
        // View should not be C-contiguous with swapped strides
        assert!(!view.is_c_contiguous());
    }
}

