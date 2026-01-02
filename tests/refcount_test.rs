//! Tests for reference counting
#![allow(clippy::arc_with_non_send_sync)]

#[cfg(test)]
mod tests {
    use raptors_core::array::Array;
    use raptors_core::types::{DType, NpyType};
    use std::sync::Arc;

    #[test]
    fn test_reference_counting_basic() {
        let shape = vec![3, 4];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        assert_eq!(Arc::strong_count(&base), 1);
        
        let view1 = Array::view_from_arc(&base, shape.clone(), vec![32, 8]).unwrap();
        assert_eq!(view1.base_reference_count(), Some(2));
        
        let view2 = Array::view_from_arc(&base, shape, vec![32, 8]).unwrap();
        assert_eq!(view2.base_reference_count(), Some(3));
        
        drop(view1);
        assert_eq!(Arc::strong_count(&base), 2); // base + view2
        
        drop(view2);
        assert_eq!(Arc::strong_count(&base), 1); // just base
    }

    #[test]
    fn test_weak_references() {
        let shape = vec![2, 2];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view = Array::view_from_arc(&base, shape, vec![8, 4]).unwrap();
        let weak = view.base_array_weak();
        
        // Weak should be None since we're using strong references
        assert!(weak.is_none());
        
        // Create weak reference directly
        let weak_base = Arc::downgrade(&base);
        assert!(weak_base.upgrade().is_some());
        
        drop(view);
        drop(base);
        
        // Now weak reference should fail
        assert!(weak_base.upgrade().is_none());
    }

    #[test]
    fn test_is_base_alive() {
        let shape = vec![3];
        let dtype = DType::new(NpyType::Float);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view = Array::view_from_arc(&base, shape, vec![4]).unwrap();
        assert!(view.is_base_alive());
        
        drop(base);
        // View still holds reference, so base should still be alive
        assert!(view.is_base_alive());
        
        drop(view);
        // Now base should be dropped
    }

    #[test]
    fn test_memory_safety() {
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        // Fill with data
        unsafe {
            let data_ptr = base.data_ptr() as *mut i32;
            for i in 0..5 {
                *data_ptr.add(i) = i as i32;
            }
        }
        
        let view = Array::view_from_arc(&base, shape, vec![4]).unwrap();
        
        // Modify through view
        unsafe {
            let view_ptr = view.data_ptr() as *mut i32;
            *view_ptr = 99;
        }
        
        // Verify base sees the change
        unsafe {
            let base_ptr = base.data_ptr() as *const i32;
            assert_eq!(*base_ptr, 99);
        }
        
        // Base should keep view alive
        drop(view);
        assert_eq!(Arc::strong_count(&base), 1);
        
        // Base data should still be valid
        unsafe {
            let base_ptr = base.data_ptr() as *const i32;
            assert_eq!(*base_ptr, 99);
        }
    }

    #[test]
    fn test_circular_reference_prevention() {
        let shape = vec![2];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view1 = Array::view_from_arc(&base, shape.clone(), vec![4]).unwrap();
        let view2 = Array::view_from_arc(&base, shape, vec![4]).unwrap();
        
        // Views reference base, but base doesn't reference views
        // This prevents circular references
        assert_eq!(Arc::strong_count(&base), 3); // base + view1 + view2
        
        drop(view1);
        drop(view2);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_reference_count_monitoring() {
        let shape = vec![3, 3];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view = Array::view_from_arc(&base, shape, vec![24, 8]).unwrap();
        
        assert_eq!(view.base_reference_count(), Some(2));
        assert_eq!(view.base_weak_count(), Some(0));
        
        let _weak = Arc::downgrade(&base);
        assert_eq!(Arc::weak_count(&base), 1);
    }

    // NumPy-style reference counting tests

    #[test]
    fn test_base_kept_alive_by_view() {
        // NumPy test: base array is kept alive while views exist
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        let view = Array::view_from_arc(&base, shape, vec![4]).unwrap();
        
        // Drop base Arc (view still holds reference)
        let base_ref = Arc::try_unwrap(base);
        assert!(base_ref.is_err()); // Should fail because view still references it
        
        drop(view);
        // Now base can be dropped
    }

    #[test]
    fn test_view_chain_reference_counting() {
        // NumPy test: view of view maintains proper reference counting
        let shape = vec![10];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Create view1
        let view1 = Array::view_from_arc(&base, vec![8], vec![8]).unwrap();
        assert_eq!(Arc::strong_count(&base), 2);
        
        // Create view2 of view1
        let base2 = Arc::new(view1);
        let view2 = Array::view_from_arc(&base2, vec![4], vec![8]).unwrap();
        
        // base should have 2 refs (original + view1)
        // base2 should have 2 refs (view1 + view2)
        assert_eq!(Arc::strong_count(&base), 2);
        assert_eq!(Arc::strong_count(&base2), 2);
        
        drop(view2);
        assert_eq!(Arc::strong_count(&base2), 1);
        
        drop(base2);
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_multiple_views_same_base_refcount() {
        // NumPy test: multiple views increment refcount correctly
        let shape = vec![20];
        let dtype = DType::new(NpyType::Float);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        let mut views = Vec::new();
        for _ in 0..5 {
            let view = Array::view_from_arc(&base, vec![10], vec![4]).unwrap();
            views.push(view);
        }
        
        // Base should have 6 references (1 original + 5 views)
        assert_eq!(Arc::strong_count(&base), 6);
        
        // Drop views one by one
        for _ in 0..5 {
            views.pop();
            assert_eq!(Arc::strong_count(&base), 1 + views.len());
        }
        
        assert_eq!(Arc::strong_count(&base), 1);
    }

    #[test]
    fn test_view_drop_before_base() {
        // NumPy test: views can be dropped before base
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape.clone(), dtype.clone()).unwrap());
        
        {
            let _view = Array::view_from_arc(&base, shape, vec![4]).unwrap();
            assert_eq!(Arc::strong_count(&base), 2);
            // View dropped here
        }
        
        // After view is dropped, base should have only 1 reference
        assert_eq!(Arc::strong_count(&base), 1);
        
        // Base should still be valid
        assert_eq!(base.size(), 5);
    }

    #[test]
    fn test_weak_reference_tracking() {
        // NumPy test: weak references don't keep base alive
        let shape = vec![3];
        let dtype = DType::new(NpyType::Double);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        let weak1 = Arc::downgrade(&base);
        let weak2 = Arc::downgrade(&base);
        
        assert_eq!(Arc::weak_count(&base), 2);
        assert!(weak1.upgrade().is_some());
        
        drop(base);
        
        // Weak references should fail after base is dropped
        assert!(weak1.upgrade().is_none());
        assert!(weak2.upgrade().is_none());
    }

    #[test]
    fn test_view_memory_safety_base_dropped() {
        // NumPy test: views prevent use-after-free
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
        
        let view = Array::view_from_arc(&base, vec![5], vec![4]).unwrap();
        
        // Even if we drop the original Arc reference, view keeps base alive
        let base_data = base.data_ptr();
        drop(base); // Drop our reference
        
        // View still holds reference, so data should be valid
        assert_eq!(view.data_ptr(), base_data);
        
        // Access through view should still work
        unsafe {
            let view_ptr = view.data_ptr() as *const i32;
            assert_eq!(*view_ptr, 0);
        }
        
        // When view is dropped, base will be dropped too
        drop(view);
    }

    #[test]
    fn test_reference_count_with_views_and_weak() {
        // NumPy test: mixed strong and weak references
        let shape = vec![7];
        let dtype = DType::new(NpyType::Float);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        let view1 = Array::view_from_arc(&base, vec![5], vec![4]).unwrap();
        let view2 = Array::view_from_arc(&base, vec![3], vec![4]).unwrap();
        let weak = Arc::downgrade(&base);
        
        assert_eq!(Arc::strong_count(&base), 3); // base + view1 + view2
        assert_eq!(Arc::weak_count(&base), 1);
        
        drop(view1);
        assert_eq!(Arc::strong_count(&base), 2);
        assert_eq!(Arc::weak_count(&base), 1);
        
        drop(view2);
        assert_eq!(Arc::strong_count(&base), 1);
        
        // Weak reference should still work
        assert!(weak.upgrade().is_some());
        
        drop(base);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_is_base_alive_with_weak() {
        // NumPy test: is_base_alive with weak references
        let shape = vec![5];
        let dtype = DType::new(NpyType::Int);
        let base = Arc::new(Array::new(shape, dtype.clone()).unwrap());
        
        // Create view with weak reference scenario
        let view = Array::view_from_arc(&base, vec![3], vec![4]).unwrap();
        assert!(view.is_base_alive());
        
        // Create weak reference
        let weak = Arc::downgrade(&base);
        drop(base);
        
        // View still holds strong reference, so base should be alive
        assert!(view.is_base_alive());
        assert!(weak.upgrade().is_some());
        
        drop(view);
        
        // Now base should be dropped
        assert!(weak.upgrade().is_none());
    }
}

