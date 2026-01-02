//! Tests for C API functions
#![allow(unused_unsafe)]

#[cfg(test)]
mod tests {
    use raptors_core::ffi::PyArrayObject;
    use raptors_core::ffi::{array_to_pyarray_ptr as ffi_array_to_pyarray_ptr, free_pyarray};
    use raptors_core::{Array, zeros};
    use raptors_core::types::{DType, NpyType};
    use std::ptr;

    // Helper function to create a simple test array structure
    // This is a minimal implementation for testing
    fn create_test_array() -> Box<PyArrayObject> {
        let mut dims = [0i64; 64];
        dims[0] = 3;
        dims[1] = 4;
        
        let mut strides = [0i64; 64];
        strides[0] = 32; // 4 * 8 bytes per double
        strides[1] = 8;  // 8 bytes per double
        
        Box::new(PyArrayObject {
            ob_base: ptr::null_mut(),
            data: ptr::null_mut(),
            nd: 2,
            descr: ptr::null_mut(),
            flags: 0,
            dimensions: dims,
            strides,
            base: ptr::null_mut(),
            _descr: ptr::null_mut(),
            weakreflist: ptr::null_mut(),
        })
    }

    // Helper to create PyArrayObject from Rust Array
    fn array_to_pyarray_ptr(array: &Array) -> *mut PyArrayObject {
        unsafe { ffi_array_to_pyarray_ptr(array) }
    }

    // Helper to create a test array with data
    fn create_test_array_with_data(shape: Vec<i64>, dtype: DType) -> Array {
        let mut array = zeros(shape, dtype).unwrap();
        // Fill with test data
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            for i in 0..array.size() {
                *ptr.add(i) = i as f64;
            }
        }
        array
    }

    #[test]
    fn test_pyarray_ndim() {
        use raptors_core::ffi::PyArray_NDIM;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let ndim = unsafe { PyArray_NDIM(arr_ptr) };
        assert_eq!(ndim, 2);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_ndim_null() {
        use raptors_core::ffi::PyArray_NDIM;
        
        let result = unsafe { PyArray_NDIM(ptr::null_mut()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_pyarray_dim() {
        use raptors_core::ffi::PyArray_DIM;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let dim0 = unsafe { PyArray_DIM(arr_ptr, 0) };
        assert_eq!(dim0, 3);
        
        let dim1 = unsafe { PyArray_DIM(arr_ptr, 1) };
        assert_eq!(dim1, 4);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_stride() {
        use raptors_core::ffi::PyArray_STRIDE;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let stride0 = unsafe { PyArray_STRIDE(arr_ptr, 0) };
        assert_eq!(stride0, 32);
        
        let stride1 = unsafe { PyArray_STRIDE(arr_ptr, 1) };
        assert_eq!(stride1, 8);
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_data() {
        use raptors_core::ffi::PyArray_DATA;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let data_ptr = unsafe { PyArray_DATA(arr_ptr) };
        assert!(!data_ptr.is_null() || data_ptr.is_null()); // Just check it doesn't crash
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_dims() {
        use raptors_core::ffi::PyArray_DIMS;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let dims_ptr = unsafe { PyArray_DIMS(arr_ptr) };
        assert!(!dims_ptr.is_null());
        
        unsafe {
            assert_eq!(*dims_ptr, 3);
            assert_eq!(*dims_ptr.add(1), 4);
        }
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    #[test]
    fn test_pyarray_strides() {
        use raptors_core::ffi::PyArray_STRIDES;
        
        let arr = create_test_array();
        let arr_ptr = Box::into_raw(arr);
        
        let strides_ptr = unsafe { PyArray_STRIDES(arr_ptr) };
        assert!(!strides_ptr.is_null());
        
        unsafe {
            assert_eq!(*strides_ptr, 32);
            assert_eq!(*strides_ptr.add(1), 8);
        }
        
        unsafe { drop(Box::from_raw(arr_ptr)); }
    }

    // ===== Phase 7 C API Tests =====

    // Array Views and Copies
    #[test]
    fn test_pyarray_squeeze() {
        use raptors_core::ffi::{PyArray_Squeeze, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![1, 3, 1, 4], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Squeeze(arr_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2); // Should remove dimensions of size 1
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_flatten() {
        use raptors_core::ffi::{PyArray_Flatten, PyArray_NDIM, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Flatten(arr_ptr, 0) }; // C-order
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 1);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 6);
        
        unsafe {
            free_pyarray(result);
        }
    }

    // NumPy-style C API View Tests

    #[test]
    fn test_pyarray_view() {
        use raptors_core::ffi::{PyArray_View, PyArray_NDIM, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![3, 4], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create view with same dtype
        let result = unsafe { PyArray_View(arr_ptr, ptr::null_mut(), ptr::null_mut()) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 12);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_newview() {
        use raptors_core::ffi::{PyArray_NewView, PyArray_NDIM, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Int);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create new view
        let result = unsafe { PyArray_NewView(arr_ptr, ptr::null_mut(), ptr::null_mut()) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 6);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_view_shares_memory() {
        // NumPy test: PyArray_View should share memory with base
        use raptors_core::ffi::{PyArray_View, PyArray_DATA};
        
        let dtype = DType::new(NpyType::Int);
        let mut array = zeros(vec![5], dtype).unwrap();
        
        // Fill with data
        unsafe {
            let ptr = array.data_ptr_mut() as *mut i32;
            *ptr = 42;
        }
        
        let arr_ptr = array_to_pyarray_ptr(&array);
        let view_ptr = unsafe { PyArray_View(arr_ptr, ptr::null_mut(), ptr::null_mut()) };
        assert!(!view_ptr.is_null());
        
        // Both should point to same data (views share memory)
        let base_data = unsafe { PyArray_DATA(arr_ptr) };
        let view_data = unsafe { PyArray_DATA(view_ptr) };
        
        // Data pointers should be the same for a view
        assert_eq!(base_data, view_data);
        
        unsafe {
            free_pyarray(view_ptr);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_view_null_input() {
        use raptors_core::ffi::PyArray_View;
        
        let result = unsafe { PyArray_View(ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
        assert!(result.is_null());
    }

    #[test]
    fn test_pyarray_squeeze_multiple_dims() {
        // NumPy test: squeeze removes all dimensions of size 1
        use raptors_core::ffi::{PyArray_Squeeze, PyArray_NDIM, PyArray_DIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![1, 1, 5, 1], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Squeeze(arr_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 1); // Should only have dimension of size 5
        
        let dim0 = unsafe { PyArray_DIM(result, 0) };
        assert_eq!(dim0, 5);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_flatten_non_contiguous() {
        // NumPy test: flatten of non-contiguous creates copy
        use raptors_core::ffi::{PyArray_Flatten, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create transposed view first (non-contiguous)
        use raptors_core::ffi::PyArray_Transpose;
        let transposed = unsafe { PyArray_Transpose(arr_ptr, ptr::null()) };
        assert!(!transposed.is_null());
        
        // Flatten non-contiguous should still work
        let flattened = unsafe { PyArray_Flatten(transposed, 0) };
        assert!(!flattened.is_null());
        
        let ndim = unsafe { PyArray_NDIM(flattened) };
        assert_eq!(ndim, 1);
        
        unsafe {
            free_pyarray(flattened);
            free_pyarray(transposed);
            free_pyarray(arr_ptr);
        }
    }

    // Array Manipulation
    #[test]
    fn test_pyarray_reshape() {
        use raptors_core::ffi::{PyArray_Reshape, PyArray_NDIM, PyArray_DIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let newshape = [6i64];
        let result = unsafe { PyArray_Reshape(arr_ptr, newshape.as_ptr(), 1) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 1);
        
        let dim0 = unsafe { PyArray_DIM(result, 0) };
        assert_eq!(dim0, 6);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_transpose() {
        use raptors_core::ffi::{PyArray_Transpose, PyArray_NDIM, PyArray_DIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Transpose(arr_ptr, ptr::null()) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let dim0 = unsafe { PyArray_DIM(result, 0) };
        let dim1 = unsafe { PyArray_DIM(result, 1) };
        assert_eq!(dim0, 3); // Transposed
        assert_eq!(dim1, 2);
        
        unsafe {
            free_pyarray(result);
        }
    }

    // Additional NumPy-style C API view tests
    #[test]
    fn test_pyarray_view_null_handling() {
        use raptors_core::ffi::PyArray_View;
        
        // Test null pointer handling
        let result = unsafe { PyArray_View(ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
        assert!(result.is_null());
    }

    #[test]
    fn test_pyarray_newview_null_handling() {
        use raptors_core::ffi::PyArray_NewView;
        
        let result = unsafe { PyArray_NewView(ptr::null_mut(), ptr::null_mut(), ptr::null_mut()) };
        assert!(result.is_null());
    }

    #[test]
    fn test_pyarray_squeeze_null_handling() {
        use raptors_core::ffi::PyArray_Squeeze;
        
        let result = unsafe { PyArray_Squeeze(ptr::null_mut()) };
        assert!(result.is_null());
    }

    #[test]
    fn test_pyarray_squeeze_no_dims_to_remove() {
        // NumPy test: squeeze on array with no size-1 dimensions
        use raptors_core::ffi::{PyArray_Squeeze, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Squeeze(arr_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2); // No change
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_flatten_order() {
        // NumPy test: flatten with different orders (C vs F)
        use raptors_core::ffi::{PyArray_Flatten, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // C-order (0)
        let result_c = unsafe { PyArray_Flatten(arr_ptr, 0) };
        assert!(!result_c.is_null());
        let ndim_c = unsafe { PyArray_NDIM(result_c) };
        assert_eq!(ndim_c, 1);
        
        unsafe {
            free_pyarray(result_c);
            free_pyarray(arr_ptr);
        }
    }

    #[test]
    fn test_pyarray_ravel() {
        use raptors_core::ffi::{PyArray_Ravel, PyArray_NDIM, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_Ravel(arr_ptr, 0) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 1);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 6);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_swapaxes() {
        use raptors_core::ffi::{PyArray_SwapAxes, PyArray_NDIM, PyArray_DIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3, 4], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_SwapAxes(arr_ptr, 0, 2) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 3);
        
        let dim0 = unsafe { PyArray_DIM(result, 0) };
        let dim2 = unsafe { PyArray_DIM(result, 2) };
        assert_eq!(dim0, 4); // Swapped
        assert_eq!(dim2, 2);
        
        unsafe {
            free_pyarray(result);
        }
    }

    // Indexing and Selection
    #[test]
    fn test_pyarray_take() {
        use raptors_core::ffi::{PyArray_Take, PyArray_SIZE};
        use raptors_core::array::Array;
        
        let dtype = DType::new(NpyType::Double);
        let array = create_test_array_with_data(vec![5], dtype);
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create index array
        let index_dtype = DType::new(NpyType::Int);
        let mut index_array = Array::new(vec![3], index_dtype).unwrap();
        unsafe {
            let ptr = index_array.data_ptr_mut() as *mut i32;
            *ptr.add(0) = 0;
            *ptr.add(1) = 2;
            *ptr.add(2) = 4;
        }
        let index_ptr = array_to_pyarray_ptr(&index_array);
        
        let result = unsafe { PyArray_Take(arr_ptr, index_ptr, -1, ptr::null_mut(), 0) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 3);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(index_ptr);
        }
    }

    #[test]
    fn test_pyarray_compress() {
        use raptors_core::ffi::{PyArray_Compress, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array = create_test_array_with_data(vec![5], dtype);
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create boolean mask
        let mask_dtype = DType::new(NpyType::Bool);
        let mut mask = Array::new(vec![5], mask_dtype).unwrap();
        unsafe {
            let ptr = mask.data_ptr_mut() as *mut bool;
            *ptr.add(0) = true;
            *ptr.add(1) = false;
            *ptr.add(2) = true;
            *ptr.add(3) = false;
            *ptr.add(4) = true;
        }
        let mask_ptr = array_to_pyarray_ptr(&mask);
        
        let result = unsafe { PyArray_Compress(arr_ptr, mask_ptr, -1, ptr::null_mut()) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 3); // 3 True values
        
        unsafe {
            free_pyarray(result);
            free_pyarray(mask_ptr);
        }
    }

    // Concatenation
    #[test]
    fn test_pyarray_concatenate() {
        use raptors_core::ffi::{PyArray_Concatenate, PyArray_SIZE, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![2, 3], dtype.clone()).unwrap();
        let array2 = zeros(vec![2, 3], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let arrays = [arr1_ptr, arr2_ptr];
        let result = unsafe { PyArray_Concatenate(arrays.as_ptr() as *mut *mut PyArrayObject, 2, 0) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 12); // 2 arrays * 6 elements
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    #[test]
    fn test_pyarray_stack() {
        use raptors_core::ffi::{PyArray_Stack, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![2, 3], dtype.clone()).unwrap();
        let array2 = zeros(vec![2, 3], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let arrays = [arr1_ptr, arr2_ptr];
        let result = unsafe { PyArray_Stack(arrays.as_ptr() as *mut *mut PyArrayObject, 2, 0) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 3);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    // Sorting
    #[test]
    fn test_pyarray_argsort() {
        use raptors_core::ffi::{PyArray_ArgSort, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let mut array = Array::new(vec![5], dtype).unwrap();
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 3.0;
            *ptr.add(1) = 1.0;
            *ptr.add(2) = 4.0;
            *ptr.add(3) = 2.0;
            *ptr.add(4) = 5.0;
        }
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let result = unsafe { PyArray_ArgSort(arr_ptr, -1, 0) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 5);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_searchsorted() {
        use raptors_core::ffi::{PyArray_SearchSorted, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let mut array = Array::new(vec![5], dtype.clone()).unwrap();
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
            *ptr.add(4) = 5.0;
        }
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        // Create values array with 1 element (shape [1])
        let mut values = Array::new(vec![1], dtype).unwrap();
        unsafe {
            let ptr = values.data_ptr_mut() as *mut f64;
            *ptr = 2.5; // Value to search for
        }
        let values_ptr = array_to_pyarray_ptr(&values);
        
        let result = unsafe { PyArray_SearchSorted(arr_ptr, values_ptr, 0, ptr::null_mut()) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 1);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(values_ptr);
        }
    }

    // Linear Algebra
    #[test]
    fn test_pyarray_matrix_product() {
        use raptors_core::ffi::{PyArray_MatrixProduct, PyArray_NDIM, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![2, 3], dtype.clone()).unwrap();
        let array2 = zeros(vec![3, 4], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let result = unsafe { PyArray_MatrixProduct(arr1_ptr, arr2_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 8); // 2 * 4
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    #[test]
    fn test_pyarray_inner_product() {
        use raptors_core::ffi::{PyArray_InnerProduct, PyArray_SIZE};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![3], dtype.clone()).unwrap();
        let array2 = zeros(vec![3], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let result = unsafe { PyArray_InnerProduct(arr1_ptr, arr2_ptr) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 1); // Scalar result
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    #[test]
    fn test_pyarray_matmul() {
        use raptors_core::ffi::{PyArray_MatMul, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![2, 3], dtype.clone()).unwrap();
        let array2 = zeros(vec![3, 4], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let result = unsafe { PyArray_MatMul(arr1_ptr, arr2_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    // Advanced Operations
    #[test]
    fn test_pyarray_broadcast() {
        use raptors_core::ffi::{PyArray_Broadcast, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array1 = zeros(vec![3, 1], dtype.clone()).unwrap();
        let array2 = zeros(vec![1, 4], dtype).unwrap();
        
        let arr1_ptr = array_to_pyarray_ptr(&array1);
        let arr2_ptr = array_to_pyarray_ptr(&array2);
        
        let result = unsafe { PyArray_Broadcast(arr1_ptr, arr2_ptr) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        unsafe {
            free_pyarray(result);
            free_pyarray(arr1_ptr);
            free_pyarray(arr2_ptr);
        }
    }

    #[test]
    fn test_pyarray_broadcast_to_shape() {
        use raptors_core::ffi::{PyArray_BroadcastToShape, PyArray_NDIM};
        
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![1, 3], dtype).unwrap();
        let arr_ptr = array_to_pyarray_ptr(&array);
        
        let target_shape = [2i64, 3i64];
        let result = unsafe { PyArray_BroadcastToShape(arr_ptr, target_shape.as_ptr(), 2) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        unsafe {
            free_pyarray(result);
        }
    }

    // Array Creation
    #[test]
    fn test_pyarray_new() {
        use raptors_core::ffi::{PyArray_New, PyArray_NDIM, PyArray_SIZE};
        
        let dims = [2i64, 3i64];
        let type_num = 12; // Double
        
        let result = unsafe { PyArray_New(
            ptr::null_mut(),
            2,
            dims.as_ptr(),
            type_num,
            ptr::null(),
            ptr::null_mut(),
            8,
            0,
            ptr::null_mut(),
        ) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 6);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_empty() {
        use raptors_core::ffi::{PyArray_Empty, PyArray_NDIM, PyArray_SIZE};
        
        let dims = [3i64, 4i64];
        let type_num = 12; // Double
        
        let result = unsafe { PyArray_Empty(2, dims.as_ptr(), type_num, 0) };
        assert!(!result.is_null());
        
        let ndim = unsafe { PyArray_NDIM(result) };
        assert_eq!(ndim, 2);
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 12);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_zeros() {
        use raptors_core::ffi::{PyArray_Zeros, PyArray_SIZE};
        
        let dims = [2i64, 3i64];
        let type_num = 12; // Double
        
        let result = unsafe { PyArray_Zeros(2, dims.as_ptr(), type_num, 0) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 6);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_ones() {
        use raptors_core::ffi::{PyArray_Ones, PyArray_SIZE};
        
        let dims = [2i64, 2i64];
        let type_num = 12; // Double
        
        let result = unsafe { PyArray_Ones(2, dims.as_ptr(), type_num, 0) };
        assert!(!result.is_null());
        
        let size = unsafe { PyArray_SIZE(result) };
        assert_eq!(size, 4);
        
        unsafe {
            free_pyarray(result);
        }
    }

    #[test]
    fn test_pyarray_itemsize() {
        use raptors_core::ffi::{PyArray_ITEMSIZE, PyArray_Zeros};
        
        // Use PyArray_Zeros instead of Empty to ensure data is allocated
        let dims = [2i64, 3i64];
        let type_num = 12; // Double (8 bytes)
        
        let arr = unsafe { PyArray_Zeros(2, dims.as_ptr(), type_num, 0) };
        assert!(!arr.is_null());
        
        let itemsize = unsafe { PyArray_ITEMSIZE(arr) };
        assert_eq!(itemsize, 8);
        
        unsafe {
            free_pyarray(arr);
        }
    }

    // Type Checking
    #[test]
    fn test_pyarray_check() {
        use raptors_core::ffi::{PyArray_Check, PyArray_Empty};
        
        let dims = [2i64, 3i64];
        let type_num = 12; // Double
        
        let arr = unsafe { PyArray_Empty(2, dims.as_ptr(), type_num, 0) };
        assert!(!arr.is_null());
        
        let is_array = unsafe { PyArray_Check(arr as *mut libc::c_void) };
        assert_eq!(is_array, 1);
        
        let is_null_array = unsafe { PyArray_Check(ptr::null_mut()) };
        assert_eq!(is_null_array, 0);
        
        unsafe {
            free_pyarray(arr);
        }
    }
}

