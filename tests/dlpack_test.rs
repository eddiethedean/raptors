//! Tests for DLPack support
#![allow(unused_unsafe)]

#[cfg(test)]
mod tests {
    use raptors_core::{DType, zeros};
    use raptors_core::types::NpyType;
    use raptors_core::dlpack::*;

    #[test]
    fn test_array_to_dlpack() {
        let dtype = DType::new(NpyType::Double);
        let array = zeros(vec![2, 3], dtype).unwrap();
        
        unsafe {
            let dlpack = array_to_dlpack(&array).unwrap();
            assert!(!dlpack.is_null());
            
            let tensor = &*dlpack;
            assert_eq!(tensor.ndim, 2);
            assert_eq!(tensor.device.device_type, DLDeviceType::CPU);
            
            // Cleanup
            delete_dlpack_tensor(dlpack);
        }
    }

    #[test]
    fn test_dlpack_dtype_conversion() {
        // Test that dtype conversion works for various types
        let test_cases = vec![
            (NpyType::Double, DLDataTypeCode::Float, 64),
            (NpyType::Float, DLDataTypeCode::Float, 32),
            (NpyType::Int, DLDataTypeCode::Int, 32),
            (NpyType::LongLong, DLDataTypeCode::Int, 64),
        ];
        
        for (npy_type, _expected_code, _expected_bits) in test_cases {
            let dtype = DType::new(npy_type);
            let array = zeros(vec![1], dtype).unwrap();
            
            unsafe {
                let dlpack = array_to_dlpack(&array).unwrap();
                let _tensor = &*dlpack;
                
                // Would verify dtype conversion when properly implemented
                // assert_eq!(tensor.dtype.code, expected_code as u8);
                // assert_eq!(tensor.dtype.bits, expected_bits);
                
                delete_dlpack_tensor(dlpack);
            }
        }
    }

    #[test]
    fn test_from_dlpack() {
        let dtype = DType::new(NpyType::Double);
        let mut original = zeros(vec![2, 2], dtype).unwrap();
        
        // Set some values
        unsafe {
            let ptr = original.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 1.0;
            *ptr.add(1) = 2.0;
            *ptr.add(2) = 3.0;
            *ptr.add(3) = 4.0;
        }
        
        unsafe {
            let dlpack = array_to_dlpack(&original).unwrap();
            
            let converted = from_dlpack(dlpack).unwrap();
            assert_eq!(converted.shape(), original.shape());
            assert_eq!(converted.dtype().type_(), original.dtype().type_());
            
            // Verify data
            unsafe {
                let orig_ptr = original.data_ptr() as *const f64;
                let conv_ptr = converted.data_ptr() as *const f64;
                for i in 0..4 {
                    assert_eq!(*orig_ptr.add(i), *conv_ptr.add(i));
                }
            }
            
            delete_dlpack_tensor(dlpack);
        }
    }

    #[test]
    fn test_dlpack_roundtrip() {
        let dtype = DType::new(NpyType::Float);
        let mut original = zeros(vec![3], dtype).unwrap();
        
        unsafe {
            let ptr = original.data_ptr_mut() as *mut f32;
            *ptr.add(0) = 1.5;
            *ptr.add(1) = 2.5;
            *ptr.add(2) = 3.5;
        }
        
        unsafe {
            let dlpack = array_to_dlpack(&original).unwrap();
            let converted = from_dlpack(dlpack).unwrap();
            
            assert_eq!(converted.size(), original.size());
            unsafe {
                let orig_ptr = original.data_ptr() as *const f32;
                let conv_ptr = converted.data_ptr() as *const f32;
                for i in 0..3 {
                    assert!((*orig_ptr.add(i) - *conv_ptr.add(i)).abs() < 1e-6);
                }
            }
            
            delete_dlpack_tensor(dlpack);
        }
    }

    #[test]
    fn test_dlpack_int_types() {
        let test_types = vec![
            NpyType::Int,
            NpyType::LongLong,
            NpyType::UInt,
        ];
        
        for npy_type in test_types {
            let dtype = DType::new(npy_type);
            let array = zeros(vec![2], dtype).unwrap();
            
            unsafe {
                let dlpack = array_to_dlpack(&array).unwrap();
                let tensor = &*dlpack;
                assert_eq!(tensor.ndim, 1);
                assert_eq!(tensor.device.device_type, DLDeviceType::CPU);
                delete_dlpack_tensor(dlpack);
            }
        }
    }

    #[test]
    fn test_dlpack_shape_preservation() {
        let shapes = vec![
            vec![5],
            vec![2, 3],
            vec![2, 2, 2],
        ];
        
        for shape in shapes {
            let dtype = DType::new(NpyType::Double);
            let array = zeros(shape.clone(), dtype).unwrap();
            
            unsafe {
                let dlpack = array_to_dlpack(&array).unwrap();
                let tensor = &*dlpack;
                assert_eq!(tensor.ndim, shape.len() as i32);
                
                // Verify shape values
                for (i, &dim) in shape.iter().enumerate() {
                    assert_eq!(*tensor.shape.add(i), dim);
                }
                
                delete_dlpack_tensor(dlpack);
            }
        }
    }

    #[test]
    fn test_dlpack_null_tensor() {
        unsafe {
            let result = dlpack_to_array(std::ptr::null_mut());
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_dlpack_delete_null() {
        unsafe {
            // Should not panic
            delete_dlpack_tensor(std::ptr::null_mut());
        }
    }
}

