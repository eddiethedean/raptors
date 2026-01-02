//! Tests for memory-mapped arrays

#[cfg(test)]
mod tests {
    use raptors_core::DType;
    use raptors_core::types::NpyType;
    use raptors_core::memmap::*;
    use std::path::Path;
    use std::fs;

    #[test]
    fn test_memmap_array_creation() {
        // Create a temporary file for testing
        let temp_file = Path::new("/tmp/test_memmap_array.npy");
        
        // Create test file with some data
        let test_data = [1.0f64, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let _result = memmap_array(temp_file, dtype, vec![4]);
        
        // Cleanup
        let _ = fs::remove_file(temp_file);
        
        // Would test when properly implemented
        // assert!(result.is_ok());
        // let mmap = result.unwrap();
        // assert_eq!(mmap.array().size(), 4);
    }

    #[test]
    fn test_memmap_array_file_not_found() {
        let dtype = DType::new(NpyType::Double);
        let nonexistent = Path::new("/tmp/nonexistent_file.npy");
        let result = memmap_array(nonexistent, dtype, vec![4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_memmap_flush() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_flush.npy");
        
        // Create test file
        let test_data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mut mmap = memmap_array_writable(temp_file, dtype.clone(), vec![3]).unwrap();
        
        // Modify data
        let array = mmap.array_mut();
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 99.0;
        }
        
        // Flush changes
        mmap.flush().unwrap();
        
        // Verify file was updated
        let loaded = load_memmap(temp_file, dtype, vec![3]).unwrap();
        unsafe {
            let ptr = loaded.array().data_ptr() as *const f64;
            assert_eq!(*ptr, 99.0);
        }
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_save_memmap() {
        use std::fs;
        use raptors_core::zeros;
        use raptors_core::memmap::save_memmap;
        
        let temp_file = Path::new("/tmp/test_save_memmap.npy");
        let dtype = DType::new(NpyType::Double);
        let mut array = zeros(vec![3], dtype.clone()).unwrap();
        
        unsafe {
            let ptr = array.data_ptr_mut() as *mut f64;
            *ptr.add(0) = 10.0;
            *ptr.add(1) = 20.0;
            *ptr.add(2) = 30.0;
        }
        
        let mmap = save_memmap(&array, temp_file).unwrap();
        assert_eq!(mmap.array().size(), 3);
        
        // Verify data
        unsafe {
            let ptr = mmap.array().data_ptr() as *const f64;
            assert_eq!(*ptr.add(0), 10.0);
            assert_eq!(*ptr.add(1), 20.0);
            assert_eq!(*ptr.add(2), 30.0);
        }
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_sync() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_sync.npy");
        
        let test_data = [1.0f64, 2.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap = memmap_array(temp_file, dtype, vec![2]).unwrap();
        
        // Sync should not fail
        mmap.sync().unwrap();
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_file_too_small() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_small.npy");
        
        // Create file with insufficient data
        fs::write(temp_file, vec![0u8; 4]).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let result = memmap_array(temp_file, dtype, vec![10]); // Need 80 bytes for 10 doubles
        
        assert!(result.is_err());
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_writable_creates_file() {
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_create.npy");
        
        // File doesn't exist yet
        assert!(!temp_file.exists());
        
        let dtype = DType::new(NpyType::Double);
        let result = memmap_array_writable(temp_file, dtype, vec![3]);
        
        // Should create file
        assert!(result.is_ok());
        assert!(temp_file.exists());
        
        let mmap = result.unwrap();
        assert_eq!(mmap.array().size(), 3);
        assert_eq!(mmap.mode(), MapMode::ReadWrite);
        
        let _ = fs::remove_file(temp_file);
    }

    // NumPy-style memory-mapped array tests

    #[test]
    fn test_memmap_readonly_prevents_write() {
        // NumPy test: read-only mmap should not allow modifications
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_readonly.npy");
        
        let test_data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap = memmap_array(temp_file, dtype, vec![3]).unwrap();
        
        assert_eq!(mmap.mode(), MapMode::ReadOnly);
        
        // Attempting to modify through array should be prevented by writeable flag
        // In read-only mode, the array should not be writeable
        assert!(!mmap.array().is_writeable());
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_copy_on_write() {
        // NumPy test: copy-on-write mode
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_cow.npy");
        
        let test_data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap = MemMapArray::new(temp_file, dtype, vec![3], MapMode::CopyOnWrite).unwrap();
        
        assert_eq!(mmap.mode(), MapMode::CopyOnWrite);
        
        // Copy-on-write should allow modifications but not affect original file
        // Modifications would create a copy in memory
        assert_eq!(mmap.array().size(), 3);
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_flush_async() {
        // NumPy test: async flush for better performance
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_flush_async.npy");
        
        let test_data = [1.0f64, 2.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mut mmap = memmap_array_writable(temp_file, dtype, vec![2]).unwrap();
        
        // Modify
        unsafe {
            let ptr = mmap.array_mut().data_ptr_mut() as *mut f64;
            *ptr = 42.0;
        }
        
        // Async flush
        mmap.flush_async().unwrap();
        
        // Sync to ensure completion
        mmap.sync().unwrap();
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_multiple_views() {
        // NumPy test: multiple memory maps of same file
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_multiple.npy");
        
        let test_data = [1.0f64, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap1 = memmap_array(temp_file, dtype.clone(), vec![4]).unwrap();
        let mmap2 = memmap_array(temp_file, dtype, vec![4]).unwrap();
        
        // Both should see the same data
        unsafe {
            let ptr1 = mmap1.array().data_ptr() as *const f64;
            let ptr2 = mmap2.array().data_ptr() as *const f64;
            assert_eq!(*ptr1, *ptr2);
        }
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_different_dtypes() {
        // NumPy test: memory map with different dtypes
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_dtype.npy");
        
        // Create file with integer data
        let test_data = [1i32, 2, 3, 4, 5, 6];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<i32>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Int);
        let mmap = memmap_array(temp_file, dtype, vec![6]).unwrap();
        
        assert_eq!(mmap.array().size(), 6);
        assert_eq!(mmap.array().dtype().type_(), NpyType::Int);
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_large_array() {
        // NumPy test: memory map with large array (>1MB)
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_large.npy");
        
        // Create large file (1MB = 131072 doubles)
        let size = 131072;
        let dtype = DType::new(NpyType::Double);
        let itemsize = dtype.itemsize();
        let file_size = size * itemsize;
        
        // Create file with appropriate size
        let file = std::fs::File::create(temp_file).unwrap();
        file.set_len(file_size as u64).unwrap();
        
        // Memory map it
        let mmap = memmap_array(temp_file, dtype, vec![size as i64]).unwrap();
        
        assert_eq!(mmap.array().size(), size);
        
        // Verify we can access elements
        unsafe {
            let ptr = mmap.array().data_ptr() as *const f64;
            let _val = *ptr; // Should not crash
        }
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_2d_array() {
        // NumPy test: memory map 2D array
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_2d.npy");
        
        // Create 2D array data (2x3 = 6 doubles)
        let test_data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        let mmap = memmap_array(temp_file, dtype, vec![2, 3]).unwrap();
        
        assert_eq!(mmap.array().shape(), &[2, 3]);
        assert_eq!(mmap.array().ndim(), 2);
        
        let _ = fs::remove_file(temp_file);
    }

    #[test]
    fn test_memmap_persists_changes() {
        // NumPy test: changes persist after flush
        use std::fs;
        let temp_file = Path::new("/tmp/test_memmap_persist.npy");
        
        // Create initial file
        let test_data = [1.0f64, 2.0, 3.0];
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                test_data.as_ptr() as *const u8,
                test_data.len() * std::mem::size_of::<f64>(),
            ).to_vec()
        };
        fs::write(temp_file, bytes).unwrap();
        
        let dtype = DType::new(NpyType::Double);
        {
            let mut mmap = memmap_array_writable(temp_file, dtype.clone(), vec![3]).unwrap();
            
            // Modify
            unsafe {
                let ptr = mmap.array_mut().data_ptr_mut() as *mut f64;
                *ptr.add(1) = 99.0;
            }
            
            // Flush and drop
            mmap.flush().unwrap();
        }
        
        // Reload and verify changes persisted
        let mmap2 = memmap_array(temp_file, dtype, vec![3]).unwrap();
        unsafe {
            let ptr = mmap2.array().data_ptr() as *const f64;
            assert_eq!(*ptr.add(1), 99.0);
        }
        
        let _ = fs::remove_file(temp_file);
    }
}

