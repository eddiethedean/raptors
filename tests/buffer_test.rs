//! Tests for buffer protocol functionality

use raptors_core::{
    array::Array,
    buffer::{export_buffer, import_buffer, BufferError, FormatString},
    types::{DType, NpyType},
};

#[test]
fn test_export_buffer_basic() {
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    let buffer_info = export_buffer(&array).unwrap();
    
    assert_eq!(buffer_info.shape, vec![3, 4]);
    assert!(!buffer_info.read_only); // New arrays are writeable
    assert_eq!(buffer_info.format, "d"); // Double format
}

#[test]
fn test_export_buffer_read_only() {
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    // Make array read-only
    array.setflags(raptors_core::array::ArrayFlags::WRITEABLE, false);
    
    let buffer_info = export_buffer(&array).unwrap();
    
    assert!(buffer_info.read_only);
}

#[test]
fn test_export_buffer_int() {
    let array = Array::new(vec![2, 3], DType::new(NpyType::Int)).unwrap();
    
    let buffer_info = export_buffer(&array).unwrap();
    
    assert_eq!(buffer_info.format, "i"); // Int format
}

#[test]
fn test_export_buffer_float() {
    let array = Array::new(vec![2, 2], DType::new(NpyType::Float)).unwrap();
    
    let buffer_info = export_buffer(&array).unwrap();
    
    assert_eq!(buffer_info.format, "f"); // Float format
}

#[test]
fn test_export_buffer_array_method() {
    let array = Array::new(vec![3, 3], DType::new(NpyType::Double)).unwrap();
    
    let buffer_info = array.to_buffer().unwrap();
    
    assert_eq!(buffer_info.shape, vec![3, 3]);
}

#[test]
fn test_import_buffer_basic() {
    // Create a buffer manually
    let mut data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let ptr = data.as_mut_ptr() as *mut u8;
    
    unsafe {
        let array = import_buffer(
            ptr,
            "d",
            vec![2, 2],
            None,
            false,
        ).unwrap();
        
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.dtype().type_(), NpyType::Double);
        
        // Verify data is accessible
        let array_data = array.data_ptr() as *const f64;
        assert!((*array_data.add(0) - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_import_buffer_array_method() {
    let mut data: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
    let ptr = data.as_mut_ptr() as *mut u8;
    
    unsafe {
        let array = Array::from_buffer(
            ptr,
            "i",
            vec![2, 3],
            None,
            false,
        ).unwrap();
        
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array.dtype().type_(), NpyType::Int);
    }
}

#[test]
fn test_format_string_parse() {
    let format = FormatString::parse("d").unwrap();
    assert_eq!(format.type_char, 'd');
    assert_eq!(format.endian, None);
    
    let format = FormatString::parse("<i").unwrap();
    assert_eq!(format.type_char, 'i');
    assert_eq!(format.endian, Some('<'));
    
    let format = FormatString::parse(">f").unwrap();
    assert_eq!(format.type_char, 'f');
    assert_eq!(format.endian, Some('>'));
}

#[test]
fn test_format_string_invalid() {
    let result = FormatString::parse("x");
    assert!(result.is_err());
    
    let result = FormatString::parse("");
    assert!(result.is_err());
}

#[test]
fn test_export_import_roundtrip() {
    let mut original = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    // Fill with test data
    unsafe {
        let data = original.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.5;
        *data.add(1) = 2.5;
        *data.add(2) = 3.5;
        *data.add(3) = 4.5;
    }
    
    // Export
    let buffer_info = export_buffer(&original).unwrap();
    
    // Import (using the same memory - this is just a test of the API)
    // In real usage, you'd import from external memory
    unsafe {
        let imported = import_buffer(
            buffer_info.ptr,
            &buffer_info.format,
            buffer_info.shape.clone(),
            Some(buffer_info.strides.clone()),
            buffer_info.read_only,
        ).unwrap();
        
        // Should have same shape
        assert_eq!(imported.shape(), &[2, 2]);
        
        // Data should be the same (same memory location)
        let imported_data = imported.data_ptr() as *const f64;
        assert!((*imported_data.add(0) - 1.5).abs() < 1e-10);
    }
}

#[test]
fn test_buffer_unsupported_type() {
    // String type not supported in buffer protocol
    let array = Array::new(vec![2], DType::new(NpyType::String)).unwrap();
    
    let result = export_buffer(&array);
    assert!(result.is_err());
}

// NumPy/Python-style buffer protocol tests

#[test]
fn test_buffer_format_endianness() {
    // Test endianness indicators in format strings
    let format = FormatString::parse("=d").unwrap(); // Native endian
    assert_eq!(format.type_char, 'd');
    assert_eq!(format.endian, Some('='));
    
    let format = FormatString::parse("!d").unwrap(); // Network (big) endian
    assert_eq!(format.endian, Some('!'));
}

#[test]
fn test_buffer_all_numeric_types() {
    // Test all supported numeric types for buffer protocol
    let types = vec![
        (NpyType::Byte, "b"),
        (NpyType::UByte, "B"),
        (NpyType::Short, "h"),
        (NpyType::UShort, "H"),
        (NpyType::Int, "i"),
        (NpyType::UInt, "I"),
        (NpyType::Long, "l"),
        (NpyType::ULong, "L"),
        (NpyType::Float, "f"),
        (NpyType::Double, "d"),
    ];
    
    for (npy_type, expected_format) in types {
        let array = Array::new(vec![2, 2], DType::new(npy_type)).unwrap();
        let buffer_info = export_buffer(&array).unwrap();
        assert_eq!(buffer_info.format, expected_format);
    }
}

#[test]
fn test_buffer_shape_preservation() {
    // Test that buffer preserves shape information
    let array = Array::new(vec![2, 3, 4], DType::new(NpyType::Double)).unwrap();
    let buffer_info = export_buffer(&array).unwrap();
    
    assert_eq!(buffer_info.shape, vec![2, 3, 4]);
    assert_eq!(buffer_info.strides.len(), 3);
}

#[test]
fn test_buffer_size_calculation() {
    // Test buffer size calculation
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let buffer_info = export_buffer(&array).unwrap();
    
    // Size should be 3 * 4 * 8 bytes = 96 bytes
    assert_eq!(buffer_info.size, 96);
}

#[test]
fn test_buffer_readonly_flag_preserved() {
    // Test that read-only flag is correctly set and preserved
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    // Initially writeable
    let buffer_info1 = export_buffer(&array).unwrap();
    assert!(!buffer_info1.read_only);
    
    // Make read-only
    array.setflags(raptors_core::array::ArrayFlags::WRITEABLE, false);
    let buffer_info2 = export_buffer(&array).unwrap();
    assert!(buffer_info2.read_only);
}

#[test]
fn test_buffer_format_roundtrip() {
    // Test format string generation and parsing
    let array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    let buffer_info = export_buffer(&array).unwrap();
    
    // Parse the format we generated
    let parsed = FormatString::parse(&buffer_info.format).unwrap();
    assert_eq!(parsed.type_char, 'd');
}

#[test]
fn test_buffer_contiguous_arrays() {
    // Test buffer export for C-contiguous arrays
    let array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    let buffer_info = export_buffer(&array).unwrap();
    
    // C-contiguous arrays should have standard strides
    assert_eq!(buffer_info.strides.len(), 2);
    // Stride for last dimension should be itemsize (8 for double)
    assert_eq!(buffer_info.strides[1], 8);
}

#[test]
fn test_buffer_multiple_export_same_array() {
    // Test exporting same array multiple times
    let array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    let buffer_info1 = export_buffer(&array).unwrap();
    let buffer_info2 = export_buffer(&array).unwrap();
    
    // Should get same pointer (same array)
    assert_eq!(buffer_info1.ptr, buffer_info2.ptr);
    assert_eq!(buffer_info1.size, buffer_info2.size);
}

