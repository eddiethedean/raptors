//! Tests for text file I/O functionality

use std::fs;
use std::path::Path;
use raptors_core::{
    array::Array,
    io::{save_text, load_text, SaveTextOptions, LoadTextOptions},
    types::{DType, NpyType},
};

#[test]
fn test_save_load_text_basic() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_array.txt");
    
    // Create a simple 2D array
    let mut array = Array::new(vec![3, 4], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        for i in 0..12 {
            *data.add(i) = i as f64;
        }
    }
    
    // Save
    let options = SaveTextOptions::default();
    save_text(&file_path, &array, options).unwrap();
    
    // Load
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[3, 4]);
    unsafe {
        let loaded_data = loaded.data_ptr() as *const f64;
        for i in 0..12 {
            assert!((*loaded_data.add(i) - i as f64).abs() < 1e-10);
        }
    }
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_csv() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_array.csv");
    
    let mut array = Array::new(vec![2, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.5;
        *data.add(1) = 2.5;
        *data.add(2) = 3.5;
        *data.add(3) = 4.5;
        *data.add(4) = 5.5;
        *data.add(5) = 6.5;
    }
    
    // Save with comma delimiter
    let options = SaveTextOptions {
        delimiter: ",".to_string(),
        ..Default::default()
    };
    save_text(&file_path, &array, options).unwrap();
    
    // Load with comma delimiter
    let load_options = LoadTextOptions {
        delimiter: Some(",".to_string()),
        ..Default::default()
    };
    let loaded = load_text(&file_path, load_options).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_with_header() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_with_header.txt");
    
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.0;
        *data.add(1) = 2.0;
        *data.add(2) = 3.0;
        *data.add(3) = 4.0;
    }
    
    // Save with header
    let options = SaveTextOptions {
        header: Some("Test Header".to_string()),
        ..Default::default()
    };
    save_text(&file_path, &array, options).unwrap();
    
    // Load with skiprows
    let load_options = LoadTextOptions {
        skiprows: 1,
        ..Default::default()
    };
    let loaded = load_text(&file_path, load_options).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_integers() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_integers.txt");
    
    let mut array = Array::new(vec![2, 3], DType::new(NpyType::Long)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut i64;
        for i in 0..6 {
            *data.add(i) = i as i64;
        }
    }
    
    // Save
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    // Load (should infer integer type)
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_1d() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_1d.txt");
    
    let mut array = Array::new(vec![5], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        for i in 0..5 {
            *data.add(i) = (i + 1) as f64;
        }
    }
    
    // Save
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    // Load
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[5, 1]); // 1D becomes 2D with 1 column
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_tab_delimited() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_tab.txt");
    
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.0;
        *data.add(1) = 2.0;
        *data.add(2) = 3.0;
        *data.add(3) = 4.0;
    }
    
    // Save with tab delimiter
    let options = SaveTextOptions {
        delimiter: "\t".to_string(),
        ..Default::default()
    };
    save_text(&file_path, &array, options).unwrap();
    
    // Load (should auto-detect tab)
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_with_comments() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_comments.txt");
    
    // Create file with comments
    let content = "# This is a comment\n1.0 2.0\n# Another comment\n3.0 4.0\n";
    fs::write(&file_path, content).unwrap();
    
    // Load (should skip comment lines)
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_skiprows() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_skiprows.txt");
    
    // Create file with header rows
    let content = "header1 header2\n1.0 2.0\n3.0 4.0\n";
    fs::write(&file_path, content).unwrap();
    
    // Load with skiprows
    let load_options = LoadTextOptions {
        skiprows: 1,
        ..Default::default()
    };
    let loaded = load_text(&file_path, load_options).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_type_inference_float() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_float_inference.txt");
    
    // Create file with float values
    let content = "1.5 2.5\n3.5 4.5\n";
    fs::write(&file_path, content).unwrap();
    
    // Load (should infer float type)
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.dtype().type_(), NpyType::Double);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_type_inference_int() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_int_inference.txt");
    
    // Create file with integer values
    let content = "1 2\n3 4\n";
    fs::write(&file_path, content).unwrap();
    
    // Load (should infer integer type)
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.dtype().type_(), NpyType::Long);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_explicit_dtype() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_explicit_dtype.txt");
    
    // Create file with integer values
    let content = "1 2\n3 4\n";
    fs::write(&file_path, content).unwrap();
    
    // Load with explicit float dtype
    let load_options = LoadTextOptions {
        dtype: Some(DType::new(NpyType::Float)),
        ..Default::default()
    };
    let loaded = load_text(&file_path, load_options).unwrap();
    
    assert_eq!(loaded.dtype().type_(), NpyType::Float);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_float32() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_float32.txt");
    
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Float)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f32;
        *data.add(0) = 1.5;
        *data.add(1) = 2.5;
        *data.add(2) = 3.5;
        *data.add(3) = 4.5;
    }
    
    // Save
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    // Load
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_save_load_text_precision() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_precision.txt");
    
    let mut array = Array::new(vec![1, 1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data = 3.14159265358979323846;
    }
    
    // Save with default format (should preserve precision)
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    // Load
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    
    unsafe {
        let loaded_data = loaded.data_ptr() as *const f64;
        // Should preserve high precision
        assert!((*loaded_data - 3.14159265358979323846).abs() < 1e-10);
    }
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_load_text_empty_file_error() {
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_empty.txt");
    
    // Create empty file
    fs::write(&file_path, "").unwrap();
    
    // Load should error
    let result = load_text(&file_path, LoadTextOptions::default());
    assert!(result.is_err());
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

// NumPy-style tests converted from numpy/tests/test_io.py patterns

#[test]
fn test_savetxt_format_string() {
    // NumPy test: custom format string
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_format.txt");
    
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.23456789;
        *data.add(1) = 2.34567890;
        *data.add(2) = 3.45678901;
        *data.add(3) = 4.56789012;
    }
    
    // Save with custom format
    let options = SaveTextOptions {
        fmt: "%.3f".to_string(),
        ..Default::default()
    };
    save_text(&file_path, &array, options).unwrap();
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_loadtxt_whitespace_delimited() {
    // NumPy test: whitespace-delimited (default)
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_whitespace.txt");
    
    // Create file with multiple spaces as delimiter
    let content = "1.0   2.0   3.0\n4.0   5.0   6.0\n";
    fs::write(&file_path, content).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    assert_eq!(loaded.shape(), &[2, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_savetxt_header_footer() {
    // NumPy test: header and footer
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_header_footer.txt");
    
    let mut array = Array::new(vec![2, 2], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.0;
        *data.add(1) = 2.0;
        *data.add(2) = 3.0;
        *data.add(3) = 4.0;
    }
    
    let options = SaveTextOptions {
        header: Some("Header line".to_string()),
        footer: Some("Footer line".to_string()),
        ..Default::default()
    };
    save_text(&file_path, &array, options).unwrap();
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_loadtxt_mixed_numeric_types() {
    // NumPy test: mixed integers and floats should infer float
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_mixed.txt");
    
    let content = "1 2.5 3\n4 5.0 6\n";
    fs::write(&file_path, content).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    // Should infer float due to mixed types
    assert_eq!(loaded.dtype().type_(), NpyType::Double);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_loadtxt_scientific_notation() {
    // NumPy test: scientific notation parsing
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_scientific.txt");
    
    let content = "1e0 2e1 3e-1\n4e2 5e-2 6e3\n";
    fs::write(&file_path, content).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    assert_eq!(loaded.shape(), &[2, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_savetxt_single_row() {
    // NumPy test: single row array
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_single_row.txt");
    
    let mut array = Array::new(vec![1, 3], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data.add(0) = 1.0;
        *data.add(1) = 2.0;
        *data.add(2) = 3.0;
    }
    
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    assert_eq!(loaded.shape(), &[1, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_loadtxt_trailing_whitespace() {
    // NumPy test: trailing whitespace should be ignored
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_trailing_ws.txt");
    
    let content = "1.0 2.0 3.0   \n4.0 5.0 6.0\t\n";
    fs::write(&file_path, content).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    assert_eq!(loaded.shape(), &[2, 3]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_loadtxt_custom_comment_char() {
    // NumPy test: custom comment character
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_custom_comment.txt");
    
    let content = "% This is a comment\n1.0 2.0\n% Another comment\n3.0 4.0\n";
    fs::write(&file_path, content).unwrap();
    
    let load_options = LoadTextOptions {
        comments: "%".to_string(),
        ..Default::default()
    };
    let loaded = load_text(&file_path, load_options).unwrap();
    assert_eq!(loaded.shape(), &[2, 2]);
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

#[test]
fn test_savetxt_large_precision() {
    // NumPy test: very large numbers
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_large.txt");
    
    let mut array = Array::new(vec![1, 1], DType::new(NpyType::Double)).unwrap();
    
    unsafe {
        let data = array.data_ptr_mut() as *mut f64;
        *data = 1e20;
    }
    
    save_text(&file_path, &array, SaveTextOptions::default()).unwrap();
    
    let loaded = load_text(&file_path, LoadTextOptions::default()).unwrap();
    unsafe {
        let loaded_data = loaded.data_ptr() as *const f64;
        assert!((*loaded_data - 1e20).abs() < 1e10); // Large numbers have less precision
    }
    
    // Cleanup
    let _ = fs::remove_file(&file_path);
}

