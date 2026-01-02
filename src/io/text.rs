//! Text file I/O implementation
//!
//! This module provides text-based file I/O functionality for arrays,
//! equivalent to NumPy's savetxt and loadtxt functions

use crate::array::{Array, ArrayError};
use crate::types::{DType, NpyType};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Text I/O error
#[derive(Debug, Clone)]
pub enum TextIoError {
    /// Array error
    ArrayError(ArrayError),
    /// File I/O error
    FileError(String),
    /// Parse error
    ParseError(String),
    /// Unsupported operation
    Unsupported(String),
}

impl std::fmt::Display for TextIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextIoError::ArrayError(e) => write!(f, "Array error: {}", e),
            TextIoError::FileError(msg) => write!(f, "File error: {}", msg),
            TextIoError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            TextIoError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for TextIoError {}

impl From<ArrayError> for TextIoError {
    fn from(err: ArrayError) -> Self {
        TextIoError::ArrayError(err)
    }
}

/// Options for saving text files
#[derive(Debug, Clone)]
pub struct SaveTextOptions {
    /// Delimiter (default: space)
    pub delimiter: String,
    /// Format string for numbers (default: "%.18e")
    pub fmt: String,
    /// Header line (optional)
    pub header: Option<String>,
    /// Footer line (optional)
    pub footer: Option<String>,
    /// Comments prefix (default: "#")
    pub comments: String,
}

impl Default for SaveTextOptions {
    fn default() -> Self {
        SaveTextOptions {
            delimiter: " ".to_string(),
            fmt: "%.18e".to_string(),
            header: None,
            footer: None,
            comments: "#".to_string(),
        }
    }
}

/// Options for loading text files
#[derive(Debug, Clone)]
pub struct LoadTextOptions {
    /// Delimiter (auto-detect if None)
    pub delimiter: Option<String>,
    /// Number of header rows to skip
    pub skiprows: usize,
    /// Optional dtype hint
    pub dtype: Option<DType>,
    /// Comments prefix (default: "#")
    pub comments: String,
}

impl Default for LoadTextOptions {
    fn default() -> Self {
        LoadTextOptions {
            delimiter: None,
            skiprows: 0,
            dtype: None,
            comments: "#".to_string(),
        }
    }
}

/// Save array to text file
///
/// Saves an array to a text file, equivalent to NumPy's savetxt.
/// For multi-dimensional arrays, flattens to 2D (rows x columns).
pub fn save_text<P: AsRef<Path>>(
    path: P,
    array: &Array,
    options: SaveTextOptions,
) -> Result<(), TextIoError> {
    let shape = array.shape();
    let ndim = shape.len();
    
    if ndim == 0 {
        return Err(TextIoError::Unsupported("Cannot save 0-dimensional array".to_string()));
    }
    
    // Flatten to 2D if needed
    let (rows, cols) = if ndim == 1 {
        (shape[0], 1)
    } else if ndim == 2 {
        (shape[0], shape[1])
    } else {
        // For ND arrays, flatten all but last dimension
        let rows: i64 = shape[..ndim - 1].iter().product();
        (rows, shape[ndim - 1])
    };
    
    let file = File::create(path).map_err(|e| {
        TextIoError::FileError(format!("Failed to create file: {}", e))
    })?;
    let mut writer = BufWriter::new(file);
    
    // Write header
    if let Some(ref header) = options.header {
        writeln!(writer, "{}{}", options.comments, header).map_err(|e| {
            TextIoError::FileError(format!("Failed to write header: {}", e))
        })?;
    }
    
    // Write data
    match array.dtype().type_() {
        NpyType::Double => write_float_data(&mut writer, array, rows, cols, &options)?,
        NpyType::Float => write_float_data(&mut writer, array, rows, cols, &options)?,
        NpyType::Int => write_int_data(&mut writer, array, rows, cols, &options)?,
        NpyType::Long => write_int_data(&mut writer, array, rows, cols, &options)?,
        _ => return Err(TextIoError::Unsupported(
            format!("Unsupported dtype for text I/O: {:?}", array.dtype().type_())
        )),
    }
    
    // Write footer
    if let Some(ref footer) = options.footer {
        writeln!(writer, "{}{}", options.comments, footer).map_err(|e| {
            TextIoError::FileError(format!("Failed to write footer: {}", e))
        })?;
    }
    
    writer.flush().map_err(|e| {
        TextIoError::FileError(format!("Failed to flush file: {}", e))
    })?;
    
    Ok(())
}

/// Write float data to file
fn write_float_data<W: Write>(
    writer: &mut BufWriter<W>,
    array: &Array,
    rows: i64,
    cols: i64,
    options: &SaveTextOptions,
) -> Result<(), TextIoError> {
    unsafe {
        let data_ptr = array.data_ptr();
        let itemsize = array.itemsize();
        
        for i in 0..rows {
            for j in 0..cols {
                let offset = (i * cols + j) as usize * itemsize;
                let value = match array.dtype().type_() {
                    NpyType::Double => {
                        *(data_ptr.add(offset) as *const f64)
                    }
                    NpyType::Float => {
                        *(data_ptr.add(offset) as *const f32) as f64
                    }
                    _ => return Err(TextIoError::Unsupported("Invalid float type".to_string())),
                };
                
                if j > 0 {
                    write!(writer, "{}", options.delimiter).map_err(|e| {
                        TextIoError::FileError(format!("Failed to write delimiter: {}", e))
                    })?;
                }
                
                write!(writer, "{:.18e}", value).map_err(|e| {
                    TextIoError::FileError(format!("Failed to write value: {}", e))
                })?;
            }
            writeln!(writer).map_err(|e| {
                TextIoError::FileError(format!("Failed to write newline: {}", e))
            })?;
        }
    }
    
    Ok(())
}

/// Write integer data to file
fn write_int_data<W: Write>(
    writer: &mut BufWriter<W>,
    array: &Array,
    rows: i64,
    cols: i64,
    options: &SaveTextOptions,
) -> Result<(), TextIoError> {
    unsafe {
        let data_ptr = array.data_ptr();
        let itemsize = array.itemsize();
        
        for i in 0..rows {
            for j in 0..cols {
                let offset = (i * cols + j) as usize * itemsize;
                let value: i64 = match array.dtype().type_() {
                    NpyType::Int => {
                        *(data_ptr.add(offset) as *const i32) as i64
                    }
                    NpyType::Long => {
                        *(data_ptr.add(offset) as *const i64)
                    }
                    _ => return Err(TextIoError::Unsupported("Invalid int type".to_string())),
                };
                
                if j > 0 {
                    write!(writer, "{}", options.delimiter).map_err(|e| {
                        TextIoError::FileError(format!("Failed to write delimiter: {}", e))
                    })?;
                }
                
                write!(writer, "{}", value).map_err(|e| {
                    TextIoError::FileError(format!("Failed to write value: {}", e))
                })?;
            }
            writeln!(writer).map_err(|e| {
                TextIoError::FileError(format!("Failed to write newline: {}", e))
            })?;
        }
    }
    
    Ok(())
}

/// Load array from text file
///
/// Loads an array from a text file, equivalent to NumPy's loadtxt.
/// Automatically detects delimiter and infers dtype.
pub fn load_text<P: AsRef<Path>>(
    path: P,
    options: LoadTextOptions,
) -> Result<Array, TextIoError> {
    let file = File::open(path).map_err(|e| {
        TextIoError::FileError(format!("Failed to open file: {}", e))
    })?;
    let reader = BufReader::new(file);
    
    let lines: Vec<String> = reader.lines()
        .enumerate()
        .filter_map(|(i, line_result)| {
            if i < options.skiprows {
                return None; // Skip header rows
            }
            match line_result {
                Ok(line) => {
                    let trimmed = line.trim();
                    // Skip empty lines and comments
                    if trimmed.is_empty() || trimmed.starts_with(&options.comments) {
                        None
                    } else {
                        Some(line)
                    }
                }
                Err(_) => None,
            }
        })
        .collect();
    
    if lines.is_empty() {
        return Err(TextIoError::ParseError("No data rows found".to_string()));
    }
    
    // Detect delimiter
    let delimiter = options.delimiter.clone().unwrap_or_else(|| {
        detect_delimiter(&lines[0])
    });
    
    // Parse first line to determine number of columns
    let first_values: Vec<&str> = lines[0].split(&delimiter)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    
    if first_values.is_empty() {
        return Err(TextIoError::ParseError("No values found in first line".to_string()));
    }
    
    let num_cols = first_values.len();
    let num_rows = lines.len();
    
    // Parse all values and infer dtype
    let dtype = options.dtype.clone().unwrap_or_else(|| {
        infer_dtype_from_text(&lines, &delimiter)
    });
    
    // Create array
    let shape = vec![num_rows as i64, num_cols as i64];
    let mut array = Array::new(shape, dtype.clone())?;
    
    // Parse and store values
    parse_and_store_values(&mut array, &lines, &delimiter, num_cols)?;
    
    Ok(array)
}

/// Detect delimiter from line
fn detect_delimiter(line: &str) -> String {
    // Try common delimiters in order
    for delim in [",", "\t", " ", ";"].iter() {
        if line.contains(delim) {
            return delim.to_string();
        }
    }
    // Default to space
    " ".to_string()
}

/// Infer dtype from text content
fn infer_dtype_from_text(lines: &[String], delimiter: &str) -> DType {
    // Check first few lines to infer type
    let mut is_float = false;
    
    for line in lines.iter().take(10.min(lines.len())) {
        let values: Vec<&str> = line.split(delimiter)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .take(5) // Check first 5 values per line
            .collect();
        
        for value in values {
            if value.contains('.') || value.contains('e') || value.contains('E') {
                is_float = true;
                break;
            }
        }
        if is_float {
            break;
        }
    }
    
    if is_float {
        DType::new(NpyType::Double)
    } else {
        DType::new(NpyType::Long)
    }
}

/// Parse values from lines and store in array
fn parse_and_store_values(
    array: &mut Array,
    lines: &[String],
    delimiter: &str,
    num_cols: usize,
) -> Result<(), TextIoError> {
    match array.dtype().type_() {
        NpyType::Double => {
            unsafe {
                let data_ptr = array.data_ptr_mut() as *mut f64;
                for (i, line) in lines.iter().enumerate() {
                    let values: Vec<&str> = line.split(delimiter)
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    for (j, value_str) in values.iter().take(num_cols).enumerate() {
                        let value: f64 = value_str.parse().map_err(|_| {
                            TextIoError::ParseError(format!("Failed to parse '{}' as float", value_str))
                        })?;
                        *data_ptr.add(i * num_cols + j) = value;
                    }
                }
            }
        }
        NpyType::Float => {
            unsafe {
                let data_ptr = array.data_ptr_mut() as *mut f32;
                for (i, line) in lines.iter().enumerate() {
                    let values: Vec<&str> = line.split(delimiter)
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    for (j, value_str) in values.iter().take(num_cols).enumerate() {
                        let value: f32 = value_str.parse().map_err(|_| {
                            TextIoError::ParseError(format!("Failed to parse '{}' as float", value_str))
                        })?;
                        *data_ptr.add(i * num_cols + j) = value;
                    }
                }
            }
        }
        NpyType::Long => {
            unsafe {
                let data_ptr = array.data_ptr_mut() as *mut i64;
                for (i, line) in lines.iter().enumerate() {
                    let values: Vec<&str> = line.split(delimiter)
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    for (j, value_str) in values.iter().take(num_cols).enumerate() {
                        let value: i64 = value_str.parse().map_err(|_| {
                            TextIoError::ParseError(format!("Failed to parse '{}' as integer", value_str))
                        })?;
                        *data_ptr.add(i * num_cols + j) = value;
                    }
                }
            }
        }
        NpyType::Int => {
            unsafe {
                let data_ptr = array.data_ptr_mut() as *mut i32;
                for (i, line) in lines.iter().enumerate() {
                    let values: Vec<&str> = line.split(delimiter)
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    
                    for (j, value_str) in values.iter().take(num_cols).enumerate() {
                        let value: i32 = value_str.parse().map_err(|_| {
                            TextIoError::ParseError(format!("Failed to parse '{}' as integer", value_str))
                        })?;
                        *data_ptr.add(i * num_cols + j) = value;
                    }
                }
            }
        }
        _ => {
            return Err(TextIoError::Unsupported(
                format!("Unsupported dtype for text loading: {:?}", array.dtype().type_())
            ));
        }
    }
    
    Ok(())
}

