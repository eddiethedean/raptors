//! Data type (dtype) implementation
//!
//! This module provides dtype functionality, equivalent to NumPy's
//! dtype system from descriptor.c and related files

use std::fmt;

/// NumPy-compatible type enumeration
///
/// This matches NumPy's NPY_TYPES enum from ndarraytypes.h
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NpyType {
    /// Boolean type
    Bool = 0,
    /// Signed 8-bit integer
    Byte,
    /// Unsigned 8-bit integer
    UByte,
    /// Signed 16-bit integer
    Short,
    /// Unsigned 16-bit integer
    UShort,
    /// Signed 32-bit integer
    Int,
    /// Unsigned 32-bit integer
    UInt,
    /// Signed 64-bit integer (platform-dependent)
    Long,
    /// Unsigned 64-bit integer (platform-dependent)
    ULong,
    /// Signed 64-bit integer
    LongLong,
    /// Unsigned 64-bit integer
    ULongLong,
    /// 32-bit floating point
    Float,
    /// 64-bit floating point
    Double,
    /// Extended precision floating point
    LongDouble,
    /// Complex 64-bit floating point (2x 32-bit float)
    CFloat,
    /// Complex 128-bit floating point (2x 64-bit double)
    CDouble,
    /// Complex extended precision floating point
    CLongDouble,
    /// Python object reference
    Object = 17,
    /// String type
    String,
    /// Unicode string type
    Unicode,
    /// Void type
    Void,
    /// Datetime type
    DateTime,
    /// Timedelta type
    Timedelta,
    /// 16-bit floating point (half precision)
    Half,
    /// Character type (deprecated)
    Char, // Deprecated
}

/// Data type descriptor
///
/// This represents a NumPy dtype, equivalent to PyArray_Descr
#[derive(Debug, Clone)]
pub struct DType {
    /// The base type
    type_: NpyType,
    /// Size in bytes
    itemsize: usize,
    /// Alignment requirement
    align: usize,
    /// Type name
    name: String,
    /// Custom type ID (if this is a custom type)
    custom_type_id: Option<u32>,
    /// Custom type metadata (optional)
    #[allow(dead_code)] // Reserved for future use
    custom_metadata: Option<String>,
}

impl DType {
    /// Create a new dtype
    pub fn new(type_: NpyType) -> Self {
        let (itemsize, align, name) = match type_ {
            NpyType::Bool => (1, 1, "bool".to_string()),
            NpyType::Byte => (1, 1, "int8".to_string()),
            NpyType::UByte => (1, 1, "uint8".to_string()),
            NpyType::Short => (2, 2, "int16".to_string()),
            NpyType::UShort => (2, 2, "uint16".to_string()),
            NpyType::Int => (4, 4, "int32".to_string()),
            NpyType::UInt => (4, 4, "uint32".to_string()),
            NpyType::Long => (std::mem::size_of::<i64>(), 8, "int64".to_string()),
            NpyType::ULong => (std::mem::size_of::<u64>(), 8, "uint64".to_string()),
            NpyType::LongLong => (8, 8, "int64".to_string()),
            NpyType::ULongLong => (8, 8, "uint64".to_string()),
            NpyType::Float => (4, 4, "float32".to_string()),
            NpyType::Double => (8, 8, "float64".to_string()),
            NpyType::LongDouble => (16, 16, "float128".to_string()),
            NpyType::CFloat => (8, 4, "complex64".to_string()),
            NpyType::CDouble => (16, 8, "complex128".to_string()),
            NpyType::CLongDouble => (32, 16, "complex256".to_string()),
            NpyType::Half => (2, 2, "float16".to_string()),
            NpyType::String => (1, 1, "string".to_string()), // Variable length, default to 1
            NpyType::Unicode => (4, 4, "unicode".to_string()), // Variable length, default to 4 bytes per char
            _ => (8, 8, "object".to_string()), // Default for unimplemented types
        };
        
        DType {
            type_,
            itemsize,
            align,
            name,
            custom_type_id: None,
            custom_metadata: None,
        }
    }
    
    /// Get the itemsize in bytes
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    
    /// Get the alignment requirement
    pub fn align(&self) -> usize {
        self.align
    }
    
    /// Get the type name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the base type
    pub fn type_(&self) -> NpyType {
        self.type_
    }
    
    /// Get custom type ID (if this is a custom type)
    pub fn custom_type_id(&self) -> Option<u32> {
        self.custom_type_id
    }
    
    /// Create a custom dtype
    pub fn custom(custom_type_id: u32, itemsize: usize, align: usize, name: String) -> Self {
        DType {
            type_: NpyType::Void, // Use Void as placeholder for custom types
            itemsize,
            align,
            name,
            custom_type_id: Some(custom_type_id),
            custom_metadata: None,
        }
    }
    
    /// Create a string dtype with custom itemsize
    /// 
    /// This is used for fixed-width string arrays where all strings
    /// have the same maximum width
    pub fn string_with_itemsize(itemsize: usize) -> Self {
        DType {
            type_: NpyType::String,
            itemsize,
            align: 1,
            name: format!("string{}", itemsize),
            custom_type_id: None,
            custom_metadata: None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::new(NpyType::Double)
    }
}

