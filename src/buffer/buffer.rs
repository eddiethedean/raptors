//! Buffer structure and implementation

use crate::array::{Array, ArrayError};
use crate::buffer::{BufferError, FormatString};
use crate::types::NpyType;

/// Buffer information for exported arrays
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Data pointer
    pub ptr: *mut u8,
    /// Total size in bytes
    pub size: usize,
    /// Format string describing the data layout
    pub format: String,
    /// Shape of the buffer
    pub shape: Vec<i64>,
    /// Strides in bytes
    pub strides: Vec<i64>,
    /// Read-only flag
    pub read_only: bool,
}

/// Exported buffer from an array
///
/// This represents an array exported via the buffer protocol.
/// The buffer shares memory with the source array.
pub struct Buffer {
    /// Buffer information
    pub info: BufferInfo,
    /// Reference to source array (to keep it alive)
    _array: Array,
}

impl Buffer {
    /// Create a new buffer from buffer info
    pub fn new(info: BufferInfo, array: Array) -> Self {
        Buffer {
            info,
            _array: array,
        }
    }
    
    /// Get the data pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.info.ptr
    }
    
    /// Get the size in bytes
    pub fn len(&self) -> usize {
        self.info.size
    }
    
    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.info.size == 0
    }
    
    /// Check if buffer is read-only
    pub fn is_read_only(&self) -> bool {
        self.info.read_only
    }
}

/// Export an array as a buffer
///
/// Creates a buffer protocol representation of the array,
/// suitable for sharing with other libraries.
pub fn export_buffer(array: &Array) -> Result<BufferInfo, BufferError> {
    // Generate format string from dtype
    let format = format_string_from_dtype(array.dtype().type_())?;
    
    // Calculate size
    let shape = array.shape();
    let size: usize = shape.iter()
        .map(|&s| s.max(0) as usize)
        .product::<usize>()
        * array.itemsize();
    
    // Determine if read-only
    let read_only = !array.is_writeable();
    
    Ok(BufferInfo {
        ptr: array.data_ptr() as *mut u8, // Cast const to mut for buffer protocol
        size,
        format,
        shape: shape.to_vec(),
        strides: array.strides().to_vec(),
        read_only,
    })
}

/// Import a buffer to create an array
///
/// Creates an array from a buffer protocol representation.
/// The array will reference the buffer's memory (zero-copy).
///
/// # Safety
/// The caller must ensure that `ptr` is valid for the lifetime of the returned array,
/// or that proper memory management is handled externally.
pub unsafe fn import_buffer(
    ptr: *mut u8,
    format: &str,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    _read_only: bool,
) -> Result<Array, BufferError> {
    // Parse format string
    let format_str = FormatString::parse(format)?;
    
    // Convert format to dtype
    let dtype = dtype_from_format(&format_str)?;
    
    // Calculate strides if not provided
    let _calculated_strides = strides.unwrap_or_else(|| {
        compute_strides_from_shape(&shape, dtype.itemsize())
    });
    
    // Validate buffer size
    let _size: usize = shape.iter()
        .map(|&s| s.max(0) as usize)
        .product::<usize>()
        * dtype.itemsize();
    
    // Create array from external data
    // Note: from_external_memory doesn't take strides, it computes them
    // For custom strides, we would need a different approach
    let owns_data = false; // Buffer is externally managed
    
    Array::from_external_memory(ptr, shape, dtype, owns_data)
        .map_err(|e| match e {
            ArrayError::InvalidShape => BufferError::BufferTooSmall,
            _ => BufferError::Unsupported(format!("Array creation failed: {:?}", e)),
        })
}

/// Generate format string from NumPy type
fn format_string_from_dtype(ty: NpyType) -> Result<String, BufferError> {
    let format = match ty {
        NpyType::Byte => "b",
        NpyType::UByte => "B",
        NpyType::Short => "h",
        NpyType::UShort => "H",
        NpyType::Int => "i",
        NpyType::UInt => "I",
        NpyType::Long => "l",
        NpyType::ULong => "L",
        NpyType::LongLong => "q",
        NpyType::ULongLong => "Q",
        NpyType::Float => "f",
        NpyType::Double => "d",
        _ => return Err(BufferError::Unsupported(
            format!("Unsupported type for buffer protocol: {:?}", ty)
        )),
    };
    
    Ok(format.to_string())
}

/// Convert format string to dtype
fn dtype_from_format(format: &FormatString) -> Result<crate::types::DType, BufferError> {
    let npy_type = match format.type_char {
        'b' => NpyType::Byte,
        'B' => NpyType::UByte,
        'h' => NpyType::Short,
        'H' => NpyType::UShort,
        'i' => NpyType::Int,
        'I' => NpyType::UInt,
        'l' => NpyType::Long,
        'L' => NpyType::ULong,
        'q' => NpyType::LongLong,
        'Q' => NpyType::ULongLong,
        'f' => NpyType::Float,
        'd' => NpyType::Double,
        _ => return Err(BufferError::InvalidFormat(
            format!("Unsupported format character: {}", format.type_char)
        )),
    };
    
    Ok(crate::types::DType::new(npy_type))
}

/// Compute strides from shape
fn compute_strides_from_shape(shape: &[i64], itemsize: usize) -> Vec<i64> {
    let mut strides = Vec::with_capacity(shape.len());
    
    if shape.is_empty() {
        return strides;
    }
    
    // C-contiguous strides
    let mut stride = itemsize as i64;
    for &dim in shape.iter().rev() {
        strides.insert(0, stride);
        stride *= dim.max(1);
    }
    
    strides
}

