//! Structured dtype definitions

use crate::types::DType;
use crate::array::ArrayError;

/// Structured array error
#[derive(Debug, Clone)]
pub enum StructuredError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid field name
    InvalidFieldName,
    /// Field not found
    FieldNotFound(String),
    /// Invalid offset
    InvalidOffset,
}

impl std::fmt::Display for StructuredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuredError::ArrayError(e) => write!(f, "Array error: {}", e),
            StructuredError::InvalidFieldName => write!(f, "Invalid field name"),
            StructuredError::FieldNotFound(name) => write!(f, "Field not found: {}", name),
            StructuredError::InvalidOffset => write!(f, "Invalid field offset"),
        }
    }
}

impl std::error::Error for StructuredError {}

impl From<ArrayError> for StructuredError {
    fn from(err: ArrayError) -> Self {
        StructuredError::ArrayError(err)
    }
}

/// Field definition for structured dtype
#[derive(Debug, Clone)]
pub struct Field {
    /// Field name
    pub name: String,
    /// Field dtype
    pub dtype: DType,
    /// Byte offset of field in structure
    pub offset: usize,
}

/// Structured dtype definition
///
/// Represents a structured/compound dtype with multiple named fields
pub struct StructuredDType {
    /// Field definitions
    fields: Vec<Field>,
    /// Total size in bytes
    itemsize: usize,
}

impl StructuredDType {
    /// Create a new structured dtype from field definitions
    ///
    /// # Arguments
    /// * `fields` - Vector of (name, dtype) pairs
    ///
    /// # Returns
    /// * `Ok(StructuredDType)` if successful
    /// * `Err(StructuredError)` if creation fails
    pub fn new(fields: Vec<(String, DType)>) -> Result<Self, StructuredError> {
        if fields.is_empty() {
            return Err(StructuredError::InvalidFieldName);
        }
        
        // Calculate offsets and total size
        let mut struct_fields = Vec::with_capacity(fields.len());
        let mut current_offset: usize = 0;
        
        for (name, dtype) in fields {
            if name.is_empty() {
                return Err(StructuredError::InvalidFieldName);
            }
            
            let field_size = dtype.itemsize();
            // Align offset to dtype alignment
            let align = dtype.align();
            current_offset = current_offset.div_ceil(align) * align;
            
            struct_fields.push(Field {
                name,
                dtype: dtype.clone(),
                offset: current_offset,
            });
            
            current_offset += field_size;
        }
        
        // Align total size
        let max_align = struct_fields.iter().map(|f| f.dtype.align()).max().unwrap_or(1);
        let itemsize = current_offset.div_ceil(max_align) * max_align;
        
        Ok(StructuredDType {
            fields: struct_fields,
            itemsize,
        })
    }
    
    /// Get number of fields
    pub fn num_fields(&self) -> usize {
        self.fields.len()
    }
    
    /// Get field by index
    pub fn get_field(&self, index: usize) -> Option<&Field> {
        self.fields.get(index)
    }
    
    /// Get field by name
    pub fn get_field_by_name(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }
    
    /// Get all field names
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }
    
    /// Get itemsize (total size of structure)
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }
    
    /// Get all fields
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}

