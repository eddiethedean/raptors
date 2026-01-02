//! User-defined types implementation
//!
//! This module provides support for custom dtypes beyond the built-in NumPy types

use crate::array::ArrayError;
use crate::types::DType;
use std::collections::HashMap;
use std::sync::Mutex;

/// Custom type trait
///
/// Types implementing this trait can be used as custom array element types
pub trait CustomType: Send + Sync {
    /// Size in bytes
    fn itemsize(&self) -> usize;
    
    /// Alignment requirement in bytes
    fn align(&self) -> usize;
    
    /// String representation of the type
    fn to_string(&self) -> String;
    
    /// Convert from bytes
    #[allow(clippy::wrong_self_convention)] // This is an intentional naming convention
    fn from_bytes(&self, bytes: &[u8]) -> Result<Vec<u8>, CustomTypeError>;
    
    /// Convert to bytes
    fn to_bytes(&self, value: &[u8]) -> Result<Vec<u8>, CustomTypeError>;
    
    /// Type name
    fn name(&self) -> &str;
}

/// Custom type error
#[derive(Debug, Clone)]
pub enum CustomTypeError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid type
    InvalidType(String),
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for CustomTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CustomTypeError::ArrayError(e) => write!(f, "Array error: {}", e),
            CustomTypeError::InvalidType(msg) => write!(f, "Invalid type: {}", msg),
            CustomTypeError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for CustomTypeError {}

impl From<ArrayError> for CustomTypeError {
    fn from(err: ArrayError) -> Self {
        CustomTypeError::ArrayError(err)
    }
}

/// Type ID for custom types
pub type CustomTypeId = u32;

/// Global type registry
static TYPE_REGISTRY: Mutex<Option<TypeRegistry>> = Mutex::new(None);

/// Type registry for custom types
pub struct TypeRegistry {
    /// Next available type ID
    next_id: CustomTypeId,
    /// Registered types
    types: HashMap<CustomTypeId, Box<dyn CustomType>>,
    /// Name to ID mapping
    name_to_id: HashMap<String, CustomTypeId>,
}

impl TypeRegistry {
    /// Create a new type registry
    fn new() -> Self {
        TypeRegistry {
            next_id: 1000, // Start custom types at 1000
            types: HashMap::new(),
            name_to_id: HashMap::new(),
        }
    }
    
    /// Get the global registry (lazy initialization)
    fn get() -> std::sync::MutexGuard<'static, Option<TypeRegistry>> {
        let mut registry = TYPE_REGISTRY.lock().unwrap();
        if registry.is_none() {
            *registry = Some(TypeRegistry::new());
        }
        registry
    }
    
    /// Register a custom type
    pub fn register<T: CustomType + 'static>(custom_type: T) -> Result<CustomTypeId, CustomTypeError> {
        let mut registry = TypeRegistry::get();
        let reg = registry.as_mut().unwrap();
        
        let id = reg.next_id;
        reg.next_id += 1;
        
        let name = custom_type.name().to_string();
        
        if reg.name_to_id.contains_key(&name) {
            return Err(CustomTypeError::InvalidType(
                format!("Type '{}' already registered", name)
            ));
        }
        
        reg.name_to_id.insert(name.clone(), id);
        reg.types.insert(id, Box::new(custom_type));
        
        Ok(id)
    }
    
    /// Get a custom type by ID
    pub fn get_by_id(_id: CustomTypeId) -> Option<Box<dyn CustomType>> {
        // Clone the type (simplified - in practice would need better cloning)
        // For now, return None as we can't easily clone trait objects
        // This is a limitation of the current implementation
        None
    }
    
    /// Get a custom type by name
    pub fn get_by_name(name: &str) -> Option<CustomTypeId> {
        let registry = TypeRegistry::get();
        let reg = registry.as_ref().unwrap();
        
        reg.name_to_id.get(name).copied()
    }
}

/// Register a custom type
///
/// Returns the type ID assigned to this custom type
pub fn register_custom_type<T: CustomType + 'static>(
    custom_type: T,
) -> Result<CustomTypeId, CustomTypeError> {
    TypeRegistry::register(custom_type)
}

/// Get custom type ID by name
pub fn get_custom_type_id(name: &str) -> Option<CustomTypeId> {
    TypeRegistry::get_by_name(name)
}

/// Create a DType for a custom type
pub fn create_custom_dtype(
    _type_id: CustomTypeId,
) -> Result<DType, CustomTypeError> {
    // Get type information from registry
    // Note: We can't easily get the type back from the registry without better cloning support
    // For now, we'll create a placeholder dtype
    // In a full implementation, would retrieve itemsize, align, and name from registry
    
    // Placeholder - would need to store this info in the registry
    Err(CustomTypeError::InvalidType(
        "Custom dtype creation requires type information retrieval from registry".to_string()
    ))
}


