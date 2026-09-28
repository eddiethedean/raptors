//! Array subclassing framework
//!
//! This module provides support for extending array types through subclassing,
//! similar to NumPy's array subclassing system.

use crate::array::{Array, ArrayError};
use crate::types::DType;
use std::any::Any;

/// Base trait for array-like types
///
/// This trait defines the common interface that all array types must implement.
/// Subclasses can override methods to customize behavior.
///
/// This trait does not establish thread safety for underlying array storage.
/// Shared ownership and mutable aliases require the rebuild's access contract.
pub trait ArrayBase {
    /// Get the underlying array data
    fn array(&self) -> &Array;
    
    /// Get mutable reference to underlying array data
    fn array_mut(&mut self) -> &mut Array;
    
    /// Get the shape of the array
    fn shape(&self) -> &[i64] {
        self.array().shape()
    }
    
    /// Get the dtype
    fn dtype(&self) -> &DType {
        self.array().dtype()
    }
    
    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.array().ndim()
    }
    
    /// Get the size (total number of elements)
    fn size(&self) -> usize {
        self.array().size()
    }
    
    /// Finalize array creation (called after array is created)
    ///
    /// This is equivalent to NumPy's __array_finalize__ method.
    /// Subclasses can override this to perform initialization.
    fn array_finalize(&mut self) -> Result<(), ArrayError> {
        Ok(())
    }
    
    /// Get type name for isinstance checks
    fn type_name(&self) -> &'static str {
        "ArrayBase"
    }
    
    /// Check if this is an instance of a given type
    fn isinstance(&self, type_name: &str) -> bool {
        self.type_name() == type_name
    }
    
    /// Get type information as Any for downcasting
    fn as_any(&self) -> &dyn Any;
    
    /// Get mutable type information as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Method resolution order (MRO) for array types
///
/// This determines the order in which methods are resolved when multiple
/// inheritance or method overriding is involved.
pub struct MethodResolutionOrder {
    /// Ordered list of type names in resolution order
    pub order: Vec<&'static str>,
}

impl MethodResolutionOrder {
    /// Create a new MRO with a single type
    pub fn new(type_name: &'static str) -> Self {
        MethodResolutionOrder {
            order: vec![type_name],
        }
    }
    
    /// Add a base type to the MRO
    pub fn with_base(mut self, base_name: &'static str) -> Self {
        self.order.push(base_name);
        self
    }
    
    /// Check if a type is in the MRO
    #[allow(clippy::manual_contains)] // Can't use contains() because order is Vec<&'static str> and type_name is &str
    pub fn contains(&self, type_name: &str) -> bool {
        self.order.iter().any(|&name| name == type_name)
    }
}

/// Subclassable array wrapper
///
/// This wraps an Array and provides subclassing capabilities.
/// Subclasses can override methods and add custom state.
pub struct SubclassableArray {
    /// The underlying array
    array: Array,
    /// Type name for isinstance checks
    type_name: &'static str,
    /// Method resolution order
    mro: MethodResolutionOrder,
    /// Custom state (optional)
    custom_state: Option<Box<dyn Any + Send + Sync>>,
}

impl SubclassableArray {
    /// Create a new subclassable array
    pub fn new(array: Array, type_name: &'static str) -> Self {
        let mro = MethodResolutionOrder::new(type_name);
        SubclassableArray {
            array,
            type_name,
            mro,
            custom_state: None,
        }
    }
    
    /// Create with custom state
    pub fn with_state(
        array: Array,
        type_name: &'static str,
        state: Box<dyn Any + Send + Sync>,
    ) -> Self {
        let mro = MethodResolutionOrder::new(type_name);
        SubclassableArray {
            array,
            type_name,
            mro,
            custom_state: Some(state),
        }
    }
    
    /// Get the type name
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }
    
    /// Get the MRO
    pub fn mro(&self) -> &MethodResolutionOrder {
        &self.mro
    }
    
    /// Get custom state
    pub fn custom_state(&self) -> Option<&(dyn Any + Send + Sync)> {
        self.custom_state.as_ref().map(|s| s.as_ref())
    }
}

impl ArrayBase for SubclassableArray {
    fn array(&self) -> &Array {
        &self.array
    }
    
    fn array_mut(&mut self) -> &mut Array {
        &mut self.array
    }
    
    fn type_name(&self) -> &'static str {
        self.type_name
    }
    
    fn isinstance(&self, type_name: &str) -> bool {
        self.mro.contains(type_name)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Example subclass: CustomArray
///
/// This demonstrates how to create a custom array subclass.
pub struct CustomArray {
    /// Wrapped array
    array: Array,
    /// Custom metadata
    metadata: String,
}

impl CustomArray {
    /// Create a new custom array
    pub fn new(array: Array, metadata: String) -> Self {
        CustomArray { array, metadata }
    }
    
    /// Get custom metadata
    pub fn metadata(&self) -> &str {
        &self.metadata
    }
}

impl ArrayBase for CustomArray {
    fn array(&self) -> &Array {
        &self.array
    }
    
    fn array_mut(&mut self) -> &mut Array {
        &mut self.array
    }
    
    fn type_name(&self) -> &'static str {
        "CustomArray"
    }
    
    fn isinstance(&self, type_name: &str) -> bool {
        matches!(type_name, "CustomArray" | "ArrayBase")
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Check if an array-like object is an instance of a type
pub fn isinstance<T: ArrayBase>(array: &T, type_name: &str) -> bool {
    array.isinstance(type_name)
}
