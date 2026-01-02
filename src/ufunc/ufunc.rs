//! Universal function (ufunc) core structure
//!
//! This module provides the core ufunc structure and registration system,
//! equivalent to NumPy's ufunc_object.c

use crate::types::NpyType;
use crate::conversion::{promote_types, PromotionError};
use std::collections::HashMap;

/// Ufunc error
#[derive(Debug, Clone)]
pub enum UfuncError {
    /// Type promotion error
    PromotionError(PromotionError),
    /// Invalid number of inputs
    InvalidInputs,
    /// Invalid number of outputs
    InvalidOutputs,
    /// Unsupported type
    UnsupportedType,
}

impl std::fmt::Display for UfuncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UfuncError::PromotionError(e) => write!(f, "Promotion error: {}", e),
            UfuncError::InvalidInputs => write!(f, "Invalid number of inputs"),
            UfuncError::InvalidOutputs => write!(f, "Invalid number of outputs"),
            UfuncError::UnsupportedType => write!(f, "Unsupported type"),
        }
    }
}

impl std::error::Error for UfuncError {}

impl From<PromotionError> for UfuncError {
    fn from(err: PromotionError) -> Self {
        UfuncError::PromotionError(err)
    }
}

/// Function signature for a ufunc
///
/// Specifies input and output types for a ufunc
#[derive(Debug, Clone)]
pub struct UfuncSignature {
    /// Number of inputs
    pub n_inputs: usize,
    /// Number of outputs
    pub n_outputs: usize,
    /// Input types
    pub input_types: Vec<NpyType>,
    /// Output types
    pub output_types: Vec<NpyType>,
}

impl UfuncSignature {
    /// Create a new signature
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        UfuncSignature {
            n_inputs,
            n_outputs,
            input_types: Vec::new(),
            output_types: Vec::new(),
        }
    }
    
    /// Add input type
    pub fn with_input_type(mut self, ty: NpyType) -> Self {
        self.input_types.push(ty);
        self
    }
    
    /// Add output type
    pub fn with_output_type(mut self, ty: NpyType) -> Self {
        self.output_types.push(ty);
        self
    }
}

/// Loop function type for binary operations
///
/// This is the type of function that performs the actual computation
/// Signature: (in1, in2, out, count, stride1, stride2, stride_out)
pub type LoopFunction = unsafe fn(*const u8, *const u8, *mut u8, usize, usize, usize, usize);

/// Loop function type for unary operations
///
/// This is the type of function that performs unary computations
/// Signature: (input, output, count, stride_in, stride_out)
pub type UnaryLoopFunction = unsafe fn(*const u8, *mut u8, usize, usize, usize);

/// Universal function structure
///
/// Represents a universal function that operates element-wise on arrays
pub struct Ufunc {
    /// Function name
    name: String,
    /// Number of inputs
    n_inputs: usize,
    /// Number of outputs
    n_outputs: usize,
    /// Registered binary loop functions by type signature
    loops: HashMap<Vec<NpyType>, LoopFunction>,
    /// Registered unary loop functions by type signature
    unary_loops: HashMap<Vec<NpyType>, UnaryLoopFunction>,
    /// Default signature (for type resolution)
    #[allow(dead_code)]
    default_signature: UfuncSignature,
}

impl Ufunc {
    /// Create a new ufunc
    pub fn new(name: String, n_inputs: usize, n_outputs: usize) -> Self {
        Ufunc {
            name,
            n_inputs,
            n_outputs,
            loops: HashMap::new(),
            unary_loops: HashMap::new(),
            default_signature: UfuncSignature::new(n_inputs, n_outputs),
        }
    }
    
    /// Register a binary loop function for specific types
    pub fn register_loop(&mut self, input_types: Vec<NpyType>, loop_fn: LoopFunction) {
        self.loops.insert(input_types, loop_fn);
    }
    
    /// Register a unary loop function for specific types
    pub fn register_unary_loop(&mut self, input_types: Vec<NpyType>, loop_fn: UnaryLoopFunction) {
        self.unary_loops.insert(input_types, loop_fn);
    }
    
    /// Get binary loop function for types
    pub fn get_loop(&self, input_types: &[NpyType]) -> Option<LoopFunction> {
        self.loops.get(input_types).copied()
    }
    
    /// Get unary loop function for types
    pub fn get_unary_loop(&self, input_types: &[NpyType]) -> Option<UnaryLoopFunction> {
        self.unary_loops.get(input_types).copied()
    }
    
    /// Resolve types for ufunc application
    ///
    /// Determines the output types based on input types using type promotion
    pub fn resolve_types(&self, input_types: &[NpyType]) -> Result<Vec<NpyType>, UfuncError> {
        if input_types.len() != self.n_inputs {
            return Err(UfuncError::InvalidInputs);
        }
        
        // For binary operations, promote types
        if self.n_inputs == 2 {
            let promoted = promote_types(input_types[0], input_types[1])?;
            Ok(vec![promoted; self.n_outputs])
        } else if self.n_inputs == 1 {
            // Unary operations - output type same as input (for now)
            Ok(input_types.to_vec())
        } else {
            Err(UfuncError::UnsupportedType)
        }
    }
    
    /// Get ufunc name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get number of inputs
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }
    
    /// Get number of outputs
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }
}

/// Ufunc registry
///
/// Global registry of all ufuncs
pub struct UfuncRegistry {
    ufuncs: HashMap<String, Ufunc>,
}

impl UfuncRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        UfuncRegistry {
            ufuncs: HashMap::new(),
        }
    }
    
    /// Register a ufunc
    pub fn register(&mut self, name: String, ufunc: Ufunc) {
        self.ufuncs.insert(name, ufunc);
    }
    
    /// Get a ufunc by name
    pub fn get(&self, name: &str) -> Option<&Ufunc> {
        self.ufuncs.get(name)
    }
    
    /// Get a ufunc by name (mutable)
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Ufunc> {
        self.ufuncs.get_mut(name)
    }
}

impl Default for UfuncRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// Global ufunc registry (thread-local for now, could be made thread-safe later)
thread_local! {
    static GLOBAL_UFUNC_REGISTRY: std::cell::RefCell<UfuncRegistry> = 
        std::cell::RefCell::new(UfuncRegistry::new());
}

/// Register a ufunc in the global registry
pub fn register_ufunc(name: String, ufunc: Ufunc) {
    GLOBAL_UFUNC_REGISTRY.with(|registry| {
        registry.borrow_mut().register(name, ufunc);
    });
}

/// Get a ufunc from the global registry
pub fn get_ufunc(name: &str) -> Option<Ufunc> {
    GLOBAL_UFUNC_REGISTRY.with(|registry| {
        registry.borrow().get(name).map(|uf| {
            // Clone the ufunc (simplified - in production might want reference)
            Ufunc::new(uf.name().to_string(), uf.n_inputs(), uf.n_outputs())
        })
    })
}

