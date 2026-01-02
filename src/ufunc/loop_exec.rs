//! Loop execution framework
//!
//! This module provides the framework for executing ufunc loops,
//! integrating with iterators and broadcasting

use crate::array::Array;
use crate::broadcasting::{broadcast_shapes, BroadcastError};
use crate::types::NpyType;
use crate::ufunc::{Ufunc, UfuncError, LoopFunction, UnaryLoopFunction};

/// Loop execution error
#[derive(Debug, Clone)]
pub enum LoopExecutionError {
    /// Ufunc error
    UfuncError(UfuncError),
    /// Broadcasting error
    BroadcastError(BroadcastError),
    /// Type mismatch
    TypeMismatch,
}

impl std::fmt::Display for LoopExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopExecutionError::UfuncError(e) => write!(f, "Ufunc error: {}", e),
            LoopExecutionError::BroadcastError(e) => write!(f, "Broadcast error: {}", e),
            LoopExecutionError::TypeMismatch => write!(f, "Type mismatch"),
        }
    }
}

impl std::error::Error for LoopExecutionError {}

impl From<UfuncError> for LoopExecutionError {
    fn from(err: UfuncError) -> Self {
        LoopExecutionError::UfuncError(err)
    }
}

impl From<BroadcastError> for LoopExecutionError {
    fn from(err: BroadcastError) -> Self {
        LoopExecutionError::BroadcastError(err)
    }
}

/// Execute a binary ufunc loop on arrays
///
/// This applies a ufunc to input arrays, handling broadcasting and type resolution
#[allow(dead_code)]
pub fn execute_ufunc_loop(
    ufunc: &Ufunc,
    inputs: &[&Array],
    output: &mut Array,
    loop_fn: LoopFunction,
) -> Result<(), LoopExecutionError> {
    if inputs.len() != ufunc.n_inputs() {
        return Err(LoopExecutionError::UfuncError(UfuncError::InvalidInputs));
    }
    
    // For now, simple implementation: assume arrays are same shape or broadcastable
    // Get input shapes
    let input_shapes: Vec<&[i64]> = inputs.iter().map(|a| a.shape()).collect();
    
    // Compute broadcast shape
    let broadcast_shape = if input_shapes.len() > 1 {
        broadcast_shapes(input_shapes[0], input_shapes[1])?
    } else {
        input_shapes[0].to_vec()
    };
    
    // Verify output shape matches
    if output.shape() != broadcast_shape.as_slice() {
        return Err(LoopExecutionError::TypeMismatch);
    }
    
    // For simplicity, assume arrays are same shape and contiguous
    // In full implementation, would handle broadcasting and strided iteration
    let count = output.size();
    let itemsize = output.itemsize();
    
    if count == 0 {
        return Ok(());
    }
    
    // Get data pointers
    let in1_ptr = inputs[0].data_ptr();
    let in2_ptr = if inputs.len() > 1 { inputs[1].data_ptr() } else { in1_ptr };
    let out_ptr = output.data_ptr_mut();
    
    // Execute loop
    // Note: loop_fn signature expects: (in1, in2, out, count, stride1, stride2, stride_out)
    // For contiguous arrays, strides equal itemsize
    unsafe {
        loop_fn(
            in1_ptr,
            in2_ptr,
            out_ptr,
            count,
            itemsize,
            itemsize,
            itemsize,
        );
    }
    
    Ok(())
}

/// Execute a unary ufunc loop on arrays
///
/// This applies a unary ufunc to an input array
pub fn execute_unary_ufunc_loop(
    ufunc: &Ufunc,
    input: &Array,
    output: &mut Array,
    loop_fn: UnaryLoopFunction,
) -> Result<(), LoopExecutionError> {
    if ufunc.n_inputs() != 1 {
        return Err(LoopExecutionError::UfuncError(UfuncError::InvalidInputs));
    }
    
    // Verify shapes match
    if input.shape() != output.shape() {
        return Err(LoopExecutionError::TypeMismatch);
    }
    
    let count = output.size();
    let _itemsize = output.itemsize();
    
    if count == 0 {
        return Ok(());
    }
    
    let in_ptr = input.data_ptr();
    let out_ptr = output.data_ptr_mut();
    
    // Execute unary loop
    // For contiguous arrays, stride should be 1 (one element), not itemsize
    // This matches how binary loops work (stride in elements, not bytes)
    unsafe {
        loop_fn(
            in_ptr,
            out_ptr,
            count,
            1,  // stride_in: 1 element
            1,  // stride_out: 1 element
        );
    }
    
    Ok(())
}

/// Create a ufunc loop for arrays
///
/// Sets up iteration and executes the loop function
#[allow(dead_code)]
pub fn create_ufunc_loop(
    ufunc: &Ufunc,
    inputs: &[&Array],
    output: &mut Array,
) -> Result<(), LoopExecutionError> {
    // Resolve types
    let input_types: Vec<NpyType> = inputs.iter().map(|a| a.dtype().type_()).collect();
    let _output_types = ufunc.resolve_types(&input_types)?;
    
    // Get loop function (simplified - would need proper dispatch)
    // For now, use double precision as default
    let loop_fn = ufunc.get_loop(&input_types)
        .ok_or(LoopExecutionError::UfuncError(UfuncError::UnsupportedType))?;
    
    execute_ufunc_loop(ufunc, inputs, output, loop_fn)
}

/// Create a unary ufunc loop for arrays
///
/// Sets up iteration and executes the unary loop function
pub fn create_unary_ufunc_loop(
    ufunc: &Ufunc,
    input: &Array,
    output: &mut Array,
) -> Result<(), LoopExecutionError> {
    // Resolve types (for unary, output type same as input)
    let input_type = input.dtype().type_();
    let input_types = vec![input_type];
    
    // Get unary loop function
      let loop_fn = ufunc.get_unary_loop(&input_types)
        .ok_or(LoopExecutionError::UfuncError(UfuncError::UnsupportedType))?;
    
    execute_unary_ufunc_loop(ufunc, input, output, loop_fn)
}
