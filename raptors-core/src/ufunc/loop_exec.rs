//! Loop execution framework
//!
//! This module provides the framework for executing ufunc loops,
//! integrating with iterators and broadcasting

use crate::array::Array;
use crate::broadcasting::{broadcast_strides, broadcast_shapes_multi, BroadcastError};
use crate::types::{NpyType, CustomTypeId};
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
/// This applies a ufunc to input arrays, handling broadcasting and type resolution.
/// Checks for custom type optimizations before falling back to standard execution.
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
    
    // Check for custom type optimizations
    // If all inputs and output are custom types, try optimized path
    let all_custom = inputs.iter().all(|a| a.dtype().custom_type_id().is_some())
        && output.dtype().custom_type_id().is_some();
    
    if all_custom {
        // Try to use optimized operation for custom types
        // This is a placeholder - full implementation would check each custom type
        // for optimized_operation support
        let _custom_type_ids: Vec<Option<CustomTypeId>> = inputs.iter()
            .map(|a| a.dtype().custom_type_id())
            .collect();
        // For now, fall through to standard execution
    }
    
    // Get input shapes and compute broadcast shape
    let input_shapes: Vec<&[i64]> = inputs.iter().map(|a| a.shape()).collect();
    
    // Compute broadcast shape for all inputs
    let broadcast_shape = if input_shapes.len() > 1 {
        broadcast_shapes_multi(&input_shapes)?
    } else {
        input_shapes[0].to_vec()
    };
    
    // Verify output shape matches broadcast shape
    if output.shape() != broadcast_shape.as_slice() {
        return Err(LoopExecutionError::TypeMismatch);
    }
    
    let count = output.size();
    let itemsize = output.itemsize();
    
    if count == 0 {
        return Ok(());
    }
    
    // Calculate broadcast strides for each input
    let mut input_broadcast_strides: Vec<Vec<i64>> = Vec::new();
    for (input, input_shape) in inputs.iter().zip(input_shapes.iter()) {
        let input_strides = input.strides();
        let broadcast_strides = broadcast_strides(
            input_shape,
            input_strides,
            &broadcast_shape,
        )?;
        input_broadcast_strides.push(broadcast_strides);
    }
    
    // Get output strides (clone to avoid borrowing issues)
    let output_strides = output.strides().to_vec();
    
    // Check if all arrays are contiguous (fast path)
    let all_contiguous = inputs.iter().all(|a| a.is_c_contiguous()) && output.is_c_contiguous();
    let all_same_shape = input_shapes.iter().all(|s| s == &broadcast_shape.as_slice());
    
    if all_contiguous && all_same_shape {
        // Fast path: all arrays are contiguous and same shape
        let in1_ptr = inputs[0].data_ptr();
        let in2_ptr = if inputs.len() > 1 { inputs[1].data_ptr() } else { in1_ptr };
        let out_ptr = output.data_ptr_mut();
        
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
    } else {
        // General path: handle broadcasting and strided arrays
        // Use multi-dimensional iteration with proper broadcast strides
        let in1_ptr = inputs[0].data_ptr();
        let in2_ptr = if inputs.len() > 1 { inputs[1].data_ptr() } else { in1_ptr };
        let out_ptr = output.data_ptr_mut();
        
        // Helper to compute offset from flat index using strides
        let compute_offset_from_index = |index: usize, shape: &[i64], strides: &[i64]| -> usize {
            let mut offset = 0usize;
            let mut remaining = index;
            
            // Convert flat index to coordinates (C-order)
            for (dim_idx, &dim_size) in shape.iter().enumerate() {
                if dim_idx < strides.len() {
                    let coord = remaining % (dim_size as usize);
                    offset += (coord as i64 * strides[dim_idx]) as usize;
                    remaining /= dim_size as usize;
                }
            }
            offset
        };
        
        // Iterate through all elements using flat index
        for i in 0..count {
            // Compute offsets for each input and output
            let offset1 = compute_offset_from_index(i, &broadcast_shape, &input_broadcast_strides[0]);
            let offset2 = if input_broadcast_strides.len() > 1 {
                compute_offset_from_index(i, &broadcast_shape, &input_broadcast_strides[1])
            } else {
                offset1
            };
            let offset_out = compute_offset_from_index(i, &broadcast_shape, &output_strides);
            
            // Execute loop function for this element
            unsafe {
                let elem_in1 = in1_ptr.add(offset1);
                let elem_in2 = in2_ptr.add(offset2);
                let elem_out = out_ptr.add(offset_out);
                
                loop_fn(
                    elem_in1,
                    elem_in2,
                    elem_out,
                    1, // Process one element at a time
                    itemsize,
                    itemsize,
                    itemsize,
                );
            }
        }
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
