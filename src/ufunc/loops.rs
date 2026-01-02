//! Ufunc loop implementations
//!
//! This module provides loop implementations for ufuncs,
//! equivalent to NumPy's loops.c.src

use crate::types::NpyType;

/// Add loop for double precision
pub unsafe fn add_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut f64;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 + val2;
    }
}

/// Add loop for float precision
pub unsafe fn add_loop_float(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f32;
    let in2_ptr = in2 as *const f32;
    let out_ptr = out as *mut f32;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 + val2;
    }
}

/// Add loop for integers
pub unsafe fn add_loop_int(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const i32;
    let in2_ptr = in2 as *const i32;
    let out_ptr = out as *mut i32;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 + val2;
    }
}

/// Subtract loop for double precision
pub unsafe fn subtract_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut f64;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 - val2;
    }
}

/// Multiply loop for double precision
pub unsafe fn multiply_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut f64;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 * val2;
    }
}

/// Divide loop for double precision
pub unsafe fn divide_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut f64;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 / val2;
    }
}

/// Equal comparison loop for double precision
pub unsafe fn equal_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut bool;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 == val2;
    }
}

/// Less than comparison loop for double precision
pub unsafe fn less_loop_double(
    in1: *const u8,
    in2: *const u8,
    out: *mut u8,
    count: usize,
    stride1: usize,
    stride2: usize,
    stride_out: usize,
) {
    let in1_ptr = in1 as *const f64;
    let in2_ptr = in2 as *const f64;
    let out_ptr = out as *mut bool;
    
    for i in 0..count {
        let val1 = *in1_ptr.add(i * stride1);
        let val2 = *in2_ptr.add(i * stride2);
        *out_ptr.add(i * stride_out) = val1 < val2;
    }
}

/// Get loop function for a type (simplified dispatch)
pub fn get_add_loop(ty: NpyType) -> Option<unsafe fn(*const u8, *const u8, *mut u8, usize, usize, usize, usize)> {
    match ty {
        NpyType::Double => Some(add_loop_double),
        NpyType::Float => Some(add_loop_float),
        NpyType::Int => Some(add_loop_int),
        _ => None,
    }
}

/// Get the subtract loop function for a given type
pub fn get_subtract_loop(ty: NpyType) -> Option<unsafe fn(*const u8, *const u8, *mut u8, usize, usize, usize, usize)> {
    match ty {
        NpyType::Double => Some(subtract_loop_double),
        _ => None,
    }
}

/// Get the multiply loop function for a given type
pub fn get_multiply_loop(ty: NpyType) -> Option<unsafe fn(*const u8, *const u8, *mut u8, usize, usize, usize, usize)> {
    match ty {
        NpyType::Double => Some(multiply_loop_double),
        _ => None,
    }
}

/// Get the divide loop function for a given type
pub fn get_divide_loop(ty: NpyType) -> Option<unsafe fn(*const u8, *const u8, *mut u8, usize, usize, usize, usize)> {
    match ty {
        NpyType::Double => Some(divide_loop_double),
        _ => None,
    }
}

