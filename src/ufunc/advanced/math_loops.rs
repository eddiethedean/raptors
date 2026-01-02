//! Advanced mathematical ufunc loop implementations
//!
//! This module provides unary loop implementations for mathematical functions,
//! equivalent to NumPy's loops_unary_fp.dispatch.c.src and loops_trigonometric.dispatch.cpp


// Trigonometric functions - Double precision

/// Sin loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn sin_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).sin();
    }
}

/// Cos loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn cos_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).cos();
    }
}

/// Tan loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn tan_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).tan();
    }
}

/// Arcsin loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn asin_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).asin();
    }
}

/// Arccos loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn acos_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).acos();
    }
}

/// Arctan loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn atan_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).atan();
    }
}

// Hyperbolic functions - Double precision

/// Sinh loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn sinh_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).sinh();
    }
}

/// Cosh loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn cosh_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).cosh();
    }
}

/// Tanh loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn tanh_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).tanh();
    }
}

// Exponential and logarithmic functions - Double precision

/// Exp loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn exp_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).exp();
    }
}

/// Log (natural logarithm) loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn log_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).ln();
    }
}

/// Log10 loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn log10_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).log10();
    }
}

/// Log2 loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn log2_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).log2();
    }
}

/// Sqrt loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn sqrt_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).sqrt();
    }
}

// Absolute value and sign functions - Double precision

/// Abs (absolute value) loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn abs_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).abs();
    }
}

/// Sign loop for double precision (returns -1.0, 0.0, or 1.0)
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn sign_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        let val = *in_ptr.add(i * stride_in);
        *out_ptr.add(i * stride_out) = if val > 0.0 {
            1.0
        } else if val < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

// Rounding functions - Double precision

/// Floor loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn floor_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).floor();
    }
}

/// Ceil loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn ceil_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).ceil();
    }
}

/// Round loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn round_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).round();
    }
}

/// Trunc loop for double precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f64` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f64` values
/// * The input and output slices must not overlap
pub unsafe fn trunc_loop_double(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f64;
    let out_ptr = output as *mut f64;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).trunc();
    }
}

// Float precision versions (using f32)

/// Sin loop for float precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f32` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f32` values
/// * The input and output slices must not overlap
pub unsafe fn sin_loop_float(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f32;
    let out_ptr = output as *mut f32;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).sin();
    }
}

/// Exp loop for float precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f32` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f32` values
/// * The input and output slices must not overlap
pub unsafe fn exp_loop_float(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f32;
    let out_ptr = output as *mut f32;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).exp();
    }
}

/// Log loop for float precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f32` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f32` values
/// * The input and output slices must not overlap
pub unsafe fn log_loop_float(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f32;
    let out_ptr = output as *mut f32;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).ln();
    }
}

/// Sqrt loop for float precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f32` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f32` values
/// * The input and output slices must not overlap
pub unsafe fn sqrt_loop_float(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f32;
    let out_ptr = output as *mut f32;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).sqrt();
    }
}

/// Abs loop for float precision
///
/// # Safety
///
/// * `input` must point to a valid slice of `count * stride_in` bytes containing `f32` values
/// * `output` must point to a valid slice of `count * stride_out` bytes for `f32` values
/// * The input and output slices must not overlap
pub unsafe fn abs_loop_float(
    input: *const u8,
    output: *mut u8,
    count: usize,
    stride_in: usize,
    stride_out: usize,
) {
    let in_ptr = input as *const f32;
    let out_ptr = output as *mut f32;
    
    for i in 0..count {
        *out_ptr.add(i * stride_out) = (*in_ptr.add(i * stride_in)).abs();
    }
}

