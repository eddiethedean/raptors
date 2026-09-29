//! Scalar kernels and dtype-loop selection for the numeric ufunc layer.
//!
//! All array traversal and output mutation is performed by the binding over
//! checked [`crate::View`] accessors. These functions operate only on copied
//! scalar values and never form typed references into array storage.

use crate::{DType, Scalar, StorageError};
use num_complex::Complex;
use raptors_longdouble as native_longdouble;
use std::cmp::Ordering;

#[path = "ufunc_loop_resolver.rs"]
mod loop_resolver;
#[path = "ufunc_signatures.rs"]
mod type_signatures;

pub fn signatures(name: &str) -> Vec<String> {
    type_signatures::signatures(name)
}

pub const TOP_LEVEL_UFUNC_NAMES: &[&str] = &[
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "equal",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "pow",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
];

pub fn input_count(name: &str) -> usize {
    match name {
        "add"
        | "subtract"
        | "multiply"
        | "divide"
        | "true_divide"
        | "floor_divide"
        | "mod"
        | "remainder"
        | "fmod"
        | "power"
        | "pow"
        | "float_power"
        | "divmod"
        | "maximum"
        | "minimum"
        | "fmax"
        | "fmin"
        | "copysign"
        | "hypot"
        | "arctan2"
        | "atan2"
        | "heaviside"
        | "nextafter"
        | "gcd"
        | "lcm"
        | "ldexp"
        | "bitwise_and"
        | "bitwise_or"
        | "bitwise_xor"
        | "bitwise_left_shift"
        | "bitwise_right_shift"
        | "left_shift"
        | "right_shift"
        | "equal"
        | "not_equal"
        | "less"
        | "less_equal"
        | "greater"
        | "greater_equal"
        | "logical_and"
        | "logical_or"
        | "logical_xor"
        | "logaddexp"
        | "logaddexp2" => 2,
        _ => 1,
    }
}

pub fn output_count(name: &str) -> usize {
    match name {
        "divmod" | "frexp" | "modf" => 2,
        _ => 1,
    }
}

pub fn identity(name: &str) -> Option<Scalar> {
    match name {
        "add" | "bitwise_or" | "bitwise_xor" | "gcd" | "hypot" => Some(Scalar::Int64(0)),
        "logical_or" | "logical_xor" => Some(Scalar::Bool(false)),
        "multiply" => Some(Scalar::Int64(1)),
        "logical_and" => Some(Scalar::Bool(true)),
        "bitwise_and" => Some(Scalar::Int64(-1)),
        "logaddexp" | "logaddexp2" => Some(Scalar::Float64(f64::NEG_INFINITY)),
        _ => None,
    }
}

/// Return output dtypes for one ufunc loop, or an error when the numeric dtype
/// combination is not supported by that loop.
pub fn output_dtypes(
    name: &str,
    left: DType,
    right: Option<DType>,
) -> Result<Vec<DType>, StorageError> {
    let inputs = if input_count(name) == 2 {
        vec![left, right.ok_or(StorageError::InvalidScalar)?]
    } else {
        vec![left]
    };
    resolve_loop(name, &inputs).map(|(_, outputs)| outputs)
}

/// Resolve the pinned NumPy loop for already-selected input dtypes. The result
/// contains the loop input dtypes followed by the output dtypes.
pub fn resolve_loop(
    name: &str,
    inputs: &[DType],
) -> Result<(Vec<DType>, Vec<DType>), StorageError> {
    let canonical = signature_name(name);
    let nin = input_count(canonical);
    if inputs.len() != nin {
        return Err(StorageError::InvalidScalar);
    }
    if matches!(canonical, "logical_and" | "logical_or" | "logical_xor") {
        // NumPy's logical ufuncs accept numeric operands of different kinds
        // and apply truth testing before producing boolean output. Their
        // type signatures expose same-type inner loops, but using
        // resolve_dtypes' bool/bool result here would reject valid mixed
        // inputs under the default same_kind rule.
        return Ok((inputs.to_vec(), vec![DType::Bool]));
    }
    let resolved = loop_resolver::resolve(canonical, inputs).ok_or(StorageError::InvalidScalar)?;
    let mut loop_inputs = resolved[..nin].to_vec();
    let mut outputs = resolved[nin..].to_vec();

    // The checked-in loop table is generated on macOS ARM64, where C
    // longdouble is an alias for binary64. Preserve its loop choices there,
    // but restore the wider dtype promotion NumPy uses on targets with a real
    // extended longdouble format.
    if DType::LongDouble.itemsize() > DType::Float64.itemsize()
        && inputs
            .iter()
            .any(|dtype| matches!(dtype, DType::LongDouble | DType::ComplexLongDouble))
    {
        for dtype in loop_inputs.iter_mut().chain(outputs.iter_mut()) {
            *dtype = match dtype.kind() {
                "f" => DType::LongDouble,
                "c" => DType::ComplexLongDouble,
                _ => *dtype,
            };
        }
    }

    Ok((loop_inputs, outputs))
}

fn signature_name(name: &str) -> &str {
    match name {
        "abs" => "absolute",
        "acos" => "arccos",
        "acosh" => "arccosh",
        "asin" => "arcsin",
        "asinh" => "arcsinh",
        "atan" => "arctan",
        "atan2" => "arctan2",
        "atanh" => "arctanh",
        "bitwise_invert" | "bitwise_not" => "invert",
        "bitwise_left_shift" => "left_shift",
        "bitwise_right_shift" => "right_shift",
        "conj" => "conjugate",
        "true_divide" => "divide",
        "mod" => "remainder",
        "pow" => "power",
        _ => name,
    }
}

pub fn unary(name: &str, value: Scalar, output: DType) -> Result<Vec<Scalar>, StorageError> {
    if native_longdouble::has_extended_native() {
        if matches!(name, "frexp" | "modf") && output == DType::LongDouble {
            let text = long_double_text(&value)?;
            if name == "frexp" {
                if let Some((fraction, exponent)) = native_longdouble::frexp(&text) {
                    return Ok(vec![Scalar::LongDouble(fraction), Scalar::Int32(exponent)]);
                }
            } else if let Some((fraction, integral)) = native_longdouble::modf(&text) {
                return Ok(vec![
                    Scalar::LongDouble(fraction),
                    Scalar::LongDouble(integral),
                ]);
            }
        }

        if matches!(name, "isfinite" | "isinf" | "isnan") {
            match &value {
                Scalar::LongDouble(text) => {
                    let result = match name {
                        "isfinite" => native_longdouble::classify(text, 4),
                        "isinf" => native_longdouble::classify(text, 1),
                        _ => native_longdouble::classify(text, 0),
                    };
                    if let Some(result) = result {
                        return Ok(vec![Scalar::Bool(result)]);
                    }
                }
                Scalar::ComplexLongDouble(real, imag) => {
                    let real_infinite = native_longdouble::classify(real, 1).unwrap_or(false);
                    let imag_infinite = native_longdouble::classify(imag, 1).unwrap_or(false);
                    let real_nan = native_longdouble::classify(real, 0).unwrap_or(false);
                    let imag_nan = native_longdouble::classify(imag, 0).unwrap_or(false);
                    let result = match name {
                        "isfinite" => !(real_infinite || imag_infinite || real_nan || imag_nan),
                        "isinf" => real_infinite || imag_infinite,
                        _ => real_nan || imag_nan,
                    };
                    return Ok(vec![Scalar::Bool(result)]);
                }
                _ => {}
            }
        }

        if name == "signbit" {
            if let Scalar::LongDouble(text) = &value {
                if let Some(result) = native_longdouble::classify(text, 2) {
                    return Ok(vec![Scalar::Bool(result)]);
                }
            }
        }

        if output == DType::LongDouble {
            if value.dtype().kind() == "c" {
                let (real, imag) = complex_long_double_parts(&value)?;
                if let Some((result, _)) = native_longdouble::unary_complex(name, &real, &imag) {
                    return Ok(vec![Scalar::LongDouble(result)]);
                }
            } else {
                let text = long_double_text(&value)?;
                if let Some(result) = native_longdouble::unary_real(name, &text) {
                    return Ok(vec![Scalar::LongDouble(result)]);
                }
            }
        } else if output == DType::ComplexLongDouble {
            let (real, imag) = complex_long_double_parts(&value)?;
            if let Some((real, imag)) = native_longdouble::unary_complex(name, &real, &imag) {
                return Ok(vec![Scalar::ComplexLongDouble(real, imag)]);
            }
        }
    }

    let result = match name {
        "negative" if value.dtype().kind() == "c" => unary_float_or_complex(name, value, output)?,
        "negative" => integer_unary(&value, output, i128::wrapping_neg, |x| -x)?,
        "positive" => value.cast(output)?,
        "fabs" => Scalar::Float64(value.as_f64()?.abs()).cast(output)?,
        "absolute" | "abs" => absolute(&value, output)?,
        "square" if is_integer(value.dtype()) => {
            integer_unary(&value, output, |x| x.wrapping_mul(x), |x| x * x)?
        }
        "floor" | "ceil" | "trunc" if is_integer(value.dtype()) => value.cast(output)?,
        "reciprocal" if is_integer(value.dtype()) => integer_reciprocal(&value, output)?,
        "conj" | "conjugate" => conjugate(value, output)?,
        "invert" | "bitwise_not" | "bitwise_invert" => bitwise_invert(value, output)?,
        "bitwise_count" => bitwise_count(value, output)?,
        "logical_not" => Scalar::Bool(!value.truthy()),
        "isfinite" => Scalar::Bool(scalar_is_finite(&value)),
        "isinf" => Scalar::Bool(scalar_is_infinite(&value)),
        "isnan" => Scalar::Bool(scalar_is_nan(&value)),
        "signbit" => Scalar::Bool(value.as_f64()?.is_sign_negative()),
        "sign" => sign(value, output)?,
        "frexp" => return frexp(value, output),
        "modf" => return modf(value, output),
        "reciprocal" => unary_float_or_complex(name, value, output)?,
        _ => unary_float_or_complex(name, value, output)?,
    };
    Ok(vec![result])
}

pub fn binary(
    name: &str,
    left: Scalar,
    right: Scalar,
    output: DType,
) -> Result<Vec<Scalar>, StorageError> {
    if is_comparison(name) {
        return Ok(vec![Scalar::Bool(compare(name, &left, &right)?)]);
    }
    if is_predicate(name) {
        let value = match name {
            "logical_and" => left.truthy() && right.truthy(),
            "logical_or" => left.truthy() || right.truthy(),
            "logical_xor" => left.truthy() ^ right.truthy(),
            _ => return Err(StorageError::InvalidScalar),
        };
        return Ok(vec![Scalar::Bool(value)]);
    }
    if is_bitwise(name) {
        return Ok(vec![bitwise_binary(name, left, right, output)?]);
    }
    if native_longdouble::has_extended_native() && output == DType::LongDouble {
        let left_text = long_double_text(&left)?;
        let right_text = long_double_text(&right)?;
        if let Some((first, second)) = native_longdouble::binary_real(name, &left_text, &right_text)
        {
            return if name == "divmod" {
                Ok(vec![Scalar::LongDouble(first), Scalar::LongDouble(second)])
            } else {
                Ok(vec![Scalar::LongDouble(first)])
            };
        }
    } else if native_longdouble::has_extended_native() && output == DType::ComplexLongDouble {
        let (left_real, left_imag) = complex_long_double_parts(&left)?;
        let (right_real, right_imag) = complex_long_double_parts(&right)?;
        if let Some((real, imag)) = native_longdouble::binary_complex(
            name,
            &left_real,
            &left_imag,
            &right_real,
            &right_imag,
        ) {
            return Ok(vec![Scalar::ComplexLongDouble(real, imag)]);
        }
    }
    if name == "divmod" {
        let q = binary("floor_divide", left.clone(), right.clone(), output)?;
        let r = binary("remainder", left, right, output)?;
        return Ok(vec![q[0].clone(), r[0].clone()]);
    }
    if is_integer(output) && left.dtype().kind() != "f" && left.dtype().kind() != "c" {
        return integer_binary(name, &left, &right, output);
    }
    if name == "ldexp" {
        let exponent = signed_value(&right)?;
        if output == DType::Float16 {
            let value = scale_pow2_f32(left.as_f64()? as f32, exponent);
            return Ok(vec![Scalar::Float32(value).cast(output)?]);
        }
        if output == DType::Float32 {
            if let Scalar::Float32(value) = left {
                return Ok(vec![Scalar::Float32(scale_pow2_f32(value, exponent))]);
            }
        }
        return Ok(vec![Scalar::Float64(scale_pow2_f64(
            left.as_f64()?,
            exponent,
        ))
        .cast(output)?]);
    }
    if output == DType::Float16 {
        let a = left.as_f64()? as f32;
        let b = right.as_f64()? as f32;
        if name == "nextafter" {
            let value = next_after_dtype(a as f64, b as f64, output);
            return Ok(vec![Scalar::Float64(value).cast(output)?]);
        }
        return Ok(vec![
            Scalar::Float32(binary_float32(name, a, b)?).cast(output)?
        ]);
    }
    if output == DType::Float32 {
        if let (Scalar::Float32(a), Scalar::Float32(b)) = (&left, &right) {
            return Ok(vec![Scalar::Float32(binary_float32(name, *a, *b)?)]);
        }
    }
    if output == DType::Complex64 {
        if let (Scalar::Complex64(ar, ai), Scalar::Complex64(br, bi)) = (&left, &right) {
            return Ok(vec![binary_complex64(
                name,
                Complex::new(*ar, *ai),
                Complex::new(*br, *bi),
            )?]);
        }
    }
    let complex = left.dtype().kind() == "c" || right.dtype().kind() == "c";
    let (left, right) = if complex {
        (left.as_complex()?, right.as_complex()?)
    } else {
        ((left.as_f64()?, 0.0), (right.as_f64()?, 0.0))
    };
    if output.kind() == "c" {
        let a = Complex::new(left.0, left.1);
        let b = Complex::new(right.0, right.1);
        let result = match name {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" | "true_divide" => a / b,
            "power" | "pow" | "float_power" => {
                if a.re == 0.0 && a.im == 0.0 && b.re < 0.0 {
                    Complex::new(f64::NAN, f64::NAN)
                } else {
                    a.powc(b)
                }
            }
            "maximum" | "minimum" | "fmax" | "fmin" => complex_extreme(name, a, b)?,
            "mod" | "remainder" | "fmod" | "floor_divide" | "heaviside" => {
                return Err(StorageError::InvalidScalar)
            }
            _ => return Err(StorageError::InvalidScalar),
        };
        return Ok(vec![Scalar::Complex128(result.re, result.im).cast(output)?]);
    }
    let (a, b) = (left.0, right.0);
    let result = match name {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" | "true_divide" => a / b,
        "floor_divide" => floor_divide(a, b),
        "mod" | "remainder" => float_remainder(a, b),
        "fmod" => a % b,
        "power" | "pow" | "float_power" => a.powf(b),
        "maximum" => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a >= b {
                a
            } else {
                b
            }
        }
        "minimum" => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a <= b {
                a
            } else {
                b
            }
        }
        "fmax" => {
            if b.is_nan() {
                a
            } else if a.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a >= b {
                a
            } else {
                b
            }
        }
        "fmin" => {
            if b.is_nan() {
                a
            } else if a.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a <= b {
                a
            } else {
                b
            }
        }
        "copysign" => a.copysign(b),
        "hypot" => a.hypot(b),
        "arctan2" | "atan2" => a.atan2(b),
        "heaviside" => {
            if a.is_nan() {
                a
            } else if a < 0.0 {
                0.0
            } else if a > 0.0 {
                1.0
            } else {
                b
            }
        }
        "logaddexp" => logaddexp(a, b),
        "logaddexp2" => logaddexp2(a, b),
        "nextafter" => next_after_dtype(a, b, output),
        "gcd" => gcd(a as i128, b as i128) as f64,
        "lcm" => lcm(a as i128, b as i128) as f64,
        "ldexp" => a * 2f64.powf(b),
        _ => return Err(StorageError::InvalidScalar),
    };
    if name == "divmod" {
        return Ok(vec![
            Scalar::Float64((a / b).floor()).cast(output)?,
            Scalar::Float64(float_remainder(a, b)).cast(output)?,
        ]);
    }
    Ok(vec![Scalar::Float64(result).cast(output)?])
}

fn is_integer(dtype: DType) -> bool {
    matches!(dtype.kind(), "i" | "u" | "b")
}

fn long_double_text(value: &Scalar) -> Result<String, StorageError> {
    match value.cast(DType::LongDouble)? {
        Scalar::LongDouble(value) => Ok(value),
        _ => Err(StorageError::InvalidScalar),
    }
}

fn complex_long_double_parts(value: &Scalar) -> Result<(String, String), StorageError> {
    match value.cast(DType::ComplexLongDouble)? {
        Scalar::ComplexLongDouble(real, imag) => Ok((real, imag)),
        _ => Err(StorageError::InvalidScalar),
    }
}
fn is_comparison(name: &str) -> bool {
    matches!(
        name,
        "equal" | "not_equal" | "less" | "less_equal" | "greater" | "greater_equal"
    )
}
fn is_predicate(name: &str) -> bool {
    matches!(name, "logical_and" | "logical_or" | "logical_xor")
}
fn is_bitwise(name: &str) -> bool {
    matches!(
        name,
        "bitwise_and"
            | "bitwise_or"
            | "bitwise_xor"
            | "bitwise_left_shift"
            | "bitwise_right_shift"
            | "left_shift"
            | "right_shift"
            | "invert"
            | "bitwise_not"
            | "bitwise_invert"
            | "bitwise_count"
    )
}
fn signed_value(value: &Scalar) -> Result<i128, StorageError> {
    Ok(match value {
        Scalar::Bool(v) => i128::from(*v),
        Scalar::Int8(v) => *v as i128,
        Scalar::Int16(v) => *v as i128,
        Scalar::Int32(v) => *v as i128,
        Scalar::Int64(v) => *v as i128,
        Scalar::UInt8(v) => *v as i128,
        Scalar::UInt16(v) => *v as i128,
        Scalar::UInt32(v) => *v as i128,
        Scalar::UInt64(v) => *v as i128,
        _ => return Err(StorageError::InvalidScalar),
    })
}
fn unsigned_value(value: &Scalar) -> Result<u128, StorageError> {
    Ok(match value {
        Scalar::Bool(v) => u128::from(*v),
        Scalar::Int8(v) => *v as u8 as u128,
        Scalar::Int16(v) => *v as u16 as u128,
        Scalar::Int32(v) => *v as u32 as u128,
        Scalar::Int64(v) => *v as u64 as u128,
        Scalar::UInt8(v) => *v as u128,
        Scalar::UInt16(v) => *v as u128,
        Scalar::UInt32(v) => *v as u128,
        Scalar::UInt64(v) => *v as u128,
        _ => return Err(StorageError::InvalidScalar),
    })
}
fn signed_result(value: i128, dtype: DType) -> Scalar {
    match dtype {
        DType::Int8 => Scalar::Int8(value as i8),
        DType::Int16 => Scalar::Int16(value as i16),
        DType::Int32 => Scalar::Int32(value as i32),
        DType::Int64 => Scalar::Int64(value as i64),
        DType::Bool => Scalar::Bool(value != 0),
        DType::UInt8 => Scalar::UInt8(value as u8),
        DType::UInt16 => Scalar::UInt16(value as u16),
        DType::UInt32 => Scalar::UInt32(value as u32),
        DType::UInt64 => Scalar::UInt64(value as u64),
        _ => unreachable!(),
    }
}
fn unsigned_result(value: u128, dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(value != 0),
        DType::Int8 => Scalar::Int8(value as i8),
        DType::UInt8 => Scalar::UInt8(value as u8),
        DType::Int16 => Scalar::Int16(value as i16),
        DType::UInt16 => Scalar::UInt16(value as u16),
        DType::Int32 => Scalar::Int32(value as i32),
        DType::UInt32 => Scalar::UInt32(value as u32),
        DType::Int64 => Scalar::Int64(value as i64),
        DType::UInt64 => Scalar::UInt64(value as u64),
        _ => unreachable!(),
    }
}
fn int_width(dtype: DType) -> u32 {
    (dtype.itemsize() * 8) as u32
}

fn bitwise_binary(
    name: &str,
    left: Scalar,
    right: Scalar,
    dtype: DType,
) -> Result<Scalar, StorageError> {
    let a = unsigned_value(&left)?;
    let b = unsigned_value(&right)?;
    let width = int_width(dtype).min(64);
    let mask = if width == 64 {
        u64::MAX as u128
    } else {
        (1u128 << width) - 1
    };
    let signed_shift = signed_value(&right).ok();
    let unsigned_shift = unsigned_value(&right).ok();
    let negative_shift = signed_shift.is_some_and(|shift| shift < 0);
    let shift = signed_shift
        .filter(|shift| *shift >= 0)
        .map(|shift| shift as u128)
        .or(unsigned_shift)
        .unwrap_or(u128::MAX);
    let shift = shift.min(width as u128) as u32;
    let value = match name {
        "bitwise_and" => a & b,
        "bitwise_or" => a | b,
        "bitwise_xor" => a ^ b,
        "left_shift" | "bitwise_left_shift" => {
            if negative_shift || shift == width {
                0
            } else {
                a.wrapping_shl(shift)
            }
        }
        "right_shift" | "bitwise_right_shift" => {
            if negative_shift || shift == width {
                if is_signed_dtype(dtype) && signed_value(&left)? < 0 {
                    mask
                } else {
                    0
                }
            } else if is_signed_dtype(dtype) {
                (signed_value(&left)? >> shift) as u64 as u128
            } else {
                a >> shift
            }
        }
        _ => return Err(StorageError::InvalidScalar),
    } & mask;
    Ok(unsigned_result(value, dtype))
}

fn is_signed_dtype(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
    )
}

fn bitwise_invert(value: Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if dtype == DType::Bool {
        return Ok(Scalar::Bool(!value.truthy()));
    }
    let mask = if int_width(dtype) >= 64 {
        u64::MAX as u128
    } else {
        (1u128 << int_width(dtype)) - 1
    };
    Ok(unsigned_result(!unsigned_value(&value)? & mask, dtype))
}
fn bitwise_count(value: Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    let count = match value {
        Scalar::Bool(value) => u32::from(value),
        Scalar::Int8(value) => value.unsigned_abs().count_ones(),
        Scalar::Int16(value) => value.unsigned_abs().count_ones(),
        Scalar::Int32(value) => value.unsigned_abs().count_ones(),
        Scalar::Int64(value) => value.unsigned_abs().count_ones(),
        Scalar::UInt8(value) => value.count_ones(),
        Scalar::UInt16(value) => value.count_ones(),
        Scalar::UInt32(value) => value.count_ones(),
        Scalar::UInt64(value) => value.count_ones(),
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(unsigned_result(count as u128, dtype))
}
fn absolute(value: &Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if let Scalar::Complex64(re, im) = value {
        return Scalar::Float32(re.hypot(*im)).cast(dtype);
    }
    if value.dtype().kind() == "c" {
        return Scalar::Float64(value.as_complex()?.0.hypot(value.as_complex()?.1)).cast(dtype);
    }
    if is_integer(value.dtype()) {
        let raw = signed_value(value)?;
        return Ok(signed_result(raw.wrapping_abs(), dtype));
    }
    Scalar::Float64(value.as_f64()?.abs()).cast(dtype)
}
fn conjugate(value: Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if value.dtype().kind() != "c" {
        return value.cast(dtype);
    }
    let (re, im) = value.as_complex()?;
    Scalar::Complex128(re, -im).cast(dtype)
}
fn sign(value: Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if let Scalar::Float16(x) = &value {
        let x = *x;
        let result = if x.is_nan() {
            x
        } else if x == 0.0 {
            0.0
        } else if x.is_sign_negative() {
            -1.0
        } else {
            1.0
        };
        return Scalar::Float16(result).cast(dtype);
    }
    if let Scalar::Complex64(re, im) = value {
        let magnitude = re.hypot(im);
        let result = if magnitude.is_nan() {
            (f32::NAN, f32::NAN)
        } else if magnitude.is_infinite() {
            if re.is_infinite() {
                if im.is_infinite() {
                    (f32::NAN, f32::NAN)
                } else {
                    (if re.is_sign_negative() { -1.0 } else { 1.0 }, 0.0)
                }
            } else {
                (0.0, if im.is_sign_negative() { -1.0 } else { 1.0 })
            }
        } else if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            (re / magnitude, im / magnitude)
        };
        return Scalar::Complex64(result.0, result.1).cast(dtype);
    }
    if value.dtype().kind() == "c" {
        let (re, im) = value.as_complex()?;
        let magnitude = re.hypot(im);
        let result = if magnitude.is_nan() {
            (f64::NAN, f64::NAN)
        } else if magnitude.is_infinite() {
            if re.is_infinite() {
                if im.is_infinite() {
                    (f64::NAN, f64::NAN)
                } else {
                    (if re.is_sign_negative() { -1.0 } else { 1.0 }, 0.0)
                }
            } else {
                (0.0, if im.is_sign_negative() { -1.0 } else { 1.0 })
            }
        } else if magnitude == 0.0 {
            (0.0, 0.0)
        } else {
            (re / magnitude, im / magnitude)
        };
        return Scalar::Complex128(result.0, result.1).cast(dtype);
    }
    let x = value.as_f64()?;
    Scalar::Float64(if x.is_nan() {
        x
    } else if x == 0.0 {
        0.0
    } else {
        x.signum()
    })
    .cast(dtype)
}

fn unary_float_or_complex(name: &str, value: Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if dtype == DType::Float16 {
        if let Scalar::Float16(value) = &value {
            let result = if name == "spacing" {
                spacing_f16(*value)
            } else {
                unary_float32(name, *value)?
            };
            return Scalar::Float32(result).cast(dtype);
        }
    }
    if dtype == DType::Float32 {
        if let Scalar::Float32(value) = value {
            return Ok(Scalar::Float32(unary_float32(name, value)?));
        }
    }
    if dtype == DType::Complex64 {
        if let Scalar::Complex64(re, im) = value {
            return unary_complex64(name, Complex::new(re, im));
        }
    }
    if value.dtype().kind() == "c" {
        let (re, im) = value.as_complex()?;
        let z = Complex::new(re, im);
        let result = match name {
            "acos" | "arccos" => complex_acos_f64(z),
            "acosh" | "arccosh" => z.acosh(),
            "asin" | "arcsin" => complex_asin_f64(z),
            "asinh" | "arcsinh" => z.asinh(),
            "atan" | "arctan" => complex_atan_f64(z),
            "atanh" | "arctanh" => complex_atanh_f64(z),
            "cos" => z.cos(),
            "cosh" => z.cosh(),
            "exp" => z.exp(),
            "exp2" => Complex::new(2.0, 0.0).powc(z),
            "expm1" => complex_expm1_f64(z),
            "rint" => Complex::new(re.round_ties_even(), im.round_ties_even()),
            "log" => z.ln(),
            "log1p" => complex_log1p_f64(z),
            "log10" => z.log10(),
            "log2" => z.log(2.0),
            "sin" => z.sin(),
            "sinh" => z.sinh(),
            "sqrt" => z.sqrt(),
            "tan" => z.tan(),
            "tanh" => z.tanh(),
            "reciprocal" => Complex::new(1.0, 0.0) / z,
            "square" => z * z,
            "positive" => z,
            "negative" => -z,
            "conj" | "conjugate" => z.conj(),
            _ => return Err(StorageError::InvalidScalar),
        };
        return Scalar::Complex128(result.re, result.im).cast(dtype);
    }
    let x = value.as_f64()?;
    let result = match name {
        "acos" | "arccos" => x.acos(),
        "acosh" | "arccosh" => x.acosh(),
        "asin" | "arcsin" => x.asin(),
        "asinh" | "arcsinh" => x.asinh(),
        "atan" | "arctan" => x.atan(),
        "atanh" | "arctanh" => x.atanh(),
        "cbrt" => x.cbrt(),
        "ceil" => x.ceil(),
        "cos" => x.cos(),
        "cosh" => x.cosh(),
        "deg2rad" | "radians" => x.to_radians(),
        "degrees" | "rad2deg" => x.to_degrees(),
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "expm1" => x.exp_m1(),
        "floor" => x.floor(),
        "log" => x.ln(),
        "log10" => x.log10(),
        "log1p" => x.ln_1p(),
        "log2" => x.log2(),
        "negative" => -x,
        "positive" => x,
        "reciprocal" => 1.0 / x,
        "rint" => x.round_ties_even(),
        "sign" => x.signum(),
        "sin" => x.sin(),
        "sinh" => x.sinh(),
        "sqrt" => x.sqrt(),
        "square" => x * x,
        "tan" => x.tan(),
        "tanh" => x.tanh(),
        "trunc" => x.trunc(),
        "spacing" if !x.is_finite() => f64::NAN,
        "spacing" => {
            let direction = if x.is_sign_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            next_after_dtype(x, direction, dtype) - x
        }
        _ => return Err(StorageError::InvalidScalar),
    };
    Scalar::Float64(result).cast(dtype)
}

fn unary_float32(name: &str, x: f32) -> Result<f32, StorageError> {
    Ok(match name {
        "acos" | "arccos" => x.acos(),
        "acosh" | "arccosh" => x.acosh(),
        "asin" | "arcsin" => x.asin(),
        "asinh" | "arcsinh" => x.asinh(),
        "atan" | "arctan" => x.atan(),
        "atanh" | "arctanh" => x.atanh(),
        "cbrt" => x.cbrt(),
        "ceil" => x.ceil(),
        "cos" => x.cos(),
        "cosh" => x.cosh(),
        "deg2rad" | "radians" => x.to_radians(),
        "degrees" | "rad2deg" => x.to_degrees(),
        "exp" => x.exp(),
        "exp2" => x.exp2(),
        "expm1" => x.exp_m1(),
        "floor" => x.floor(),
        "log" => x.ln(),
        "log10" => x.log10(),
        "log1p" => x.ln_1p(),
        "log2" => x.log2(),
        "negative" => -x,
        "positive" => x,
        "reciprocal" => 1.0 / x,
        "rint" => x.round_ties_even(),
        "sin" => x.sin(),
        "sinh" => x.sinh(),
        "sqrt" => x.sqrt(),
        "square" => x * x,
        "tan" => x.tan(),
        "tanh" => x.tanh(),
        "trunc" => x.trunc(),
        "spacing" if !x.is_finite() => f32::NAN,
        "spacing" => next_after_f32(x, f32::INFINITY.copysign(x)) - x,
        _ => return Err(StorageError::InvalidScalar),
    })
}

fn unary_complex64(name: &str, z: Complex<f32>) -> Result<Scalar, StorageError> {
    let result = match name {
        "acos" | "arccos" => complex_acos_f32(z),
        "acosh" | "arccosh" => z.acosh(),
        "asin" | "arcsin" => complex_asin_f32(z),
        "asinh" | "arcsinh" => z.asinh(),
        "atan" | "arctan" => complex_atan_f32(z),
        "atanh" | "arctanh" => complex_atanh_f32(z),
        "cos" => z.cos(),
        "cosh" => z.cosh(),
        "exp" => z.exp(),
        "exp2" => Complex::new(2.0, 0.0).powc(z),
        "expm1" => complex_expm1_f32(z),
        "rint" => Complex::new(z.re.round_ties_even(), z.im.round_ties_even()),
        "log" => z.ln(),
        "log1p" => complex_log1p_f32(z),
        "log10" => z.log10(),
        "log2" => z.log(2.0),
        "sin" => z.sin(),
        "sinh" => z.sinh(),
        "sqrt" => z.sqrt(),
        "tan" => z.tan(),
        "tanh" => z.tanh(),
        "reciprocal" => Complex::new(1.0, 0.0) / z,
        "square" => z * z,
        "positive" => z,
        "negative" => -z,
        "conj" | "conjugate" => z.conj(),
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(Scalar::Complex64(result.re, result.im))
}

fn complex_asin_f32(z: Complex<f32>) -> Complex<f32> {
    let result = complex_asin_f64(Complex::new(z.re as f64, z.im as f64));
    Complex::new(result.re as f32, result.im as f32)
}

fn complex_acos_f32(z: Complex<f32>) -> Complex<f32> {
    let result = complex_asin_f64(Complex::new(z.re as f64, z.im as f64));
    Complex::new(
        (std::f64::consts::FRAC_PI_2 - result.re) as f32,
        (-result.im) as f32,
    )
}

fn complex_atan_f32(z: Complex<f32>) -> Complex<f32> {
    let result = complex_atan_f64(Complex::new(z.re as f64, z.im as f64));
    Complex::new(result.re as f32, result.im as f32)
}

fn complex_atanh_f32(z: Complex<f32>) -> Complex<f32> {
    let result = complex_atanh_f64(Complex::new(z.re as f64, z.im as f64));
    Complex::new(result.re as f32, result.im as f32)
}

fn complex_atan_f64(z: Complex<f64>) -> Complex<f64> {
    if !z.re.is_finite()
        || !z.im.is_finite()
        || z.re.abs() >= f64::MAX.sqrt() / 4.0
        || z.im.abs() >= f64::MAX.sqrt() / 4.0
    {
        return z.atan();
    }

    let denominator = 1.0 - z.re * z.re - z.im * z.im;
    let magnitude = z.re * z.re + (z.im - 1.0) * (z.im - 1.0);
    let imaginary = if magnitude == 0.0 {
        0.25 * ((z.re * z.re + (z.im + 1.0) * (z.im + 1.0)).ln() - magnitude.ln())
    } else {
        0.25 * (4.0 * z.im / magnitude).ln_1p()
    };
    Complex::new((2.0 * z.re).atan2(denominator) * 0.5, imaginary)
}

fn complex_atanh_f64(z: Complex<f64>) -> Complex<f64> {
    if !z.re.is_finite()
        || !z.im.is_finite()
        || z.re.abs() >= f64::MAX.sqrt() / 4.0
        || z.im.abs() >= f64::MAX.sqrt() / 4.0
    {
        return z.atanh();
    }

    let denominator = 1.0 - z.re * z.re - z.im * z.im;
    let magnitude = (1.0 - z.re) * (1.0 - z.re) + z.im * z.im;
    let real = if magnitude == 0.0 {
        0.25 * (((1.0 + z.re) * (1.0 + z.re) + z.im * z.im).ln() - magnitude.ln())
    } else {
        0.25 * (4.0 * z.re / magnitude).ln_1p()
    };
    Complex::new(real, (2.0 * z.im).atan2(denominator) * 0.5)
}

fn complex_asin_f64(z: Complex<f64>) -> Complex<f64> {
    if !z.re.is_finite() || !z.im.is_finite() {
        return z.asin();
    }

    // The sum-of-distances identity is stable for large finite values and
    // avoids the branch-cut sign loss in num-complex's general formula.
    let alpha = 0.5 * (z.re + 1.0).hypot(z.im) + 0.5 * (z.re - 1.0).hypot(z.im);
    let sine_real = (z.re / alpha).clamp(-1.0, 1.0);
    let real = sine_real.asin();

    // acosh(alpha) loses very small imaginary parts when alpha rounds to 1.
    // Recover those with sinh(v) = y / cos(u), using the already-resolved
    // real component u = asin(x / alpha).
    let cosine_real =
        (((alpha - z.re) / alpha).max(0.0) * ((alpha + z.re) / alpha).max(0.0)).sqrt();
    let imaginary_magnitude = if cosine_real > 0.0 {
        (z.im / cosine_real).asinh()
    } else {
        acosh_one_plus(alpha)
    };
    Complex::new(real, imaginary_magnitude.copysign(z.im))
}

fn complex_acos_f64(z: Complex<f64>) -> Complex<f64> {
    let result = complex_asin_f64(z);
    Complex::new(std::f64::consts::FRAC_PI_2 - result.re, -result.im)
}

fn acosh_one_plus(value: f64) -> f64 {
    if value <= 1.0 {
        return 0.0;
    }
    if value > f64::MAX.sqrt() * 0.5 {
        return value.ln() + std::f64::consts::LN_2;
    }
    let delta = value - 1.0;
    (delta + (delta * (value + 1.0)).sqrt()).ln_1p()
}

fn complex_expm1_f32(z: Complex<f32>) -> Complex<f32> {
    let sin_half_imag = (z.im * 0.5).sin();
    let imag = if z.im == 0.0 {
        z.im
    } else {
        z.re.exp() * z.im.sin()
    };
    Complex::new(
        z.re.exp_m1() * z.im.cos() - 2.0 * sin_half_imag * sin_half_imag,
        imag,
    )
}

fn complex_log1p_f32(z: Complex<f32>) -> Complex<f32> {
    let real = if z.re.hypot(z.im) < 0.5 {
        0.5 * (2.0 * z.re + z.re * z.re + z.im * z.im).ln_1p()
    } else {
        (1.0 + z.re).hypot(z.im).ln()
    };
    Complex::new(real, z.im.atan2(1.0 + z.re))
}

fn complex_expm1_f64(z: Complex<f64>) -> Complex<f64> {
    let sin_half_imag = (z.im * 0.5).sin();
    let imag = if z.im == 0.0 {
        z.im
    } else {
        z.re.exp() * z.im.sin()
    };
    Complex::new(
        z.re.exp_m1() * z.im.cos() - 2.0 * sin_half_imag * sin_half_imag,
        imag,
    )
}

fn complex_log1p_f64(z: Complex<f64>) -> Complex<f64> {
    let real = if z.re.hypot(z.im) < 0.5 {
        0.5 * (2.0 * z.re + z.re * z.re + z.im * z.im).ln_1p()
    } else {
        (1.0 + z.re).hypot(z.im).ln()
    };
    Complex::new(real, z.im.atan2(1.0 + z.re))
}

// Match NumPy 2.5.3's scalar tie handling for signed zero. Its x86 maximum and
// minimum kernels keep the right operand; AArch64 kernels select by sign. The
// fmax/fmin tie behavior also follows the platform's scalar math implementation.
fn extrema_zero_tie_selects_left(name: &str, left_is_negative: bool) -> bool {
    let _ = left_is_negative;
    match name {
        "maximum" => {
            #[cfg(target_arch = "aarch64")]
            {
                !left_is_negative
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                false
            }
        }
        "minimum" => {
            #[cfg(target_arch = "aarch64")]
            {
                left_is_negative
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                false
            }
        }
        "fmax" => {
            #[cfg(any(target_arch = "aarch64", target_os = "windows", target_env = "musl"))]
            {
                !left_is_negative
            }
            #[cfg(not(any(target_arch = "aarch64", target_os = "windows", target_env = "musl")))]
            {
                true
            }
        }
        "fmin" => {
            #[cfg(any(target_arch = "aarch64", target_os = "windows", target_env = "musl"))]
            {
                left_is_negative
            }
            #[cfg(not(any(target_arch = "aarch64", target_os = "windows", target_env = "musl")))]
            {
                true
            }
        }
        _ => true,
    }
}

fn binary_float32(name: &str, a: f32, b: f32) -> Result<f32, StorageError> {
    Ok(match name {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" | "true_divide" => a / b,
        "floor_divide" => floor_divide_f32(a, b),
        "remainder" | "mod" => float_remainder_f32(a, b),
        "fmod" => a % b,
        "power" | "pow" | "float_power" => a.powf(b),
        "maximum" => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a >= b {
                a
            } else {
                b
            }
        }
        "minimum" => {
            if a.is_nan() {
                a
            } else if b.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a <= b {
                a
            } else {
                b
            }
        }
        "fmax" => {
            if b.is_nan() {
                a
            } else if a.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a >= b {
                a
            } else {
                b
            }
        }
        "fmin" => {
            if b.is_nan() {
                a
            } else if a.is_nan() {
                b
            } else if a == 0.0 && b == 0.0 {
                if extrema_zero_tie_selects_left(name, a.is_sign_negative()) {
                    a
                } else {
                    b
                }
            } else if a <= b {
                a
            } else {
                b
            }
        }
        "copysign" => a.copysign(b),
        "hypot" => a.hypot(b),
        "arctan2" | "atan2" => a.atan2(b),
        "heaviside" => {
            if a.is_nan() {
                a
            } else if a < 0.0 {
                0.0
            } else if a > 0.0 {
                1.0
            } else {
                b
            }
        }
        "logaddexp" => logaddexp_f32(a, b),
        "logaddexp2" => logaddexp2_f32(a, b),
        "nextafter" => next_after_f32(a, b),
        "ldexp" => a * 2.0f32.powf(b),
        _ => return Err(StorageError::InvalidScalar),
    })
}

fn binary_complex64(name: &str, a: Complex<f32>, b: Complex<f32>) -> Result<Scalar, StorageError> {
    let result = match name {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" | "true_divide" => a / b,
        "power" | "pow" | "float_power" => {
            if a.re == 0.0 && a.im == 0.0 && b.re < 0.0 {
                Complex::new(f32::NAN, f32::NAN)
            } else {
                a.powc(b)
            }
        }
        "maximum" | "minimum" | "fmax" | "fmin" => complex_extreme_f32(name, a, b)?,
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(Scalar::Complex64(result.re, result.im))
}

fn float_remainder_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() || b == 0.0 || a.is_infinite() {
        return f32::NAN;
    }
    if b.is_infinite() {
        if a == 0.0 {
            return 0.0f32.copysign(b);
        }
        return if a.is_sign_negative() == b.is_sign_negative() {
            a
        } else {
            b
        };
    }
    let remainder = a % b;
    if remainder == 0.0 {
        return 0.0f32.copysign(b);
    }
    if remainder.is_sign_negative() != b.is_sign_negative() {
        remainder + b
    } else {
        remainder
    }
}

fn floor_divide_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if b == 0.0 {
        return a / b;
    }
    if a.is_infinite() {
        return f32::NAN;
    }
    if b.is_infinite() {
        if a == 0.0 {
            return 0.0f32.copysign(a / b);
        }
        return if a.is_sign_negative() != b.is_sign_negative() {
            -1.0
        } else {
            0.0
        };
    }
    let modulus = a % b;
    let mut division = (a - modulus) / b;
    if modulus != 0.0 && b.is_sign_negative() != modulus.is_sign_negative() {
        division -= 1.0;
    }
    if division != 0.0 {
        let mut quotient = division.floor();
        if division - quotient > 0.5 {
            quotient += 1.0;
        }
        quotient
    } else {
        0.0f32.copysign(a / b)
    }
}

fn logaddexp_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == f32::INFINITY || b == f32::INFINITY {
        return f32::INFINITY;
    }
    if a == f32::NEG_INFINITY {
        return if b == 0.0 { 0.0 } else { b };
    }
    if b == f32::NEG_INFINITY {
        return if a == 0.0 { 0.0 } else { a };
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

fn logaddexp2_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == f32::INFINITY || b == f32::INFINITY {
        return f32::INFINITY;
    }
    if a == f32::NEG_INFINITY {
        return if b == 0.0 { 0.0 } else { b };
    }
    if b == f32::NEG_INFINITY {
        return if a == 0.0 { 0.0 } else { a };
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp2().ln_1p() / std::f32::consts::LN_2
}

fn scale_pow2_f32(mut value: f32, mut exponent: i128) -> f32 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    if exponent > 512 {
        return f32::INFINITY.copysign(value);
    }
    if exponent < -512 {
        return 0.0f32.copysign(value);
    }
    while exponent > 127 {
        value *= 2.0f32.powi(127);
        exponent -= 127;
    }
    while exponent < -126 {
        value *= 2.0f32.powi(-126);
        exponent += 126;
    }
    value * 2.0f32.powi(exponent as i32)
}

fn scale_pow2_f64(mut value: f64, mut exponent: i128) -> f64 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }
    if exponent > 4096 {
        return f64::INFINITY.copysign(value);
    }
    if exponent < -4096 {
        return 0.0f64.copysign(value);
    }
    while exponent > 1023 {
        value *= 2.0f64.powi(1023);
        exponent -= 1023;
    }
    while exponent < -1022 {
        value *= 2.0f64.powi(-1022);
        exponent += 1022;
    }
    value * 2.0f64.powi(exponent as i32)
}

fn integer_reciprocal(value: &Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    if dtype.kind() == "u" || dtype == DType::Bool {
        let value = unsigned_value(value)?;
        Ok(unsigned_result(
            1u128.checked_div(value).unwrap_or(0),
            dtype,
        ))
    } else {
        let value = signed_value(value)?;
        Ok(signed_result(1i128.checked_div(value).unwrap_or(0), dtype))
    }
}

fn scalar_is_nan(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_nan();
    }
    match value {
        Scalar::Float16(x) | Scalar::Float32(x) => x.is_nan(),
        Scalar::Float64(x) => x.is_nan(),
        Scalar::Complex64(re, im) => re.is_nan() || im.is_nan(),
        Scalar::Complex128(re, im) => re.is_nan() || im.is_nan(),
        Scalar::LongDouble(x) => native_longdouble::classify(x, 0)
            .unwrap_or_else(|| x.parse::<f64>().map(|n| n.is_nan()).unwrap_or(false)),
        Scalar::ComplexLongDouble(re, im) => {
            native_longdouble::classify(re, 0)
                .unwrap_or_else(|| re.parse::<f64>().map(|n| n.is_nan()).unwrap_or(false))
                || native_longdouble::classify(im, 0)
                    .unwrap_or_else(|| im.parse::<f64>().map(|n| n.is_nan()).unwrap_or(false))
        }
        _ => false,
    }
}
fn scalar_is_infinite(value: &Scalar) -> bool {
    if matches!(
        value,
        Scalar::LongDouble(_) | Scalar::ComplexLongDouble(_, _)
    ) {
        return value.is_infinite();
    }
    match value {
        Scalar::Float16(x) | Scalar::Float32(x) => x.is_infinite(),
        Scalar::Float64(x) => x.is_infinite(),
        Scalar::Complex64(re, im) => re.is_infinite() || im.is_infinite(),
        Scalar::Complex128(re, im) => re.is_infinite() || im.is_infinite(),
        Scalar::LongDouble(x) => native_longdouble::classify(x, 1)
            .unwrap_or_else(|| x.parse::<f64>().map(|n| n.is_infinite()).unwrap_or(false)),
        Scalar::ComplexLongDouble(re, im) => {
            native_longdouble::classify(re, 1)
                .unwrap_or_else(|| re.parse::<f64>().map(|n| n.is_infinite()).unwrap_or(false))
                || native_longdouble::classify(im, 1)
                    .unwrap_or_else(|| im.parse::<f64>().map(|n| n.is_infinite()).unwrap_or(false))
        }
        _ => false,
    }
}
fn scalar_is_finite(value: &Scalar) -> bool {
    !scalar_is_nan(value) && !scalar_is_infinite(value)
}

fn compare(name: &str, left: &Scalar, right: &Scalar) -> Result<bool, StorageError> {
    let ordering = if native_longdouble::has_extended_native()
        && (matches!(left.dtype(), DType::LongDouble | DType::ComplexLongDouble)
            || matches!(right.dtype(), DType::LongDouble | DType::ComplexLongDouble))
    {
        let comparison = if left.dtype().kind() == "c" || right.dtype().kind() == "c" {
            let (left_real, left_imag) = complex_long_double_parts(left)?;
            let (right_real, right_imag) = complex_long_double_parts(right)?;
            native_longdouble::compare_complex(&left_real, &left_imag, &right_real, &right_imag)
        } else {
            native_longdouble::compare(&long_double_text(left)?, &long_double_text(right)?)
        };
        match comparison.flatten() {
            Some(-1) => Some(Ordering::Less),
            Some(0) => Some(Ordering::Equal),
            Some(1) => Some(Ordering::Greater),
            _ => None,
        }
    } else if left.dtype().kind() == "c" || right.dtype().kind() == "c" {
        let (left_re, left_im) = left.as_complex()?;
        let (right_re, right_im) = right.as_complex()?;
        if left_im.is_nan() || right_im.is_nan() {
            None
        } else {
            left_re.partial_cmp(&right_re).and_then(|real_order| {
                if real_order == Ordering::Equal {
                    left_im.partial_cmp(&right_im)
                } else {
                    Some(real_order)
                }
            })
        }
    } else if is_integer(left.dtype()) && is_integer(right.dtype()) {
        Some(integer_ordering(left, right)?)
    } else if is_integer(left.dtype()) {
        compare_integer_float(left, right.as_f64()?)
    } else if is_integer(right.dtype()) {
        compare_integer_float(right, left.as_f64()?).map(Ordering::reverse)
    } else {
        left.as_f64()?.partial_cmp(&right.as_f64()?)
    };
    let result = match name {
        "equal" => ordering == Some(Ordering::Equal),
        "not_equal" => ordering != Some(Ordering::Equal),
        "less" => ordering == Some(Ordering::Less),
        "less_equal" => matches!(ordering, Some(Ordering::Less | Ordering::Equal)),
        "greater" => ordering == Some(Ordering::Greater),
        "greater_equal" => matches!(ordering, Some(Ordering::Greater | Ordering::Equal)),
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(result)
}

fn integer_ordering(left: &Scalar, right: &Scalar) -> Result<Ordering, StorageError> {
    let left_signed = is_signed_dtype(left.dtype());
    let right_signed = is_signed_dtype(right.dtype());
    match (left_signed, right_signed) {
        (true, true) => Ok(signed_value(left)?.cmp(&signed_value(right)?)),
        (false, false) => Ok(unsigned_value(left)?.cmp(&unsigned_value(right)?)),
        (true, false) => {
            let signed = signed_value(left)?;
            if signed < 0 {
                Ok(Ordering::Less)
            } else {
                Ok((signed as u128).cmp(&unsigned_value(right)?))
            }
        }
        (false, true) => {
            let signed = signed_value(right)?;
            if signed < 0 {
                Ok(Ordering::Greater)
            } else {
                Ok(unsigned_value(left)?.cmp(&(signed as u128)))
            }
        }
    }
}

fn compare_integer_float(integer: &Scalar, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    let integer = if is_signed_dtype(integer.dtype()) {
        signed_value(integer).ok()?
    } else {
        i128::try_from(unsigned_value(integer).ok()?).ok()?
    };
    let truncated = float.trunc();
    if truncated < i128::MIN as f64 {
        return Some(Ordering::Greater);
    }
    if truncated >= 2f64.powi(127) {
        return Some(Ordering::Less);
    }
    let truncated_integer = truncated as i128;
    let ordering = integer.cmp(&truncated_integer);
    Some(match ordering {
        Ordering::Equal if float == truncated => Ordering::Equal,
        Ordering::Equal if float.is_sign_negative() => Ordering::Greater,
        Ordering::Equal => Ordering::Less,
        other => other,
    })
}

fn integer_binary(
    name: &str,
    left: &Scalar,
    right: &Scalar,
    dtype: DType,
) -> Result<Vec<Scalar>, StorageError> {
    if dtype.kind() == "u" || dtype == DType::Bool {
        let a = unsigned_value(left)?;
        let b = unsigned_value(right)?;
        let value = match name {
            "add" => a.wrapping_add(b),
            "subtract" => a.wrapping_sub(b),
            "multiply" => a.wrapping_mul(b),
            "floor_divide" => a.checked_div(b).unwrap_or(0),
            "remainder" | "mod" | "fmod" => {
                if b == 0 {
                    0
                } else {
                    a % b
                }
            }
            "power" | "pow" => {
                if (b as i128) < 0 {
                    return Err(StorageError::InvalidScalar);
                }
                wrapping_pow_unsigned(a, b)
            }
            "maximum" | "fmax" => a.max(b),
            "minimum" | "fmin" => a.min(b),
            "gcd" => gcd_unsigned(a, b),
            "lcm" => lcm_unsigned(a, b),
            _ => return Err(StorageError::InvalidScalar),
        };
        return Ok(vec![unsigned_result(value, dtype)]);
    }
    let a = signed_value(left)?;
    let b = signed_value(right)?;
    let value = match name {
        "add" => a.wrapping_add(b),
        "subtract" => a.wrapping_sub(b),
        "multiply" => a.wrapping_mul(b),
        "floor_divide" => {
            if b == 0 {
                0
            } else {
                floor_quotient(a, b)?
            }
        }
        "remainder" | "mod" => {
            if b == 0 {
                0
            } else {
                floor_remainder(a, b)?
            }
        }
        "fmod" => {
            if b == 0 {
                0
            } else {
                a % b
            }
        }
        "power" | "pow" => {
            if b < 0 {
                return Err(StorageError::InvalidScalar);
            }
            wrapping_pow_signed(a, b as u128)
        }
        "maximum" | "fmax" => a.max(b),
        "minimum" | "fmin" => a.min(b),
        "gcd" => gcd(a, b),
        "lcm" => lcm(a, b),
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(vec![signed_result(value, dtype)])
}

fn integer_unary<F, G>(
    value: &Scalar,
    dtype: DType,
    int_fn: F,
    float_fn: G,
) -> Result<Scalar, StorageError>
where
    F: FnOnce(i128) -> i128,
    G: FnOnce(f64) -> f64,
{
    if is_integer(dtype) {
        Ok(signed_result(int_fn(signed_value(value)?), dtype))
    } else {
        Ok(Scalar::Float64(float_fn(value.as_f64()?)).cast(dtype)?)
    }
}

fn frexp(value: Scalar, output: DType) -> Result<Vec<Scalar>, StorageError> {
    if matches!(output, DType::Float16 | DType::Float32) {
        let (fraction, exponent) = frexp_f32(value.as_f64()? as f32);
        return Ok(vec![
            Scalar::Float32(fraction).cast(output)?,
            Scalar::Int32(exponent),
        ]);
    }
    let x = value.as_f64()?;
    let (fraction, exponent) = frexp_f64(x);
    Ok(vec![
        Scalar::Float64(fraction).cast(output)?,
        Scalar::Int32(exponent),
    ])
}

fn frexp_f32(value: f32) -> (f32, i32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0);
    }

    const FRACTION_MASK: u32 = (1_u32 << 23) - 1;
    const SIGN_MASK: u32 = 1_u32 << 31;
    const HALF_EXPONENT: u32 = 126;
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 23) & 0xff) as i32;
    if exponent_bits == 0 {
        let (fraction, exponent) = frexp_f32(value * 2.0_f32.powi(25));
        return (fraction, exponent - 25);
    }

    let fraction_bits = (bits & (SIGN_MASK | FRACTION_MASK)) | (HALF_EXPONENT << 23);
    (
        f32::from_bits(fraction_bits),
        exponent_bits - HALF_EXPONENT as i32,
    )
}

/// Split a finite binary64 value without constructing `2**exponent`.
///
/// The direct division overflows for values near `f64::MAX`; subnormal values
/// also need normalization before their exponent bits can be used.
fn frexp_f64(value: f64) -> (f64, i32) {
    if value == 0.0 || !value.is_finite() {
        return (value, 0);
    }

    const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
    const SIGN_MASK: u64 = 1_u64 << 63;
    const HALF_EXPONENT: u64 = 1022;
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    if exponent_bits == 0 {
        let (fraction, exponent) = frexp_f64(value * 2.0_f64.powi(54));
        return (fraction, exponent - 54);
    }

    let fraction_bits = (bits & (SIGN_MASK | FRACTION_MASK)) | (HALF_EXPONENT << 52);
    (
        f64::from_bits(fraction_bits),
        exponent_bits - HALF_EXPONENT as i32,
    )
}
fn modf(value: Scalar, output: DType) -> Result<Vec<Scalar>, StorageError> {
    if matches!(output, DType::Float16 | DType::Float32) {
        let x = value.as_f64()? as f32;
        let integral = x.trunc();
        let fractional = if x == integral {
            0.0f32.copysign(x)
        } else {
            x - integral
        };
        return Ok(vec![
            Scalar::Float32(fractional).cast(output)?,
            Scalar::Float32(integral).cast(output)?,
        ]);
    }
    let x = value.as_f64()?;
    let integral = x.trunc();
    let fractional = if x == integral {
        0.0f64.copysign(x)
    } else {
        x - integral
    };
    Ok(vec![
        Scalar::Float64(fractional).cast(output)?,
        Scalar::Float64(integral).cast(output)?,
    ])
}
fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.wrapping_abs();
    b = b.wrapping_abs();
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}
fn lcm(a: i128, b: i128) -> i128 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd(a, b)).wrapping_mul(b).wrapping_abs()
    }
}
fn gcd_unsigned(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}
fn wrapping_pow_signed(mut base: i128, mut exponent: u128) -> i128 {
    let mut result = 1i128;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.wrapping_mul(base);
        }
    }
    result
}
fn wrapping_pow_unsigned(mut base: u128, mut exponent: u128) -> u128 {
    let mut result = 1u128;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = result.wrapping_mul(base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base.wrapping_mul(base);
        }
    }
    result
}
fn lcm_unsigned(a: u128, b: u128) -> u128 {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd_unsigned(a, b)).wrapping_mul(b)
    }
}
fn floor_quotient(a: i128, b: i128) -> Result<i128, StorageError> {
    if b == 0 {
        return Err(StorageError::InvalidScalar);
    }
    let quotient = a / b;
    let remainder = a % b;
    Ok(if remainder != 0 && (remainder < 0) != (b < 0) {
        quotient - 1
    } else {
        quotient
    })
}
fn floor_remainder(a: i128, b: i128) -> Result<i128, StorageError> {
    let quotient = floor_quotient(a, b)?;
    Ok(a - quotient * b)
}
fn float_remainder(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() || b == 0.0 || a.is_infinite() {
        return f64::NAN;
    }
    if b.is_infinite() {
        if a == 0.0 {
            return 0.0f64.copysign(b);
        }
        return if a.is_sign_negative() == b.is_sign_negative() {
            a
        } else {
            b
        };
    }
    let remainder = a % b;
    if remainder == 0.0 {
        return 0.0f64.copysign(b);
    }
    if remainder.is_sign_negative() != b.is_sign_negative() {
        remainder + b
    } else {
        remainder
    }
}

fn floor_divide(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if b == 0.0 {
        return a / b;
    }
    if a.is_infinite() {
        return f64::NAN;
    }
    if b.is_infinite() {
        if a == 0.0 {
            return 0.0f64.copysign(a / b);
        }
        return if a.is_sign_negative() != b.is_sign_negative() {
            -1.0
        } else {
            0.0
        };
    }
    let modulus = a % b;
    let mut division = (a - modulus) / b;
    if modulus != 0.0 && b.is_sign_negative() != modulus.is_sign_negative() {
        division -= 1.0;
    }
    if division != 0.0 {
        let mut quotient = division.floor();
        if division - quotient > 0.5 {
            quotient += 1.0;
        }
        quotient
    } else {
        0.0f64.copysign(a / b)
    }
}
fn logaddexp(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    if a == f64::NEG_INFINITY {
        return if b == 0.0 { 0.0 } else { b };
    }
    if b == f64::NEG_INFINITY {
        return if a == 0.0 { 0.0 } else { a };
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

fn logaddexp2(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    if a == f64::NEG_INFINITY {
        return if b == 0.0 { 0.0 } else { b };
    }
    if b == f64::NEG_INFINITY {
        return if a == 0.0 { 0.0 } else { a };
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp2().ln_1p() / std::f64::consts::LN_2
}
fn complex_extreme(
    name: &str,
    left: Complex<f64>,
    right: Complex<f64>,
) -> Result<Complex<f64>, StorageError> {
    let left_nan = left.re.is_nan() || left.im.is_nan();
    let right_nan = right.re.is_nan() || right.im.is_nan();
    if matches!(name, "fmax" | "fmin") {
        if right_nan {
            return Ok(left);
        }
        if left_nan {
            return Ok(right);
        }
    } else {
        if left_nan {
            return Ok(left);
        }
        if right_nan {
            return Ok(right);
        }
    }
    let order = left
        .re
        .partial_cmp(&right.re)
        .and_then(|order| {
            if order == Ordering::Equal {
                left.im.partial_cmp(&right.im)
            } else {
                Some(order)
            }
        })
        .ok_or(StorageError::InvalidScalar)?;
    let choose_left = match name {
        "maximum" | "fmax" => order != Ordering::Less,
        "minimum" | "fmin" => order != Ordering::Greater,
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(if choose_left { left } else { right })
}
fn complex_extreme_f32(
    name: &str,
    left: Complex<f32>,
    right: Complex<f32>,
) -> Result<Complex<f32>, StorageError> {
    let left_nan = left.re.is_nan() || left.im.is_nan();
    let right_nan = right.re.is_nan() || right.im.is_nan();
    if matches!(name, "fmax" | "fmin") {
        if right_nan {
            return Ok(left);
        }
        if left_nan {
            return Ok(right);
        }
    } else {
        if left_nan {
            return Ok(left);
        }
        if right_nan {
            return Ok(right);
        }
    }
    let order = left
        .re
        .partial_cmp(&right.re)
        .and_then(|order| {
            if order == Ordering::Equal {
                left.im.partial_cmp(&right.im)
            } else {
                Some(order)
            }
        })
        .ok_or(StorageError::InvalidScalar)?;
    let choose_left = match name {
        "maximum" | "fmax" => order != Ordering::Less,
        "minimum" | "fmin" => order != Ordering::Greater,
        _ => return Err(StorageError::InvalidScalar),
    };
    Ok(if choose_left { left } else { right })
}
fn next_after_dtype(x: f64, y: f64, dtype: DType) -> f64 {
    match dtype {
        DType::Float16 => next_after_f16(half::f16::from_f64(x), half::f16::from_f64(y)).to_f64(),
        DType::Float32 => next_after_f32(x as f32, y as f32) as f64,
        _ => next_after_f64(x, y),
    }
}
fn next_after_f16(x: half::f16, y: half::f16) -> half::f16 {
    if x.is_nan() || y.is_nan() {
        return half::f16::NAN;
    }
    if x == y {
        return y;
    }
    if x == half::f16::ZERO {
        return half::f16::from_bits(if y.is_sign_negative() { 0x8001 } else { 0x0001 });
    }
    let bits = x.to_bits();
    half::f16::from_bits(if (y > x) == (x > half::f16::ZERO) {
        bits + 1
    } else {
        bits - 1
    })
}
fn spacing_f16(value: f32) -> f32 {
    if !value.is_finite() {
        return f32::NAN;
    }
    let input = half::f16::from_f32(value);
    let direction = half::f16::INFINITY.copysign(input);
    (next_after_f16(input, direction).to_f32()) - value
}
fn next_after_f32(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return f32::from_bits(if y.is_sign_negative() { 0x8000_0001 } else { 1 });
    }
    let bits = x.to_bits();
    f32::from_bits(if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    })
}
fn next_after_f64(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return if y.is_sign_negative() {
            -f64::from_bits(1)
        } else {
            f64::from_bits(1)
        };
    }
    let bits = x.to_bits();
    f64::from_bits(if (y > x) == (x > 0.0) {
        bits + 1
    } else {
        bits - 1
    })
}
