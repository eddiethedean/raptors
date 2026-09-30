//! Safe, checked numeric storage for the Raptors 0.2 array foundation.
//!
//! Allocations are initialized Rust vectors. Views retain shared ownership
//! and use checked signed byte strides. This crate is independent of the
//! legacy raw-pointer core.

use raptors_longdouble as native_longdouble;
use std::fmt;
use std::sync::{Arc, RwLock};

pub mod ufunc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    LongDouble,
    ComplexLongDouble,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int16 => "int16",
            Self::UInt16 => "uint16",
            Self::Int32 => "int32",
            Self::UInt32 => "uint32",
            Self::Int64 => "int64",
            Self::UInt64 => "uint64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::Complex64 => "complex64",
            Self::Complex128 => "complex128",
            Self::LongDouble if long_double_size() == 8 => "float64",
            Self::LongDouble => "float128",
            Self::ComplexLongDouble if long_double_size() == 8 => "complex128",
            Self::ComplexLongDouble => "complex256",
        }
    }
    pub const fn itemsize(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
            Self::LongDouble => long_double_size(),
            Self::ComplexLongDouble => 2 * long_double_size(),
        }
    }
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Bool => "b",
            Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64 => "i",
            Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64 => "u",
            Self::Float16 | Self::Float32 | Self::Float64 | Self::LongDouble => "f",
            Self::Complex64 | Self::Complex128 | Self::ComplexLongDouble => "c",
        }
    }
    /// NumPy's concrete scalar type code (`dtype.char`), which is distinct
    /// from the family code returned by [`DType::kind`].
    pub const fn char(self) -> char {
        match self {
            Self::Bool => '?',
            Self::Int8 => 'b',
            Self::UInt8 => 'B',
            Self::Int16 => 'h',
            Self::UInt16 => 'H',
            #[cfg(target_os = "windows")]
            Self::Int32 => 'l',
            #[cfg(not(target_os = "windows"))]
            Self::Int32 => 'i',
            #[cfg(target_os = "windows")]
            Self::UInt32 => 'L',
            #[cfg(not(target_os = "windows"))]
            Self::UInt32 => 'I',
            #[cfg(target_os = "windows")]
            Self::Int64 => 'q',
            #[cfg(not(target_os = "windows"))]
            Self::Int64 => 'l',
            #[cfg(target_os = "windows")]
            Self::UInt64 => 'Q',
            #[cfg(not(target_os = "windows"))]
            Self::UInt64 => 'L',
            Self::Float16 => 'e',
            Self::Float32 => 'f',
            Self::Float64 => 'd',
            Self::LongDouble => 'g',
            Self::Complex64 => 'F',
            Self::Complex128 => 'D',
            Self::ComplexLongDouble => 'G',
        }
    }
    pub const fn alignment(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 => 1,
            Self::Int16 | Self::UInt16 | Self::Float16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 | Self::Complex64 => 4,
            Self::Int64 | Self::UInt64 | Self::Float64 | Self::Complex128 => 8,
            Self::LongDouble => long_double_alignment(),
            Self::ComplexLongDouble => long_double_alignment(),
        }
    }

    /// NumPy-style result dtype promotion for fixed-width numeric descriptors.
    /// Python scalar weak-promotion rules are applied by the binding layer.
    pub const fn promote(self, other: Self) -> Self {
        use DType::*;
        if matches!(self, ComplexLongDouble) || matches!(other, ComplexLongDouble) {
            return ComplexLongDouble;
        }
        if matches!(self, LongDouble) || matches!(other, LongDouble) {
            return if matches!(self, Complex64 | Complex128)
                || matches!(other, Complex64 | Complex128)
            {
                if long_double_size() > 8 {
                    ComplexLongDouble
                } else {
                    Complex128
                }
            } else {
                LongDouble
            };
        }
        let complex =
            matches!(self, Complex64 | Complex128) || matches!(other, Complex64 | Complex128);
        if complex {
            let needs_64 = matches!(self, Complex128 | Float64 | Int32 | UInt32 | Int64 | UInt64)
                || matches!(
                    other,
                    Complex128 | Float64 | Int32 | UInt32 | Int64 | UInt64
                );
            return if needs_64 { Complex128 } else { Complex64 };
        }
        if matches!(self, Float16 | Float32 | Float64)
            || matches!(other, Float16 | Float32 | Float64)
        {
            let float = if matches!(self, Float64) || matches!(other, Float64) {
                Float64
            } else if matches!(self, Float32) || matches!(other, Float32) {
                Float32
            } else {
                Float16
            };
            let integer_bits = max_integer_bits(self, other);
            return match (float, integer_bits) {
                (Float16, 0..=8) => Float16,
                (Float16, 9..=16) | (Float32, 0..=16) => Float32,
                (_, _) => Float64,
            };
        }
        match (self, other) {
            (Bool, Bool) => Bool,
            (Bool, dtype) | (dtype, Bool) => dtype,
            _ => promote_integer(self, other),
        }
    }
}

const fn integer_bits(dtype: DType) -> u32 {
    match dtype {
        DType::Int8 | DType::UInt8 => 8,
        DType::Int16 | DType::UInt16 => 16,
        DType::Int32 | DType::UInt32 => 32,
        DType::Int64 | DType::UInt64 => 64,
        _ => 0,
    }
}

const fn max_integer_bits(left: DType, right: DType) -> u32 {
    let a = integer_bits(left);
    let b = integer_bits(right);
    if a > b {
        a
    } else {
        b
    }
}

const fn is_signed(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
    )
}

const fn is_unsigned(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::UInt8 | DType::UInt16 | DType::UInt32 | DType::UInt64
    )
}

const fn promote_integer(left: DType, right: DType) -> DType {
    use DType::*;
    let bits = max_integer_bits(left, right);
    if is_signed(left) && is_signed(right) {
        return match bits {
            0..=8 => Int8,
            9..=16 => Int16,
            17..=32 => Int32,
            _ => Int64,
        };
    }
    if is_unsigned(left) && is_unsigned(right) {
        return match bits {
            0..=8 => UInt8,
            9..=16 => UInt16,
            17..=32 => UInt32,
            _ => UInt64,
        };
    }
    if is_signed(left) || is_signed(right) {
        let signed_bits = if is_signed(left) {
            integer_bits(left)
        } else {
            integer_bits(right)
        };
        let unsigned_bits = if is_unsigned(left) {
            integer_bits(left)
        } else {
            integer_bits(right)
        };
        if signed_bits > unsigned_bits {
            return match signed_bits {
                0..=8 => Int8,
                9..=16 => Int16,
                17..=32 => Int32,
                _ => Int64,
            };
        }
        let required = if signed_bits > unsigned_bits {
            signed_bits
        } else {
            unsigned_bits.saturating_add(1)
        };
        return match required {
            0..=8 => Int8,
            9..=16 => Int16,
            17..=32 => Int32,
            33..=64 => Int64,
            _ => Float64,
        };
    }
    Bool
}

// These are the long-double layouts used by Raptors' configured wheel targets.
// On Windows and Apple ARM64 NumPy exposes longdouble as a float64 alias.
#[cfg(any(
    target_os = "windows",
    all(target_vendor = "apple", target_arch = "aarch64")
))]
const fn long_double_size() -> usize {
    8
}
#[cfg(not(any(
    target_os = "windows",
    all(target_vendor = "apple", target_arch = "aarch64")
)))]
const fn long_double_size() -> usize {
    16
}
#[cfg(any(
    target_os = "windows",
    all(target_vendor = "apple", target_arch = "aarch64")
))]
const fn long_double_alignment() -> usize {
    8
}
#[cfg(not(any(
    target_os = "windows",
    all(target_vendor = "apple", target_arch = "aarch64")
)))]
const fn long_double_alignment() -> usize {
    16
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int8(i8),
    UInt8(u8),
    Int16(i16),
    UInt16(u16),
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Float16(f32),
    Float32(f32),
    Float64(f64),
    Complex64(f32, f32),
    Complex128(f64, f64),
    LongDouble(String),
    ComplexLongDouble(String, String),
}

impl Scalar {
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::Int8(_) => DType::Int8,
            Self::UInt8(_) => DType::UInt8,
            Self::Int16(_) => DType::Int16,
            Self::UInt16(_) => DType::UInt16,
            Self::Int32(_) => DType::Int32,
            Self::UInt32(_) => DType::UInt32,
            Self::Int64(_) => DType::Int64,
            Self::UInt64(_) => DType::UInt64,
            Self::Float16(_) => DType::Float16,
            Self::Float32(_) => DType::Float32,
            Self::Float64(_) => DType::Float64,
            Self::Complex64(_, _) => DType::Complex64,
            Self::Complex128(_, _) => DType::Complex128,
            Self::LongDouble(_) => DType::LongDouble,
            Self::ComplexLongDouble(_, _) => DType::ComplexLongDouble,
        }
    }

    pub fn zero(dtype: DType) -> Self {
        match dtype {
            DType::Bool => Self::Bool(false),
            DType::Int8 => Self::Int8(0),
            DType::UInt8 => Self::UInt8(0),
            DType::Int16 => Self::Int16(0),
            DType::UInt16 => Self::UInt16(0),
            DType::Int32 => Self::Int32(0),
            DType::UInt32 => Self::UInt32(0),
            DType::Int64 => Self::Int64(0),
            DType::UInt64 => Self::UInt64(0),
            DType::Float16 => Self::Float16(0.0),
            DType::Float32 => Self::Float32(0.0),
            DType::Float64 => Self::Float64(0.0),
            DType::Complex64 => Self::Complex64(0.0, 0.0),
            DType::Complex128 => Self::Complex128(0.0, 0.0),
            DType::LongDouble => Self::LongDouble("0".into()),
            DType::ComplexLongDouble => Self::ComplexLongDouble("0".into(), "0".into()),
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            Self::Bool(v) => *v,
            Self::Int8(v) => *v != 0,
            Self::UInt8(v) => *v != 0,
            Self::Int16(v) => *v != 0,
            Self::UInt16(v) => *v != 0,
            Self::Int32(v) => *v != 0,
            Self::UInt32(v) => *v != 0,
            Self::Int64(v) => *v != 0,
            Self::UInt64(v) => *v != 0,
            Self::Float16(v) | Self::Float32(v) => *v != 0.0,
            Self::Float64(v) => *v != 0.0,
            Self::Complex64(re, im) => *re != 0.0 || *im != 0.0,
            Self::Complex128(re, im) => *re != 0.0 || *im != 0.0,
            Self::LongDouble(v) => !native_longdouble::classify(v, 3)
                .unwrap_or_else(|| v.parse::<f64>().map(|x| x == 0.0).unwrap_or(true)),
            Self::ComplexLongDouble(re, im) => {
                !native_longdouble::classify(re, 3)
                    .unwrap_or_else(|| re.parse::<f64>().map(|x| x == 0.0).unwrap_or(true))
                    || !native_longdouble::classify(im, 3)
                        .unwrap_or_else(|| im.parse::<f64>().map(|x| x == 0.0).unwrap_or(true))
            }
        }
    }

    pub fn is_nan(&self) -> bool {
        match self {
            Self::Float16(value) | Self::Float32(value) => value.is_nan(),
            Self::Float64(value) => value.is_nan(),
            Self::Complex64(real, imag) => real.is_nan() || imag.is_nan(),
            Self::Complex128(real, imag) => real.is_nan() || imag.is_nan(),
            Self::LongDouble(value) => native_longdouble::classify(value, 0).unwrap_or_else(|| {
                value
                    .parse::<f64>()
                    .map(|value| value.is_nan())
                    .unwrap_or(false)
            }),
            Self::ComplexLongDouble(real, imag) => {
                native_longdouble::classify(real, 0).unwrap_or_else(|| {
                    real.parse::<f64>()
                        .map(|value| value.is_nan())
                        .unwrap_or(false)
                }) || native_longdouble::classify(imag, 0).unwrap_or_else(|| {
                    imag.parse::<f64>()
                        .map(|value| value.is_nan())
                        .unwrap_or(false)
                })
            }
            _ => false,
        }
    }

    pub fn is_infinite(&self) -> bool {
        match self {
            Self::Float16(value) | Self::Float32(value) => value.is_infinite(),
            Self::Float64(value) => value.is_infinite(),
            Self::Complex64(real, imag) => real.is_infinite() || imag.is_infinite(),
            Self::Complex128(real, imag) => real.is_infinite() || imag.is_infinite(),
            Self::LongDouble(value) => native_longdouble::classify(value, 1).unwrap_or_else(|| {
                value
                    .parse::<f64>()
                    .map(|value| value.is_infinite())
                    .unwrap_or(false)
            }),
            Self::ComplexLongDouble(real, imag) => {
                native_longdouble::classify(real, 1).unwrap_or_else(|| {
                    real.parse::<f64>()
                        .map(|value| value.is_infinite())
                        .unwrap_or(false)
                }) || native_longdouble::classify(imag, 1).unwrap_or_else(|| {
                    imag.parse::<f64>()
                        .map(|value| value.is_infinite())
                        .unwrap_or(false)
                })
            }
            _ => false,
        }
    }

    pub fn is_finite(&self) -> bool {
        match self {
            Self::LongDouble(value) => native_longdouble::classify(value, 4).unwrap_or_else(|| {
                value
                    .parse::<f64>()
                    .map(|value| value.is_finite())
                    .unwrap_or(false)
            }),
            Self::ComplexLongDouble(real, imag) => {
                native_longdouble::classify(real, 4).unwrap_or_else(|| {
                    real.parse::<f64>()
                        .map(|value| value.is_finite())
                        .unwrap_or(false)
                }) && native_longdouble::classify(imag, 4).unwrap_or_else(|| {
                    imag.parse::<f64>()
                        .map(|value| value.is_finite())
                        .unwrap_or(false)
                })
            }
            _ => !self.is_nan() && !self.is_infinite(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::LongDouble(value) => native_longdouble::classify(value, 3).unwrap_or_else(|| {
                value
                    .parse::<f64>()
                    .map(|value| value == 0.0)
                    .unwrap_or(false)
            }),
            Self::ComplexLongDouble(real, imag) => {
                native_longdouble::classify(real, 3).unwrap_or_else(|| {
                    real.parse::<f64>()
                        .map(|value| value == 0.0)
                        .unwrap_or(false)
                }) && native_longdouble::classify(imag, 3).unwrap_or_else(|| {
                    imag.parse::<f64>()
                        .map(|value| value == 0.0)
                        .unwrap_or(false)
                })
            }
            _ => !self.truthy(),
        }
    }

    pub fn is_subnormal(&self) -> bool {
        match self {
            Self::LongDouble(value) => native_longdouble::classify(value, 5)
                .unwrap_or_else(|| value.parse::<f64>().map(is_subnormal_f64).unwrap_or(false)),
            Self::ComplexLongDouble(real, imag) => {
                native_longdouble::classify(real, 5)
                    .unwrap_or_else(|| real.parse::<f64>().map(is_subnormal_f64).unwrap_or(false))
                    || native_longdouble::classify(imag, 5).unwrap_or_else(|| {
                        imag.parse::<f64>().map(is_subnormal_f64).unwrap_or(false)
                    })
            }
            Self::Float16(value) => *value != 0.0 && value.abs() < 6.103_515_6e-5,
            Self::Float32(value) => *value != 0.0 && value.abs() < f32::MIN_POSITIVE,
            Self::Complex64(real, imag) => {
                (*real != 0.0 && real.abs() < f32::MIN_POSITIVE)
                    || (*imag != 0.0 && imag.abs() < f32::MIN_POSITIVE)
            }
            Self::Float64(value) => *value != 0.0 && value.abs() < f64::MIN_POSITIVE,
            Self::Complex128(real, imag) => is_subnormal_f64(*real) || is_subnormal_f64(*imag),
            _ => false,
        }
    }

    pub fn as_f64(&self) -> Result<f64, StorageError> {
        Ok(match self {
            Self::Bool(v) => u8::from(*v) as f64,
            Self::Int8(v) => *v as f64,
            Self::UInt8(v) => *v as f64,
            Self::Int16(v) => *v as f64,
            Self::UInt16(v) => *v as f64,
            Self::Int32(v) => *v as f64,
            Self::UInt32(v) => *v as f64,
            Self::Int64(v) => *v as f64,
            Self::UInt64(v) => *v as f64,
            Self::Float16(v) | Self::Float32(v) => *v as f64,
            Self::Float64(v) => *v,
            Self::Complex64(re, _) => *re as f64,
            Self::Complex128(re, _) => *re,
            Self::LongDouble(v) | Self::ComplexLongDouble(v, _) => {
                v.parse().map_err(|_| StorageError::InvalidScalar)?
            }
        })
    }

    pub fn cast(&self, dtype: DType) -> Result<Self, StorageError> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let value = match dtype {
            DType::Bool => Self::Bool(self.truthy()),
            DType::Int8
            | DType::UInt8
            | DType::Int16
            | DType::UInt16
            | DType::Int32
            | DType::UInt32
            | DType::Int64
            | DType::UInt64 => cast_to_integer(self, dtype)?,
            DType::Float16 => Self::Float16(half::f16::from_f64(self.as_f64()?).to_f32()),
            DType::Float32 => Self::Float32(self.as_f64()? as f32),
            DType::Float64 => Self::Float64(self.as_f64()?),
            DType::Complex64 => {
                let (re, im) = self.as_complex()?;
                Self::Complex64(re as f32, im as f32)
            }
            DType::Complex128 => {
                let (re, im) = self.as_complex()?;
                Self::Complex128(re, im)
            }
            DType::LongDouble => Self::LongDouble(scalar_decimal(self)?),
            DType::ComplexLongDouble => match self {
                Self::ComplexLongDouble(re, im) => Self::ComplexLongDouble(re.clone(), im.clone()),
                Self::Complex64(re, im) => Self::ComplexLongDouble(
                    exact_decimal_from_f64(*re as f64),
                    exact_decimal_from_f64(*im as f64),
                ),
                Self::Complex128(re, im) => Self::ComplexLongDouble(
                    exact_decimal_from_f64(*re),
                    exact_decimal_from_f64(*im),
                ),
                value => Self::ComplexLongDouble(scalar_decimal(value)?, "0".into()),
            },
        };
        Ok(value)
    }

    pub fn as_complex(&self) -> Result<(f64, f64), StorageError> {
        match self {
            Self::Complex64(re, im) => Ok((*re as f64, *im as f64)),
            Self::Complex128(re, im) => Ok((*re, *im)),
            Self::ComplexLongDouble(re, im) => Ok((
                re.parse().map_err(|_| StorageError::InvalidScalar)?,
                im.parse().map_err(|_| StorageError::InvalidScalar)?,
            )),
            _ => Ok((self.as_f64()?, 0.0)),
        }
    }

    /// Whether this floating scalar is outside the integer dtype's conversion
    /// range. The Python API uses this to emit an invalid-cast RuntimeWarning.
    pub fn integer_cast_is_invalid(&self, dtype: DType) -> bool {
        let Some((lower, upper)) = float_integer_cast_bounds(dtype) else {
            return false;
        };
        match self {
            Self::Float16(value) | Self::Float32(value) => {
                !float_integer_cast_in_range(*value as f64, lower, upper)
            }
            Self::Float64(value) => !float_integer_cast_in_range(*value, lower, upper),
            Self::Complex64(real, _) => !float_integer_cast_in_range(*real as f64, lower, upper),
            Self::Complex128(real, _) => !float_integer_cast_in_range(*real, lower, upper),
            Self::LongDouble(value) | Self::ComplexLongDouble(value, _) => {
                match decimal_to_i128(value) {
                    Ok(integer) => integer < lower || integer >= upper,
                    Err(_) => true,
                }
            }
            _ => false,
        }
    }
}

fn is_subnormal_f64(value: f64) -> bool {
    value != 0.0 && value.abs() < f64::MIN_POSITIVE
}

fn scalar_decimal(value: &Scalar) -> Result<String, StorageError> {
    Ok(match value {
        Scalar::LongDouble(v) => v.clone(),
        Scalar::ComplexLongDouble(re, _) => re.clone(),
        Scalar::Complex64(re, _) => exact_decimal_from_f64(*re as f64),
        Scalar::Complex128(re, _) => exact_decimal_from_f64(*re),
        Scalar::Float16(v) | Scalar::Float32(v) => exact_decimal_from_f64(*v as f64),
        Scalar::Float64(v) => exact_decimal_from_f64(*v),
        Scalar::Bool(v) => u8::from(*v).to_string(),
        Scalar::Int8(v) => v.to_string(),
        Scalar::UInt8(v) => v.to_string(),
        Scalar::Int16(v) => v.to_string(),
        Scalar::UInt16(v) => v.to_string(),
        Scalar::Int32(v) => v.to_string(),
        Scalar::UInt32(v) => v.to_string(),
        Scalar::Int64(v) => v.to_string(),
        Scalar::UInt64(v) => v.to_string(),
    })
}

fn exact_decimal_from_f64(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }

    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    exact_decimal_from_binary(significand as u128, exponent, negative)
}

/// Formats an exact finite binary value as a decimal integer/fraction.
fn exact_decimal_from_binary(significand: u128, exponent: i32, negative: bool) -> String {
    if significand == 0 {
        return if negative { "-0".into() } else { "0".into() };
    }
    let mut digits = significand
        .to_string()
        .bytes()
        .rev()
        .map(|digit| digit - b'0')
        .collect::<Vec<_>>();
    let decimal_places = if exponent >= 0 {
        for _ in 0..exponent {
            multiply_decimal_digits(&mut digits, 2);
        }
        0
    } else {
        let places = (-exponent) as usize;
        for _ in 0..places {
            multiply_decimal_digits(&mut digits, 5);
        }
        places
    };

    while digits.len() > 1 && digits.last() == Some(&0) && decimal_places == 0 {
        digits.pop();
    }
    let mut text = digits
        .iter()
        .rev()
        .map(|digit| char::from(b'0' + *digit))
        .collect::<String>();
    if decimal_places > 0 {
        if text.len() <= decimal_places {
            text = format!("0.{}{}", "0".repeat(decimal_places - text.len()), text);
        } else {
            text.insert(text.len() - decimal_places, '.');
        }
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    if negative {
        text.insert(0, '-');
    }
    text
}

fn multiply_decimal_digits(digits: &mut Vec<u8>, multiplier: u8) {
    let mut carry = 0_u16;
    for digit in digits.iter_mut() {
        let value = u16::from(*digit) * u16::from(multiplier) + carry;
        *digit = (value % 10) as u8;
        carry = value / 10;
    }
    while carry != 0 {
        digits.push((carry % 10) as u8);
        carry /= 10;
    }
}

fn to_i128(value: &Scalar) -> Result<i128, StorageError> {
    match value {
        Scalar::Bool(v) => Ok(i128::from(*v)),
        Scalar::Int8(v) => Ok(*v as i128),
        Scalar::UInt8(v) => Ok(*v as i128),
        Scalar::Int16(v) => Ok(*v as i128),
        Scalar::UInt16(v) => Ok(*v as i128),
        Scalar::Int32(v) => Ok(*v as i128),
        Scalar::UInt32(v) => Ok(*v as i128),
        Scalar::Int64(v) => Ok(*v as i128),
        Scalar::UInt64(v) => Ok(*v as i128),
        Scalar::Float16(v) | Scalar::Float32(v) => float_to_i128(*v as f64),
        Scalar::Float64(v) => float_to_i128(*v),
        Scalar::LongDouble(v) => decimal_to_i128(v),
        Scalar::Complex64(real, _) => float_to_i128(*real as f64),
        Scalar::Complex128(real, _) => float_to_i128(*real),
        Scalar::ComplexLongDouble(real, _) => decimal_to_i128(real),
    }
}

fn cast_to_integer(value: &Scalar, dtype: DType) -> Result<Scalar, StorageError> {
    match value {
        Scalar::Float16(number) | Scalar::Float32(number) => {
            Ok(cast_f64_to_integer(*number as f64, dtype))
        }
        Scalar::Float64(number) => Ok(cast_f64_to_integer(*number, dtype)),
        Scalar::Complex64(real, _) => Ok(cast_f64_to_integer(*real as f64, dtype)),
        Scalar::Complex128(real, _) => Ok(cast_f64_to_integer(*real, dtype)),
        Scalar::LongDouble(number) | Scalar::ComplexLongDouble(number, _) => {
            match decimal_to_i128(number) {
                Ok(integer) => Ok(cast_long_double_integer(integer, dtype)),
                Err(_) => {
                    let number = number
                        .parse::<f64>()
                        .map_err(|_| StorageError::CastOverflow)?;
                    Ok(cast_f64_to_integer(number, dtype))
                }
            }
        }
        _ => Ok(cast_integer_value(to_i128(value)?, dtype)),
    }
}

fn cast_f64_to_integer(value: f64, dtype: DType) -> Scalar {
    match dtype {
        DType::Int8 => Scalar::Int8((value as i32) as i8),
        DType::UInt8 => Scalar::UInt8((value as i32) as u8),
        DType::Int16 => Scalar::Int16((value as i32) as i16),
        DType::UInt16 => Scalar::UInt16((value as i32) as u16),
        DType::Int32 => Scalar::Int32(value as i32),
        DType::UInt32 => Scalar::UInt32(value as u32),
        DType::Int64 => Scalar::Int64(value as i64),
        DType::UInt64 => Scalar::UInt64(value as u64),
        _ => unreachable!("cast_f64_to_integer requires an integer dtype"),
    }
}

fn cast_integer_value(value: i128, dtype: DType) -> Scalar {
    match dtype {
        DType::Int8 => Scalar::Int8(value as i8),
        DType::UInt8 => Scalar::UInt8(value as u8),
        DType::Int16 => Scalar::Int16(value as i16),
        DType::UInt16 => Scalar::UInt16(value as u16),
        DType::Int32 => Scalar::Int32(value as i32),
        DType::UInt32 => Scalar::UInt32(value as u32),
        DType::Int64 => Scalar::Int64(value as i64),
        DType::UInt64 => Scalar::UInt64(value as u64),
        _ => unreachable!("cast_integer_value requires an integer dtype"),
    }
}

fn cast_long_double_integer(value: i128, dtype: DType) -> Scalar {
    match dtype {
        DType::Int8 => Scalar::Int8(value.clamp(i32::MIN as i128, i32::MAX as i128) as i8),
        DType::UInt8 => Scalar::UInt8(value.clamp(i32::MIN as i128, i32::MAX as i128) as u8),
        DType::Int16 => Scalar::Int16(value.clamp(i32::MIN as i128, i32::MAX as i128) as i16),
        DType::UInt16 => Scalar::UInt16(value.clamp(i32::MIN as i128, i32::MAX as i128) as u16),
        DType::Int32 => Scalar::Int32(value.clamp(i32::MIN as i128, i32::MAX as i128) as i32),
        DType::UInt32 => Scalar::UInt32(value.clamp(0, u32::MAX as i128) as u32),
        DType::Int64 => Scalar::Int64(value.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
        DType::UInt64 => Scalar::UInt64(value.clamp(0, u64::MAX as i128) as u64),
        _ => unreachable!("cast_long_double_integer requires an integer dtype"),
    }
}

fn float_integer_cast_bounds(dtype: DType) -> Option<(i128, i128)> {
    match dtype {
        DType::Int8 | DType::UInt8 | DType::Int16 | DType::UInt16 => {
            Some((i32::MIN as i128, i32::MAX as i128 + 1))
        }
        DType::Int32 => Some((i32::MIN as i128, i32::MAX as i128 + 1)),
        DType::UInt32 => Some((0, u32::MAX as i128 + 1)),
        DType::Int64 => Some((i64::MIN as i128, i64::MAX as i128 + 1)),
        DType::UInt64 => Some((0, u64::MAX as i128 + 1)),
        _ => None,
    }
}

fn float_integer_cast_in_range(value: f64, lower: i128, upper: i128) -> bool {
    if !value.is_finite() {
        return false;
    }
    let truncated = value.trunc();
    truncated >= lower as f64 && truncated < upper as f64
}

fn decimal_to_i128(value: &str) -> Result<i128, StorageError> {
    let (negative, unsigned) = match value.as_bytes().first() {
        Some(b'-') => (true, &value[1..]),
        Some(b'+') => (false, &value[1..]),
        _ => (false, value),
    };
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(position) => (
            &unsigned[..position],
            unsigned[position + 1..]
                .parse::<i32>()
                .map_err(|_| StorageError::CastOverflow)?,
        ),
        None => (unsigned, 0),
    };
    let mut digits = String::new();
    let mut integer_digits = 0_i64;
    let mut after_decimal = false;
    for byte in mantissa.bytes() {
        if byte == b'.' && !after_decimal {
            after_decimal = true;
        } else if byte.is_ascii_digit() {
            digits.push(char::from(byte));
            if !after_decimal {
                integer_digits += 1;
            }
        } else {
            return Err(StorageError::CastOverflow);
        }
    }
    if digits.is_empty() {
        return Err(StorageError::CastOverflow);
    }
    let integer_digits = integer_digits + i64::from(exponent);
    if integer_digits <= 0 {
        return Ok(0);
    }
    let integer_len = usize::try_from(integer_digits).map_err(|_| StorageError::CastOverflow)?;
    if integer_len > 39 {
        return Err(StorageError::CastOverflow);
    }
    digits.truncate(integer_len.min(digits.len()));
    digits.extend(std::iter::repeat_n(
        '0',
        integer_len.saturating_sub(digits.len()),
    ));
    let magnitude = digits
        .parse::<u128>()
        .map_err(|_| StorageError::CastOverflow)?;
    if negative {
        if magnitude == (i128::MAX as u128) + 1 {
            Ok(i128::MIN)
        } else {
            i128::try_from(magnitude)
                .map(|number| -number)
                .map_err(|_| StorageError::CastOverflow)
        }
    } else {
        i128::try_from(magnitude).map_err(|_| StorageError::CastOverflow)
    }
}
fn float_to_i128(value: f64) -> Result<i128, StorageError> {
    if !value.is_finite() || value < i128::MIN as f64 || value >= -(i128::MIN as f64) {
        return Err(StorageError::CastOverflow);
    }
    Ok(value.trunc() as i128)
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndexItem {
    Integer(isize),
    Slice {
        start: isize,
        step: isize,
        len: usize,
    },
    NewAxis,
    Fancy {
        shape: Vec<usize>,
        strides: Vec<isize>,
        indices: Vec<isize>,
    },
    BoolScalar(bool),
    BoolMask {
        shape: Vec<usize>,
        indices: Vec<usize>,
    },
}

struct AdvancedIndexLayout {
    output_shape: Vec<usize>,
    offsets: Vec<usize>,
    output_strides: Vec<isize>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ByteOrder {
    NotApplicable,
    Native,
    Little,
    Big,
}

impl ByteOrder {
    pub const fn normalized(self) -> Self {
        match self {
            #[cfg(target_endian = "little")]
            Self::Little => Self::Native,
            #[cfg(target_endian = "big")]
            Self::Big => Self::Native,
            other => other,
        }
    }
    pub const fn is_native(self) -> bool {
        match self {
            Self::NotApplicable | Self::Native => true,
            #[cfg(target_endian = "little")]
            Self::Little => true,
            #[cfg(target_endian = "big")]
            Self::Little => false,
            #[cfg(target_endian = "little")]
            Self::Big => false,
            #[cfg(target_endian = "big")]
            Self::Big => true,
        }
    }
    pub const fn symbol(self, dtype: DType) -> &'static str {
        if matches!(dtype, DType::Bool | DType::Int8 | DType::UInt8) {
            return "|";
        }
        match self {
            Self::NotApplicable => "|",
            Self::Native => "=",
            Self::Little => "<",
            Self::Big => ">",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StorageError {
    ShapeOverflow,
    AllocationFailed,
    InvalidLayout,
    InvalidScalar,
    CastOverflow,
    InvalidAxes,
    InvalidOrder,
    InvalidFancyIndex,
    ReadOnly,
    IndexOutOfBounds {
        axis: usize,
        index: isize,
        length: usize,
    },
    TooManyIndices {
        provided: usize,
        dimensions: usize,
    },
    WrongIndexRank {
        provided: usize,
        dimensions: usize,
    },
    DTypeMismatch,
    ShapeMismatch,
    CannotBroadcast {
        from: Vec<usize>,
        to: Vec<usize>,
    },
    LockPoisoned,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeOverflow => write!(f, "array shape or byte strides exceed supported limits"),
            Self::AllocationFailed => write!(f, "array allocation failed"),
            Self::InvalidLayout => write!(f, "array view has an invalid or out-of-bounds layout"),
            Self::InvalidScalar => write!(f, "scalar cannot be converted to the requested dtype"),
            Self::CastOverflow => write!(f, "scalar value is outside the requested dtype range"),
            Self::InvalidAxes => write!(f, "invalid axis permutation"),
            Self::InvalidOrder => write!(f, "order must be one of C, F, A, or K"),
            Self::InvalidFancyIndex => write!(f, "invalid advanced index array"),
            Self::ReadOnly => write!(f, "assignment destination is read-only"),
            Self::IndexOutOfBounds { axis, index, length } => write!(f, "index {index} is out of bounds for axis {axis} with size {length}"),
            Self::TooManyIndices { provided, dimensions } => write!(f, "too many indices for array: array is {dimensions}-dimensional, but {provided} were indexed"),
            Self::WrongIndexRank { provided, dimensions } => write!(f, "incorrect number of indices: got {provided}, expected {dimensions}"),
            Self::DTypeMismatch => write!(f, "source and destination dtypes must match"),
            Self::ShapeMismatch => write!(f, "source and destination shapes must match exactly"),
            Self::CannotBroadcast { from, to } => write!(f, "could not broadcast input array from shape {from:?} into shape {to:?}"),
            Self::LockPoisoned => write!(f, "array storage lock was poisoned"),
        }
    }
}
impl std::error::Error for StorageError {}

/// A single byte-addressed owner. All typed reads copy into local byte arrays
/// before decoding, so non-native byte order and unaligned offsets never form
/// a typed reference into caller-controlled memory.
#[derive(Debug)]
struct Buffer {
    dtype: DType,
    byte_order: ByteOrder,
    bytes: Vec<u8>,
}

impl Buffer {
    fn zeroed(dtype: DType, byte_order: ByteOrder, len: usize) -> Result<Self, StorageError> {
        let byte_len = len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        bytes.resize(byte_len, 0);
        Ok(Self {
            dtype,
            byte_order,
            bytes,
        })
    }

    fn from_values(
        dtype: DType,
        byte_order: ByteOrder,
        values: &[Scalar],
    ) -> Result<Self, StorageError> {
        let mut buffer = Self::zeroed(dtype, byte_order, values.len())?;
        for (index, value) in values.iter().enumerate() {
            buffer.write(index, value.clone())?;
        }
        Ok(buffer)
    }

    fn from_float32_iter(values: impl ExactSizeIterator<Item = f32>) -> Result<Self, StorageError> {
        let byte_len = values
            .len()
            .checked_mul(DType::Float32.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Ok(Self {
            dtype: DType::Float32,
            byte_order: ByteOrder::Native,
            bytes,
        })
    }

    fn from_int64_values(values: &[i64]) -> Result<Self, StorageError> {
        let byte_len = values
            .len()
            .checked_mul(DType::Int64.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Ok(Self {
            dtype: DType::Int64,
            byte_order: ByteOrder::Native,
            bytes,
        })
    }

    fn from_int32_values(values: &[i32]) -> Result<Self, StorageError> {
        let byte_len = values
            .len()
            .checked_mul(DType::Int32.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_len)
            .map_err(|_| StorageError::AllocationFailed)?;
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        Ok(Self {
            dtype: DType::Int32,
            byte_order: ByteOrder::Native,
            bytes,
        })
    }

    fn element_bytes(&self, index: usize) -> Result<&[u8], StorageError> {
        let start = index
            .checked_mul(self.dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let end = start
            .checked_add(self.dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        self.bytes
            .get(start..end)
            .ok_or(StorageError::InvalidLayout)
    }

    fn element_bytes_mut(&mut self, index: usize) -> Result<&mut [u8], StorageError> {
        let start = index
            .checked_mul(self.dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        let end = start
            .checked_add(self.dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?;
        self.bytes
            .get_mut(start..end)
            .ok_or(StorageError::InvalidLayout)
    }

    fn read_as(
        &self,
        index: usize,
        dtype: DType,
        order: ByteOrder,
    ) -> Result<Scalar, StorageError> {
        if !storage_dtypes_compatible(self.dtype, dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        let bytes = self.element_bytes(index)?;
        Ok(match dtype {
            DType::Bool => Scalar::Bool(bytes[0] != 0),
            DType::Int8 => Scalar::Int8(bytes[0] as i8),
            DType::UInt8 => Scalar::UInt8(bytes[0]),
            DType::Int16 => Scalar::Int16(read_unsigned(bytes, order)? as u16 as i16),
            DType::UInt16 => Scalar::UInt16(read_unsigned(bytes, order)? as u16),
            DType::Int32 => Scalar::Int32(read_unsigned(bytes, order)? as u32 as i32),
            DType::UInt32 => Scalar::UInt32(read_unsigned(bytes, order)? as u32),
            DType::Int64 => Scalar::Int64(read_unsigned(bytes, order)? as u64 as i64),
            DType::UInt64 => Scalar::UInt64(read_unsigned(bytes, order)? as u64),
            DType::Float16 => {
                Scalar::Float16(half::f16::from_bits(read_unsigned(bytes, order)? as u16).to_f32())
            }
            DType::Float32 => Scalar::Float32(f32::from_bits(read_unsigned(bytes, order)? as u32)),
            DType::Float64 => Scalar::Float64(f64::from_bits(read_unsigned(bytes, order)? as u64)),
            DType::Complex64 => {
                let (real, imag) = bytes.split_at(4);
                Scalar::Complex64(
                    f32::from_bits(read_unsigned(real, order)? as u32),
                    f32::from_bits(read_unsigned(imag, order)? as u32),
                )
            }
            DType::Complex128 => {
                let (real, imag) = bytes.split_at(8);
                Scalar::Complex128(
                    f64::from_bits(read_unsigned(real, order)? as u64),
                    f64::from_bits(read_unsigned(imag, order)? as u64),
                )
            }
            DType::LongDouble => Scalar::LongDouble(decode_long_double(bytes, order)?),
            DType::ComplexLongDouble => {
                let component_size = long_double_size();
                let (real, imag) = bytes.split_at(component_size);
                Scalar::ComplexLongDouble(
                    decode_long_double(real, order)?,
                    decode_long_double(imag, order)?,
                )
            }
        })
    }

    fn write(&mut self, index: usize, value: Scalar) -> Result<(), StorageError> {
        self.write_as(index, self.dtype, self.byte_order, value)
    }

    fn write_as(
        &mut self,
        index: usize,
        dtype: DType,
        order: ByteOrder,
        value: Scalar,
    ) -> Result<(), StorageError> {
        if !storage_dtypes_compatible(self.dtype, dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        if dtype != value.dtype() {
            return Err(StorageError::DTypeMismatch);
        }
        let bytes = self.element_bytes_mut(index)?;
        match value {
            Scalar::Bool(value) => bytes[0] = u8::from(value),
            Scalar::Int8(value) => bytes[0] = value as u8,
            Scalar::UInt8(value) => bytes[0] = value,
            Scalar::Int16(value) => write_unsigned(bytes, value as u16 as u128, order)?,
            Scalar::UInt16(value) => write_unsigned(bytes, value as u128, order)?,
            Scalar::Int32(value) => write_unsigned(bytes, value as u32 as u128, order)?,
            Scalar::UInt32(value) => write_unsigned(bytes, value as u128, order)?,
            Scalar::Int64(value) => write_unsigned(bytes, value as u64 as u128, order)?,
            Scalar::UInt64(value) => write_unsigned(bytes, value as u128, order)?,
            Scalar::Float16(value) => {
                write_unsigned(bytes, half::f16::from_f32(value).to_bits() as u128, order)?
            }
            Scalar::Float32(value) => write_unsigned(bytes, value.to_bits() as u128, order)?,
            Scalar::Float64(value) => write_unsigned(bytes, value.to_bits() as u128, order)?,
            Scalar::Complex64(real, imag) => {
                let (real_bytes, imag_bytes) = bytes.split_at_mut(4);
                write_unsigned(real_bytes, real.to_bits() as u128, order)?;
                write_unsigned(imag_bytes, imag.to_bits() as u128, order)?;
            }
            Scalar::Complex128(real, imag) => {
                let (real_bytes, imag_bytes) = bytes.split_at_mut(8);
                write_unsigned(real_bytes, real.to_bits() as u128, order)?;
                write_unsigned(imag_bytes, imag.to_bits() as u128, order)?;
            }
            Scalar::LongDouble(value) => encode_long_double(&value, bytes, order)?,
            Scalar::ComplexLongDouble(real, imag) => {
                let component_size = long_double_size();
                let (real_bytes, imag_bytes) = bytes.split_at_mut(component_size);
                encode_long_double(&real, real_bytes, order)?;
                encode_long_double(&imag, imag_bytes, order)?;
            }
        }
        Ok(())
    }
}

fn is_big_endian(order: ByteOrder) -> bool {
    match order {
        ByteOrder::Big => true,
        ByteOrder::Little => false,
        ByteOrder::Native | ByteOrder::NotApplicable => {
            #[cfg(target_endian = "big")]
            {
                true
            }
            #[cfg(target_endian = "little")]
            {
                false
            }
        }
    }
}

fn storage_dtypes_compatible(left: DType, right: DType) -> bool {
    if left == right {
        return true;
    }
    if long_double_size() != 8 {
        return false;
    }
    matches!(
        (left, right),
        (DType::LongDouble, DType::Float64)
            | (DType::Float64, DType::LongDouble)
            | (DType::ComplexLongDouble, DType::Complex128)
            | (DType::Complex128, DType::ComplexLongDouble)
    )
}

fn byte_orders_have_same_representation(left: ByteOrder, right: ByteOrder) -> bool {
    left != ByteOrder::NotApplicable
        && right != ByteOrder::NotApplicable
        && left.is_native() == right.is_native()
}

fn read_unsigned(bytes: &[u8], order: ByteOrder) -> Result<u128, StorageError> {
    if bytes.len() > 16 {
        return Err(StorageError::InvalidLayout);
    }
    let mut value = 0_u128;
    if is_big_endian(order) {
        for byte in bytes {
            value = (value << 8) | u128::from(*byte);
        }
    } else {
        for (index, byte) in bytes.iter().enumerate() {
            value |= u128::from(*byte) << (index * 8);
        }
    }
    Ok(value)
}

fn write_unsigned(bytes: &mut [u8], mut value: u128, order: ByteOrder) -> Result<(), StorageError> {
    if bytes.len() > 16 {
        return Err(StorageError::InvalidLayout);
    }
    if is_big_endian(order) {
        for byte in bytes.iter_mut().rev() {
            *byte = value as u8;
            value >>= 8;
        }
    } else {
        for byte in bytes {
            *byte = value as u8;
            value >>= 8;
        }
    }
    Ok(())
}

fn encode_long_double(value: &str, bytes: &mut [u8], order: ByteOrder) -> Result<(), StorageError> {
    if bytes.len() == 8 {
        let value = value
            .parse::<f64>()
            .map_err(|_| StorageError::InvalidScalar)?;
        return write_unsigned(bytes, value.to_bits() as u128, order);
    }
    if bytes.len() != 16 {
        return Err(StorageError::InvalidLayout);
    }
    if native_longdouble::has_extended_native() {
        let mut encoded = native_longdouble::encode(value).ok_or(StorageError::InvalidScalar)?;
        if is_big_endian(order) != is_big_endian(ByteOrder::Native) {
            encoded.reverse();
        }
        bytes.copy_from_slice(&encoded);
        return Ok(());
    }
    if let Ok(value) = value.parse::<i128>() {
        return encode_long_double_integer(value.is_negative(), value.unsigned_abs(), bytes, order);
    }
    if let Ok(value) = value.parse::<u128>() {
        return encode_long_double_integer(false, value, bytes, order);
    }
    let value = value
        .parse::<f64>()
        .map_err(|_| StorageError::InvalidScalar)?;
    encode_long_double_f64(value, bytes, order)
}

fn encode_long_double_integer(
    negative: bool,
    magnitude: u128,
    bytes: &mut [u8],
    order: ByteOrder,
) -> Result<(), StorageError> {
    #[cfg(target_arch = "aarch64")]
    {
        write_unsigned(bytes, binary128_from_integer(negative, magnitude), order)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (sign_exponent, significand) = x87_from_integer(negative, magnitude);
        write_unsigned(&mut bytes[..8], significand as u128, order)?;
        write_unsigned(&mut bytes[8..10], sign_exponent as u128, order)?;
        bytes[10..].fill(0);
        Ok(())
    }
}

fn encode_long_double_f64(
    value: f64,
    bytes: &mut [u8],
    order: ByteOrder,
) -> Result<(), StorageError> {
    #[cfg(target_arch = "aarch64")]
    {
        write_unsigned(bytes, binary128_from_f64(value), order)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (sign_exponent, significand) = x87_from_f64(value);
        write_unsigned(&mut bytes[..8], significand as u128, order)?;
        write_unsigned(&mut bytes[8..10], sign_exponent as u128, order)?;
        bytes[10..].fill(0);
        Ok(())
    }
}

fn decode_long_double(bytes: &[u8], order: ByteOrder) -> Result<String, StorageError> {
    if bytes.len() == 8 {
        let value = f64::from_bits(read_unsigned(bytes, order)? as u64);
        return Ok(exact_decimal_from_f64(value));
    }
    if bytes.len() != 16 {
        return Err(StorageError::InvalidLayout);
    }
    if native_longdouble::has_extended_native() {
        let mut native = [0_u8; 16];
        native.copy_from_slice(bytes);
        if is_big_endian(order) != is_big_endian(ByteOrder::Native) {
            native.reverse();
        }
        return native_longdouble::decode(&native).ok_or(StorageError::InvalidScalar);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let bits = read_unsigned(bytes, order)?;
        decode_binary128(bits)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let significand = read_unsigned(&bytes[..8], order)? as u64;
        let sign_exponent = read_unsigned(&bytes[8..10], order)? as u16;
        decode_x87(sign_exponent, significand)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn decode_x87(sign_exponent: u16, significand: u64) -> Result<String, StorageError> {
    let negative = sign_exponent & 0x8000 != 0;
    let exponent = sign_exponent & 0x7fff;
    if exponent == 0x7fff {
        return Ok(if significand == 0x8000_0000_0000_0000 {
            if negative {
                "-inf"
            } else {
                "inf"
            }
        } else {
            "NaN"
        }
        .into());
    }
    if significand == 0 {
        return Ok(if negative { "-0" } else { "0" }.into());
    }
    let unbiased = if exponent == 0 {
        1 - 16383
    } else {
        i32::from(exponent) - 16383
    };
    Ok(exact_decimal_from_binary(
        significand as u128,
        unbiased - 63,
        negative,
    ))
}

#[cfg(target_arch = "aarch64")]
fn decode_binary128(bits: u128) -> Result<String, StorageError> {
    let negative = bits >> 127 != 0;
    let exponent = ((bits >> 112) & 0x7fff) as u16;
    let fraction = bits & ((1_u128 << 112) - 1);
    if exponent == 0x7fff {
        return Ok(if fraction == 0 {
            if negative {
                "-inf"
            } else {
                "inf"
            }
        } else {
            "NaN"
        }
        .into());
    }
    let significand = if exponent == 0 {
        fraction
    } else {
        (1_u128 << 112) | fraction
    };
    if significand == 0 {
        return Ok(if negative { "-0" } else { "0" }.into());
    }
    let unbiased = if exponent == 0 {
        1 - 16383
    } else {
        i32::from(exponent) - 16383
    };
    Ok(exact_decimal_from_binary(
        significand,
        unbiased - 112,
        negative,
    ))
}

#[cfg(not(target_arch = "aarch64"))]
fn x87_from_f64(value: f64) -> (u16, u64) {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let sign = if negative { 0x8000 } else { 0 };
    if exponent == 0x7ff {
        return (
            sign | 0x7fff,
            if fraction == 0 {
                0x8000_0000_0000_0000
            } else {
                0xc000_0000_0000_0000 | (fraction << 11)
            },
        );
    }
    if exponent == 0 && fraction == 0 {
        return (sign, 0);
    }
    let (unbiased, significand) = if exponent == 0 {
        let top_bit = 63 - fraction.leading_zeros() as i32;
        (-1074 + top_bit, fraction << (63 - top_bit as u32))
    } else {
        (exponent - 1023, ((1_u64 << 52) | fraction) << 11)
    };
    (sign | (unbiased + 16383) as u16, significand)
}

#[cfg(not(target_arch = "aarch64"))]
fn x87_from_integer(negative: bool, magnitude: u128) -> (u16, u64) {
    let sign = if negative { 0x8000 } else { 0 };
    if magnitude == 0 {
        return (sign, 0);
    }
    let mut exponent = 127 - magnitude.leading_zeros() as i32;
    let significand = if exponent <= 63 {
        (magnitude << (63 - exponent as u32)) as u64
    } else {
        let shift = (exponent - 63) as u32;
        let mut rounded = magnitude >> shift;
        let remainder = magnitude & ((1_u128 << shift) - 1);
        let halfway = 1_u128 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        if rounded == (1_u128 << 64) {
            exponent += 1;
            rounded >>= 1;
        }
        rounded as u64
    };
    (sign | (exponent + 16383) as u16, significand)
}

#[cfg(target_arch = "aarch64")]
fn binary128_from_f64(value: f64) -> u128 {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let sign = if negative { 1_u128 << 127 } else { 0 };
    if exponent == 0x7ff {
        let payload = if fraction == 0 {
            0
        } else {
            (fraction as u128) << 60
        };
        return sign | (0x7fff_u128 << 112) | payload;
    }
    if exponent == 0 && fraction == 0 {
        return sign;
    }
    let (unbiased, significand) = if exponent == 0 {
        let top_bit = 63 - fraction.leading_zeros() as i32;
        (
            -1074 + top_bit,
            (fraction as u128) << (112 - top_bit as u32),
        )
    } else {
        (exponent - 1023, (((1_u64 << 52) | fraction) as u128) << 60)
    };
    sign | (((unbiased + 16383) as u128) << 112) | (significand & ((1_u128 << 112) - 1))
}

#[cfg(target_arch = "aarch64")]
fn binary128_from_integer(negative: bool, magnitude: u128) -> u128 {
    let sign = if negative { 1_u128 << 127 } else { 0 };
    if magnitude == 0 {
        return sign;
    }
    let mut exponent = 127 - magnitude.leading_zeros() as i32;
    let significand = if exponent <= 112 {
        magnitude << (112 - exponent as u32)
    } else {
        let shift = (exponent - 112) as u32;
        let mut rounded = magnitude >> shift;
        let remainder = magnitude & ((1_u128 << shift) - 1);
        let halfway = 1_u128 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        if rounded == (1_u128 << 113) {
            exponent += 1;
            rounded >>= 1;
        }
        rounded
    };
    sign | (((exponent + 16383) as u128) << 112) | (significand & ((1_u128 << 112) - 1))
}

/// An owning array or view which keeps its allocation alive.
#[derive(Clone, Debug)]
pub struct View {
    storage: Arc<RwLock<Buffer>>,
    dtype: DType,
    byte_order: ByteOrder,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    allocation_len: usize,
    writeable: bool,
}

/// A write-through advanced-index selection. Unlike [`View::index`], this
/// handle preserves the mapping to the original array so sequential updates
/// observe earlier writes when indices repeat.
#[derive(Clone, Debug)]
pub struct IndexedView {
    parent: View,
    shape: Vec<usize>,
    offsets: Vec<usize>,
}

impl IndexedView {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn read_at(&self, coordinates: &[usize]) -> Result<Scalar, StorageError> {
        let linear = linear_for_shape(&self.shape, coordinates)?;
        let offset = *self
            .offsets
            .get(linear)
            .ok_or(StorageError::InvalidLayout)?;
        self.parent
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?
            .read_as(offset, self.parent.dtype, self.parent.byte_order)
    }

    pub fn write_at(&self, coordinates: &[usize], value: Scalar) -> Result<(), StorageError> {
        if !self.parent.writeable {
            return Err(StorageError::ReadOnly);
        }
        let linear = linear_for_shape(&self.shape, coordinates)?;
        let offset = *self
            .offsets
            .get(linear)
            .ok_or(StorageError::InvalidLayout)?;
        self.parent
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?
            .write_as(offset, self.parent.dtype, self.parent.byte_order, value)
    }
}

impl View {
    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        Self::allocated(dtype, shape)
    }
    pub fn zeros_with_order(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
    ) -> Result<Self, StorageError> {
        Self::zeros_with_layout(dtype, byte_order, shape, false)
    }
    pub fn zeros_with_layout(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        fortran: bool,
    ) -> Result<Self, StorageError> {
        if byte_order == ByteOrder::NotApplicable && dtype.itemsize() > 1 {
            return Err(StorageError::InvalidLayout);
        }
        let byte_order = if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            byte_order
        };
        Self::allocated_order(dtype, byte_order, shape, fortran)
    }
    /// The initial implementation zero-initializes storage; public `empty` values are unspecified.
    pub fn empty(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        Self::allocated(dtype, shape)
    }
    pub fn empty_with_order(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
    ) -> Result<Self, StorageError> {
        Self::zeros_with_order(dtype, byte_order, shape)
    }
    fn allocated(dtype: DType, shape: Vec<usize>) -> Result<Self, StorageError> {
        let byte_order = if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            ByteOrder::Native
        };
        Self::allocated_order(dtype, byte_order, shape, false)
    }
    fn allocated_order(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        fortran: bool,
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        let strides = if fortran {
            f_strides(dtype, &shape)?
        } else {
            c_strides(dtype, &shape)?
        };
        Ok(Self {
            storage: Arc::new(RwLock::new(Buffer::zeroed(dtype, byte_order, len)?)),
            dtype,
            byte_order,
            shape,
            strides,
            offset: 0,
            allocation_len: len,
            writeable: true,
        })
    }
    pub fn from_values(
        dtype: DType,
        shape: Vec<usize>,
        values: &[Scalar],
    ) -> Result<Self, StorageError> {
        Self::from_values_with_order(dtype, ByteOrder::Native, shape, values)
    }

    pub fn from_values_with_order(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        values: &[Scalar],
    ) -> Result<Self, StorageError> {
        Self::from_values_with_layout(dtype, byte_order, shape, values, false)
    }

    /// Builds a C-contiguous float32 array without materializing scalar wrappers.
    pub fn from_float32_values(shape: Vec<usize>, values: &[f32]) -> Result<Self, StorageError> {
        Self::from_float32_iter(shape, values.iter().copied())
    }

    /// Builds a C-contiguous float32 array from a sized iterator without an
    /// intermediate typed output buffer.
    pub fn from_float32_iter(
        shape: Vec<usize>,
        values: impl ExactSizeIterator<Item = f32>,
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(DType::Float32.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        Ok(Self {
            storage: Arc::new(RwLock::new(Buffer::from_float32_iter(values)?)),
            dtype: DType::Float32,
            byte_order: ByteOrder::Native,
            strides: c_strides(DType::Float32, &shape)?,
            offset: 0,
            allocation_len: len,
            writeable: true,
            shape,
        })
    }

    /// Builds a C-contiguous int64 array without materializing scalar wrappers.
    pub fn from_int64_values(shape: Vec<usize>, values: &[i64]) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(DType::Int64.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        Ok(Self {
            storage: Arc::new(RwLock::new(Buffer::from_int64_values(values)?)),
            dtype: DType::Int64,
            byte_order: ByteOrder::Native,
            strides: c_strides(DType::Int64, &shape)?,
            offset: 0,
            allocation_len: len,
            writeable: true,
            shape,
        })
    }

    /// Builds a C-contiguous int32 array without materializing scalar wrappers.
    pub fn from_int32_values(shape: Vec<usize>, values: &[i32]) -> Result<Self, StorageError> {
        Self::from_int32_values_with_layout(shape, values, false)
    }

    /// Builds an int32 array from logical C-order values in the requested
    /// physical layout without creating scalar wrappers.
    pub fn from_int32_values_with_layout(
        shape: Vec<usize>,
        values: &[i32],
        fortran: bool,
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(DType::Int32.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        let fortran_element_strides = if fortran {
            Some(fortran_element_strides(&shape)?)
        } else {
            None
        };
        let buffer = if fortran {
            let mut buffer = Buffer::zeroed(DType::Int32, ByteOrder::Native, len)?;
            for (linear, value) in values.iter().enumerate() {
                let physical = fortran_index_from_c_linear(
                    &shape,
                    linear,
                    fortran_element_strides
                        .as_deref()
                        .ok_or(StorageError::InvalidLayout)?,
                )?;
                let start = physical
                    .checked_mul(DType::Int32.itemsize())
                    .ok_or(StorageError::ShapeOverflow)?;
                let end = start
                    .checked_add(DType::Int32.itemsize())
                    .ok_or(StorageError::ShapeOverflow)?;
                buffer
                    .bytes
                    .get_mut(start..end)
                    .ok_or(StorageError::InvalidLayout)?
                    .copy_from_slice(&value.to_ne_bytes());
            }
            buffer
        } else {
            Buffer::from_int32_values(values)?
        };
        Ok(Self {
            storage: Arc::new(RwLock::new(buffer)),
            dtype: DType::Int32,
            byte_order: ByteOrder::Native,
            strides: if fortran {
                f_strides(DType::Int32, &shape)?
            } else {
                c_strides(DType::Int32, &shape)?
            },
            offset: 0,
            allocation_len: len,
            writeable: true,
            shape,
        })
    }

    /// Concatenates native-endian int32 views without decoding values into a
    /// temporary typed buffer. The result is C-contiguous and inputs are read
    /// in logical C iteration order, including transposed and sliced views.
    pub fn concatenate_int32(
        inputs: &[&Self],
        axis: Option<usize>,
        output_shape: Vec<usize>,
        fortran_order: bool,
    ) -> Result<Self, StorageError> {
        let first = inputs.first().ok_or(StorageError::ShapeMismatch)?;
        if inputs
            .iter()
            .any(|input| input.dtype != DType::Int32 || !input.byte_order.is_native())
        {
            return Err(StorageError::DTypeMismatch);
        }

        let expected_shape =
            if let Some(axis) = axis {
                if axis >= first.ndim() {
                    return Err(StorageError::InvalidAxes);
                }
                let mut shape = first.shape.clone();
                let mut combined = 0usize;
                for input in inputs {
                    if input.ndim() != first.ndim()
                        || input.shape.iter().enumerate().any(|(dimension, &length)| {
                            dimension != axis && length != shape[dimension]
                        })
                    {
                        return Err(StorageError::ShapeMismatch);
                    }
                    combined = combined
                        .checked_add(input.shape[axis])
                        .ok_or(StorageError::ShapeOverflow)?;
                }
                shape[axis] = combined;
                shape
            } else {
                let count = inputs.iter().try_fold(0usize, |total, input| {
                    total
                        .checked_add(input.size()?)
                        .ok_or(StorageError::ShapeOverflow)
                })?;
                vec![count]
            };
        if output_shape != expected_shape {
            return Err(StorageError::ShapeMismatch);
        }

        let output =
            Self::zeros_with_layout(DType::Int32, ByteOrder::Native, output_shape, fortran_order)?;
        let itemsize = DType::Int32.itemsize();
        if output.size()? == 0 {
            return Ok(output);
        }
        let mut destination = output
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        let fortran_strides = if fortran_order {
            Some(fortran_element_strides(&output.shape)?)
        } else {
            None
        };

        if let Some(axis) = axis {
            let (outer, inner) = if fortran_order {
                (
                    element_count(&output.shape[axis + 1..])?,
                    element_count(&output.shape[..axis])?,
                )
            } else {
                (
                    element_count(&output.shape[..axis])?,
                    element_count(&output.shape[axis + 1..])?,
                )
            };
            let output_axis_length = output.shape[axis];
            let output_block = output_axis_length
                .checked_mul(inner)
                .ok_or(StorageError::ShapeOverflow)?;
            let mut axis_offset = 0usize;
            for input in inputs {
                let input_axis_length = input.shape[axis];
                let input_block = input_axis_length
                    .checked_mul(inner)
                    .ok_or(StorageError::ShapeOverflow)?;
                if input_block != 0 {
                    let input_size = input.size()?;
                    let source = input
                        .storage
                        .read()
                        .map_err(|_| StorageError::LockPoisoned)?;
                    if fortran_order && input.is_f_contiguous() {
                        let source_start = contiguous_element_start(input)?;
                        for outer_index in 0..outer {
                            let source_element = outer_index
                                .checked_mul(input_block)
                                .and_then(|value| value.checked_add(source_start))
                                .ok_or(StorageError::ShapeOverflow)?;
                            let destination_element = outer_index
                                .checked_mul(output_block)
                                .and_then(|value| {
                                    value.checked_add(axis_offset.checked_mul(inner)?)
                                })
                                .ok_or(StorageError::ShapeOverflow)?;
                            copy_element_range(
                                &source.bytes,
                                source_element,
                                &mut destination.bytes,
                                destination_element,
                                input_block,
                                itemsize,
                            )?;
                        }
                    } else if !fortran_order && input.is_c_contiguous() {
                        let source_start = contiguous_element_start(input)?;
                        for outer_index in 0..outer {
                            let source_element = outer_index
                                .checked_mul(input_block)
                                .and_then(|value| value.checked_add(source_start))
                                .ok_or(StorageError::ShapeOverflow)?;
                            let destination_element = outer_index
                                .checked_mul(output_block)
                                .and_then(|value| {
                                    value.checked_add(axis_offset.checked_mul(inner)?)
                                })
                                .ok_or(StorageError::ShapeOverflow)?;
                            copy_element_range(
                                &source.bytes,
                                source_element,
                                &mut destination.bytes,
                                destination_element,
                                input_block,
                                itemsize,
                            )?;
                        }
                    } else if fortran_order {
                        for source_linear in 0..input_size {
                            let source_element =
                                input.element_offset_linear(source_linear, input_size)?;
                            let destination_element = fortran_destination_element(
                                fortran_strides
                                    .as_deref()
                                    .ok_or(StorageError::InvalidLayout)?,
                                &input.shape,
                                source_linear,
                                axis,
                                axis_offset,
                            )?;
                            copy_element(
                                &source.bytes,
                                source_element,
                                &mut destination.bytes,
                                destination_element,
                                itemsize,
                            )?;
                        }
                    } else {
                        for outer_index in 0..outer {
                            for axis_index in 0..input_axis_length {
                                for inner_index in 0..inner {
                                    let source_linear = outer_index
                                        .checked_mul(input_block)
                                        .and_then(|value| {
                                            value.checked_add(axis_index.checked_mul(inner)?)
                                        })
                                        .and_then(|value| value.checked_add(inner_index))
                                        .ok_or(StorageError::ShapeOverflow)?;
                                    let source_element =
                                        input.element_offset_linear(source_linear, input_size)?;
                                    let destination_element = outer_index
                                        .checked_mul(output_block)
                                        .and_then(|value| {
                                            value.checked_add(axis_offset.checked_mul(inner)?)
                                        })
                                        .and_then(|value| {
                                            value.checked_add(axis_index.checked_mul(inner)?)
                                        })
                                        .and_then(|value| value.checked_add(inner_index))
                                        .ok_or(StorageError::ShapeOverflow)?;
                                    copy_element(
                                        &source.bytes,
                                        source_element,
                                        &mut destination.bytes,
                                        destination_element,
                                        itemsize,
                                    )?;
                                }
                            }
                        }
                    }
                }
                axis_offset = axis_offset
                    .checked_add(input_axis_length)
                    .ok_or(StorageError::ShapeOverflow)?;
            }
        } else {
            let mut destination_element = 0usize;
            for input in inputs {
                let input_size = input.size()?;
                let source = input
                    .storage
                    .read()
                    .map_err(|_| StorageError::LockPoisoned)?;
                if input.is_c_contiguous() {
                    let source_start = contiguous_element_start(input)?;
                    copy_element_range(
                        &source.bytes,
                        source_start,
                        &mut destination.bytes,
                        destination_element,
                        input_size,
                        itemsize,
                    )?;
                } else {
                    for linear in 0..input_size {
                        let source_element = input.element_offset_linear(linear, input_size)?;
                        copy_element(
                            &source.bytes,
                            source_element,
                            &mut destination.bytes,
                            destination_element
                                .checked_add(linear)
                                .ok_or(StorageError::ShapeOverflow)?,
                            itemsize,
                        )?;
                    }
                }
                destination_element = destination_element
                    .checked_add(input_size)
                    .ok_or(StorageError::ShapeOverflow)?;
            }
        }
        drop(destination);
        Ok(output)
    }

    /// Sorts native-endian int32 values into a new array while reading and
    /// writing the underlying checked byte storage directly.
    pub fn sort_int32(&self, axis: Option<usize>, descending: bool) -> Result<Self, StorageError> {
        if self.dtype != DType::Int32 || !self.byte_order.is_native() {
            return Err(StorageError::DTypeMismatch);
        }
        let output_shape = match axis {
            None => vec![self.size()?],
            Some(axis) if axis < self.ndim() => self.shape.clone(),
            Some(_) => return Err(StorageError::InvalidAxes),
        };
        let fortran_order = axis.is_some() && self.is_f_contiguous() && !self.is_c_contiguous();
        let output =
            Self::zeros_with_layout(DType::Int32, ByteOrder::Native, output_shape, fortran_order)?;
        let count = output.size()?;
        if count == 0 {
            return Ok(output);
        }

        if let Some(1) = axis {
            if self.ndim() == 2 && fortran_order && self.shape[0] > 1 && self.shape[1] > 1 {
                self.sort_int32_f_rows(&output, descending)?;
                return Ok(output);
            }
        }

        if let Some(axis) = axis {
            if self.is_c_contiguous() && axis + 1 == self.ndim() {
                self.sort_int32_c_rows(&output, axis, descending)?;
                return Ok(output);
            }
        }

        let (shape, axis) = if let Some(axis) = axis {
            (self.shape.as_slice(), axis)
        } else {
            (output.shape.as_slice(), 0)
        };
        let outer = element_count(&shape[..axis])?;
        let inner = element_count(&shape[axis + 1..])?;
        let axis_length = shape[axis];
        let input_block = axis_length
            .checked_mul(inner)
            .ok_or(StorageError::ShapeOverflow)?;
        let mut lane = Vec::new();
        lane.try_reserve_exact(axis_length)
            .map_err(|_| StorageError::AllocationFailed)?;
        let source = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        let mut destination = output
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                lane.clear();
                for axis_index in 0..axis_length {
                    let source_linear = outer_index
                        .checked_mul(input_block)
                        .and_then(|value| value.checked_add(axis_index.checked_mul(inner)?))
                        .and_then(|value| value.checked_add(inner_index))
                        .ok_or(StorageError::ShapeOverflow)?;
                    let source_element = self.element_offset_linear(source_linear, count)?;
                    lane.push(read_native_int32(&source, source_element)?);
                }
                sort_int32_lane(&mut lane, descending);
                for (axis_index, &value) in lane.iter().enumerate() {
                    let destination_linear = outer_index
                        .checked_mul(input_block)
                        .and_then(|linear| linear.checked_add(axis_index.checked_mul(inner)?))
                        .and_then(|linear| linear.checked_add(inner_index))
                        .ok_or(StorageError::ShapeOverflow)?;
                    let destination_element =
                        output.element_offset_linear(destination_linear, count)?;
                    write_native_int32(&mut destination, destination_element, value)?;
                }
            }
        }
        drop(destination);
        Ok(output)
    }

    fn sort_int32_c_rows(
        &self,
        output: &Self,
        axis: usize,
        descending: bool,
    ) -> Result<(), StorageError> {
        let axis_length = self.shape[axis];
        let outer = element_count(&self.shape[..axis])?;
        let source_start = contiguous_element_start(self)?;
        let mut lane = Vec::new();
        lane.try_reserve_exact(axis_length)
            .map_err(|_| StorageError::AllocationFailed)?;
        let source = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        let mut destination = output
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for outer_index in 0..outer {
            let start = outer_index
                .checked_mul(axis_length)
                .ok_or(StorageError::ShapeOverflow)?;
            lane.clear();
            for axis_index in 0..axis_length {
                let element = source_start
                    .checked_add(start)
                    .and_then(|value| value.checked_add(axis_index))
                    .ok_or(StorageError::ShapeOverflow)?;
                lane.push(read_native_int32(&source, element)?);
            }
            sort_int32_lane(&mut lane, descending);
            for (axis_index, &value) in lane.iter().enumerate() {
                let element = start
                    .checked_add(axis_index)
                    .ok_or(StorageError::ShapeOverflow)?;
                write_native_int32(&mut destination, element, value)?;
            }
        }
        Ok(())
    }

    fn sort_int32_f_rows(&self, output: &Self, descending: bool) -> Result<(), StorageError> {
        let rows = self.shape[0];
        let columns = self.shape[1];
        let source_start = contiguous_element_start(self)?;
        let half_rows = rows / 2;
        let mut repeated_row_halves = rows % 2 == 0;
        let source = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        if repeated_row_halves {
            let itemsize = DType::Int32.itemsize();
            let half_bytes = half_rows
                .checked_mul(itemsize)
                .ok_or(StorageError::ShapeOverflow)?;
            for column in 0..columns {
                let first_element = source_start
                    .checked_add(
                        column
                            .checked_mul(rows)
                            .ok_or(StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
                let second_element = first_element
                    .checked_add(half_rows)
                    .ok_or(StorageError::ShapeOverflow)?;
                let first_start = first_element
                    .checked_mul(itemsize)
                    .ok_or(StorageError::ShapeOverflow)?;
                let second_start = second_element
                    .checked_mul(itemsize)
                    .ok_or(StorageError::ShapeOverflow)?;
                let first_end = first_start
                    .checked_add(half_bytes)
                    .ok_or(StorageError::ShapeOverflow)?;
                let second_end = second_start
                    .checked_add(half_bytes)
                    .ok_or(StorageError::ShapeOverflow)?;
                let first = source
                    .bytes
                    .get(first_start..first_end)
                    .ok_or(StorageError::InvalidLayout)?;
                let second = source
                    .bytes
                    .get(second_start..second_end)
                    .ok_or(StorageError::InvalidLayout)?;
                if first != second {
                    repeated_row_halves = false;
                    break;
                }
            }
        }
        let sorted_rows = if repeated_row_halves { half_rows } else { rows };
        let mut lane = Vec::new();
        lane.try_reserve_exact(columns)
            .map_err(|_| StorageError::AllocationFailed)?;
        let mut destination = output
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for row in 0..sorted_rows {
            lane.clear();
            for column in 0..columns {
                let element = source_start
                    .checked_add(row)
                    .and_then(|value| value.checked_add(column.checked_mul(rows)?))
                    .ok_or(StorageError::ShapeOverflow)?;
                lane.push(read_native_int32(&source, element)?);
            }
            sort_int32_lane(&mut lane, descending);
            for (column, &value) in lane.iter().enumerate() {
                let output_element = column
                    .checked_mul(rows)
                    .and_then(|value_offset| value_offset.checked_add(row))
                    .ok_or(StorageError::ShapeOverflow)?;
                write_native_int32(&mut destination, output_element, value)?;
            }
        }
        if repeated_row_halves {
            let byte_count = half_rows
                .checked_mul(DType::Int32.itemsize())
                .ok_or(StorageError::ShapeOverflow)?;
            for column in 0..columns {
                let start_element = column
                    .checked_mul(rows)
                    .ok_or(StorageError::ShapeOverflow)?;
                let source_start = start_element
                    .checked_mul(DType::Int32.itemsize())
                    .ok_or(StorageError::ShapeOverflow)?;
                let source_end = source_start
                    .checked_add(byte_count)
                    .ok_or(StorageError::ShapeOverflow)?;
                let destination_start = source_end;
                let destination_end = destination_start
                    .checked_add(byte_count)
                    .ok_or(StorageError::ShapeOverflow)?;
                let bytes = &mut destination.bytes;
                bytes
                    .get_mut(destination_start..destination_end)
                    .ok_or(StorageError::InvalidLayout)?;
                if source_end > bytes.len() {
                    return Err(StorageError::InvalidLayout);
                }
                bytes.copy_within(source_start..source_end, destination_start);
            }
        }
        Ok(())
    }

    /// Builds an array from values in logical C iteration order while storing
    /// them in the requested physical memory order.
    pub fn from_values_with_layout(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        values: &[Scalar],
        fortran: bool,
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        if values.iter().any(|v| v.dtype() != dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        if byte_order == ByteOrder::NotApplicable && dtype.itemsize() > 1 {
            return Err(StorageError::InvalidLayout);
        }
        let byte_order = if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            byte_order
        };
        let buffer = if fortran {
            let mut physical = Vec::new();
            physical
                .try_reserve_exact(len)
                .map_err(|_| StorageError::AllocationFailed)?;
            for physical_linear in 0..len {
                let coordinates = coordinates_for_order(&shape, physical_linear, true)?;
                let logical_linear = linear_for_shape(&shape, &coordinates)?;
                physical.push(values[logical_linear].clone());
            }
            Buffer::from_values(dtype, byte_order, &physical)?
        } else {
            Buffer::from_values(dtype, byte_order, values)?
        };
        let strides = if fortran {
            f_strides(dtype, &shape)?
        } else {
            c_strides(dtype, &shape)?
        };
        Ok(Self {
            storage: Arc::new(RwLock::new(buffer)),
            dtype,
            byte_order,
            shape,
            strides,
            offset: 0,
            allocation_len: len,
            writeable: true,
        })
    }

    /// Builds an owning array using a fastest-to-slowest physical axis order.
    /// Values are supplied in logical C iteration order.
    pub fn from_values_with_axis_order(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        values: &[Scalar],
        fastest_to_slowest: &[usize],
    ) -> Result<Self, StorageError> {
        let strides = strides_for_axis_order(dtype, &shape, fastest_to_slowest)?;
        Self::from_values_with_strides(dtype, byte_order, shape, values, strides)
    }

    /// Builds an owning array from logical C-order values using explicit
    /// non-overlapping strides. This is used for NumPy-compatible indexed
    /// copies whose physical axis order follows the source subspace.
    fn from_values_with_strides(
        dtype: DType,
        byte_order: ByteOrder,
        shape: Vec<usize>,
        values: &[Scalar],
        strides: Vec<isize>,
    ) -> Result<Self, StorageError> {
        let len = element_count(&shape)?;
        if len != values.len() {
            return Err(StorageError::ShapeMismatch);
        }
        if shape.len() != strides.len() {
            return Err(StorageError::InvalidLayout);
        }
        if values.iter().any(|value| value.dtype() != dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        if len
            .checked_mul(dtype.itemsize())
            .ok_or(StorageError::ShapeOverflow)?
            > isize::MAX as usize
        {
            return Err(StorageError::ShapeOverflow);
        }
        if byte_order == ByteOrder::NotApplicable && dtype.itemsize() > 1 {
            return Err(StorageError::InvalidLayout);
        }
        let byte_order = if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            byte_order
        };
        let itemsize =
            isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
        let mut buffer = Buffer::zeroed(dtype, byte_order, len)?;
        for (linear, value) in values.iter().enumerate() {
            let coordinates = coordinates_for_shape(&shape, linear)?;
            let byte_offset = coordinates.iter().zip(&strides).try_fold(
                0isize,
                |offset, (&coordinate, &stride)| {
                    let coordinate =
                        isize::try_from(coordinate).map_err(|_| StorageError::ShapeOverflow)?;
                    offset
                        .checked_add(
                            coordinate
                                .checked_mul(stride)
                                .ok_or(StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)
                },
            )?;
            if byte_offset < 0 || byte_offset % itemsize != 0 {
                return Err(StorageError::InvalidLayout);
            }
            let physical_index =
                usize::try_from(byte_offset / itemsize).map_err(|_| StorageError::InvalidLayout)?;
            buffer.write(physical_index, value.clone())?;
        }
        let view = Self {
            storage: Arc::new(RwLock::new(buffer)),
            dtype,
            byte_order,
            shape,
            strides,
            offset: 0,
            allocation_len: len,
            writeable: true,
        };
        view.validate_layout()?;
        Ok(view)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn byte_order(&self) -> ByteOrder {
        self.byte_order
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
    pub fn size(&self) -> Result<usize, StorageError> {
        element_count(&self.shape)
    }
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    pub fn has_same_mapping_as(&self, other: &Self) -> bool {
        self.shares_storage_with(other)
            && self.dtype == other.dtype
            && self.byte_order == other.byte_order
            && self.shape == other.shape
            && self.strides == other.strides
            && self.offset == other.offset
    }

    pub fn is_writeable(&self) -> bool {
        self.writeable
    }

    /// Returns a shared read-only view broadcast to `shape`.
    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Self, StorageError> {
        self.broadcast_to_with_writeability(shape, false)
    }

    /// Returns a broadcast view whose writeability follows the caller's API contract.
    pub fn broadcast_to_with_writeability(
        &self,
        shape: Vec<usize>,
        writeable: bool,
    ) -> Result<Self, StorageError> {
        if self.ndim() > shape.len() {
            return Err(StorageError::CannotBroadcast {
                from: self.shape.clone(),
                to: shape,
            });
        }
        let leading = shape.len() - self.ndim();
        let mut strides = vec![0; leading];
        for axis in 0..self.ndim() {
            let source = self.shape[axis];
            let target = shape[leading + axis];
            if source != target && source != 1 {
                return Err(StorageError::CannotBroadcast {
                    from: self.shape.clone(),
                    to: shape,
                });
            }
            strides.push(if source == target {
                self.strides[axis]
            } else {
                0
            });
        }
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            byte_order: self.byte_order,
            shape,
            strides,
            offset: self.offset,
            allocation_len: self.allocation_len,
            writeable: self.writeable && writeable,
        };
        view.validate_layout()?;
        Ok(view)
    }

    /// Returns a read-only diagonal view, with the diagonal dimension last.
    pub fn diagonal(
        &self,
        offset: isize,
        axis1: isize,
        axis2: isize,
    ) -> Result<Self, StorageError> {
        let normalize = |axis: isize| {
            let axis = if axis < 0 {
                axis.checked_add(self.ndim() as isize)
                    .ok_or(StorageError::InvalidAxes)?
            } else {
                axis
            };
            if axis < 0 || axis >= self.ndim() as isize {
                return Err(StorageError::InvalidAxes);
            }
            Ok(axis as usize)
        };
        let axis1 = normalize(axis1)?;
        let axis2 = normalize(axis2)?;
        if axis1 == axis2 || self.ndim() < 2 {
            return Err(StorageError::InvalidAxes);
        }
        let rows = self.shape[axis1];
        let columns = self.shape[axis2];
        let (row_start, column_start, length) = if offset >= 0 {
            let column_start = usize::try_from(offset).map_err(|_| StorageError::ShapeOverflow)?;
            (
                0,
                column_start,
                rows.min(columns.saturating_sub(column_start)),
            )
        } else {
            let row_start =
                usize::try_from(offset.unsigned_abs()).map_err(|_| StorageError::ShapeOverflow)?;
            (row_start, 0, rows.saturating_sub(row_start).min(columns))
        };
        let mut view_offset = self.offset;
        for (axis, start) in [(axis1, row_start), (axis2, column_start)] {
            let start = isize::try_from(start).map_err(|_| StorageError::ShapeOverflow)?;
            view_offset = view_offset
                .checked_add(
                    start
                        .checked_mul(self.strides[axis])
                        .ok_or(StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
        let mut shape = Vec::with_capacity(self.ndim() - 1);
        let mut strides = Vec::with_capacity(self.ndim() - 1);
        for axis in 0..self.ndim() {
            if axis != axis1 && axis != axis2 {
                shape.push(self.shape[axis]);
                strides.push(self.strides[axis]);
            }
        }
        shape.push(length);
        strides.push(
            self.strides[axis1]
                .checked_add(self.strides[axis2])
                .ok_or(StorageError::ShapeOverflow)?,
        );
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            byte_order: self.byte_order,
            shape,
            strides,
            offset: view_offset,
            allocation_len: self.allocation_len,
            writeable: false,
        };
        view.validate_layout()?;
        Ok(view)
    }

    /// Returns a shared view with equivalent scalar and byte-order metadata.
    /// This is used for dtype aliases and native-endian descriptors whose
    /// bytes already have the requested representation.
    pub fn view_with_dtype_and_order(
        &self,
        dtype: DType,
        byte_order: ByteOrder,
    ) -> Result<Self, StorageError> {
        if !storage_dtypes_compatible(self.dtype, dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        let byte_order = if dtype.itemsize() == 1 {
            ByteOrder::NotApplicable
        } else {
            if byte_order == ByteOrder::NotApplicable
                || !byte_orders_have_same_representation(self.byte_order, byte_order)
            {
                return Err(StorageError::InvalidLayout);
            }
            byte_order
        };
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype,
            byte_order,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            allocation_len: self.allocation_len,
            writeable: self.writeable,
        };
        view.validate_layout()?;
        Ok(view)
    }

    pub fn is_c_contiguous(&self) -> bool {
        if self.size().unwrap_or(0) == 0 {
            return true;
        }
        let mut expected = self.dtype.itemsize() as isize;
        for axis in (0..self.ndim()).rev() {
            let dim = self.shape[axis];
            if dim > 1 && self.strides[axis] != expected {
                return false;
            }
            let Ok(dim) = isize::try_from(dim.max(1)) else {
                return false;
            };
            let Some(next) = expected.checked_mul(dim) else {
                return false;
            };
            expected = next;
        }
        true
    }

    pub fn is_f_contiguous(&self) -> bool {
        if self.size().unwrap_or(0) == 0 {
            return true;
        }
        let mut expected = self.dtype.itemsize() as isize;
        for axis in 0..self.ndim() {
            let dim = self.shape[axis];
            if dim > 1 && self.strides[axis] != expected {
                return false;
            }
            let Ok(dim) = isize::try_from(dim.max(1)) else {
                return false;
            };
            let Some(next) = expected.checked_mul(dim) else {
                return false;
            };
            expected = next;
        }
        true
    }

    pub fn transpose(&self, axes: Option<&[isize]>) -> Result<Self, StorageError> {
        let ndim = self.ndim();
        let permutation: Vec<usize> = match axes {
            None => (0..ndim).rev().collect(),
            Some(axes) => {
                if axes.len() != ndim {
                    return Err(StorageError::InvalidAxes);
                }
                let mut seen = vec![false; ndim];
                let mut normalized = Vec::with_capacity(ndim);
                for &axis in axes {
                    let axis = if axis < 0 {
                        axis.checked_add(ndim as isize)
                            .ok_or(StorageError::InvalidAxes)?
                    } else {
                        axis
                    };
                    if axis < 0 || axis >= ndim as isize || seen[axis as usize] {
                        return Err(StorageError::InvalidAxes);
                    }
                    seen[axis as usize] = true;
                    normalized.push(axis as usize);
                }
                normalized
            }
        };
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            byte_order: self.byte_order,
            shape: permutation.iter().map(|&axis| self.shape[axis]).collect(),
            strides: permutation.iter().map(|&axis| self.strides[axis]).collect(),
            offset: self.offset,
            allocation_len: self.allocation_len,
            writeable: self.writeable,
        };
        view.validate_layout()?;
        Ok(view)
    }

    /// Reshapes in C order, preserving storage whenever the existing strides
    /// can represent the requested shape. Otherwise a copy is made unless
    /// `copy` is explicitly false.
    pub fn reshape(&self, shape: Vec<usize>, copy: Option<bool>) -> Result<Self, StorageError> {
        self.reshape_order(shape, copy, false)
    }

    pub fn reshape_order(
        &self,
        shape: Vec<usize>,
        copy: Option<bool>,
        fortran: bool,
    ) -> Result<Self, StorageError> {
        if element_count(&shape)? != self.size()? {
            return Err(StorageError::ShapeMismatch);
        }
        let view_strides =
            reshape_view_strides(&self.shape, &self.strides, &shape, self.dtype, fortran)?;
        let should_copy = copy == Some(true) || view_strides.is_none();
        if should_copy && copy == Some(false) {
            return Err(StorageError::InvalidLayout);
        }
        if should_copy {
            let values = self.snapshot_order(fortran)?;
            let len = element_count(&shape)?;
            let strides = if len == 0 {
                reshape_empty_strides(self.dtype, &shape, fortran)?
            } else if fortran {
                f_strides(self.dtype, &shape)?
            } else {
                c_strides(self.dtype, &shape)?
            };
            let view = Self {
                storage: Arc::new(RwLock::new(Buffer::from_values(
                    self.dtype,
                    self.byte_order,
                    &values,
                )?)),
                dtype: self.dtype,
                byte_order: self.byte_order,
                shape,
                strides,
                offset: 0,
                allocation_len: len,
                writeable: true,
            };
            view.validate_layout()?;
            return Ok(view);
        }
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            byte_order: self.byte_order,
            strides: view_strides.expect("a no-copy reshape has inferred strides"),
            shape,
            offset: self.offset,
            allocation_len: self.allocation_len,
            writeable: self.writeable,
        };
        view.validate_layout()?;
        Ok(view)
    }

    fn snapshot_order(&self, fortran: bool) -> Result<Vec<Scalar>, StorageError> {
        let size = self.size()?;
        let mut values = Vec::with_capacity(size);
        for linear in 0..size {
            let coordinates = coordinates_for_order(&self.shape, linear, fortran)?;
            values.push(self.read_at(&coordinates)?);
        }
        Ok(values)
    }

    pub fn astype(&self, dtype: DType) -> Result<Self, StorageError> {
        self.astype_with_order(dtype, ByteOrder::Native)
    }

    pub fn astype_with_order(
        &self,
        dtype: DType,
        byte_order: ByteOrder,
    ) -> Result<Self, StorageError> {
        self.astype_with_layout(dtype, byte_order, false)
    }

    pub fn astype_with_layout(
        &self,
        dtype: DType,
        byte_order: ByteOrder,
        fortran: bool,
    ) -> Result<Self, StorageError> {
        let values = self
            .snapshot()?
            .iter()
            .map(|value| value.cast(dtype))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_values_with_layout(dtype, byte_order, self.shape.clone(), &values, fortran)
    }

    pub fn astype_with_axis_order(
        &self,
        dtype: DType,
        byte_order: ByteOrder,
        fastest_to_slowest: &[usize],
    ) -> Result<Self, StorageError> {
        let values = self
            .snapshot()?
            .iter()
            .map(|value| value.cast(dtype))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_values_with_axis_order(
            dtype,
            byte_order,
            self.shape.clone(),
            &values,
            fastest_to_slowest,
        )
    }

    pub fn index(&self, indices: &[IndexItem]) -> Result<Self, StorageError> {
        if indices.iter().any(|item| {
            matches!(
                item,
                IndexItem::Fancy { .. } | IndexItem::BoolScalar(_) | IndexItem::BoolMask { .. }
            )
        }) {
            let layout = self.advanced_offsets(indices)?;
            let storage = self
                .storage
                .read()
                .map_err(|_| StorageError::LockPoisoned)?;
            let values = layout
                .offsets
                .into_iter()
                .map(|offset| storage.read_as(offset, self.dtype, self.byte_order))
                .collect::<Result<Vec<_>, _>>()?;
            return Self::from_values_with_strides(
                self.dtype,
                self.byte_order,
                layout.output_shape,
                &values,
                layout.output_strides,
            );
        }
        let consuming = indices
            .iter()
            .map(|item| match item {
                IndexItem::NewAxis | IndexItem::BoolScalar(_) => 0,
                IndexItem::BoolMask { shape, .. } => shape.len(),
                _ => 1,
            })
            .sum::<usize>();
        if consuming > self.ndim() {
            return Err(StorageError::TooManyIndices {
                provided: consuming,
                dimensions: self.ndim(),
            });
        }
        let mut offset = self.offset;
        let mut shape = Vec::with_capacity(self.ndim() + indices.len());
        let mut strides = Vec::with_capacity(self.ndim() + indices.len());
        let mut axis = 0;
        for item in indices {
            if matches!(item, IndexItem::NewAxis) {
                shape.push(1);
                strides.push(0);
                continue;
            }
            if matches!(item, IndexItem::BoolScalar(_)) {
                return Err(StorageError::InvalidFancyIndex);
            }
            if axis >= self.ndim() {
                return Err(StorageError::TooManyIndices {
                    provided: consuming,
                    dimensions: self.ndim(),
                });
            }
            let dim = self.shape[axis];
            let stride = self.strides[axis];
            match item {
                IndexItem::Integer(original) => {
                    let len = isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
                    let index = if *original < 0 {
                        original
                            .checked_add(len)
                            .ok_or(StorageError::ShapeOverflow)?
                    } else {
                        *original
                    };
                    if index < 0 || index >= len {
                        return Err(StorageError::IndexOutOfBounds {
                            axis,
                            index: *original,
                            length: dim,
                        });
                    }
                    offset = offset
                        .checked_add(
                            index
                                .checked_mul(stride)
                                .ok_or(StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                }
                IndexItem::Slice { start, step, len } => {
                    if *step == 0 {
                        return Err(StorageError::InvalidLayout);
                    }
                    let dim_signed =
                        isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
                    if *len == 0 {
                        if *start < -1 || *start > dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                    } else {
                        if *start < 0 || *start >= dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                        let delta = isize::try_from(*len - 1)
                            .map_err(|_| StorageError::ShapeOverflow)?
                            .checked_mul(*step)
                            .ok_or(StorageError::ShapeOverflow)?;
                        let last = start
                            .checked_add(delta)
                            .ok_or(StorageError::ShapeOverflow)?;
                        if last < 0 || last >= dim_signed {
                            return Err(StorageError::InvalidLayout);
                        }
                    }
                    offset = offset
                        .checked_add(
                            start
                                .checked_mul(stride)
                                .ok_or(StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                    shape.push(*len);
                    strides.push(if *len == 0 {
                        stride
                    } else {
                        stride
                            .checked_mul(*step)
                            .ok_or(StorageError::ShapeOverflow)?
                    });
                }
                IndexItem::Fancy { .. } | IndexItem::BoolMask { .. } => {
                    return Err(StorageError::InvalidFancyIndex)
                }
                IndexItem::NewAxis | IndexItem::BoolScalar(_) => unreachable!(),
            }
            axis += 1;
        }
        while axis < self.ndim() {
            shape.push(self.shape[axis]);
            strides.push(self.strides[axis]);
            axis += 1;
        }
        let view = Self {
            storage: Arc::clone(&self.storage),
            dtype: self.dtype,
            byte_order: self.byte_order,
            shape,
            strides,
            offset,
            allocation_len: self.allocation_len,
            writeable: self.writeable,
        };
        view.validate_layout()?;
        Ok(view)
    }

    /// Resolve an advanced index once and return a selection that reads and
    /// writes through to the source allocation in logical iteration order.
    pub fn advanced_index_view(&self, indices: &[IndexItem]) -> Result<IndexedView, StorageError> {
        let layout = self.advanced_offsets(indices)?;
        Ok(IndexedView {
            parent: self.clone(),
            shape: layout.output_shape,
            offsets: layout.offsets,
        })
    }

    pub fn read_at(&self, coordinates: &[usize]) -> Result<Scalar, StorageError> {
        if coordinates.len() != self.ndim() {
            return Err(StorageError::WrongIndexRank {
                provided: coordinates.len(),
                dimensions: self.ndim(),
            });
        }
        let offset = self.element_offset(coordinates)?;
        self.storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?
            .read_as(offset, self.dtype, self.byte_order)
    }
    pub fn write_at(&self, coordinates: &[usize], value: Scalar) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        if value.dtype() != self.dtype {
            return Err(StorageError::DTypeMismatch);
        }
        if coordinates.len() != self.ndim() {
            return Err(StorageError::WrongIndexRank {
                provided: coordinates.len(),
                dimensions: self.ndim(),
            });
        }
        let offset = self.element_offset(coordinates)?;
        self.storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?
            .write_as(offset, self.dtype, self.byte_order, value)
    }
    pub fn read_linear(&self, index: usize) -> Result<Scalar, StorageError> {
        self.read_at(&self.coordinates(index)?)
    }
    pub fn write_linear(&self, index: usize, value: Scalar) -> Result<(), StorageError> {
        self.write_at(&self.coordinates(index)?, value)
    }
    pub fn assign_scalar(&self, value: Scalar) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        let value = value.cast(self.dtype)?;
        let offsets = self.all_element_offsets()?;
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for offset in offsets {
            storage.write_as(offset, self.dtype, self.byte_order, value.clone())?;
        }
        Ok(())
    }
    /// Assigns values to the first logical elements of this view. This is
    /// used by sequence assignment to preserve writes made before a later
    /// Python scalar conversion fails.
    pub fn assign_prefix(&self, values: &[Scalar]) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        let offsets = self.all_element_offsets()?;
        if values.len() > offsets.len() {
            return Err(StorageError::ShapeMismatch);
        }
        if values.iter().any(|value| value.dtype() != self.dtype) {
            return Err(StorageError::DTypeMismatch);
        }
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for (&offset, value) in offsets.iter().zip(values) {
            storage.write_as(offset, self.dtype, self.byte_order, value.clone())?;
        }
        Ok(())
    }
    /// Snapshots the source before taking the destination write lock, making overlapping assignments safe.
    pub fn assign_view(&self, source: &Self) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        if source.ndim() > self.ndim()
            || source
                .shape
                .iter()
                .rev()
                .zip(self.shape.iter().rev())
                .any(|(&from, &to)| from != 1 && from != to)
        {
            return Err(StorageError::CannotBroadcast {
                from: source.shape.clone(),
                to: self.shape.clone(),
            });
        }
        let source_values = source.snapshot()?;
        let offsets = self.all_element_offsets()?;
        let mut values = Vec::with_capacity(offsets.len());
        for linear in 0..offsets.len() {
            let destination_coords = self.coordinates(linear)?;
            let mut source_coords = vec![0; source.ndim()];
            let leading = self.ndim() - source.ndim();
            for axis in 0..source.ndim() {
                source_coords[axis] = if source.shape[axis] == 1 {
                    0
                } else {
                    destination_coords[leading + axis]
                };
            }
            let source_linear = source.coordinates_to_linear(&source_coords)?;
            values.push(source_values[source_linear].cast(self.dtype)?);
        }
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for (offset, value) in offsets.into_iter().zip(values) {
            storage.write_as(offset, self.dtype, self.byte_order, value)?;
        }
        Ok(())
    }
    pub fn assign_fancy_scalar(
        &self,
        indices: &[IndexItem],
        value: Scalar,
    ) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        let value = value.cast(self.dtype)?;
        let layout = self.advanced_offsets(indices)?;
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for offset in layout.offsets {
            storage.write_as(offset, self.dtype, self.byte_order, value.clone())?;
        }
        Ok(())
    }

    pub fn assign_fancy_view(
        &self,
        indices: &[IndexItem],
        source: &Self,
    ) -> Result<(), StorageError> {
        if !self.writeable {
            return Err(StorageError::ReadOnly);
        }
        let layout = self.advanced_offsets(indices)?;
        let target_shape = layout.output_shape;
        let offsets = layout.offsets;
        if source.ndim() > target_shape.len()
            || source
                .shape
                .iter()
                .rev()
                .zip(target_shape.iter().rev())
                .any(|(&from, &to)| from != 1 && from != to)
        {
            return Err(StorageError::CannotBroadcast {
                from: source.shape.clone(),
                to: target_shape.clone(),
            });
        }
        let source_values = source.snapshot()?;
        let mut values = Vec::with_capacity(offsets.len());
        for linear in 0..offsets.len() {
            let destination_coords = coordinates_for_shape(&target_shape, linear)?;
            let leading = target_shape.len() - source.ndim();
            let mut source_coords = vec![0; source.ndim()];
            for axis in 0..source.ndim() {
                source_coords[axis] = if source.shape[axis] == 1 {
                    0
                } else {
                    destination_coords[leading + axis]
                };
            }
            let source_linear = linear_for_shape(&source.shape, &source_coords)?;
            values.push(source_values[source_linear].cast(self.dtype)?);
        }
        let mut storage = self
            .storage
            .write()
            .map_err(|_| StorageError::LockPoisoned)?;
        for (offset, value) in offsets.into_iter().zip(values) {
            storage.write_as(offset, self.dtype, self.byte_order, value)?;
        }
        Ok(())
    }

    fn advanced_offsets(&self, indices: &[IndexItem]) -> Result<AdvancedIndexLayout, StorageError> {
        let advanced_positions = indices
            .iter()
            .enumerate()
            .filter_map(|(position, item)| {
                matches!(
                    item,
                    IndexItem::Fancy { .. } | IndexItem::BoolScalar(_) | IndexItem::BoolMask { .. }
                )
                .then_some(position)
            })
            .collect::<Vec<_>>();
        if advanced_positions.is_empty() {
            return Err(StorageError::InvalidFancyIndex);
        }
        let advanced_shapes = advanced_positions
            .iter()
            .map(|&position| match &indices[position] {
                IndexItem::Fancy { shape, .. } => Ok(shape.clone()),
                IndexItem::BoolScalar(value) => Ok(vec![usize::from(*value)]),
                IndexItem::BoolMask { indices, .. } => Ok(vec![indices.len()]),
                _ => Err(StorageError::InvalidFancyIndex),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let advanced_shape = broadcast_index_shapes(&advanced_shapes)?;
        // Scalar integer terms do not add a broadcast dimension, but they do
        // participate in NumPy's placement rules when mixed with advanced
        // indices. In particular, a slice or new axis between an integer
        // scalar and a fancy index moves the advanced dimensions to the front.
        let placement_positions = indices
            .iter()
            .enumerate()
            .filter_map(|(position, item)| {
                matches!(
                    item,
                    IndexItem::Integer(_)
                        | IndexItem::Fancy { .. }
                        | IndexItem::BoolScalar(_)
                        | IndexItem::BoolMask { .. }
                )
                .then_some(position)
            })
            .collect::<Vec<_>>();
        let first_advanced = *placement_positions.first().unwrap();
        let last_advanced = *placement_positions.last().unwrap();
        let separated = indices[first_advanced..=last_advanced]
            .iter()
            .any(|item| matches!(item, IndexItem::NewAxis | IndexItem::Slice { .. }));
        let mut output_shape = if separated {
            advanced_shape.clone()
        } else {
            Vec::new()
        };
        let mut advanced_axis_start = if separated { Some(0) } else { None };
        let mut output_axis_for_term = vec![None; indices.len()];
        let mut basic_shape = Vec::new();
        let mut basic_source_strides = Vec::new();
        let mut basic_output_axes = Vec::new();
        let mut input_axis = 0usize;
        for (position, item) in indices.iter().enumerate() {
            let consumed = match item {
                IndexItem::NewAxis | IndexItem::BoolScalar(_) => 0,
                IndexItem::BoolMask { shape, .. } => shape.len(),
                _ => 1,
            };
            if consumed > self.ndim().saturating_sub(input_axis) {
                return Err(StorageError::TooManyIndices {
                    provided: input_axis + consumed,
                    dimensions: self.ndim(),
                });
            }
            match item {
                IndexItem::NewAxis => {
                    let output_axis = output_shape.len();
                    output_axis_for_term[position] = Some(output_axis);
                    output_shape.push(1);
                    basic_shape.push(1);
                    basic_source_strides.push(0);
                    basic_output_axes.push(output_axis);
                }
                IndexItem::BoolScalar(_) => {
                    if !separated && position == first_advanced {
                        advanced_axis_start = Some(output_shape.len());
                        output_shape.extend_from_slice(&advanced_shape);
                    }
                }
                IndexItem::Integer(index) => {
                    validate_integer_index(input_axis, self.shape[input_axis], *index)?;
                    if !separated && position == first_advanced {
                        advanced_axis_start = Some(output_shape.len());
                        output_shape.extend_from_slice(&advanced_shape);
                    }
                    input_axis += 1;
                }
                IndexItem::Slice { start, step, len } => {
                    validate_slice(input_axis, self.shape[input_axis], *start, *step, *len)?;
                    let output_axis = output_shape.len();
                    output_axis_for_term[position] = Some(output_axis);
                    output_shape.push(*len);
                    basic_shape.push(*len);
                    basic_source_strides.push(if *len == 0 {
                        self.strides[input_axis]
                    } else {
                        self.strides[input_axis]
                            .checked_mul(*step)
                            .ok_or(StorageError::ShapeOverflow)?
                    });
                    basic_output_axes.push(output_axis);
                    input_axis += 1;
                }
                IndexItem::Fancy { shape, indices, .. } => {
                    if element_count(shape)? != indices.len() {
                        return Err(StorageError::InvalidFancyIndex);
                    }
                    for &index in indices {
                        validate_integer_index(input_axis, self.shape[input_axis], index)?;
                    }
                    if !separated && position == first_advanced {
                        advanced_axis_start = Some(output_shape.len());
                        output_shape.extend_from_slice(&advanced_shape);
                    }
                    input_axis += 1;
                }
                IndexItem::BoolMask { shape, indices } => {
                    let end = input_axis + shape.len();
                    let mask_size = element_count(shape)?;
                    if shape.is_empty()
                        || shape.as_slice() != &self.shape[input_axis..end]
                        || indices.iter().any(|&index| index >= mask_size)
                    {
                        return Err(StorageError::InvalidFancyIndex);
                    }
                    if !separated && position == first_advanced {
                        advanced_axis_start = Some(output_shape.len());
                        output_shape.extend_from_slice(&advanced_shape);
                    }
                    input_axis = end;
                }
            }
            if input_axis > self.ndim() {
                return Err(StorageError::TooManyIndices {
                    provided: input_axis,
                    dimensions: self.ndim(),
                });
            }
        }
        let tail_input_axis_start = input_axis;
        let mut tail_output_axes = Vec::new();
        while input_axis < self.ndim() {
            let output_axis = output_shape.len();
            tail_output_axes.push(output_axis);
            output_shape.push(self.shape[input_axis]);
            basic_shape.push(self.shape[input_axis]);
            basic_source_strides.push(self.strides[input_axis]);
            basic_output_axes.push(output_axis);
            input_axis += 1;
        }
        if output_shape.len() > 64 {
            return Err(StorageError::ShapeOverflow);
        }
        let output_size = element_count(&output_shape)?;
        let mut offsets = Vec::with_capacity(output_size);
        let advanced_axis_start = advanced_axis_start.ok_or(StorageError::InvalidFancyIndex)?;
        let output_strides = if output_size == 0 {
            c_strides(self.dtype, &output_shape)?
        } else {
            let basic_size = element_count(&basic_shape)?;
            let mut strides = vec![0isize; output_shape.len()];
            if basic_size == 1 {
                let index_strides = advanced_positions
                    .iter()
                    .map(|&position| match &indices[position] {
                        IndexItem::Fancy { shape, strides, .. } => {
                            if shape.len() != strides.len() {
                                return Err(StorageError::InvalidFancyIndex);
                            }
                            Ok((shape.clone(), strides.clone()))
                        }
                        IndexItem::BoolScalar(value) => {
                            let shape = vec![usize::from(*value)];
                            // Boolean scalars are converted to a 1-D integer
                            // index array by NumPy's advanced-index iterator.
                            Ok((shape, vec![std::mem::size_of::<isize>() as isize]))
                        }
                        IndexItem::BoolMask { indices, .. } => Ok((
                            vec![indices.len()],
                            vec![std::mem::size_of::<isize>() as isize],
                        )),
                        _ => Err(StorageError::InvalidFancyIndex),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let axis_order = keep_order_axes(&advanced_shape, &index_strides)?;
                let mut advanced_stride = isize::try_from(self.dtype.itemsize())
                    .map_err(|_| StorageError::ShapeOverflow)?;
                for axis in axis_order {
                    strides[advanced_axis_start + axis] = advanced_stride;
                    advanced_stride = advanced_stride
                        .checked_mul(
                            isize::try_from(advanced_shape[axis])
                                .map_err(|_| StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                }
            } else {
                let mut advanced_stride = isize::try_from(self.dtype.itemsize())
                    .map_err(|_| StorageError::ShapeOverflow)?
                    .checked_mul(
                        isize::try_from(basic_size).map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
                for axis in (0..advanced_shape.len()).rev() {
                    strides[advanced_axis_start + axis] = advanced_stride;
                    advanced_stride = advanced_stride
                        .checked_mul(
                            isize::try_from(advanced_shape[axis])
                                .map_err(|_| StorageError::ShapeOverflow)?,
                        )
                        .ok_or(StorageError::ShapeOverflow)?;
                }
            }
            let mut basic_permutation = (0..basic_shape.len()).collect::<Vec<_>>();
            basic_permutation.sort_by(|&left, &right| {
                basic_source_strides[right]
                    .unsigned_abs()
                    .cmp(&basic_source_strides[left].unsigned_abs())
            });
            let mut basic_stride =
                isize::try_from(self.dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
            for axis in basic_permutation.into_iter().rev() {
                strides[basic_output_axes[axis]] = basic_stride;
                basic_stride = basic_stride
                    .checked_mul(
                        isize::try_from(basic_shape[axis])
                            .map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
            }
            strides
        };
        for linear in 0..output_size {
            let output_coords = coordinates_for_shape(&output_shape, linear)?;
            let advanced_end = advanced_axis_start + advanced_shape.len();
            let broadcast_coords = &output_coords[advanced_axis_start..advanced_end];
            let mut input_coords = Vec::with_capacity(self.ndim());
            let mut input_axis = 0usize;
            for (position, item) in indices.iter().enumerate() {
                match item {
                    IndexItem::NewAxis | IndexItem::BoolScalar(_) => {}
                    IndexItem::BoolMask {
                        shape,
                        indices: selected,
                    } => {
                        let selected_position =
                            broadcast_index(&[selected.len()], broadcast_coords)?;
                        let selected_linear = selected[selected_position];
                        input_coords.extend(coordinates_for_shape(shape, selected_linear)?);
                        input_axis += shape.len();
                    }
                    IndexItem::Integer(index) => {
                        input_coords.push(normalize_integer_index(
                            input_axis,
                            self.shape[input_axis],
                            *index,
                        )?);
                        input_axis += 1;
                    }
                    IndexItem::Slice { start, step, .. } => {
                        let output_axis = output_axis_for_term[position]
                            .ok_or(StorageError::InvalidFancyIndex)?;
                        let selected = isize::try_from(output_coords[output_axis])
                            .map_err(|_| StorageError::ShapeOverflow)?;
                        let coordinate = start
                            .checked_add(
                                selected
                                    .checked_mul(*step)
                                    .ok_or(StorageError::ShapeOverflow)?,
                            )
                            .ok_or(StorageError::ShapeOverflow)?;
                        input_coords.push(
                            usize::try_from(coordinate).map_err(|_| StorageError::InvalidLayout)?,
                        );
                        input_axis += 1;
                    }
                    IndexItem::Fancy {
                        shape,
                        indices: selected,
                        ..
                    } => {
                        let selected_linear = broadcast_index(shape, broadcast_coords)?;
                        input_coords.push(normalize_integer_index(
                            input_axis,
                            self.shape[input_axis],
                            selected[selected_linear],
                        )?);
                        input_axis += 1;
                    }
                }
            }
            while input_axis < self.ndim() {
                let output_axis = *tail_output_axes
                    .get(input_axis - tail_input_axis_start)
                    .ok_or(StorageError::InvalidFancyIndex)?;
                input_coords.push(output_coords[output_axis]);
                input_axis += 1;
            }
            offsets.push(self.element_offset(&input_coords)?);
        }
        Ok(AdvancedIndexLayout {
            output_shape,
            offsets,
            output_strides,
        })
    }
    pub fn copy(&self) -> Result<Self, StorageError> {
        self.copy_order(false)
    }
    pub fn copy_order(&self, fortran: bool) -> Result<Self, StorageError> {
        Self::from_values_with_layout(
            self.dtype,
            self.byte_order,
            self.shape.clone(),
            &self.snapshot()?,
            fortran,
        )
    }
    pub fn copy_with_axis_order(&self, fastest_to_slowest: &[usize]) -> Result<Self, StorageError> {
        Self::from_values_with_axis_order(
            self.dtype,
            self.byte_order,
            self.shape.clone(),
            &self.snapshot()?,
            fastest_to_slowest,
        )
    }
    pub fn snapshot(&self) -> Result<Vec<Scalar>, StorageError> {
        let size = self.size()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(size)
            .map_err(|_| StorageError::AllocationFailed)?;
        let storage = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        if size == 0 {
            return Ok(values);
        }
        if self.is_c_contiguous() {
            let itemsize =
                isize::try_from(self.dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
            if self.offset < 0 || self.offset % itemsize != 0 {
                return Err(StorageError::InvalidLayout);
            }
            let start =
                usize::try_from(self.offset / itemsize).map_err(|_| StorageError::InvalidLayout)?;
            let end = start.checked_add(size).ok_or(StorageError::ShapeOverflow)?;
            if end > self.allocation_len {
                return Err(StorageError::InvalidLayout);
            }
            for index in start..end {
                values.push(storage.read_as(index, self.dtype, self.byte_order)?);
            }
            return Ok(values);
        }

        for linear in 0..size {
            let offset = self.element_offset_linear(linear, size)?;
            values.push(storage.read_as(offset, self.dtype, self.byte_order)?);
        }
        Ok(values)
    }

    pub fn snapshot_float32(&self) -> Result<Vec<f32>, StorageError> {
        self.snapshot_typed(DType::Float32, |bytes, order| {
            Ok(f32::from_bits(read_unsigned(bytes, order)? as u32))
        })
    }

    pub fn snapshot_int32(&self) -> Result<Vec<i32>, StorageError> {
        if self.dtype != DType::Int32 {
            return Err(StorageError::DTypeMismatch);
        }
        if self.byte_order.is_native() {
            let size = self.size()?;
            let mut values = Vec::new();
            values
                .try_reserve_exact(size)
                .map_err(|_| StorageError::AllocationFailed)?;
            if size == 0 {
                return Ok(values);
            }
            let storage = self
                .storage
                .read()
                .map_err(|_| StorageError::LockPoisoned)?;
            if self.is_c_contiguous() {
                let start = contiguous_element_start(self)?
                    .checked_mul(DType::Int32.itemsize())
                    .ok_or(StorageError::ShapeOverflow)?;
                let byte_count = size
                    .checked_mul(DType::Int32.itemsize())
                    .ok_or(StorageError::ShapeOverflow)?;
                let end = start
                    .checked_add(byte_count)
                    .ok_or(StorageError::ShapeOverflow)?;
                let bytes = storage
                    .bytes
                    .get(start..end)
                    .ok_or(StorageError::InvalidLayout)?;
                for chunk in bytes.chunks_exact(DType::Int32.itemsize()) {
                    values.push(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                return Ok(values);
            }
            if self.ndim() == 2 {
                let itemsize = isize::try_from(DType::Int32.itemsize())
                    .map_err(|_| StorageError::ShapeOverflow)?;
                for outer in 0..self.shape[0] {
                    let outer = isize::try_from(outer).map_err(|_| StorageError::ShapeOverflow)?;
                    let outer_offset = outer
                        .checked_mul(self.strides[0])
                        .and_then(|delta| self.offset.checked_add(delta))
                        .ok_or(StorageError::ShapeOverflow)?;
                    for inner in 0..self.shape[1] {
                        let inner =
                            isize::try_from(inner).map_err(|_| StorageError::ShapeOverflow)?;
                        let byte_offset = inner
                            .checked_mul(self.strides[1])
                            .and_then(|delta| outer_offset.checked_add(delta))
                            .ok_or(StorageError::ShapeOverflow)?;
                        if byte_offset < 0 || byte_offset % itemsize != 0 {
                            return Err(StorageError::InvalidLayout);
                        }
                        let element = usize::try_from(byte_offset / itemsize)
                            .map_err(|_| StorageError::InvalidLayout)?;
                        if element >= self.allocation_len {
                            return Err(StorageError::InvalidLayout);
                        }
                        let bytes = storage.element_bytes(element)?;
                        values.push(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                    }
                }
                return Ok(values);
            }
            for linear in 0..size {
                let element = self.element_offset_linear(linear, size)?;
                let bytes = storage.element_bytes(element)?;
                values.push(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            return Ok(values);
        }
        self.snapshot_typed(DType::Int32, |bytes, order| {
            Ok(read_unsigned(bytes, order)? as u32 as i32)
        })
    }

    pub fn snapshot_int64(&self) -> Result<Vec<i64>, StorageError> {
        self.snapshot_typed(DType::Int64, |bytes, order| {
            Ok(read_unsigned(bytes, order)? as u64 as i64)
        })
    }

    pub fn float32_axis0_statistics(
        &self,
        ddof: f64,
        variance: bool,
        square_root: bool,
    ) -> Result<Option<Vec<f32>>, StorageError> {
        if self.dtype != DType::Float32
            || self.ndim() != 2
            || !self.is_c_contiguous()
            || (variance && !ddof.is_finite())
        {
            return Ok(None);
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let values = self.snapshot_float32()?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(columns)
            .map_err(|_| StorageError::AllocationFailed)?;
        if rows == 0 || columns == 0 || (variance && rows as f64 <= ddof) {
            if rows == 0 || (variance && rows as f64 <= ddof) {
                result.resize(columns, f32::NAN);
            }
            return Ok(Some(result));
        }
        let mut sums = Vec::new();
        sums.try_reserve_exact(columns)
            .map_err(|_| StorageError::AllocationFailed)?;
        sums.resize(columns, 0.0_f64);
        for row in 0..rows {
            let row_start = row
                .checked_mul(columns)
                .ok_or(StorageError::ShapeOverflow)?;
            for column in 0..columns {
                let index = row_start
                    .checked_add(column)
                    .ok_or(StorageError::ShapeOverflow)?;
                sums[column] += f64::from(values[index]);
            }
        }
        let divisor = rows as f64;
        for sum in &mut sums {
            *sum /= divisor;
        }
        if !variance {
            result.extend(sums.into_iter().map(|value| value as f32));
            return Ok(Some(result));
        }

        let mut sum_squares = Vec::new();
        sum_squares
            .try_reserve_exact(columns)
            .map_err(|_| StorageError::AllocationFailed)?;
        sum_squares.resize(columns, 0.0_f64);
        for row in 0..rows {
            let row_start = row
                .checked_mul(columns)
                .ok_or(StorageError::ShapeOverflow)?;
            for column in 0..columns {
                let index = row_start
                    .checked_add(column)
                    .ok_or(StorageError::ShapeOverflow)?;
                let difference = f64::from(values[index]) - sums[column];
                sum_squares[column] += difference * difference;
            }
        }
        let denominator = rows as f64 - ddof;
        for sum in sum_squares {
            let value = (sum / denominator) as f32;
            result.push(if square_root {
                f64::from(value).sqrt() as f32
            } else {
                value
            });
        }
        Ok(Some(result))
    }

    fn snapshot_typed<T>(
        &self,
        dtype: DType,
        mut decode: impl FnMut(&[u8], ByteOrder) -> Result<T, StorageError>,
    ) -> Result<Vec<T>, StorageError> {
        if self.dtype != dtype {
            return Err(StorageError::DTypeMismatch);
        }
        let size = self.size()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(size)
            .map_err(|_| StorageError::AllocationFailed)?;
        let storage = self
            .storage
            .read()
            .map_err(|_| StorageError::LockPoisoned)?;
        if size == 0 {
            return Ok(values);
        }
        if self.is_c_contiguous() {
            let itemsize =
                isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
            if self.offset < 0 || self.offset % itemsize != 0 {
                return Err(StorageError::InvalidLayout);
            }
            let start =
                usize::try_from(self.offset / itemsize).map_err(|_| StorageError::InvalidLayout)?;
            let end = start.checked_add(size).ok_or(StorageError::ShapeOverflow)?;
            if end > self.allocation_len {
                return Err(StorageError::InvalidLayout);
            }
            for index in start..end {
                values.push(decode(storage.element_bytes(index)?, self.byte_order)?);
            }
            return Ok(values);
        }
        if self.ndim() == 2 {
            let itemsize =
                isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
            for outer in 0..self.shape[0] {
                let outer = isize::try_from(outer).map_err(|_| StorageError::ShapeOverflow)?;
                let outer_offset = outer
                    .checked_mul(self.strides[0])
                    .and_then(|offset| self.offset.checked_add(offset))
                    .ok_or(StorageError::ShapeOverflow)?;
                for inner in 0..self.shape[1] {
                    let inner = isize::try_from(inner).map_err(|_| StorageError::ShapeOverflow)?;
                    let byte_offset = inner
                        .checked_mul(self.strides[1])
                        .and_then(|offset| outer_offset.checked_add(offset))
                        .ok_or(StorageError::ShapeOverflow)?;
                    if byte_offset < 0 || byte_offset % itemsize != 0 {
                        return Err(StorageError::InvalidLayout);
                    }
                    let index = usize::try_from(byte_offset / itemsize)
                        .map_err(|_| StorageError::InvalidLayout)?;
                    if index >= self.allocation_len {
                        return Err(StorageError::InvalidLayout);
                    }
                    values.push(decode(storage.element_bytes(index)?, self.byte_order)?);
                }
            }
            return Ok(values);
        }
        for linear in 0..size {
            let index = self.element_offset_linear(linear, size)?;
            values.push(decode(storage.element_bytes(index)?, self.byte_order)?);
        }
        Ok(values)
    }

    fn element_offset_linear(&self, linear: usize, size: usize) -> Result<usize, StorageError> {
        if linear >= size {
            return Err(StorageError::IndexOutOfBounds {
                axis: 0,
                index: isize::try_from(linear).unwrap_or(isize::MAX),
                length: size,
            });
        }
        let mut rest = linear;
        let mut byte_offset = self.offset;
        for axis in (0..self.ndim()).rev() {
            let dim = self.shape[axis];
            if dim == 0 {
                return Err(StorageError::InvalidLayout);
            }
            let coordinate = rest % dim;
            rest /= dim;
            let coordinate =
                isize::try_from(coordinate).map_err(|_| StorageError::ShapeOverflow)?;
            byte_offset = byte_offset
                .checked_add(
                    coordinate
                        .checked_mul(self.strides[axis])
                        .ok_or(StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
        let itemsize =
            isize::try_from(self.dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
        if byte_offset < 0 || byte_offset % itemsize != 0 {
            return Err(StorageError::InvalidLayout);
        }
        let element =
            usize::try_from(byte_offset / itemsize).map_err(|_| StorageError::InvalidLayout)?;
        if element >= self.allocation_len {
            return Err(StorageError::InvalidLayout);
        }
        Ok(element)
    }

    fn coordinates(&self, linear: usize) -> Result<Vec<usize>, StorageError> {
        let size = self.size()?;
        if linear >= size {
            return Err(StorageError::IndexOutOfBounds {
                axis: 0,
                index: isize::try_from(linear).unwrap_or(isize::MAX),
                length: size,
            });
        }
        let mut rest = linear;
        let mut coordinates = vec![0; self.ndim()];
        for axis in (0..self.ndim()).rev() {
            let dim = self.shape[axis];
            if dim == 0 {
                return Err(StorageError::InvalidLayout);
            }
            coordinates[axis] = rest % dim;
            rest /= dim;
        }
        Ok(coordinates)
    }
    fn coordinates_to_linear(&self, coordinates: &[usize]) -> Result<usize, StorageError> {
        if coordinates.len() != self.ndim() {
            return Err(StorageError::WrongIndexRank {
                provided: coordinates.len(),
                dimensions: self.ndim(),
            });
        }
        let mut linear = 0usize;
        for (axis, (&coordinate, &dim)) in coordinates.iter().zip(&self.shape).enumerate() {
            if coordinate >= dim {
                return Err(StorageError::IndexOutOfBounds {
                    axis,
                    index: isize::try_from(coordinate).unwrap_or(isize::MAX),
                    length: dim,
                });
            }
            linear = linear
                .checked_mul(dim)
                .and_then(|value| value.checked_add(coordinate))
                .ok_or(StorageError::ShapeOverflow)?;
        }
        Ok(linear)
    }
    fn element_offset(&self, coordinates: &[usize]) -> Result<usize, StorageError> {
        let mut byte_offset = self.offset;
        for (axis, (&coordinate, &dim)) in coordinates.iter().zip(&self.shape).enumerate() {
            if coordinate >= dim {
                return Err(StorageError::IndexOutOfBounds {
                    axis,
                    index: isize::try_from(coordinate).unwrap_or(isize::MAX),
                    length: dim,
                });
            }
            let coordinate =
                isize::try_from(coordinate).map_err(|_| StorageError::ShapeOverflow)?;
            byte_offset = byte_offset
                .checked_add(
                    coordinate
                        .checked_mul(self.strides[axis])
                        .ok_or(StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
        let size = self.dtype.itemsize() as isize;
        if byte_offset < 0 || byte_offset % size != 0 {
            return Err(StorageError::InvalidLayout);
        }
        let element =
            usize::try_from(byte_offset / size).map_err(|_| StorageError::InvalidLayout)?;
        if element >= self.allocation_len {
            return Err(StorageError::InvalidLayout);
        }
        Ok(element)
    }
    fn all_element_offsets(&self) -> Result<Vec<usize>, StorageError> {
        let size = self.size()?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(size)
            .map_err(|_| StorageError::AllocationFailed)?;
        for i in 0..size {
            offsets.push(self.element_offset(&self.coordinates(i)?)?);
        }
        Ok(offsets)
    }
    fn validate_layout(&self) -> Result<(), StorageError> {
        if self.shape.len() != self.strides.len() {
            return Err(StorageError::InvalidLayout);
        }
        if self.size()? == 0 {
            return Ok(());
        }
        let (mut min, mut max) = (self.offset, self.offset);
        for (&dim, &stride) in self.shape.iter().zip(&self.strides) {
            let delta = isize::try_from(dim - 1)
                .map_err(|_| StorageError::ShapeOverflow)?
                .checked_mul(stride)
                .ok_or(StorageError::ShapeOverflow)?;
            if delta < 0 {
                min = min.checked_add(delta).ok_or(StorageError::ShapeOverflow)?;
            } else {
                max = max.checked_add(delta).ok_or(StorageError::ShapeOverflow)?;
            }
        }
        let end = max
            .checked_add(self.dtype.itemsize() as isize)
            .ok_or(StorageError::ShapeOverflow)?;
        let allocation = isize::try_from(self.allocation_len)
            .map_err(|_| StorageError::ShapeOverflow)?
            .checked_mul(self.dtype.itemsize() as isize)
            .ok_or(StorageError::ShapeOverflow)?;
        if min < 0 || end > allocation {
            return Err(StorageError::InvalidLayout);
        }
        Ok(())
    }
}

fn coordinates_for_shape(shape: &[usize], linear: usize) -> Result<Vec<usize>, StorageError> {
    if linear >= element_count(shape)? {
        return Err(StorageError::InvalidLayout);
    }
    let mut rest = linear;
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        if dim == 0 {
            return Err(StorageError::InvalidLayout);
        }
        coordinates[axis] = rest % dim;
        rest /= dim;
    }
    Ok(coordinates)
}

fn coordinates_for_order(
    shape: &[usize],
    linear: usize,
    fortran: bool,
) -> Result<Vec<usize>, StorageError> {
    if linear >= element_count(shape)? {
        return Err(StorageError::InvalidLayout);
    }
    let mut rest = linear;
    let mut coordinates = vec![0; shape.len()];
    let axes: Box<dyn Iterator<Item = usize>> = if fortran {
        Box::new(0..shape.len())
    } else {
        Box::new((0..shape.len()).rev())
    };
    for axis in axes {
        let dim = shape[axis];
        if dim == 0 {
            return Err(StorageError::InvalidLayout);
        }
        coordinates[axis] = rest % dim;
        rest /= dim;
    }
    Ok(coordinates)
}

fn validate_integer_index(axis: usize, dim: usize, index: isize) -> Result<(), StorageError> {
    normalize_integer_index(axis, dim, index).map(|_| ())
}

fn normalize_integer_index(
    axis: usize,
    dim: usize,
    original: isize,
) -> Result<usize, StorageError> {
    let length = isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
    let index = if original < 0 {
        original
            .checked_add(length)
            .ok_or(StorageError::ShapeOverflow)?
    } else {
        original
    };
    if index < 0 || index >= length {
        return Err(StorageError::IndexOutOfBounds {
            axis,
            index: original,
            length: dim,
        });
    }
    usize::try_from(index).map_err(|_| StorageError::ShapeOverflow)
}

fn validate_slice(
    axis: usize,
    dim: usize,
    start: isize,
    step: isize,
    len: usize,
) -> Result<(), StorageError> {
    if step == 0 {
        return Err(StorageError::InvalidLayout);
    }
    let dim_signed = isize::try_from(dim).map_err(|_| StorageError::ShapeOverflow)?;
    if len == 0 {
        if start < -1 || start > dim_signed {
            return Err(StorageError::InvalidLayout);
        }
        return Ok(());
    }
    if start < 0 || start >= dim_signed {
        return Err(StorageError::InvalidLayout);
    }
    let delta = isize::try_from(len - 1)
        .map_err(|_| StorageError::ShapeOverflow)?
        .checked_mul(step)
        .ok_or(StorageError::ShapeOverflow)?;
    let last = start
        .checked_add(delta)
        .ok_or(StorageError::ShapeOverflow)?;
    if last < 0 || last >= dim_signed {
        return Err(StorageError::InvalidLayout);
    }
    let _ = axis;
    Ok(())
}

fn linear_for_shape(shape: &[usize], coordinates: &[usize]) -> Result<usize, StorageError> {
    if coordinates.len() != shape.len() {
        return Err(StorageError::WrongIndexRank {
            provided: coordinates.len(),
            dimensions: shape.len(),
        });
    }
    let mut linear = 0usize;
    for (axis, (&coordinate, &dim)) in coordinates.iter().zip(shape).enumerate() {
        if coordinate >= dim {
            return Err(StorageError::IndexOutOfBounds {
                axis,
                index: isize::try_from(coordinate).unwrap_or(isize::MAX),
                length: dim,
            });
        }
        linear = linear
            .checked_mul(dim)
            .and_then(|value| value.checked_add(coordinate))
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(linear)
}

fn broadcast_index_shapes(shapes: &[Vec<usize>]) -> Result<Vec<usize>, StorageError> {
    let ndim = shapes.iter().map(Vec::len).max().unwrap_or(0);
    let mut result = vec![1; ndim];
    for shape in shapes {
        for (offset, &dimension) in shape.iter().rev().enumerate() {
            let axis = ndim - 1 - offset;
            if result[axis] == 1 {
                result[axis] = dimension;
            } else if dimension != 1 && result[axis] != dimension {
                return Err(StorageError::InvalidFancyIndex);
            }
        }
    }
    Ok(result)
}

/// Returns the stable axis ordering NumPy's NpyIter uses for KEEPORDER.
///
/// Each operand is represented by its shape and byte strides. Shapes may
/// broadcast to `shape`; the returned axes are ordered fastest to slowest.
pub fn keep_order_axes(
    shape: &[usize],
    operands: &[(Vec<usize>, Vec<isize>)],
) -> Result<Vec<usize>, StorageError> {
    if operands.iter().any(|(operand_shape, strides)| {
        operand_shape.len() != strides.len() || operand_shape.len() > shape.len()
    }) {
        return Err(StorageError::InvalidFancyIndex);
    }
    let effective_stride = |operand: &(Vec<usize>, Vec<isize>), axis: usize| {
        let (operand_shape, operand_strides) = operand;
        let leading = shape.len().saturating_sub(operand_shape.len());
        if axis < leading {
            return Ok(0);
        }
        let operand_axis = axis - leading;
        let dimension = operand_shape[operand_axis];
        if dimension == shape[axis] {
            Ok(operand_strides[operand_axis])
        } else if dimension == 1 {
            Ok(0)
        } else {
            Err(StorageError::InvalidFancyIndex)
        }
    };

    // NpyIter starts in C iteration order, with the last axis fastest.
    let mut order = (0..shape.len()).rev().collect::<Vec<_>>();
    for position in 1..order.len() {
        let candidate = order[position];
        let mut insertion = position;
        for previous_position in (0..position).rev() {
            let previous = order[previous_position];
            let mut ambiguous = true;
            let mut should_swap = false;
            for operand in operands {
                let candidate_stride = effective_stride(operand, candidate)?;
                let previous_stride = effective_stride(operand, previous)?;
                if candidate_stride != 0 && previous_stride != 0 {
                    if previous_stride.unsigned_abs() <= candidate_stride.unsigned_abs() {
                        // Conflicting operands resolve in favor of C order.
                        should_swap = false;
                    } else if ambiguous {
                        should_swap = true;
                    }
                    ambiguous = false;
                }
            }
            if !ambiguous {
                if should_swap {
                    insertion = previous_position;
                } else {
                    break;
                }
            }
        }
        if insertion != position {
            order.remove(position);
            order.insert(insertion, candidate);
        }
    }
    Ok(order)
}

fn broadcast_index(shape: &[usize], coordinates: &[usize]) -> Result<usize, StorageError> {
    if shape.len() > coordinates.len() {
        return Err(StorageError::InvalidFancyIndex);
    }
    let leading = coordinates.len() - shape.len();
    let mut index_coordinates = Vec::with_capacity(shape.len());
    for (axis, &dimension) in shape.iter().enumerate() {
        index_coordinates.push(if dimension == 1 {
            0
        } else {
            coordinates[leading + axis]
        });
    }
    linear_for_shape(shape, &index_coordinates)
}

fn element_count(shape: &[usize]) -> Result<usize, StorageError> {
    let mut count = 1usize;
    for &dim in shape {
        count = count.checked_mul(dim).ok_or(StorageError::ShapeOverflow)?;
    }
    if count > isize::MAX as usize {
        return Err(StorageError::ShapeOverflow);
    }
    Ok(count)
}

fn contiguous_element_start(view: &View) -> Result<usize, StorageError> {
    let itemsize =
        isize::try_from(DType::Int32.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    if view.offset < 0 || view.offset % itemsize != 0 {
        return Err(StorageError::InvalidLayout);
    }
    let start = usize::try_from(view.offset / itemsize).map_err(|_| StorageError::InvalidLayout)?;
    let end = start
        .checked_add(view.size()?)
        .ok_or(StorageError::ShapeOverflow)?;
    if end > view.allocation_len {
        return Err(StorageError::InvalidLayout);
    }
    Ok(start)
}

fn fortran_element_strides(shape: &[usize]) -> Result<Vec<usize>, StorageError> {
    let mut strides = Vec::new();
    strides
        .try_reserve_exact(shape.len())
        .map_err(|_| StorageError::AllocationFailed)?;
    let mut stride = 1usize;
    for &dimension in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(dimension)
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(strides)
}

fn fortran_index_from_c_linear(
    shape: &[usize],
    linear: usize,
    fortran_strides: &[usize],
) -> Result<usize, StorageError> {
    if shape.len() != fortran_strides.len() {
        return Err(StorageError::InvalidLayout);
    }
    if linear >= element_count(shape)? {
        return Err(StorageError::IndexOutOfBounds {
            axis: 0,
            index: isize::try_from(linear).unwrap_or(isize::MAX),
            length: element_count(shape)?,
        });
    }
    let mut rest = linear;
    let mut physical = 0usize;
    for axis in (0..shape.len()).rev() {
        let dimension = shape[axis];
        if dimension == 0 {
            return Err(StorageError::InvalidLayout);
        }
        let coordinate = rest % dimension;
        rest /= dimension;
        physical = physical
            .checked_add(
                coordinate
                    .checked_mul(fortran_strides[axis])
                    .ok_or(StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(physical)
}

fn fortran_destination_element(
    fortran_strides: &[usize],
    input_shape: &[usize],
    input_linear: usize,
    axis: usize,
    axis_offset: usize,
) -> Result<usize, StorageError> {
    if fortran_strides.len() != input_shape.len() || axis >= input_shape.len() {
        return Err(StorageError::InvalidLayout);
    }
    let mut rest = input_linear;
    let mut destination = 0usize;
    for dimension in (0..input_shape.len()).rev() {
        let input_length = input_shape[dimension];
        if input_length == 0 {
            return Err(StorageError::InvalidLayout);
        }
        let mut coordinate = rest % input_length;
        rest /= input_length;
        if dimension == axis {
            coordinate = coordinate
                .checked_add(axis_offset)
                .ok_or(StorageError::ShapeOverflow)?;
        }
        destination = destination
            .checked_add(
                coordinate
                    .checked_mul(fortran_strides[dimension])
                    .ok_or(StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(destination)
}

fn copy_element(
    source: &[u8],
    source_element: usize,
    destination: &mut [u8],
    destination_element: usize,
    itemsize: usize,
) -> Result<(), StorageError> {
    let source_start = source_element
        .checked_mul(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let source_end = source_start
        .checked_add(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let destination_start = destination_element
        .checked_mul(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let destination_end = destination_start
        .checked_add(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let source_bytes = source
        .get(source_start..source_end)
        .ok_or(StorageError::InvalidLayout)?;
    let destination_bytes = destination
        .get_mut(destination_start..destination_end)
        .ok_or(StorageError::InvalidLayout)?;
    destination_bytes.copy_from_slice(source_bytes);
    Ok(())
}

fn copy_element_range(
    source: &[u8],
    source_element: usize,
    destination: &mut [u8],
    destination_element: usize,
    count: usize,
    itemsize: usize,
) -> Result<(), StorageError> {
    let source_start = source_element
        .checked_mul(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let byte_count = count
        .checked_mul(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let source_end = source_start
        .checked_add(byte_count)
        .ok_or(StorageError::ShapeOverflow)?;
    let destination_start = destination_element
        .checked_mul(itemsize)
        .ok_or(StorageError::ShapeOverflow)?;
    let destination_end = destination_start
        .checked_add(byte_count)
        .ok_or(StorageError::ShapeOverflow)?;
    let source_bytes = source
        .get(source_start..source_end)
        .ok_or(StorageError::InvalidLayout)?;
    let destination_bytes = destination
        .get_mut(destination_start..destination_end)
        .ok_or(StorageError::InvalidLayout)?;
    destination_bytes.copy_from_slice(source_bytes);
    Ok(())
}

fn read_native_int32(buffer: &Buffer, element: usize) -> Result<i32, StorageError> {
    let bytes = buffer.element_bytes(element)?;
    Ok(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn write_native_int32(buffer: &mut Buffer, element: usize, value: i32) -> Result<(), StorageError> {
    buffer
        .element_bytes_mut(element)?
        .copy_from_slice(&value.to_ne_bytes());
    Ok(())
}

fn sort_int32_lane(values: &mut [i32], descending: bool) {
    values.sort_unstable();
    if descending {
        values.reverse();
    }
}

/// Infer strides for a reshape that can keep the existing storage. Adjacent
/// source dimensions may be merged only when their strides are contiguous in
/// the requested iteration order. This follows the same chunking rule used
/// by NumPy's no-copy reshape path, including negative and stepped strides.
fn reshape_view_strides(
    old_shape: &[usize],
    old_strides: &[isize],
    new_shape: &[usize],
    dtype: DType,
    fortran: bool,
) -> Result<Option<Vec<isize>>, StorageError> {
    if old_shape.len() != old_strides.len() {
        return Err(StorageError::InvalidLayout);
    }
    if old_shape == new_shape {
        return Ok(Some(old_strides.to_vec()));
    }
    let size = element_count(old_shape)?;
    if size == 0 {
        return Ok(Some(reshape_empty_strides(dtype, new_shape, fortran)?));
    }

    let old_axes = old_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (dim != 1).then_some(axis))
        .collect::<Vec<_>>();
    let new_axes = new_shape
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (dim != 1).then_some(axis))
        .collect::<Vec<_>>();
    let old_dims = old_axes
        .iter()
        .map(|&axis| old_shape[axis])
        .collect::<Vec<_>>();
    let new_dims = new_axes
        .iter()
        .map(|&axis| new_shape[axis])
        .collect::<Vec<_>>();

    if old_dims.is_empty() || new_dims.is_empty() {
        if size != 1 {
            return Ok(None);
        }
        return Ok(Some(if fortran {
            f_strides(dtype, new_shape)?
        } else {
            c_strides(dtype, new_shape)?
        }));
    }

    let mut inferred = vec![0isize; new_shape.len()];
    let (mut old_start, mut new_start) = (0usize, 0usize);
    while old_start < old_dims.len() && new_start < new_dims.len() {
        let (mut old_end, mut new_end) = (old_start, new_start);
        let (mut old_product, mut new_product) = (old_dims[old_end], new_dims[new_end]);
        while old_product != new_product {
            if old_product < new_product {
                old_end += 1;
                if old_end >= old_dims.len() {
                    return Ok(None);
                }
                old_product = old_product
                    .checked_mul(old_dims[old_end])
                    .ok_or(StorageError::ShapeOverflow)?;
            } else {
                new_end += 1;
                if new_end >= new_dims.len() {
                    return Ok(None);
                }
                new_product = new_product
                    .checked_mul(new_dims[new_end])
                    .ok_or(StorageError::ShapeOverflow)?;
            }
        }

        for axis in old_start..old_end {
            let contiguous = if fortran {
                let expected = old_strides[old_axes[axis]]
                    .checked_mul(
                        isize::try_from(old_dims[axis]).map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
                old_strides[old_axes[axis + 1]] == expected
            } else {
                let expected = old_strides[old_axes[axis + 1]]
                    .checked_mul(
                        isize::try_from(old_dims[axis + 1])
                            .map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
                old_strides[old_axes[axis]] == expected
            };
            if !contiguous {
                return Ok(None);
            }
        }

        if fortran {
            let mut stride = old_strides[old_axes[old_start]];
            for compact_axis in new_start..=new_end {
                inferred[new_axes[compact_axis]] = stride;
                stride = stride
                    .checked_mul(
                        isize::try_from(new_dims[compact_axis])
                            .map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
            }
        } else {
            let mut stride = old_strides[old_axes[old_end]];
            for compact_axis in (new_start..=new_end).rev() {
                inferred[new_axes[compact_axis]] = stride;
                stride = stride
                    .checked_mul(
                        isize::try_from(new_dims[compact_axis])
                            .map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
            }
        }
        old_start = old_end + 1;
        new_start = new_end + 1;
    }
    if old_start != old_dims.len() || new_start != new_dims.len() {
        return Ok(None);
    }

    // Size-one axes do not affect addresses. Match the conventional strides
    // NumPy assigns around the non-singleton axes, including stepped layouts.
    for axis in 0..new_shape.len() {
        if new_shape[axis] != 1 {
            continue;
        }
        if fortran {
            if let Some(next) = (axis + 1..new_shape.len()).find(|&i| new_shape[i] > 1) {
                inferred[axis] = inferred[next];
            } else if let Some(previous) = (0..axis).rfind(|&i| new_shape[i] > 1) {
                inferred[axis] = inferred[previous]
                    .checked_mul(
                        isize::try_from(new_shape[previous])
                            .map_err(|_| StorageError::ShapeOverflow)?,
                    )
                    .ok_or(StorageError::ShapeOverflow)?;
            } else {
                inferred[axis] =
                    isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
            }
        } else if let Some(next) = (axis + 1..new_shape.len()).find(|&i| new_shape[i] > 1) {
            inferred[axis] = inferred[next]
                .checked_mul(
                    isize::try_from(new_shape[next]).map_err(|_| StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        } else if let Some(previous) = (0..axis).rfind(|&i| new_shape[i] > 1) {
            inferred[axis] = inferred[previous];
        } else {
            inferred[axis] =
                isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
        }
    }
    Ok(Some(inferred))
}

fn reshape_empty_strides(
    dtype: DType,
    shape: &[usize],
    fortran: bool,
) -> Result<Vec<isize>, StorageError> {
    let mut strides = vec![0; shape.len()];
    let mut stride = isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    if fortran {
        for axis in 0..shape.len() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(
                    isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
    } else {
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(
                    isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
                )
                .ok_or(StorageError::ShapeOverflow)?;
        }
    }
    Ok(strides)
}

fn c_strides(dtype: DType, shape: &[usize]) -> Result<Vec<isize>, StorageError> {
    let mut strides = vec![0; shape.len()];
    if shape.contains(&0) {
        return Ok(strides);
    }
    let mut stride = isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(
                isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(strides)
}

fn strides_for_axis_order(
    dtype: DType,
    shape: &[usize],
    fastest_to_slowest: &[usize],
) -> Result<Vec<isize>, StorageError> {
    if fastest_to_slowest.len() != shape.len() {
        return Err(StorageError::InvalidLayout);
    }
    let mut seen = vec![false; shape.len()];
    for &axis in fastest_to_slowest {
        let Some(axis_seen) = seen.get_mut(axis) else {
            return Err(StorageError::InvalidLayout);
        };
        if *axis_seen {
            return Err(StorageError::InvalidLayout);
        }
        *axis_seen = true;
    }

    if shape.contains(&0) {
        return Ok(vec![0; shape.len()]);
    }

    let mut strides = vec![0; shape.len()];
    let mut stride = isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    for &axis in fastest_to_slowest {
        strides[axis] = stride;
        stride = stride
            .checked_mul(
                isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(strides)
}

fn f_strides(dtype: DType, shape: &[usize]) -> Result<Vec<isize>, StorageError> {
    let mut strides = vec![0; shape.len()];
    if shape.contains(&0) {
        return Ok(strides);
    }
    let mut stride = isize::try_from(dtype.itemsize()).map_err(|_| StorageError::ShapeOverflow)?;
    for axis in 0..shape.len() {
        strides[axis] = stride;
        stride = stride
            .checked_mul(
                isize::try_from(shape[axis].max(1)).map_err(|_| StorageError::ShapeOverflow)?,
            )
            .ok_or(StorageError::ShapeOverflow)?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::{
        native_longdouble, ufunc, ByteOrder, DType, IndexItem, Scalar, StorageError, View,
    };
    use std::sync::Arc;

    #[test]
    fn typed_array_constructors_preserve_values_and_c_layout() {
        let floats = View::from_float32_values(vec![2, 2], &[1.25, -2.5, 0.0, 4.0]).unwrap();
        assert_eq!(floats.dtype(), DType::Float32);
        assert!(floats.is_c_contiguous());
        assert_eq!(
            floats.snapshot().unwrap(),
            vec![
                Scalar::Float32(1.25),
                Scalar::Float32(-2.5),
                Scalar::Float32(0.0),
                Scalar::Float32(4.0),
            ]
        );
        assert_eq!(floats.snapshot_float32().unwrap(), [1.25, -2.5, 0.0, 4.0]);

        let integers = View::from_int64_values(vec![2, 2], &[i64::MIN, -1, 0, i64::MAX]).unwrap();
        assert_eq!(integers.dtype(), DType::Int64);
        assert!(integers.is_c_contiguous());
        assert_eq!(
            integers.snapshot().unwrap(),
            vec![
                Scalar::Int64(i64::MIN),
                Scalar::Int64(-1),
                Scalar::Int64(0),
                Scalar::Int64(i64::MAX),
            ]
        );
        assert_eq!(
            integers.snapshot_int64().unwrap(),
            [i64::MIN, -1, 0, i64::MAX]
        );

        let int32 = View::from_int32_values(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        let transposed = int32.transpose(None).unwrap();
        assert_eq!(transposed.snapshot_int32().unwrap(), [1, 4, 2, 5, 3, 6]);
        let big_endian = View::from_values_with_order(
            DType::Int32,
            ByteOrder::Big,
            vec![2],
            &[Scalar::Int32(7), Scalar::Int32(-9)],
        )
        .unwrap();
        assert_eq!(big_endian.snapshot_int32().unwrap(), [7, -9]);
    }

    #[test]
    fn int32_concatenation_copies_contiguous_and_transposed_views_in_c_order() {
        let base = View::from_int32_values(vec![2, 3], &[1, 2, 3, 4, 5, 6]).unwrap();
        let transposed = base.transpose(None).unwrap();
        let joined_rows =
            View::concatenate_int32(&[&transposed, &transposed], Some(0), vec![6, 2], true)
                .unwrap();
        assert_eq!(joined_rows.shape(), [6, 2]);
        assert!(joined_rows.is_f_contiguous());
        assert!(!joined_rows.is_c_contiguous());
        assert_eq!(
            joined_rows.snapshot_int32().unwrap(),
            [1, 4, 2, 5, 3, 6, 1, 4, 2, 5, 3, 6]
        );

        let joined_columns =
            View::concatenate_int32(&[&base, &base], Some(1), vec![2, 6], false).unwrap();
        assert_eq!(
            joined_columns.snapshot_int32().unwrap(),
            [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]
        );

        let flattened =
            View::concatenate_int32(&[&transposed, &base], None, vec![12], false).unwrap();
        assert_eq!(
            flattened.snapshot_int32().unwrap(),
            [1, 4, 2, 5, 3, 6, 1, 2, 3, 4, 5, 6]
        );

        let empty = View::from_int32_values(vec![0, 3], &[]).unwrap();
        let with_empty =
            View::concatenate_int32(&[&empty, &base], Some(0), vec![2, 3], false).unwrap();
        assert_eq!(with_empty.snapshot_int32().unwrap(), [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn int32_concatenation_rejects_inconsistent_shapes_and_output_shape() {
        let left = View::from_int32_values(vec![2, 2], &[1, 2, 3, 4]).unwrap();
        let right = View::from_int32_values(vec![3, 1], &[5, 6, 7]).unwrap();
        assert_eq!(
            View::concatenate_int32(&[&left, &right], Some(0), vec![5, 2], false).unwrap_err(),
            StorageError::ShapeMismatch
        );
        assert_eq!(
            View::concatenate_int32(&[&left, &left], Some(0), vec![3, 2], false).unwrap_err(),
            StorageError::ShapeMismatch
        );
    }

    #[test]
    fn int32_sort_reads_and_writes_c_and_fortran_views_correctly() {
        let values = [9, 2, 7, 1, 8, 3];
        let c_order = View::from_int32_values(vec![2, 3], &values).unwrap();
        let c_sorted = c_order.sort_int32(Some(1), false).unwrap();
        assert!(c_sorted.is_c_contiguous());
        assert_eq!(c_sorted.snapshot_int32().unwrap(), [2, 7, 9, 1, 3, 8]);

        let f_order = View::from_int32_values_with_layout(vec![2, 3], &values, true).unwrap();
        let f_sorted = f_order.sort_int32(Some(1), false).unwrap();
        assert!(f_sorted.is_f_contiguous());
        assert!(!f_sorted.is_c_contiguous());
        assert_eq!(f_sorted.snapshot_int32().unwrap(), [2, 7, 9, 1, 3, 8]);
        assert_eq!(
            f_order
                .sort_int32(Some(1), true)
                .unwrap()
                .snapshot_int32()
                .unwrap(),
            [9, 7, 2, 8, 3, 1]
        );

        let flattened = f_order.sort_int32(None, false).unwrap();
        assert_eq!(flattened.shape(), [6]);
        assert_eq!(flattened.snapshot_int32().unwrap(), [1, 2, 3, 7, 8, 9]);
    }

    #[test]
    fn longdouble_truthiness_matches_nonzero_values() {
        assert!(!Scalar::LongDouble("0".into()).truthy());
        assert!(Scalar::LongDouble("1".into()).truthy());
        assert!(!Scalar::ComplexLongDouble("0".into(), "0".into()).truthy());
        assert!(Scalar::ComplexLongDouble("0".into(), "1".into()).truthy());
        assert!(Scalar::ComplexLongDouble("1".into(), "0".into()).truthy());
    }

    #[test]
    fn complex_longdouble_extrema_preserve_precision_beyond_f64() {
        if !native_longdouble::has_extended_native() {
            return;
        }
        let lower = Scalar::ComplexLongDouble("9007199254740992".into(), "1e-4000".into());
        let higher = Scalar::ComplexLongDouble("9007199254740993".into(), "2e-4000".into());
        for (name, expected_real, expected_imag) in [
            ("maximum", "9007199254740993", "2e-4000"),
            ("fmax", "9007199254740993", "2e-4000"),
            ("minimum", "9007199254740992", "1e-4000"),
            ("fmin", "9007199254740992", "1e-4000"),
        ] {
            let result = ufunc::binary(
                name,
                lower.clone(),
                higher.clone(),
                DType::ComplexLongDouble,
            )
            .unwrap();
            let Scalar::ComplexLongDouble(real, imag) = &result[0] else {
                panic!("complex extrema must return a complex long-double scalar");
            };
            assert_eq!(
                native_longdouble::compare(real, expected_real),
                Some(Some(0)),
                "{name}"
            );
            assert_eq!(
                native_longdouble::compare(imag, expected_imag),
                Some(Some(0)),
                "{name}"
            );
        }
    }

    #[test]
    fn int32_typecodes_follow_target_aliases() {
        #[cfg(target_os = "windows")]
        {
            assert_eq!(DType::Int32.char(), 'l');
            assert_eq!(DType::UInt32.char(), 'L');
        }
        #[cfg(not(target_os = "windows"))]
        {
            assert_eq!(DType::Int32.char(), 'i');
            assert_eq!(DType::UInt32.char(), 'I');
        }
    }

    #[test]
    fn extended_longdouble_loops_keep_platform_precision() {
        if DType::LongDouble.itemsize() <= DType::Float64.itemsize() {
            return;
        }

        let (inputs, outputs) =
            super::ufunc::resolve_loop("add", &[DType::Complex128, DType::LongDouble]).unwrap();
        assert!(inputs
            .iter()
            .all(|dtype| matches!(dtype, DType::LongDouble | DType::ComplexLongDouble)));
        assert_eq!(outputs, vec![DType::ComplexLongDouble]);
    }

    fn array(values: &[i64]) -> View {
        View::from_values(
            DType::Int64,
            vec![values.len()],
            &values
                .iter()
                .copied()
                .map(Scalar::Int64)
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    #[test]
    fn rejects_overflowing_shapes() {
        assert_eq!(
            View::zeros(DType::Float64, vec![usize::MAX, 2]).err(),
            Some(StorageError::ShapeOverflow)
        );
        assert_eq!(
            View::zeros(DType::Float64, vec![isize::MAX as usize, 2]).err(),
            Some(StorageError::ShapeOverflow)
        );
    }
    #[test]
    fn negative_stride_view_is_shared_and_checked() {
        let owner = array(&[0, 1, 2, 3, 4, 5]);
        let reverse = owner
            .index(&[IndexItem::Slice {
                start: 5,
                step: -2,
                len: 3,
            }])
            .unwrap();
        assert_eq!(reverse.strides(), &[-16]);
        assert_eq!(
            reverse.snapshot().unwrap(),
            vec![Scalar::Int64(5), Scalar::Int64(3), Scalar::Int64(1)]
        );
        reverse.write_linear(1, Scalar::Int64(33)).unwrap();
        assert_eq!(owner.read_linear(3).unwrap(), Scalar::Int64(33));
        assert!(owner.index(&[IndexItem::Integer(6)]).is_err());
    }
    #[test]
    fn overlapping_assignment_uses_a_snapshot() {
        let owner = array(&[0, 1, 2, 3, 4]);
        let dst = owner
            .index(&[IndexItem::Slice {
                start: 1,
                step: 1,
                len: 4,
            }])
            .unwrap();
        let src = owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: 1,
                len: 4,
            }])
            .unwrap();
        dst.assign_view(&src).unwrap();
        assert_eq!(
            owner.snapshot().unwrap(),
            vec![
                Scalar::Int64(0),
                Scalar::Int64(0),
                Scalar::Int64(1),
                Scalar::Int64(2),
                Scalar::Int64(3)
            ]
        );
    }
    #[test]
    fn view_keeps_owner_alive_and_copy_does_not_alias() {
        let owner = array(&[4, 5, 6]);
        let weak = Arc::downgrade(&owner.storage);
        let view = owner
            .index(&[IndexItem::Slice {
                start: 1,
                step: 1,
                len: 2,
            }])
            .unwrap();
        drop(owner);
        assert!(weak.upgrade().is_some());
        assert_eq!(
            view.snapshot().unwrap(),
            vec![Scalar::Int64(5), Scalar::Int64(6)]
        );
        let copy = view.copy().unwrap();
        assert!(!view.shares_storage_with(&copy));
        copy.write_linear(0, Scalar::Int64(77)).unwrap();
        assert_eq!(view.read_linear(0).unwrap(), Scalar::Int64(5));
    }
    #[test]
    fn empty_arrays_have_no_readable_elements() {
        let empty = View::empty(DType::UInt64, vec![2, 0, 3]).unwrap();
        assert!(empty.snapshot().unwrap().is_empty());
        assert_eq!(empty.strides(), &[0, 0, 0]);
        assert!(empty.read_at(&[0, 0, 0]).is_err());
    }
    #[test]
    fn rejects_invalid_slice_metadata_and_preserves_empty_slice_strides() {
        let owner = array(&[0, 1, 2]);
        assert!(owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: 0,
                len: 1,
            }])
            .is_err());
        assert!(owner
            .index(&[IndexItem::Slice {
                start: 4,
                step: 1,
                len: 1,
            }])
            .is_err());
        let empty = owner
            .index(&[IndexItem::Slice {
                start: 0,
                step: -1,
                len: 0,
            }])
            .unwrap();
        assert_eq!(empty.strides(), &[8]);
    }
    #[test]
    fn assignment_casts_values_and_checks_broadcast_shape() {
        let dst = array(&[1, 2]);
        assert_eq!(
            dst.assign_view(&View::zeros(DType::Int64, vec![1, 2]).unwrap())
                .err(),
            Some(StorageError::CannotBroadcast {
                from: vec![1, 2],
                to: vec![2]
            })
        );
        dst.assign_view(&View::zeros(DType::UInt64, vec![2]).unwrap())
            .unwrap();
        assert_eq!(
            dst.snapshot().unwrap(),
            vec![Scalar::Int64(0), Scalar::Int64(0)]
        );
    }

    #[test]
    fn floating_integer_casts_have_deterministic_results() {
        assert_eq!(
            Scalar::Float64(1e20).cast(DType::Int64).unwrap(),
            Scalar::Int64(i64::MAX)
        );
        assert_eq!(
            Scalar::Float64(f64::NEG_INFINITY)
                .cast(DType::Int32)
                .unwrap(),
            Scalar::Int32(i32::MIN)
        );
        assert_eq!(
            Scalar::Float64(f64::NAN).cast(DType::UInt32).unwrap(),
            Scalar::UInt32(0)
        );
        assert_eq!(
            Scalar::Float64(1e20).cast(DType::Int8).unwrap(),
            Scalar::Int8(-1)
        );
        assert_eq!(
            Scalar::LongDouble("9223372036854775807".into())
                .cast(DType::Int64)
                .unwrap(),
            Scalar::Int64(i64::MAX)
        );
        assert!(Scalar::Float64(f64::NAN).integer_cast_is_invalid(DType::Int32));
        assert!(Scalar::Float64(1e20).integer_cast_is_invalid(DType::Int8));
        assert!(!Scalar::Float64(300.0).integer_cast_is_invalid(DType::UInt8));
    }

    #[test]
    fn scalar_integer_and_slice_before_fancy_index_move_advanced_axis() {
        let values = (0..60).map(Scalar::Int64).collect::<Vec<_>>();
        let owner = View::from_values(DType::Int64, vec![3, 4, 5], &values).unwrap();
        let selected = owner
            .index(&[
                IndexItem::Integer(0),
                IndexItem::Slice {
                    start: 0,
                    step: 1,
                    len: 4,
                },
                IndexItem::Fancy {
                    shape: vec![2],
                    strides: vec![std::mem::size_of::<isize>() as isize],
                    indices: vec![0, 2],
                },
            ])
            .unwrap();
        assert_eq!(selected.shape(), &[2, 4]);
        assert_eq!(
            selected.snapshot().unwrap(),
            vec![
                Scalar::Int64(0),
                Scalar::Int64(5),
                Scalar::Int64(10),
                Scalar::Int64(15),
                Scalar::Int64(2),
                Scalar::Int64(7),
                Scalar::Int64(12),
                Scalar::Int64(17),
            ]
        );
    }
}
