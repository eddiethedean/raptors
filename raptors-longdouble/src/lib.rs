//! Isolated native `long double` bridge for targets where it is wider than
//! binary64. Scalar strings cross the boundary; array storage stays in the
//! safe `raptors-storage` crate.

#[cfg(raptors_native_longdouble)]
use std::ffi::{CStr, CString};
#[cfg(raptors_native_longdouble)]
use std::os::raw::{c_char, c_int};

#[cfg(raptors_native_longdouble)]
const TEXT_CAPACITY: usize = 128;

#[cfg(raptors_native_longdouble)]
extern "C" {
    fn raptors_ld_encode(value: *const c_char, output: *mut u8, output_len: usize) -> c_int;
    fn raptors_ld_decode(
        input: *const u8,
        input_len: usize,
        output: *mut c_char,
        cap: usize,
    ) -> c_int;
    fn raptors_ld_unary_real(
        name: *const c_char,
        value: *const c_char,
        output: *mut c_char,
        cap: usize,
    ) -> c_int;
    fn raptors_ld_binary_real(
        name: *const c_char,
        left: *const c_char,
        right: *const c_char,
        first: *mut c_char,
        first_cap: usize,
        second: *mut c_char,
        second_cap: usize,
    ) -> c_int;
    fn raptors_ld_unary_complex(
        name: *const c_char,
        real: *const c_char,
        imag: *const c_char,
        out_real: *mut c_char,
        real_cap: usize,
        out_imag: *mut c_char,
        imag_cap: usize,
    ) -> c_int;
    fn raptors_ld_binary_complex(
        name: *const c_char,
        left_real: *const c_char,
        left_imag: *const c_char,
        right_real: *const c_char,
        right_imag: *const c_char,
        out_real: *mut c_char,
        real_cap: usize,
        out_imag: *mut c_char,
        imag_cap: usize,
    ) -> c_int;
    fn raptors_ld_classify(value: *const c_char, property: c_int) -> c_int;
    fn raptors_ld_logaddexp_intermediate_underflow(
        left: *const c_char,
        right: *const c_char,
        base2: c_int,
    ) -> c_int;
    fn raptors_ld_logaddexp_intermediate_overflow(
        left: *const c_char,
        right: *const c_char,
    ) -> c_int;
    fn raptors_ld_compare(left: *const c_char, right: *const c_char) -> c_int;
    fn raptors_ld_compare_complex(
        left_real: *const c_char,
        left_imag: *const c_char,
        right_real: *const c_char,
        right_imag: *const c_char,
    ) -> c_int;
    fn raptors_ld_frexp(
        value: *const c_char,
        fraction: *mut c_char,
        cap: usize,
        exponent: *mut c_int,
    ) -> c_int;
    fn raptors_ld_modf(
        value: *const c_char,
        fraction: *mut c_char,
        fraction_cap: usize,
        integral: *mut c_char,
        integral_cap: usize,
    ) -> c_int;
}

/// Whether the target has a native C `long double` wider than binary64.
pub const fn has_extended_native() -> bool {
    cfg!(raptors_native_longdouble) && !cfg!(miri)
}

/// Encode a decimal scalar using the target's native long-double layout.
pub fn encode(value: &str) -> Option<[u8; 16]> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let value = CString::new(value).ok()?;
        let mut output = [0_u8; 16];
        // SAFETY: the input is NUL-terminated and output points to 16 writable
        // bytes, matching the target layouts selected in build.rs.
        let status =
            unsafe { raptors_ld_encode(value.as_ptr(), output.as_mut_ptr(), output.len()) };
        (status == 1).then_some(output)
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = value;
        None
    }
}

/// Decode bytes in the target's native long-double layout to a round-trip
/// decimal representation.
pub fn decode(input: &[u8; 16]) -> Option<String> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let mut output = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: input is exactly 16 readable bytes and output is a live
        // writable C character buffer with the supplied capacity.
        let status = unsafe {
            raptors_ld_decode(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        (status == 1).then(|| c_buffer_to_string(&output))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = input;
        None
    }
}

pub fn unary_real(name: &str, value: &str) -> Option<String> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let name = CString::new(name).ok()?;
        let value = CString::new(value).ok()?;
        let mut output = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: both inputs are NUL-terminated and output is a bounded live
        // writable buffer; the C bridge writes at most `output.len()` bytes.
        let status = unsafe {
            raptors_ld_unary_real(
                name.as_ptr(),
                value.as_ptr(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        (status == 1).then(|| c_buffer_to_string(&output))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (name, value);
        None
    }
}

pub fn binary_real(name: &str, left: &str, right: &str) -> Option<(String, String)> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let name = CString::new(name).ok()?;
        let left = CString::new(left).ok()?;
        let right = CString::new(right).ok()?;
        let mut first = [0 as c_char; TEXT_CAPACITY];
        let mut second = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: inputs are NUL-terminated and both outputs are live bounded
        // character buffers with capacities passed to the C function.
        let status = unsafe {
            raptors_ld_binary_real(
                name.as_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                first.as_mut_ptr(),
                first.len(),
                second.as_mut_ptr(),
                second.len(),
            )
        };
        (status == 1).then(|| (c_buffer_to_string(&first), c_buffer_to_string(&second)))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (name, left, right);
        None
    }
}

pub fn unary_complex(name: &str, real: &str, imag: &str) -> Option<(String, String)> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let name = CString::new(name).ok()?;
        let real = CString::new(real).ok()?;
        let imag = CString::new(imag).ok()?;
        let mut out_real = [0 as c_char; TEXT_CAPACITY];
        let mut out_imag = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: each input is NUL-terminated and each output is a live
        // bounded character buffer whose capacity is passed to C.
        let status = unsafe {
            raptors_ld_unary_complex(
                name.as_ptr(),
                real.as_ptr(),
                imag.as_ptr(),
                out_real.as_mut_ptr(),
                out_real.len(),
                out_imag.as_mut_ptr(),
                out_imag.len(),
            )
        };
        (status == 1).then(|| (c_buffer_to_string(&out_real), c_buffer_to_string(&out_imag)))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (name, real, imag);
        None
    }
}

pub fn binary_complex(
    name: &str,
    left_real: &str,
    left_imag: &str,
    right_real: &str,
    right_imag: &str,
) -> Option<(String, String)> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let name = CString::new(name).ok()?;
        let left_real = CString::new(left_real).ok()?;
        let left_imag = CString::new(left_imag).ok()?;
        let right_real = CString::new(right_real).ok()?;
        let right_imag = CString::new(right_imag).ok()?;
        let mut out_real = [0 as c_char; TEXT_CAPACITY];
        let mut out_imag = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: inputs are NUL-terminated and output buffers are live,
        // bounded, and disjoint for the duration of the C call.
        let status = unsafe {
            raptors_ld_binary_complex(
                name.as_ptr(),
                left_real.as_ptr(),
                left_imag.as_ptr(),
                right_real.as_ptr(),
                right_imag.as_ptr(),
                out_real.as_mut_ptr(),
                out_real.len(),
                out_imag.as_mut_ptr(),
                out_imag.len(),
            )
        };
        (status == 1).then(|| (c_buffer_to_string(&out_real), c_buffer_to_string(&out_imag)))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (name, left_real, left_imag, right_real, right_imag);
        None
    }
}

pub fn classify(value: &str, property: i32) -> Option<bool> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let value = CString::new(value).ok()?;
        // SAFETY: value is a live NUL-terminated string and the property is a
        // small integer enum validated by the C bridge.
        let result = unsafe { raptors_ld_classify(value.as_ptr(), property) };
        (result >= 0).then_some(result != 0)
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (value, property);
        None
    }
}

/// Whether the exponential intermediate in logaddexp underflows the native
/// long-double normal range for these operands.
pub fn logaddexp_intermediate_underflow(left: &str, right: &str, base2: bool) -> Option<bool> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let left = CString::new(left).ok()?;
        let right = CString::new(right).ok()?;
        // SAFETY: both values are live NUL-terminated strings; base2 is a
        // boolean flag consumed by the C bridge.
        let result = unsafe {
            raptors_ld_logaddexp_intermediate_underflow(
                left.as_ptr(),
                right.as_ptr(),
                c_int::from(base2),
            )
        };
        (result >= 0).then_some(result != 0)
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (left, right, base2);
        None
    }
}

/// Whether subtracting these finite operands overflows native `long double`.
pub fn logaddexp_intermediate_overflow(left: &str, right: &str) -> Option<bool> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let left = CString::new(left).ok()?;
        let right = CString::new(right).ok()?;
        // SAFETY: both inputs are live NUL-terminated strings.
        let result =
            unsafe { raptors_ld_logaddexp_intermediate_overflow(left.as_ptr(), right.as_ptr()) };
        (result >= 0).then_some(result != 0)
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (left, right);
        None
    }
}

/// Compare finite or infinite real values: -1, 0, 1; `None` for NaN.
pub fn compare(left: &str, right: &str) -> Option<Option<i8>> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let left = CString::new(left).ok()?;
        let right = CString::new(right).ok()?;
        // SAFETY: both inputs are live NUL-terminated strings.
        let result = unsafe { raptors_ld_compare(left.as_ptr(), right.as_ptr()) };
        Some(match result {
            -1 => Some(-1),
            0 => Some(0),
            1 => Some(1),
            _ => None,
        })
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (left, right);
        None
    }
}

pub fn compare_complex(
    left_real: &str,
    left_imag: &str,
    right_real: &str,
    right_imag: &str,
) -> Option<Option<i8>> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let left_real = CString::new(left_real).ok()?;
        let left_imag = CString::new(left_imag).ok()?;
        let right_real = CString::new(right_real).ok()?;
        let right_imag = CString::new(right_imag).ok()?;
        // SAFETY: all four inputs are live NUL-terminated strings.
        let result = unsafe {
            raptors_ld_compare_complex(
                left_real.as_ptr(),
                left_imag.as_ptr(),
                right_real.as_ptr(),
                right_imag.as_ptr(),
            )
        };
        Some(match result {
            -1 => Some(-1),
            0 => Some(0),
            1 => Some(1),
            _ => None,
        })
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = (left_real, left_imag, right_real, right_imag);
        None
    }
}

pub fn frexp(value: &str) -> Option<(String, i32)> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let value = CString::new(value).ok()?;
        let mut fraction = [0 as c_char; TEXT_CAPACITY];
        let mut exponent = 0;
        // SAFETY: value is NUL-terminated and outputs point to valid writable
        // storage with the stated fraction capacity.
        let status = unsafe {
            raptors_ld_frexp(
                value.as_ptr(),
                fraction.as_mut_ptr(),
                fraction.len(),
                &mut exponent,
            )
        };
        (status == 1).then(|| (c_buffer_to_string(&fraction), exponent))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = value;
        None
    }
}

pub fn modf(value: &str) -> Option<(String, String)> {
    if !has_extended_native() {
        return None;
    }
    #[cfg(raptors_native_longdouble)]
    {
        let value = CString::new(value).ok()?;
        let mut fraction = [0 as c_char; TEXT_CAPACITY];
        let mut integral = [0 as c_char; TEXT_CAPACITY];
        // SAFETY: value is NUL-terminated and the two outputs are disjoint,
        // writable buffers whose capacities are passed to C.
        let status = unsafe {
            raptors_ld_modf(
                value.as_ptr(),
                fraction.as_mut_ptr(),
                fraction.len(),
                integral.as_mut_ptr(),
                integral.len(),
            )
        };
        (status == 1).then(|| (c_buffer_to_string(&fraction), c_buffer_to_string(&integral)))
    }
    #[cfg(not(raptors_native_longdouble))]
    {
        let _ = value;
        None
    }
}

#[cfg(raptors_native_longdouble)]
fn c_buffer_to_string(buffer: &[c_char]) -> String {
    // SAFETY: every C output function guarantees a trailing NUL on success;
    // this conversion is only used when that function returned status 1.
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::{
        binary_complex, binary_real, classify, compare, decode, encode, has_extended_native,
        logaddexp_intermediate_overflow, logaddexp_intermediate_underflow, unary_complex,
        unary_real,
    };

    #[test]
    fn native_roundtrip_and_real_kernels_keep_extended_precision() {
        if !has_extended_native() {
            return;
        }
        let original = "9007199254740993";
        let bytes = encode(original).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(compare(&decoded, original), Some(Some(0)));

        let (sum, _) = binary_real("add", original, "1").unwrap();
        assert_eq!(compare(&sum, "9007199254740994"), Some(Some(0)));

        let exponential = binary_real("multiply", "1e4000", "1").unwrap().0;
        assert_eq!(classify(&exponential, 4), Some(true));
        assert_eq!(classify(&exponential, 1), Some(false));

        for operation in [
            "absolute",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cbrt",
            "ceil",
            "conjugate",
            "cos",
            "cosh",
            "deg2rad",
            "degrees",
            "exp",
            "exp2",
            "expm1",
            "fabs",
            "floor",
            "log",
            "log10",
            "log1p",
            "log2",
            "negative",
            "positive",
            "reciprocal",
            "rint",
            "sign",
            "sin",
            "sinh",
            "spacing",
            "sqrt",
            "square",
            "tan",
            "tanh",
            "trunc",
        ] {
            assert!(unary_real(operation, "0.5").is_some(), "{operation}");
        }
        for operation in [
            "add",
            "subtract",
            "multiply",
            "divide",
            "true_divide",
            "floor_divide",
            "remainder",
            "mod",
            "fmod",
            "divmod",
            "power",
            "pow",
            "float_power",
            "maximum",
            "minimum",
            "fmax",
            "fmin",
            "copysign",
            "hypot",
            "arctan2",
            "atan2",
            "heaviside",
            "logaddexp",
            "logaddexp2",
            "nextafter",
            "ldexp",
        ] {
            assert!(
                binary_real(operation, "0.5", "1.5").is_some(),
                "{operation}"
            );
        }
    }

    #[test]
    fn ldexp_clamps_exponents_before_converting_to_c_int() {
        if !has_extended_native() {
            return;
        }
        let overflow = binary_real("ldexp", "1", "2147483648").unwrap().0;
        assert_eq!(classify(&overflow, 1), Some(true));
        let underflow = binary_real("ldexp", "1", "-2147483649").unwrap().0;
        assert_eq!(classify(&underflow, 3), Some(true));
    }

    #[test]
    fn extended_longdouble_extrema_keep_the_first_signed_zero() {
        if !has_extended_native() {
            return;
        }
        for operation in ["maximum", "minimum"] {
            let (positive_first, _) = binary_real(operation, "0", "-0").unwrap();
            let (negative_first, _) = binary_real(operation, "-0", "0").unwrap();
            assert_eq!(classify(&positive_first, 2), Some(false), "{operation}");
            assert_eq!(classify(&negative_first, 2), Some(true), "{operation}");
        }
    }

    #[test]
    fn longdouble_logaddexp_underflow_detects_extreme_exponents() {
        if !has_extended_native() {
            return;
        }
        assert_eq!(
            logaddexp_intermediate_underflow("0", "-1e10", false),
            Some(true)
        );
        assert_eq!(
            logaddexp_intermediate_underflow("0", "-1e10", true),
            Some(true)
        );
    }

    #[test]
    fn longdouble_logaddexp_overflow_detects_native_range_differences() {
        if !has_extended_native() {
            return;
        }
        assert_eq!(
            logaddexp_intermediate_overflow("1e4932", "-1e4932"),
            Some(true)
        );
        assert_eq!(logaddexp_intermediate_overflow("1e4932", "0"), Some(false));
        assert_eq!(logaddexp_intermediate_overflow("inf", "-inf"), Some(false));
    }

    #[test]
    fn native_complex_addition_keeps_extended_precision() {
        if !has_extended_native() {
            return;
        }
        let (real, imag) = binary_complex("add", "9007199254740993", "0", "1", "0").unwrap();
        assert_eq!(compare(&real, "9007199254740994"), Some(Some(0)));
        assert_eq!(compare(&imag, "0"), Some(Some(0)));
    }

    #[test]
    fn native_complex_kernel_preserves_real_and_imaginary_components() {
        if !has_extended_native() {
            return;
        }
        let (real, imag) =
            binary_complex("add", "9007199254740993", "1e-4000", "1", "1e-4000").unwrap();
        assert_eq!(compare(&real, "9007199254740994"), Some(Some(0)));
        assert_eq!(compare(&imag, "2e-4000"), Some(Some(0)));

        for operation in [
            "absolute",
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cos",
            "cosh",
            "exp",
            "exp2",
            "expm1",
            "rint",
            "log",
            "log1p",
            "log10",
            "log2",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "reciprocal",
            "square",
            "positive",
            "negative",
            "conj",
            "conjugate",
            "sign",
        ] {
            assert!(
                unary_complex(operation, "0.5", "0.25").is_some(),
                "{operation}"
            );
        }
        for operation in [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "pow",
            "float_power",
        ] {
            assert!(
                binary_complex(operation, "0.5", "0.25", "1.5", "0.75").is_some(),
                "{operation}"
            );
        }
        for operation in ["maximum", "minimum", "fmax", "fmin"] {
            assert!(
                binary_complex(operation, "0.5", "0.25", "1.5", "0.75").is_some(),
                "{operation}"
            );
        }
    }
}
