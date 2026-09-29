#include <complex.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_ld(const char *text, long double *value) {
    if (text == NULL || value == NULL) return 0;
    char *end = NULL;
    *value = strtold(text, &end);
    return end != text && end != NULL && *end == '\0';
}

static int write_ld(long double value, char *output, size_t capacity) {
    if (output == NULL || capacity == 0) return 0;
#ifdef LDBL_DECIMAL_DIG
    const int digits = LDBL_DECIMAL_DIG;
#else
    const int digits = 36;
#endif
    int written = snprintf(output, capacity, "%.*Lg", digits, value);
    return written >= 0 && (size_t)written < capacity;
}

int raptors_ld_encode(const char *text, unsigned char *output, size_t output_len) {
    if (output == NULL || output_len != 16) return 0;
    long double value;
    if (!parse_ld(text, &value) || sizeof(value) != 16 || LDBL_MANT_DIG <= 53) return 0;
    memset(output, 0, output_len);
#if LDBL_MANT_DIG == 64
    /* x87 stores 80 value bits in a 16-byte ABI slot; clear its six padding bytes. */
    memcpy(output, &value, 10);
#else
    memcpy(output, &value, sizeof(value));
#endif
    return 1;
}

int raptors_ld_decode(const unsigned char *input, size_t input_len, char *output, size_t capacity) {
    if (input == NULL || input_len != 16 || output == NULL || capacity == 0) return 0;
    long double value;
    memset(&value, 0, sizeof(value));
#if LDBL_MANT_DIG == 64
    memcpy(&value, input, 10);
#else
    memcpy(&value, input, sizeof(value));
#endif
    return write_ld(value, output, capacity);
}

static int unary_value(const char *name, long double x, long double *result) {
    if (strcmp(name, "absolute") == 0 || strcmp(name, "abs") == 0 || strcmp(name, "fabs") == 0) *result = fabsl(x);
    else if (strcmp(name, "acos") == 0 || strcmp(name, "arccos") == 0) *result = acosl(x);
    else if (strcmp(name, "acosh") == 0 || strcmp(name, "arccosh") == 0) *result = acoshl(x);
    else if (strcmp(name, "asin") == 0 || strcmp(name, "arcsin") == 0) *result = asinl(x);
    else if (strcmp(name, "asinh") == 0 || strcmp(name, "arcsinh") == 0) *result = asinhl(x);
    else if (strcmp(name, "atan") == 0 || strcmp(name, "arctan") == 0) *result = atanl(x);
    else if (strcmp(name, "atanh") == 0 || strcmp(name, "arctanh") == 0) *result = atanhl(x);
    else if (strcmp(name, "cbrt") == 0) *result = cbrtl(x);
    else if (strcmp(name, "ceil") == 0) *result = ceill(x);
    else if (strcmp(name, "cos") == 0) *result = cosl(x);
    else if (strcmp(name, "cosh") == 0) *result = coshl(x);
    else if (strcmp(name, "deg2rad") == 0 || strcmp(name, "radians") == 0) *result = x * acosl(-1.0L) / 180.0L;
    else if (strcmp(name, "degrees") == 0 || strcmp(name, "rad2deg") == 0) *result = x * 180.0L / acosl(-1.0L);
    else if (strcmp(name, "exp") == 0) *result = expl(x);
    else if (strcmp(name, "exp2") == 0) *result = exp2l(x);
    else if (strcmp(name, "expm1") == 0) *result = expm1l(x);
    else if (strcmp(name, "floor") == 0) *result = floorl(x);
    else if (strcmp(name, "log") == 0) *result = logl(x);
    else if (strcmp(name, "log10") == 0) *result = log10l(x);
    else if (strcmp(name, "log1p") == 0) *result = log1pl(x);
    else if (strcmp(name, "log2") == 0) *result = log2l(x);
    else if (strcmp(name, "negative") == 0) *result = -x;
    else if (strcmp(name, "positive") == 0 || strcmp(name, "conjugate") == 0 || strcmp(name, "conj") == 0) *result = x;
    else if (strcmp(name, "reciprocal") == 0) *result = 1.0L / x;
    else if (strcmp(name, "rint") == 0) *result = rintl(x);
    else if (strcmp(name, "sign") == 0) *result = isnan(x) ? x : (x == 0.0L ? 0.0L : copysignl(1.0L, x));
    else if (strcmp(name, "sin") == 0) *result = sinl(x);
    else if (strcmp(name, "sinh") == 0) *result = sinhl(x);
    else if (strcmp(name, "spacing") == 0) {
        if (!isfinite(x)) *result = nanl("");
        else *result = nextafterl(x, signbit(x) ? -INFINITY : INFINITY) - x;
    }
    else if (strcmp(name, "sqrt") == 0) *result = sqrtl(x);
    else if (strcmp(name, "square") == 0) *result = x * x;
    else if (strcmp(name, "tan") == 0) *result = tanl(x);
    else if (strcmp(name, "tanh") == 0) *result = tanhl(x);
    else if (strcmp(name, "trunc") == 0) *result = truncl(x);
    else return 0;
    return 1;
}

int raptors_ld_unary_real(const char *name, const char *text, char *output, size_t capacity) {
    long double x, result;
    if (!parse_ld(text, &x) || !unary_value(name, x, &result)) return 0;
    return write_ld(result, output, capacity);
}

static int binary_values(const char *name, long double a, long double b, long double *first, long double *second) {
    if (strcmp(name, "add") == 0) *first = a + b;
    else if (strcmp(name, "subtract") == 0) *first = a - b;
    else if (strcmp(name, "multiply") == 0) *first = a * b;
    else if (strcmp(name, "divide") == 0 || strcmp(name, "true_divide") == 0) *first = a / b;
    else if (strcmp(name, "floor_divide") == 0) *first = floorl(a / b);
    else if (strcmp(name, "fmod") == 0) *first = fmodl(a, b);
    else if (strcmp(name, "remainder") == 0 || strcmp(name, "mod") == 0) {
        *first = fmodl(a, b);
        if (*first != 0.0L && signbit(*first) != signbit(b)) *first += b;
        if (*first == 0.0L) *first = copysignl(0.0L, b);
    }
    else if (strcmp(name, "divmod") == 0) {
        *first = floorl(a / b);
        *second = fmodl(a, b);
        if (*second != 0.0L && signbit(*second) != signbit(b)) *second += b;
        if (*second == 0.0L) *second = copysignl(0.0L, b);
    }
    else if (strcmp(name, "power") == 0 || strcmp(name, "pow") == 0 || strcmp(name, "float_power") == 0) *first = powl(a, b);
    else if (strcmp(name, "maximum") == 0 || strcmp(name, "minimum") == 0) {
        if (isnan(a)) *first = a;
        else if (isnan(b)) *first = b;
        else if (a == 0.0L && b == 0.0L) {
#if defined(__APPLE__)
            if (strcmp(name, "maximum") == 0) *first = signbit(a) ? b : a;
            else *first = signbit(a) ? a : b;
#else
            *first = b;
#endif
        }
        else if (strcmp(name, "maximum") == 0) *first = a > b ? a : b;
        else *first = a < b ? a : b;
    }
    else if (strcmp(name, "fmax") == 0) *first = fmaxl(a, b);
    else if (strcmp(name, "fmin") == 0) *first = fminl(a, b);
    else if (strcmp(name, "copysign") == 0) *first = copysignl(a, b);
    else if (strcmp(name, "hypot") == 0) *first = hypotl(a, b);
    else if (strcmp(name, "arctan2") == 0 || strcmp(name, "atan2") == 0) *first = atan2l(a, b);
    else if (strcmp(name, "heaviside") == 0) *first = isnan(a) ? a : (a < 0.0L ? 0.0L : (a > 0.0L ? 1.0L : b));
    else if (strcmp(name, "logaddexp") == 0) {
        if (a == b) *first = a + logl(2.0L);
        else if (a > b) *first = a + log1pl(expl(b - a));
        else *first = b + log1pl(expl(a - b));
    }
    else if (strcmp(name, "logaddexp2") == 0) {
        if (a == b) *first = a + 1.0L;
        else if (a > b) *first = a + log2l(1.0L + exp2l(b - a));
        else *first = b + log2l(1.0L + exp2l(a - b));
    }
    else if (strcmp(name, "nextafter") == 0) *first = nextafterl(a, b);
    else if (strcmp(name, "ldexp") == 0) {
        int exponent;
        if (b >= (long double)INT_MAX) exponent = INT_MAX;
        else if (b <= (long double)INT_MIN) exponent = INT_MIN;
        else exponent = (int)b;
        *first = scalbnl(a, exponent);
    }
    else return 0;
    if (strcmp(name, "divmod") != 0) *second = 0.0L;
    return 1;
}

int raptors_ld_binary_real(const char *name, const char *left, const char *right, char *first, size_t first_cap, char *second, size_t second_cap) {
    long double a, b, first_value, second_value = 0.0L;
    if (!parse_ld(left, &a) || !parse_ld(right, &b) || !binary_values(name, a, b, &first_value, &second_value)) return 0;
    if (!write_ld(first_value, first, first_cap)) return 0;
    return write_ld(second_value, second, second_cap);
}

static int parse_complex(const char *real_text, const char *imag_text, long double complex *value) {
    long double real, imag;
    if (!parse_ld(real_text, &real) || !parse_ld(imag_text, &imag)) return 0;
    *value = CMPLXL(real, imag);
    return 1;
}

static int write_complex(long double complex value, char *out_real, size_t real_cap, char *out_imag, size_t imag_cap) {
    return write_ld(creall(value), out_real, real_cap) && write_ld(cimagl(value), out_imag, imag_cap);
}

static long double complex complex_atan(long double real, long double imag) {
    /* musl's catanl loses several digits for ordinary finite inputs. These
     * equivalent formulas keep each component in the long-double domain and
     * let atan2l preserve the branch selected by signed zero. */
    long double denominator = 1.0L - real * real - imag * imag;
    long double magnitude = real * real + (imag - 1.0L) * (imag - 1.0L);
    long double imaginary_result;
    if (magnitude == 0.0L) {
        imaginary_result = 0.25L * (logl(real * real + (imag + 1.0L) * (imag + 1.0L)) - logl(magnitude));
    } else {
        imaginary_result = 0.25L * log1pl((4.0L * imag) / magnitude);
    }
    return CMPLXL(0.5L * atan2l(2.0L * real, denominator), imaginary_result);
}

static long double complex complex_atanh(long double real, long double imag) {
    long double denominator = 1.0L - real * real - imag * imag;
    long double magnitude = (1.0L - real) * (1.0L - real) + imag * imag;
    long double real_result;
    if (magnitude == 0.0L) {
        real_result = 0.25L * (logl((1.0L + real) * (1.0L + real) + imag * imag) - logl(magnitude));
    } else {
        real_result = 0.25L * log1pl((4.0L * real) / magnitude);
    }
    return CMPLXL(real_result, 0.5L * atan2l(2.0L * imag, denominator));
}

static int unary_complex_value(const char *name, long double complex z, long double complex *result) {
    long double re = creall(z), im = cimagl(z);
    if (strcmp(name, "absolute") == 0 || strcmp(name, "abs") == 0) *result = CMPLXL(cabsl(z), 0.0L);
    else if (strcmp(name, "acos") == 0 || strcmp(name, "arccos") == 0) *result = cacosl(z);
    else if (strcmp(name, "acosh") == 0 || strcmp(name, "arccosh") == 0) *result = cacoshl(z);
    else if (strcmp(name, "asin") == 0 || strcmp(name, "arcsin") == 0) *result = casinl(z);
    else if (strcmp(name, "asinh") == 0 || strcmp(name, "arcsinh") == 0) *result = casinhl(z);
    else if (strcmp(name, "atan") == 0 || strcmp(name, "arctan") == 0) {
        if (isfinite(re) && isfinite(im)
            && fabsl(re) < sqrtl(LDBL_MAX) / 4.0L
            && fabsl(im) < sqrtl(LDBL_MAX) / 4.0L) *result = complex_atan(re, im);
        else *result = catanl(z);
    }
    else if (strcmp(name, "atanh") == 0 || strcmp(name, "arctanh") == 0) {
        if (isfinite(re) && isfinite(im)
            && fabsl(re) < sqrtl(LDBL_MAX) / 4.0L
            && fabsl(im) < sqrtl(LDBL_MAX) / 4.0L) *result = complex_atanh(re, im);
        else *result = catanhl(z);
    }
    else if (strcmp(name, "cos") == 0) *result = ccosl(z);
    else if (strcmp(name, "cosh") == 0) *result = ccoshl(z);
    else if (strcmp(name, "exp") == 0) *result = cexpl(z);
    else if (strcmp(name, "exp2") == 0) *result = cpowl(CMPLXL(2.0L, 0.0L), z);
    else if (strcmp(name, "expm1") == 0) {
        long double real = creall(z), imag = cimagl(z);
        *result = CMPLXL(expm1l(real) * cosl(imag) - 2.0L * sinl(imag / 2.0L) * sinl(imag / 2.0L),
                         expl(real) * sinl(imag));
    }
    else if (strcmp(name, "rint") == 0) *result = CMPLXL(rintl(re), rintl(im));
    else if (strcmp(name, "log") == 0) *result = clogl(z);
    else if (strcmp(name, "log1p") == 0) {
        long double real = creall(z), imag = cimagl(z);
        long double squared_magnitude_delta = 2.0L * real + real * real + imag * imag;
        long double log_magnitude = fabsl(real) < 0.5L && fabsl(imag) < 0.5L
            ? 0.5L * log1pl(squared_magnitude_delta)
            : logl(hypotl(1.0L + real, imag));
        *result = CMPLXL(log_magnitude, atan2l(imag, 1.0L + real));
    }
    else if (strcmp(name, "log10") == 0) *result = clogl(z) / logl(10.0L);
    else if (strcmp(name, "log2") == 0) *result = clogl(z) / logl(2.0L);
    else if (strcmp(name, "sin") == 0) *result = csinl(z);
    else if (strcmp(name, "sinh") == 0) *result = csinhl(z);
    else if (strcmp(name, "sqrt") == 0) *result = csqrtl(z);
    else if (strcmp(name, "tan") == 0) *result = ctanl(z);
    else if (strcmp(name, "tanh") == 0) *result = ctanhl(z);
    else if (strcmp(name, "reciprocal") == 0) *result = 1.0L / z;
    else if (strcmp(name, "square") == 0) *result = z * z;
    else if (strcmp(name, "positive") == 0) *result = z;
    else if (strcmp(name, "negative") == 0) *result = -z;
    else if (strcmp(name, "conj") == 0 || strcmp(name, "conjugate") == 0) *result = conjl(z);
    else if (strcmp(name, "sign") == 0) {
        long double magnitude = cabsl(z);
        if (isnan(magnitude)) *result = CMPLXL(nanl(""), nanl(""));
        else if (magnitude == 0.0L) *result = CMPLXL(0.0L, 0.0L);
        else if (isinf(magnitude)) {
            if (isinf(re) && isinf(im)) *result = CMPLXL(nanl(""), nanl(""));
            else if (isinf(re)) *result = CMPLXL(copysignl(1.0L, re), 0.0L);
            else *result = CMPLXL(0.0L, copysignl(1.0L, im));
        } else *result = z / magnitude;
    }
    else return 0;
    return 1;
}

int raptors_ld_unary_complex(const char *name, const char *real_text, const char *imag_text, char *out_real, size_t real_cap, char *out_imag, size_t imag_cap) {
    long double complex z, result;
    if (!parse_complex(real_text, imag_text, &z) || !unary_complex_value(name, z, &result)) return 0;
    return write_complex(result, out_real, real_cap, out_imag, imag_cap);
}

int raptors_ld_binary_complex(const char *name, const char *left_real, const char *left_imag, const char *right_real, const char *right_imag, char *out_real, size_t real_cap, char *out_imag, size_t imag_cap) {
    long double complex a, b, result;
    if (!parse_complex(left_real, left_imag, &a) || !parse_complex(right_real, right_imag, &b)) return 0;
    if (strcmp(name, "add") == 0) result = a + b;
    else if (strcmp(name, "subtract") == 0) result = a - b;
    else if (strcmp(name, "multiply") == 0) result = a * b;
    else if (strcmp(name, "divide") == 0 || strcmp(name, "true_divide") == 0) result = a / b;
    else if (strcmp(name, "power") == 0 || strcmp(name, "pow") == 0 || strcmp(name, "float_power") == 0) {
        if (creall(a) == 0.0L && cimagl(a) == 0.0L && creall(b) < 0.0L)
            result = CMPLXL(nanl(""), nanl(""));
        else result = cpowl(a, b);
    }
    else return 0;
    return write_complex(result, out_real, real_cap, out_imag, imag_cap);
}

int raptors_ld_classify(const char *text, int property) {
    long double value;
    if (!parse_ld(text, &value)) return -1;
    switch (property) {
        case 0: return isnan(value) != 0;
        case 1: return isinf(value) != 0;
        case 2: return signbit(value) != 0;
        case 3: return value == 0.0L;
        case 4: return isfinite(value) != 0;
        case 5: return fpclassify(value) == FP_SUBNORMAL;
        default: return -1;
    }
}

int raptors_ld_compare(const char *left, const char *right) {
    long double a, b;
    if (!parse_ld(left, &a) || !parse_ld(right, &b) || isnan(a) || isnan(b)) return 2;
    return a < b ? -1 : (a > b ? 1 : 0);
}

int raptors_ld_compare_complex(const char *left_real, const char *left_imag, const char *right_real, const char *right_imag) {
    long double complex a, b;
    if (!parse_complex(left_real, left_imag, &a) || !parse_complex(right_real, right_imag, &b)) return 2;
    long double ar = creall(a), ai = cimagl(a), br = creall(b), bi = cimagl(b);
    if (isnan(ar) || isnan(ai) || isnan(br) || isnan(bi)) return 2;
    if (ar < br) return -1;
    if (ar > br) return 1;
    if (ai < bi) return -1;
    if (ai > bi) return 1;
    return 0;
}

int raptors_ld_frexp(const char *text, char *fraction, size_t capacity, int *exponent) {
    long double value;
    if (!parse_ld(text, &value) || fraction == NULL || exponent == NULL) return 0;
    long double result = frexpl(value, exponent);
    return write_ld(result, fraction, capacity);
}

int raptors_ld_modf(const char *text, char *fraction, size_t fraction_cap, char *integral, size_t integral_cap) {
    long double value, integer;
    if (!parse_ld(text, &value) || fraction == NULL || integral == NULL) return 0;
    long double result = modfl(value, &integer);
    return write_ld(result, fraction, fraction_cap) && write_ld(integer, integral, integral_cap);
}
