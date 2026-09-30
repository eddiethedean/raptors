# Raptors 0.3 numeric ufuncs

Raptors is a Rust-backed Python array library rebuilding toward NumPy's public
Python functionality through `import raptors as np`. Version 0.3.0 builds on
the numeric array and dtype foundation with 101 top-level elementwise ufunc
names, numeric operators, and applicable ufunc methods. It is an explicitly
limited subset, not a general NumPy replacement, and makes no performance
claim.

The supported numeric dtypes are `bool`, signed and unsigned 8/16/32/64-bit
integers, `float16`/`float32`/`float64`, `complex64`/`complex128`, and the
platform's `longdouble`/`clongdouble` representations. Dtype aliases, byte
order, alignment, shape, strides, contiguity, casting, promotion, reshape,
transpose, integer/slice/boolean/fancy indexing, broadcast assignment, and
copy behavior are covered by the [0.2 compatibility manifest](https://github.com/eddiethedean/raptors/blob/main/compat/raptors-0.2.json).

The 0.3 contract covers the ufunc loops offered by NumPy 2.5.3 for supported
numeric dtypes, including `out`, `where`, casting, broadcasting, reductions,
accumulation, `outer`, indexed `at`, and floating-error controls. See the
[0.3 compatibility manifest](https://github.com/eddiethedean/raptors/blob/main/compat/raptors-0.3.json) for exact signatures,
known limits, platform cases, and verification evidence.

```python
import raptors

a = raptors.array([[1, 2], [3, 4]])
column = a[:, 1]
as_float = a.astype(raptors.float32)
```

Construction accepts Python boolean, integer, float, and complex values in
rectangular list/tuple nesting. `zeros` and `empty` allocate numeric arrays in
C or Fortran order. Scalar indexing returns Raptors scalar wrappers, not
NumPy scalar classes; wrappers do not implement full scalar arithmetic or
protocol compatibility.

Array-level routines such as `sum` and `mean`, generalized matrix ufuncs,
datetime/timedelta, object/string/structured dtypes, NumPy interoperation,
foreign buffers, and the NumPy C ABI remain outside 0.3. NumPy 2.5.3 is used
only by the development test suite; the installed package has no NumPy runtime
dependency.

Use GIL-enabled CPython 3.12–3.14. Build and test instructions are in
[BUILD.md](https://github.com/eddiethedean/raptors/blob/main/raptors-python/BUILD.md)
and [TESTING.md](https://github.com/eddiethedean/raptors/blob/main/raptors-python/TESTING.md).
See the [0.3 release record](https://github.com/eddiethedean/raptors/blob/main/docs/RELEASE_0_3.md)
for the exact support boundary and verification evidence.
