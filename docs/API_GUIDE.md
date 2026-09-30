# API guide

**Status:** 0.3.0 is the latest published package. The [0.3 release record](RELEASE_0_3.md) and [0.3 contract](../compat/raptors-0.3.json) define its ufunc boundary. The following sections preserve the 0.1 and 0.2 foundation APIs; the generated NumPy inventory is a backlog map, not a conformance reference.

## Published 0.3.0 package

The release adds the reviewed 101-name set of top-level elementwise ufuncs, including applicable ufunc methods, numeric dtype loops, operators, output controls, and floating-error state. It is a limited NumPy-compatible slice. Consult the [0.3 contract](../compat/raptors-0.3.json) for exact signatures, loops, limits, and evidence.

## Previous published 0.1 package

The package uses a different import from NumPy:

```python
import raptors

a = raptors.array([[1, 2], [3, 4]], dtype=raptors.int64)
column = a[::-1, 1]
column[0] = 9
```

The published package exposes `array(data, dtype)`, `zeros(shape, dtype=None)`, `empty(shape, dtype=None)`, the dtype constants `bool_`, `int64`, `uint64`, `float32`, `float64`, and a limited `Array` with metadata, basic integer/slice indexing, scalar assignment, same-dtype exact-shape array assignment, and `copy()`.

The published 0.1 package does not expose arithmetic, broadcasting, reductions, dtype inference, reshape/transpose, advanced indexing, or NumPy interoperation. Scalar indexing returns typed Raptors wrappers, not NumPy scalar classes. Check the 0.1 manifest before using that published package in an application.

## Published 0.2.0 package

The package adds fixed-width numeric dtypes and aliases, numeric inference and promotion, casts, C/F construction and reshape order, transpose, integer/slice/boolean/fancy indexing, broadcast assignment, and endian-aware storage. Its signatures are `array(data, dtype=None, copy=True, order='K')`, `zeros(shape, dtype=None, order='C')`, and `empty(shape, dtype=None, order='C')`. Dtype objects expose `kind`, `char`, `itemsize`, `alignment`, `byteorder`, `isnative`, `str`, and `type`. The local suite has 774 passing cases with no skips on each supported CPython version; all eight hosted wheel targets passed and were published. See the [0.2 contract](../compat/raptors-0.2.json) and [release record](RELEASE_0_2.md).

## Legacy source tree

The old extension source remains for audit, but the published 0.2 Python module does not register or call it. These files are not evidence that the package supports the corresponding operations:

| Legacy area | Source | Current interpretation |
| --- | --- | --- |
| Old array and dtype bindings | [`src/lib.rs`](../raptors-python/src/lib.rs), [`array.rs`](../raptors-python/src/array.rs), [`dtype.rs`](../raptors-python/src/dtype.rs) | Retained prototype code; not compiled into the preview extension |
| Old ufuncs and iterators | [`ufunc.rs`](../raptors-python/src/ufunc.rs), [`iterators.rs`](../raptors-python/src/iterators.rs) | Retained for audit; not registered in the preview |
| Old NumPy adapters | [`numpy_interop.rs`](../raptors-python/src/numpy_interop.rs) | Not part of the 0.2 package API |
| Old core crate | [`raptors-core`](../raptors-core/) | Legacy implementation; not the new safe storage layer |

The current PyO3 module is [`preview.rs`](../raptors-python/src/preview.rs), backed by [`raptors-storage`](../raptors-storage/).

## Rust implementation API

`raptors-storage` supplies byte-addressed numeric storage, endian-aware checked reads and writes, checked views, indexing, snapshot assignment, and independent copies. It is an internal foundation for the Python preview, not a stable public Rust compatibility layer. The older `raptors-core` types are unrelated to the preview's storage contract.

## Behavior reserved for later releases

- Ufunc keywords/methods, reductions, numerical accuracy, warnings, and errors.
- File formats, foreign ownership, DLPack/buffer adapters, and dispatch protocols.
- Public submodules, specialized dtypes, and the remaining API inventory.

See [architecture](ARCHITECTURE.md), [test porting](TEST_PORTING.md), [migration guidance](CONVERSION_GUIDE.md), and the [release roadmap](CONVERSION_ROADMAP.md).
