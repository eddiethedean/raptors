# API guide

**Status:** 0.1 remains the published package. The worktree targets 0.2.0 and has passed its local differential, Rust, Miri, and local wheel checks; the hosted sanitizer and eight-target wheel gate are pending. The [0.1 manifest](../compat/raptors-0.1.json) defines the published package. The [0.2 contract](../compat/raptors-0.2.json) defines the current candidate and its remaining release checks. The generated NumPy inventory is a backlog map, not a conformance reference.

## Published 0.1 package

The package uses a different import from NumPy:

```python
import raptors

a = raptors.array([[1, 2], [3, 4]], dtype=raptors.int64)
column = a[::-1, 1]
column[0] = 9
```

The published package exposes `array(data, dtype)`, `zeros(shape, dtype=None)`, `empty(shape, dtype=None)`, the dtype constants `bool_`, `int64`, `uint64`, `float32`, `float64`, and a limited `Array` with metadata, basic integer/slice indexing, scalar assignment, same-dtype exact-shape array assignment, and `copy()`.

The published 0.1 package does not expose arithmetic, broadcasting, reductions, dtype inference, reshape/transpose, advanced indexing, or NumPy interoperation. Scalar indexing returns typed Raptors wrappers, not NumPy scalar classes. Check the 0.1 manifest before using that published package in an application.

## 0.2.0 candidate worktree surface

The worktree adds fixed-width numeric dtypes and aliases, numeric inference and promotion, casts, C/F construction and reshape order, transpose, integer/slice/boolean/fancy indexing, broadcast assignment, and endian-aware storage. Its signatures are `array(data, dtype=None, copy=True, order='K')`, `zeros(shape, dtype=None, order='C')`, and `empty(shape, dtype=None, order='C')`. Dtype objects expose `kind`, `char`, `itemsize`, `alignment`, `byteorder`, `isnative`, `str`, and `type`. The local suite has 771 passing cases with no skips on each supported CPython version. Hosted cross-platform validation remains pending; see the [0.2 contract](../compat/raptors-0.2.json) and [release gate](RELEASE_0_2.md).

## Legacy source tree

The old extension source remains for audit, but the built 0.2 Python module does not register or call it. These files are not evidence that the current candidate supports the corresponding operations:

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
