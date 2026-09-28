# API guide

**Status: 0.1 preview API plus legacy source orientation.** The [rebuild plan](REBUILD_PLAN.md) defines the destination. The [0.1 manifest](../compat/raptors-0.1.json) is the authority for current availability; the generated NumPy inventory is a preliminary backlog, not a conformance reference.

## Current Python preview

The package uses a different import from NumPy:

```python
import raptors

a = raptors.array([[1, 2], [3, 4]], dtype=raptors.int64)
column = a[::-1, 1]
column[0] = 9
```

The preview exposes `array(data, dtype)`, `zeros(shape, dtype=None)`, `empty(shape, dtype=None)`, the dtype constants `bool_`, `int64`, `uint64`, `float32`, `float64`, and a limited `Array` with `shape`, `ndim`, `size`, `dtype`, `strides`, basic integer/slice indexing, scalar assignment, same-dtype exact-shape array assignment, and `copy()`.

It does not expose arithmetic, broadcasting, reductions, dtype inference, reshape/transpose, advanced indexing, or NumPy interoperation. Scalar indexing returns typed Raptors wrappers, not NumPy scalar classes. Check the manifest before using any call in an application.

## Legacy source tree

The old extension source remains for audit, but the built 0.1 Python module does not register or call it. These files are not evidence that the preview supports the corresponding operations:

| Legacy area | Source | Current interpretation |
| --- | --- | --- |
| Old array and dtype bindings | [`src/lib.rs`](../raptors-python/src/lib.rs), [`array.rs`](../raptors-python/src/array.rs), [`dtype.rs`](../raptors-python/src/dtype.rs) | Retained prototype code; not compiled into the preview extension |
| Old ufuncs and iterators | [`ufunc.rs`](../raptors-python/src/ufunc.rs), [`iterators.rs`](../raptors-python/src/iterators.rs) | Retained for audit; not registered in the preview |
| Old NumPy adapters | [`numpy_interop.rs`](../raptors-python/src/numpy_interop.rs) | Not part of the 0.1 package API |
| Old core crate | [`raptors-core`](../raptors-core/) | Legacy implementation; not the new safe storage layer |

The preview's PyO3 module is [`preview.rs`](../raptors-python/src/preview.rs), backed by [`raptors-storage`](../raptors-storage/).

## Rust implementation API

`raptors-storage` supplies typed initialized vectors, checked views, indexing, snapshot assignment, and independent copies. It is an internal foundation for the Python preview, not a stable public Rust compatibility layer. The older `raptors-core` types are unrelated to the preview's storage contract.

## Behavior still to prove in later releases

- Numeric dtype widths, promotion, casts, and complete scalar semantics.
- Reshape, transpose, broadcasting, advanced indexing, and full assignment rules.
- Ufunc keywords/methods, reductions, numerical accuracy, warnings, and errors.
- File formats, foreign ownership, DLPack/buffer adapters, and dispatch protocols.
- Public submodules, specialized dtypes, and the remaining API inventory.

See [architecture](ARCHITECTURE.md), [test porting](TEST_PORTING.md), [migration guidance](CONVERSION_GUIDE.md), and the [release roadmap](CONVERSION_ROADMAP.md).
