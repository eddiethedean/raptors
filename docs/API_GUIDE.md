# API guide

**Status: legacy API orientation and target behavior.** The [rebuild plan](REBUILD_PLAN.md) defines the destination. This is not a complete or validated NumPy compatibility reference.

## Target Python API

Applications should eventually keep their NumPy expressions and change the import:

```python
import raptors as np
```

The contract covers values, dtypes, scalar behavior, shapes, views, mutation, keywords, exceptions, warnings, protocols, and public submodules at a pinned NumPy release. Passing arrays to software that requires an actual NumPy object will need explicit adapters.

The machine-readable compatibility inventory is a v0.1 deliverable. Until it exists and its cases pass, API availability must not be presented as conformance.

## Existing Python entry points

The [module registration](../raptors-python/src/lib.rs) currently exposes `PyArray`/`Array`, `PyDType`/`DType`, an iterator, constructors, dtype constants, selected ufunc functions, and NumPy conversion helpers.

| Area | Existing names or locations | Validation needed |
| --- | --- | --- |
| Construction | `zeros`, `ones`, `empty`, `array` | Shape rules, dtype inference/conversion, initialization |
| Properties | `shape`, `dtype`, `size`, `ndim`, `itemsize`, `strides`, layout flags | Return types and view/layout semantics |
| Array operations | Arithmetic/comparison operators, indexing, assignment, copy, view, reshape, transpose, flatten | Aliasing, broadcasting, dtype promotion, errors |
| Math and reductions | Functions registered in [ufunc.rs](../raptors-python/src/ufunc.rs) | Supported dtypes, keyword behavior, axes, numerical results |
| Interoperation | `from_numpy`, `to_numpy`, conversion and DLPack methods | Copy behavior, ownership, protocol compliance |
| Custom dtypes | Registration helpers in [dtype.rs](../raptors-python/src/dtype.rs) | Full semantics and lifetime contracts |

This table maps source locations, not feature completeness. For example, a reduction with an `axis` argument does not establish support for the full NumPy signature.

A minimal legacy inspection example, after building the extension:

```python
import raptors as np

a = np.zeros([2, 3], dtype=np.float64)
print(a.shape, a.size, a.ndim)
```

Examples here describe registered entry points; they were not executed as part of the documentation update. Do not use list-to-array construction with arbitrary dtypes as a safety example: its existing `f64` copy path needs replacement.

## Existing Rust API

The core crate re-exports `Array`, `DType`, `zeros`, `ones`, and `empty`. Dtype identifiers live in `raptors_core::types`.

```rust
use raptors_core::{zeros, DType};
use raptors_core::types::NpyType;

let array = zeros(vec![2, 3], DType::new(NpyType::Double)).unwrap();
assert_eq!(array.shape(), &[2, 3]);
assert_eq!(array.size(), 6);
```

The existing [builder](../raptors-core/src/array/builder.rs), traits, and operation modules are implementation references. They may change during the rebuild. Raw-pointer access is not the proposed default safe Rust interface.

## Behavior the rebuild must prove

- Constructors preserve integer precision and requested dtypes.
- Basic slices share storage; advanced indexing and copies follow NumPy's rules.
- Deleting a parent does not invalidate a live view.
- Negative strides, zero-sized dimensions, transposes, and broadcasting are valid inputs.
- Overlapping assignment and `out=` preserve the reference result.
- Scalar promotion and ufunc methods/keywords match the pinned NumPy release.
- Errors, warnings, and numerical edge cases are compared as part of the contract.
- Unsafe metadata and uncontrolled external mutation are outside an unconditional safety guarantee.

See [architecture](ARCHITECTURE.md), [test porting](TEST_PORTING.md), and [migration guidance](CONVERSION_GUIDE.md). Public documentation will expand with passing compatibility evidence.
