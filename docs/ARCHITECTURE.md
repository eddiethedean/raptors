# Raptors architecture

**Status:** 0.2.0 is published. Its numeric storage and binding slice passed local and hosted checks, including sanitizer and all eight wheel targets. The broader execution design remains planned. The [rebuild plan](REBUILD_PLAN.md) and [release roadmap](CONVERSION_ROADMAP.md) define the acceptance gates.

## Public contract

The Python package targets NumPy's public functionality and defined behavior for a pinned release. The import may be `raptors`. A NumPy-compatible C ABI is outside the release requirement; the existing C facade must not be treated as one.

Rust APIs support the Python implementation and may change during the rebuild. Array API standard conformance can be an early test target, but does not cover the full NumPy goal.

## Current preview and retained legacy code

| Component | Current location | Status |
| --- | --- | --- |
| Checked byte-addressed storage, owners, and signed-stride views | [`raptors-storage`](../raptors-storage/) | 0.2 implementation; 12 Rust tests pass normally and under local Miri; no `unsafe` in the crate |
| Native extended-precision scalar bridge | [`raptors-longdouble`](../raptors-longdouble/) | Isolated C `long double` math and scalar encoding; never receives array pointers |
| Numeric Python API and scalar conversion | [`preview.rs`](../raptors-python/src/preview.rs), [`preview_lib.rs`](../raptors-python/src/preview_lib.rs) | Published in 0.2; numeric dtype inference, casts, views, indexing, and assignment |
| Differential and generated checks | [`tests/preview`](../raptors-python/tests/preview/) | 774 cases pass with no skips on CPython 3.12–3.14 against the pinned oracle |
| Python project and test lock | [`pyproject.toml`](../raptors-python/pyproject.toml), [`uv.lock`](../raptors-python/uv.lock) | Version 0.2.0; NumPy is development-only |

The supported Python path no longer imports the old Rust array engine. That engine remains in the repository for audit and future reference:

| Component | Current location | Rebuild treatment |
| --- | --- | --- |
| Raw data pointers, metadata, ownership flags, and views | [arrayobject.rs](../raptors-core/src/array/arrayobject.rs) | Replace the foundation; audit reuse against written invariants |
| Dtypes, promotion, and casts | [types](../raptors-core/src/types/), [conversion](../raptors-core/src/conversion/) | Validate against the pinned NumPy reference |
| Broadcasting, iterators, and kernels | [ufunc](../raptors-core/src/ufunc/), [operations](../raptors-core/src/operations/) | Consolidate around one checked execution plan |
| Python objects and conversion | [Python sources](../raptors-python/src/) | Preserve compatible behavior only after differential validation |
| Experimental C wrappers | [ffi](../raptors-core/src/ffi/) | Legacy interface; no NumPy ABI guarantee |

Manual `Send`/`Sync` implementations in the legacy bindings and shared `Arc` ownership do not establish safe concurrent mutation. Existing legacy view, allocation, and conversion code requires review before reuse.

## Proposed layers

```text
Python API and PyO3 bindings
           |
Array semantics: dtypes, indexing, broadcasting, output rules
           |
Checked execution plan
           |
Reference kernels / optimized kernels / specialized backends
           |
Shared storage, checked layouts, and guarded access
```

Python-specific exceptions, dispatch, and callbacks stay at the binding boundary. Array semantics determine output types and layout before execution. Scalar reference kernels provide an oracle for optimized implementations, alongside NumPy comparisons.

The 0.2 package is intentionally narrower than this full design. It adds the numeric foundation but has no arithmetic execution plan, ufunc kernel layer, or numerical backend. Use maintained numerical components when their semantics, licensing, platform support, and safety boundaries fit the contract. The [0.2 release record](RELEASE_0_2.md) defines its numeric boundary and passed release gates.

## Storage and layout invariants

- One allocation owner may support many view descriptors. Each descriptor carries dtype, shape, signed byte strides, and offset.
- A view keeps its storage alive after the original Python array is deleted.
- Dimensions, products, allocation sizes, and reachable byte ranges are checked before pointer arithmetic. Negative strides and empty arrays need explicit handling.
- Access guards belong to the shared allocation. Borrowing one wrapper mutably does not prove exclusive access to its bytes.
- Multi-array operations account for aliases and acquire access in a stable order. Overlapping writes use correct traversal or a temporary snapshot.
- Safe typed reads require initialized, aligned, valid data. Initially initialize buffers exposed by `empty()`; optimize only with a reviewed initialization strategy.
- Conversions preserve dtype-specific values. Large integers must not pass through a floating-point intermediate.
- Writeability and copy decisions follow the compatibility contract. Copy-on-write must not silently change observable view mutation.

## Unsafe and foreign boundaries

Restrict unsafe code to allocation, specialized kernels, and foreign adapters. Document the preconditions for each block and the checks establishing them. Safe callers must not be able to construct invalid layouts or trigger undefined behavior.

Foreign buffers require a retained owner, known bounds, and an access contract. Copy when those conditions cannot be established. A read-only export alone cannot stop mutation through another alias. Explicit `copy=False` requests must either avoid a copy or fail appropriately.

Raw-pointer escape hatches and external native libraries are trust boundaries. Full memory-safety claims cannot include arbitrary external writes.

## Threads and Python lifetimes

Parallel execution and GIL release depend on proven storage access rules. Reference counting alone is insufficient. Object arrays, callbacks, writable buffer exports, resize, and free-threaded Python need separate tests for lifetimes, reentrancy, and synchronization.

Preserve eager NumPy semantics. Async scheduling and implicit lazy fusion are not part of the rebuild foundation; see [deferred async design](raptors_vs_numpy_async_design.md).

## Evidence required

The foundation gate combines written invariants, review of unsafe code, differential cases, operation-sequence fuzzing, Miri on isolated Rust code, and supported sanitizer builds. No single tool proves soundness. See [test porting](TEST_PORTING.md) and [performance](PERFORMANCE.md) for the corresponding behavioral and measurement requirements.
