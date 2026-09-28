# Differential testing and upstream test reuse

The [rebuild plan](REBUILD_PLAN.md) requires a behavioral oracle before broad implementation. NumPy at an exact pinned release is that oracle. The differential harness, provenance registry, and machine-readable compatibility manifest are planned deliverables, not existing infrastructure.

## What exists today

- Rust integration tests in [raptors-core/tests](../raptors-core/tests/), including files named `numpy_port_*_test.rs`.
- Python API tests and a [numpy_port directory](../raptors-python/tests/numpy_port/).
- [port_numpy_test.py](../scripts/port_numpy_test.py), which generates Rust stubs.
- [generate_missing_tests.py](../scripts/generate_missing_tests.py), which generates additions by category and includes placeholders.

Audit these before reuse. Names, file counts, and generated test bodies do not prove a faithful upstream port. Use the [verification record](NUMPY_TEST_VERIFICATION.md) for actual execution evidence.

## Reference and provenance

1. Pin the NumPy wheel version and matching source revision.
2. Inventory public behavior independently of what Raptors currently implements.
3. Record each reused upstream path and test identifier at that revision.
4. Preserve license notices and attribution.
5. Adapt imports and fixtures minimally; review every changed assertion.
6. Identify tests tied to NumPy's private implementation separately from public behavioral tests.

Do not assume upstream test paths from historical documentation are still valid. Use the pinned checkout. Public helpers and submodules remain part of the declared inventory even when they are not array kernels.

## Differential harness contract

Execute the same cases through a backend fixture for NumPy and Raptors. Compare:

| Dimension | Required checks |
| --- | --- |
| Values | Exact integers and booleans; operation-specific floating criteria |
| Types | Dtype, scalar type, scalar versus array return, promotion, casting |
| Structure | Shape, applicable strides/order, flags, empty and scalar arrays |
| Mutation | Shared views, owner lifetimes, assignment, overlapping `out=` |
| Errors | Exception class, relevant warning category, error-state behavior |
| API | Signatures, keywords, return structure, dispatch and subclass behavior |

Handle NaN, infinity, signed zero, integer overflow, and out-of-range casts deliberately. Never apply a broad floating tolerance to all operations. Unspecified `empty()` values are not an equality target, but initialized-read safety is.

Use subprocesses for legacy crash containment, global state isolation, and selected lifetime tests. Capture partial results when a process crashes.

## Adversarial coverage

Generate cases with Hypothesis for Python and property testing for Rust. Cover zero-sized dimensions, negative/zero strides, non-contiguous views, large integers, mixed dtypes, invalid metadata, overlapping assignment, and repeated view/copy/write/destruction sequences.

Retain minimized failures as regression cases. Property tests supplement the NumPy oracle; they cannot replace it. Check the harness itself with intentionally incorrect outputs to ensure it detects dtype, broadcasting, aliasing, and ownership faults.

## Skips and unsupported behavior

Required extension build or import failures must fail the job. Existing Python collection can skip on import failure; correct that in v0.1.

Every remaining skip or expected failure must identify the missing behavior, reference case, milestone, and removal condition. Required supported cases cannot remain skipped at a release gate. Do not hide unknown failures behind broad exceptions or blanket skips.

## Safety and performance

Use Miri on isolated Rust storage/layout code, fuzzing for metadata and operation sequences, and sanitizer builds where supported. Review unsafe preconditions and foreign-memory contracts separately from test output.

Each optimized kernel must pass the same semantic cases as its scalar reference implementation. See [performance requirements](PERFORMANCE.md).

## Definition of done

A feature is conformant only when its declared semantic cases pass, upstream adaptations have provenance, unsupported cases remain visible, safety boundaries are reviewed, and relevant performance results are recorded. Update the manifest from captured evidence. Raw test totals are not a completion metric.
