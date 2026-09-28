# Differential testing and upstream test reuse

The [rebuild plan](REBUILD_PLAN.md) requires a behavioral oracle before broad implementation. Version 0.1 pins NumPy 2.5.3, source commit `dd88c0c19b54ad9ed3533224221285bf0873249a`, and its Python test dependencies. The new differential harness, Hypothesis cases, preview manifest, and generated API inventory are in the repository. The current 48-case suite is authored against NumPy; it does not reuse upstream test code.

The 0.1 required cases cover its explicit-dtype construction, metadata, integer/slice views, assignment, and copy. Harness fault probes can use synthetic backends for later behavior such as broadcasting; a passing probe does not claim that Raptors implements that behavior.

## Existing test material

- New preview cases in [raptors-python/tests/preview](../raptors-python/tests/preview).
- Legacy Rust integration tests in [raptors-core/tests](../raptors-core/tests/), including files named `numpy_port_*_test.rs`.
- Legacy Python API tests and a [numpy_port directory](../raptors-python/tests/numpy_port/).
- [port_numpy_test.py](../scripts/port_numpy_test.py), which generates Rust stubs.
- [generate_missing_tests.py](../scripts/generate_missing_tests.py), which generates additions by category and includes placeholders.

The legacy suites are retained for audit and do not run as 0.1 preview coverage. Names, file counts, and generated test bodies do not prove a faithful upstream port. Use the [verification record](NUMPY_TEST_VERIFICATION.md) for actual execution evidence.

## Reference and provenance

1. Pin the NumPy wheel version and matching source revision. The 0.1 pin is recorded in `compat/raptors-0.1.json` and `raptors-python/uv.lock`.
2. Inventory public names independently of what Raptors currently implements. `compat/numpy-api-2.5.3.json` is the generated name/member backlog; its future-release case plans still need per-entry semantic review.
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

Required extension build or import failures must fail the job. The preview suite's root `conftest.py` imports the built package directly and fails on import errors. A legacy nested `numpy_port` conftest still contains an import skip; that suite is not part of 0.1 validation.

Every remaining skip or expected failure must identify the missing behavior, reference case, milestone, and removal condition. Required supported cases cannot remain skipped at a release gate. Do not hide unknown failures behind broad exceptions or blanket skips.

## Safety and performance

Use Miri on isolated Rust storage/layout code, fuzzing for metadata and operation sequences, and sanitizer builds where supported. Review unsafe preconditions and foreign-memory contracts separately from test output.

Each optimized kernel must pass the same semantic cases as its scalar reference implementation. See [performance requirements](PERFORMANCE.md).

## Definition of done

A feature is conformant only when its declared semantic cases pass, upstream adaptations have provenance, unsupported cases remain visible, safety boundaries are reviewed, and relevant performance results are recorded. Update the manifest from captured evidence. Raw test totals are not a completion metric.
