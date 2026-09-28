# Raptors rebuild plan

Date: 2026-09-28

Status: accepted project direction; the 0.1 preview is published, with its hosted release gates passed. Releases 0.2–0.9 remain planned. See the [0.x release roadmap](CONVERSION_ROADMAP.md) for versioned deliverables and the [documentation index](README.md) for aligned development and validation guides. This plan supersedes the legacy completion roadmap and async-service positioning.

The 0.1 result includes the pinned NumPy oracle, generated API inventory, compatibility manifest, legacy baseline record, differential/property harness, new safe storage crate, PyO3 preview, wheel checks, benchmark, Miri job, and tag-triggered trusted-publisher workflow. CPython 3.12–3.14 suites pass locally on macOS ARM64; the tagged release passed hosted platform and AddressSanitizer jobs. The current workflow builds eight `cp312-abi3` target wheels for later releases, and its build-only matrix passed in [run 36468681074](https://github.com/eddiethedean/raptors/actions/runs/36468681074). See [release 0.1](RELEASE_0_1.md) for exact scope and evidence.

## Goal and recommendation

Build a Rust-backed Python package usable as `import raptors as np`, with the same public functionality and defined behavior as a pinned NumPy release. The user has confirmed that the import can differ. Replacing NumPy's binary interface underneath precompiled extensions is not a release requirement.

Recommend a new storage, dtype, and execution foundation, introduced alongside the legacy implementation until it passes its acceptance gates. Preserve the old implementation as reference material and a source of regression cases. Reuse code only after validating its semantics and safety. Do not extend the existing ownership and FFI design by default.

Three goals must be demonstrated separately:

1. **Compatibility:** matching values, dtypes, shapes, mutation, views, errors, warnings, protocols, and public APIs.
2. **Memory safety:** a sound safe Rust API, checked array metadata, and explicitly reviewed boundaries for unsafe code and foreign memory.
3. **Performance:** reproducible improvements on declared Python workloads, including allocation and conversion costs.

A more capable model can help implement this. It cannot replace an independent behavioral reference, a sound design, or executable acceptance gates.

## What the existing implementation teaches us

These are observations from source inspection, not a complete audit:

- `raptors-python/src/lib.rs::array_from_list` allocates using the requested dtype, then copies all values through an `f64` pointer. Narrow dtypes can overrun the allocation, and converting every input through `f64` can lose integer precision.
- `raptors-python/src/array.rs` manually implements `Send` and `Sync`, justified by `Arc`. Reference counting alone does not synchronize mutable array storage.
- `raptors-core/src/ffi/mod.rs::PyArrayObject` differs from NumPy's documented structure. Matching C function names did not establish NumPy ABI compatibility.
- `raptors-core/src/operations/arithmetic.rs` copies inputs even when no dtype conversion is required. The execution design introduces unnecessary memory traffic before computation.
- The legacy `numpy_port` conftest skips on import failure. The new preview test root imports the package directly and fails when it cannot load.
- `docs/CONVERSION_ROADMAP.md` declares broad completion alongside known failures and missing behavior. Test counts and the presence of modules were being treated as proof of compatibility.

Prior local checks established that five core array integration tests pass and the Python crate compiles with warnings. They do not establish overall correctness. A full baseline is a v0.1 deliverable.

## Compatibility contract

The [0.1 execution plan](RELEASE_0_1.md) pins NumPy **2.5.3** at `dd88c0c19b54ad9ed3533224221285bf0873249a` and GIL-enabled CPython **3.12–3.14** as the first support range. The Python dependency lock and source submodule match that reference. Version 0.1.0 shipped twelve version-specific wheels for Linux x86-64, macOS x86-64/ARM64, and Windows x86-64. The current release workflow expands later releases to eight `cp312-abi3` wheels across manylinux, musllinux, macOS, and Windows x86-64/ARM64. Keep a separate compatibility job for a newer NumPy release so upstream changes cannot silently change the reference.

The eventual inventory includes:

- Module functions, signatures, keyword behavior, constants, exceptions, warnings, and importable public submodules.
- `ndarray` methods and properties; NumPy scalar types and their Python interactions.
- Dtype parsing, inference, promotion, casting, byte order, alignment, structured fields, strings, datetime/timedelta, and object storage.
- Indexing, assignment, broadcasting, views, copy behavior, writeability, strides, memory order, and overlap handling.
- Ufunc objects and their methods and keywords, including `out`, `where`, `dtype`, casting rules, `reduce`, `accumulate`, `reduceat`, `outer`, and `at`.
- Reductions, sorting/searching, statistics, linear algebra, FFT, random generators/state, file I/O, masked arrays, polynomials, and public testing/typing helpers.
- Array conversion and dispatch protocols, subclass behavior, serialization, and interoperation with external arrays.

Record every inventory entry in a machine-readable compatibility manifest with reference version, semantic cases, implementation status, evidence, limitations, and milestone. The 0.1 generated inventory records preliminary release assignments and required case plans; later entries still need per-API semantic review before implementation. Suggested states: unimplemented, partial, conformant, and delegated. A delegated NumPy call is not native implementation coverage.

The Array API standard may provide an early conformance suite, but its smaller surface does not define the final NumPy compatibility goal.

Match defined behavior for valid inputs. Reject metadata that would access memory outside its allocation. NumPy escape hatches such as arbitrary pointer manipulation cannot carry an unconditional safety guarantee. Record these deliberate restrictions in the compatibility manifest.

Integration with software that insists on an actual `numpy.ndarray` uses explicit adapters. Passing Raptors objects directly to every existing extension is a separate capability, not implied by changing the import in application code. NumPy is initially a test oracle and optional interoperability dependency; the native numeric implementation must run with NumPy fallback disabled.

## Proposed implementation architecture

Keep Python policy, array semantics, and storage machinery distinct. Suggested boundaries, finalized after the foundation prototype:

- **Python package / PyO3 bindings:** public signatures, Python scalar conversion, exceptions, protocol dispatch, and Python object lifetimes.
- **Array semantics:** dtype resolution, shape/index normalization, broadcasting, output planning, and operation contracts.
- **Storage and layout:** allocation ownership, checked offsets, shared views, access guards, and foreign-buffer adapters.
- **Execution:** one validated iteration plan used by scalar reference kernels and optimized kernels.
- **Specialized backends:** linear algebra, FFT, and other algorithms behind narrow interfaces and the same conformance tests.

Evaluate maintained Rust primitives and numerical backends before writing replacements. Adoption requires evidence for negative/zero strides, dynamic dtypes, shared mutation, licensing, supported platforms, and unsafe boundaries. Do not force NumPy semantics into an incompatible container merely to reuse a crate.

### Storage rules that must be settled first

1. **One allocation owner, many descriptors.** A view contains a shared storage handle, dtype, shape, signed byte strides, and offset. Basic slicing shares storage. A view keeps its allocation alive after its parent is dropped.
2. **Check all arithmetic.** Validate dimensions, element counts, byte sizes, offsets, and reachable address ranges, including zero-sized dimensions and negative strides. Bound allocations and offsets by platform limits before pointer arithmetic.
3. **Control mutation at the allocation level.** Use access guards shared by every view of the same allocation. Do not create an exclusive Rust reference simply because one Python wrapper is mutably borrowed. Detect aliases across all inputs and outputs; acquire access in a stable order to avoid deadlocks.
4. **Handle overlapping operations deliberately.** Operations such as `a[1:] = a[:-1]` and overlapping `out=` need NumPy-compatible results. When disjoint access cannot be established, use an appropriately ordered operation or a temporary snapshot. Preserve observable view sharing.
5. **Initialization is part of safety.** `empty()` may return unspecified values, but safe Rust must never read uninitialized typed values. Initially initialize exposed numeric buffers; measure that cost. Any later uninitialized-storage optimization requires a reviewed initialization proof and cannot expose uninitialized reads. Object storage requires valid initialized references.
6. **Represent types faithfully.** Never convert every scalar through `f64`. Preserve large integers, complex values, booleans, and dtype-specific conversion errors. Account for alignment and endianness before typed access.
7. **Constrain unsafe code.** Isolate raw allocation, specialized kernels, and foreign interfaces. Each unsafe block must state its preconditions and identify the validation that establishes them. Deny unsafe code in modules that do not need it.
8. **Treat foreign ownership separately.** Default to copying when lifetime, bounds, or exclusive access cannot be established. Zero-copy adapters need an explicit contract and a retained owner; a read-only view flag does not prevent writes through another foreign alias. Honor explicit `copy=False` by succeeding without a copy or raising the correct error.
9. **Thread safety precedes parallelism.** Do not add blanket `Send`/`Sync` implementations to silence compiler errors. Release the GIL only when the storage access model supports it. Free-threaded Python, object callbacks, writable buffer exports, resize, and reentrancy each need separate proof and tests.

External native libraries and consumers of raw pointers remain trust boundaries. Tests, Miri, sanitizers, and Rust types provide complementary evidence; no passing test suite proves the entire system memory-safe.

## Engineering work streams and exit gates

The detailed engineering work below rolls up into the versioned releases in [CONVERSION_ROADMAP.md](CONVERSION_ROADMAP.md). The roadmap owns version numbers; these work streams describe dependencies and technical exit gates.

| Engineering work stream | Release |
| --- | --- |
| A: baseline, pinned contract, and API inventory; B: differential harness | 0.1 |
| C: safe storage preview and array foundation | 0.1-0.2 |
| D: complete dtype and numeric behavior | 0.2-0.3 |
| E: measured optimization | 0.4 |
| F: expand the public API and specialized modules | 0.5-0.8 |
| G: conformance closure, applications, and release validation | 0.9 |

### Work stream A — Establish a truthful baseline (v0.1)

Deliverables:

- Preserve the current revision and inventory reusable algorithms and tests.
- Reproduce the existing Rust and Python suites in isolated environments; record crashes, failures, skips, and missing dependencies. Run crash-prone legacy tests in subprocesses.
- Lock the selected NumPy 2.5.3 source/wheels, CPython 3.12–3.14 support matrix, platform/tool versions, and an initial set of representative application workloads.
- Create the compatibility manifest and a Python benchmark harness comparing Raptors and NumPy.
- Make required build/import failures fatal; verify tests import the newly built artifact. Move generated files into ignored build locations.

**Exit gate:** a reproducible baseline report and an agreed compatibility contract. No inherited claim of completion is accepted without evidence.

### Work stream B — Build the behavioral test harness (v0.1)

Run the same test cases against NumPy and Raptors through a shared backend fixture, with separate subprocesses where crash containment or global state requires it.

Compare more than numeric output: dtype, scalar versus array return, shape, relevant strides/flags, view sharing, mutation of aliases, exceptions, and warnings. Use exact comparisons for integers and structural properties, and operation-specific floating-point criteria. Preserve NaN, infinity, signed-zero, overflow, and error-state behavior. Never use one broad tolerance to hide discrepancies.

Use curated upstream tests with their provenance and license notices preserved. Adapt imports and fixtures minimally; audit changes to assertions. Add generated cases using Hypothesis for Python and property testing for Rust. Shrink failures into permanent regressions. Exercise sequences of views, writes, copies, and owner destruction, not just isolated operations.

**Exit gate:** the harness detects intentional faults in dtype selection, broadcasting, overlap, and view ownership. It fails for a missing extension and reports all skipped/unsupported cases explicitly. The 0.1 preview's harness probes passed locally and its tagged release passed hosted wheel tests on CPython 3.12–3.14.

### Work stream C — Prove the safe array foundation (v0.1 preview, v0.2 completion)

Implement the new storage/layout design with a deliberately small dtype set: boolean, signed and unsigned 64-bit integers, and 32/64-bit floats. The [0.1 preview](RELEASE_0_1.md) covers explicit-dtype construction, scalar and empty arrays, integer/basic-slice views, same-dtype assignment, and copy. Complete transpose, reshape, broader assignment and layout behavior in 0.2.

Test allocation bounds, invalid metadata, negative and zero strides, non-contiguous access, parent destruction, repeated aliases, overlapping assignment, and read/write conflicts. Use Miri on isolated Rust storage/layout code, fuzz its constructors and operation sequences, and run sanitizer builds where supported. Test Python lifetimes through the actual extension.

**Exit gate:** the 0.1 preview passes its declared owner-lifetime, stride, overlap, bounds, and initialization cases with documented invariants, reviewed unsafe boundaries, and recorded Miri/sanitizer results. Local Rust/Miri tests and hosted sanitizer/wheel checks passed for 0.1. Complete the remaining foundation cases in 0.2 before growing the operation catalog.

### Work stream D — Deliver one complete numeric path (v0.2-v0.3)

Implement construction → view/slice → broadcast arithmetic → reduction → conversion back to Python, all through the same execution and dtype machinery. Include scalar promotion, tuple/negative axes, `keepdims`, and the applicable ufunc keywords. Extend to the remaining numeric widths and complex types before declaring numeric compatibility.

Required demonstrations include:

- Large integer construction without a floating-point intermediate.
- Reverse-strided and transposed arrays in arithmetic and reductions.
- Mixed dtypes and Python scalar promotion matching the pinned reference.
- `np.add(a, b, out=a)` and partially overlapping inputs/outputs.
- A sliced view that remains valid after its original Python owner is deleted.
- NaN, infinity, zero-length dimensions, and invalid input behavior.

**Exit gate:** v0.2 passes declared numeric dtype, array layout, indexing, and mutation cases. By v0.3, every promised numeric ufunc case passes the differential suite with fallback disabled. Publish baseline latency, allocation count, and peak memory before optimizing.

### Work stream E — Establish performance wins (v0.4)

Profile complete Python calls. Remove redundant copies, metadata allocations, and per-element Python interaction first. Then specialize contiguous loops, collapse dimensions in iteration plans, add validated SIMD paths, and introduce parallel thresholds backed by measurements. Keep a scalar reference implementation for every optimized kernel.

Benchmark tiny, medium, and large arrays; contiguous and strided layouts; broadcasting; allocation and `out=` paths; mixed dtypes; reductions; and real workflows. Record CPU, OS, compiler, Python/NumPy versions, numerical backend, and thread counts. Compare both controlled single-thread settings and practical defaults. Include warmup, repeated samples, variability, conversion overhead, and native peak memory.

Proposed first performance gate: at least three preregistered application workloads improve by 20% or more in median end-to-end time, with repeatable evidence beyond measurement noise. No regression over 10% on the declared common-operation suite may remain unexplained and unaccepted. These are proposed product thresholds, fixed before tuning, not claims about current performance.

Do not promise every NumPy operation will be faster. NumPy already has CPU-dispatched SIMD kernels. Keep eager semantics: implicit lazy execution or fusion must not alter when mutation, errors, or numerical rounding become observable.

**Exit gate:** reproducible correctness-preserving wins and a public report showing both wins and regressions.

### Work stream F — Expand to full public functionality (v0.5-v0.8)

Add complete semantic groups in dependency order:

1. Remaining indexing/assignment and ufunc behavior; reductions, concatenation, sorting/searching, statistics, and set operations.
2. Linear algebra, FFT, and einsum, using evaluated backends where appropriate. Define numerical tolerances, backend thread behavior, and error mappings.
3. Random generators, seed/state serialization, and stream compatibility for the pinned reference where specified. Statistical similarity alone is insufficient when exact reproducibility is promised.
4. NPY/NPZ, text I/O, memory mapping, serialization, and buffer/DLPack interoperation, with explicit lifetime and bounds contracts.
5. Datetime/timedelta, strings, structured/record dtypes, and object arrays. Object operations require Python reference management and callback/reentrancy handling.
6. Masked arrays, polynomials, remaining public helpers/submodules, dispatch protocols, subclassing, typing, and public testing utilities.

The manifest determines the precise remaining backlog, including platform-specific types and behavior. These groups are not declarations of completion and may be reordered by measured user needs. Temporary delegation is tracked separately and cannot silently inflate native coverage or performance results.

**Exit gate for each group:** applicable upstream tests, differential and generated cases, safety checks, Python examples, and benchmark evidence. The full compatibility label waits until the complete declared public surface passes; earlier releases state their exact supported subset.

### Work stream G — Release and maintain compatibility (v0.9)

- Build and install release wheels in clean environments for the declared Linux, macOS, and Windows architecture matrix.
- Validate application scripts with only the import changed; maintain tests for explicit NumPy adapters used by downstream packages.
- Test packaging, imports, public submodules, type hints, serialization, and installation without unintended runtime NumPy dependencies.
- Generate compatibility documentation from the manifest and performance reports from captured measurements.
- Run fast deterministic gates on every change, broader differential/platform suites in scheduled CI, and bounded nightly fuzzing with retained regression corpora.
- Require a safety review for storage, pointer, aliasing, or concurrency changes. Revalidate the relevant gates when the NumPy reference version changes.

**Exit gate:** release evidence is reproducible from a clean checkout, all required tests run, and compatibility, safety boundaries, and performance claims match that evidence.

## Development discipline for model-assisted work

For each change:

1. State the exact NumPy behavior and the invariant being changed.
2. Capture reference cases from the pinned NumPy version and add meaningful adversarial cases.
3. Implement the smallest coherent change.
4. Review semantics and unsafe assumptions separately from implementation; do not use generated comments as proof.
5. Run the relevant correctness, safety, and performance gates.
6. Update the compatibility manifest using actual test results.

Do not weaken expected results, add blanket skips, catch all errors to return plausible output, or label a feature complete to make a milestone green. Missing runtime functionality must produce a clear unsupported status until implemented. Keep changes small enough for substantive review. Broad feature generation and simultaneous rewrites of interdependent foundations recreate the original risk.

## Current implementation checkpoint

The 0.1 work streams have local and hosted artifacts and results recorded in [RELEASE_0_1.md](RELEASE_0_1.md) and [NUMPY_TEST_VERIFICATION.md](NUMPY_TEST_VERIFICATION.md). Version 0.1.0 passed its twelve-cell OS/Python wheel matrix and Linux AddressSanitizer gate before publication. Run the current eight-target stable-ABI workflow before each later release. Estimate later work from the reviewed inventory and the measured difficulty of each complete numeric path. Full NumPy functionality is a sustained library engineering effort; a fixed short rewrite schedule would be speculation.

## Reference material

- [NumPy C structures](https://numpy.org/doc/stable/reference/c-api/types-and-structures.html): real object layout and ownership fields; explains why the existing C facade is not an ABI implementation.
- [NumPy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html): shared mutation and resizing hazards, including free-threaded Python.
- [NEP 50: scalar promotion](https://numpy.org/neps/nep-0050-scalar-promotion.html): Python scalar promotion semantics.
- [NEP 18: array function dispatch](https://numpy.org/neps/nep-0018-array-function-protocol.html): public dispatch protocol for alternative array implementations.
- [NumPy SIMD infrastructure](https://numpy.org/doc/stable/reference/simd/index.html): the optimized baseline Raptors must measure against.
- [NumPy downstream compatibility](https://numpy.org/doc/stable/dev/depending_on_numpy.html): API/ABI versioning and downstream extension considerations.

Pin versioned documentation and source revisions alongside the chosen NumPy release for v0.1; stable documentation URLs move over time.
