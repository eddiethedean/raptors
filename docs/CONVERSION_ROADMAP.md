# Raptors 0.x release roadmap

This roadmap turns the [rebuild plan](REBUILD_PLAN.md) into versioned releases. Raptors targets NumPy's public Python functionality through `import raptors as np`, backed by a new Rust storage and execution foundation. The import can differ; replacing NumPy's binary interface underneath precompiled extensions is not part of the target.

Full public functionality remains the destination. Each 0.x release is an explicitly limited development release, and its compatibility manifest must say what works and what does not. A release number is a gate, not a date estimate or evidence that work has passed.

**Current status:** v0.1's declared preview is published. The 0.2.0 numeric foundation passed local checks, main CI, hosted Miri and AddressSanitizer, and the eight-target stable-ABI wheel gate; it is ready as a release candidate. The old engine remains a legacy prototype. No full compatibility, general memory-safety, or performance-advantage claim is supported.

## Releases

| Release | Theme | What it delivers | Release gate | Status |
| --- | --- | --- | --- | --- |
| **0.1** | Safe array preview | [Detailed 0.1 scope and gate](RELEASE_0_1.md): NumPy 2.5.3 contract, generated public inventory and manifest, recorded legacy baseline, fail-closed differential harness, and new checked storage exposed through a narrow Python preview. Explicit `bool`, `int64`, `uint64`, `float32`, and `float64` construction; metadata; basic integer/slice views; same-dtype assignment; and copy. Arithmetic and reductions move to later releases. | Every declared preview case matches the pinned oracle with no skips or fallback; harness fault probes, negative-stride/overlap/owner-lifetime cases, safety checks and review, clean wheel installs without NumPy, and pre-tag release validation all pass on the declared matrix. | Published as `raptors==0.1.0`; see the release record |
| **0.2** | Array and dtype foundation | [Detailed 0.2 execution plan](RELEASE_0_2.md): checked numeric storage and layout; all built-in numeric widths and complex values, including available platform-dependent variants; dtype inference, casts, promotion, scalar interactions, byte order, alignment, reshape/transpose, indexing, and assignment. | Reviewed 0.2 API contract and differential cases cover every declared numeric dtype and layout, aliases, parent destruction, overlapping assignment, invalid shapes/casts, and zero-sized arrays; Miri/sanitizer findings resolved; safe Rust access cannot create an out-of-bounds, uninitialized, or misaligned typed read/write. | Release candidate; hosted gates passed |
| **0.3** | Numeric ufuncs | Numeric ufunc functions and objects, beginning with elementwise add/subtract/multiply, then broadcasting, comparisons, trigonometric/logarithmic/exponential functions, `out`, `where`, dtype/casting controls, reductions and ufunc methods such as `reduce`, `accumulate`, `reduceat`, `outer`, and `at`. | Pinned-reference signature, result, type, error, warning, and mutation cases pass across numeric dtypes and strided/broadcast inputs; native execution passes with NumPy fallback disabled. | Pending |
| **0.4** | Core array routines | Numeric creation and manipulation functions, `sum`/`mean`, concatenate/stack/split, sorting/searching/partitioning, set operations, statistics, histograms, and remaining numeric indexing/shape routines identified by the manifest. | Each public routine has reviewed reference cases for boundaries, axes, output shape/dtype, and mutation; the declared suite passes; performance and memory costs are recorded for representative workloads. | Pending |
| **0.5** | Scientific algorithms | Linear algebra, matrix products, einsum, FFT, polynomials, and numeric random generators/state. Evaluate established non-NumPy kernels where they meet semantics, licensing, safety, and platform requirements. | Numerical tolerances and backend/thread behavior are documented; edge and ill-conditioned cases are compared to NumPy; random seeding/state matches the pinned contract where promised; end-to-end correctness precedes any speed claim. | Pending |
| **0.6** | Data, files, and interoperation | NPY/NPZ and text I/O, memory maps, buffer and DLPack interoperation, array conversion protocols, and explicit adapters for consumers that require actual NumPy arrays. | Round-trip/version/error behavior passes; ownership, bounds, copies, writeability, and foreign-alias access are tested; explicit no-copy requests succeed without copying or raise the reference-compatible error. | Pending |
| **0.7** | Specialized data models | Datetime/timedelta, strings, structured/record dtypes, object arrays, masked arrays, and related iterator and user-defined dtype behavior. | Dtype layout, conversion, Python reference counts/callbacks, masking, views, serialization, and error behavior have coverage; object operations preserve Python lifetime and reentrancy invariants. | Pending |
| **0.8** | Public API implementation complete | Implement the remaining public NumPy functions, scalar classes, constants, importable submodules, dispatch and subclass protocols, typing, public test helpers, and configuration/reporting surfaces in the pinned inventory. | Every in-scope entry has an implementation and focused reference cases; no public name is silently missing. The full differential matrix is reserved for the 0.9 closure gate. | Pending |
| **0.9** | Compatibility and release candidate | Close the pinned NumPy conformance matrix; validate representative applications with only the import changed; finish safety, platform, packaging, documentation, and performance evidence. No new API family is added in this release. | Clean release builds/install tests pass on every declared platform; all required differential, generated, safety, and application gates pass; all skips/failures and delegated kernels are visible; every performance claim has reproducible measurements. | Pending |

Reaching 0.8 means the planned public surface is implemented, not that it is correct. Reaching 0.9 requires passing the full gates. The compatibility label applies only to the exact NumPy reference version and public surface recorded in the manifest. Updating that reference later requires rerunning and, where needed, revising the gates.

Release 0.1's [execution plan](RELEASE_0_1.md) defines its exact preview boundary, work order, adversarial cases, evidence bundle, and known blockers. It supersedes the earlier 0.1 arithmetic and reduction preview; those operations depend on the 0.2 dtype/layout foundation and the 0.3–0.4 numeric operation work.

Release 0.2's [execution plan](RELEASE_0_2.md) defines its numeric foundation boundary, dependency order, platform dtype decisions, adversarial cases, and release gate. Its public method scope requires semantic review of the generated inventory before implementation; preliminary 0.2 assignments are not a completion target by themselves.

## Release rules

- Do not publish or tag a 0.x release until its gate passes. Internal builds and pre-release wheels may be used for review, but must identify themselves as previews.
- A failed gate keeps the release pending. Do not rename or split a version to disguise a failure; update the roadmap and feature manifest with the reason.
- Every release publishes its supported subset and known limits. Features do not inherit completion from old modules, tests, or roadmap checkmarks.
- A test counts only when it exercises the behavior and has reviewed provenance. Generated placeholders and raw test totals are not evidence.
- Numerical benchmarks use equivalent Python calls, inputs, dtypes, layouts, output reuse, thread settings, and correctness criteria. Include conversion, allocation, variability, and native memory. Report regressions as well as wins.
- Numeric functionality must not silently fall back to NumPy. If an evaluated non-NumPy backend is delegated, identify it in the manifest and measurements.
- Miri, sanitizers, fuzzing, and code review are complementary safety evidence. Preserve explicit boundaries around arbitrary raw pointers and external native code.
- Work is sequenced by dependency and evidence, not calendar. The public API inventory created for 0.1 may adjust later version assignments; changes must preserve these exit gates and keep the full goal intact.

## Full scope

The 0.x sequence covers NumPy's declared public Python surface at the pinned reference release: constructors and dtype/scalar behavior; ndarray and ufunc contracts; indexing, views, mutation, and numerical routines; scientific and random modules; files and protocols; specialized dtypes and arrays; submodules and public helpers. The [dtype architecture plan](DTYPE_ARCHITECTURE.md) maps all eleven `dtype.kind` groups and NumPy 2.x `StringDType` to their shared descriptor design and release dependencies. Release 0.1 creates the exact inventory, including APIs absent from this summary table.

The target is an import-level replacement for applications whose required behavior has reached a conformant release. An extension demanding the actual `numpy.ndarray` object uses an explicit adapter. Replacing NumPy's C/ABI for precompiled extensions, private internal APIs, and arbitrary out-of-bounds pointer tricks are not part of the public Python contract. Where a documented public API exposes foreign or raw memory, its safe ownership and bounds policy must be explicit and tested.

## Final release and deferred work

After 0.9 passes, prepare 1.0 as the stable release that freezes the verified public compatibility contract and begins normal compatibility/deprecation policy. 1.0 is not a promise to exceed NumPy on every workload; it publishes only performance results supported by the benchmark report.

GPU execution, JIT compilation, distributed arrays, and async job APIs remain separate future proposals. They are not prerequisites for NumPy compatibility and must not delay or weaken the 0.x release gates.

## Current evidence

The [verification record](NUMPY_TEST_VERIFICATION.md) separates the legacy baseline from preview evidence. The local CPython 3.12–3.14 preview suites each have 48 passing cases; seven Rust storage tests pass normally and under Miri; and the tagged `v0.1.0` workflow passed the hosted twelve-wheel matrix and sanitizer gate. The full NumPy suite, general memory-safety guarantee, and a speed advantage have not been established.

See the [rebuild plan](REBUILD_PLAN.md) for architecture and implementation discipline, [test porting](TEST_PORTING.md) for differential cases, and [performance](PERFORMANCE.md) for the measurement method.
