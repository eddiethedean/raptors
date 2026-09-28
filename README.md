# Raptors

Raptors is rebuilding a Rust-backed Python package with the goal of providing NumPy's public functionality through `import raptors as np`, with a sound memory model and measured performance improvements.

**Status: rebuild planning; the current implementation is an experimental legacy prototype.** Full NumPy compatibility, memory safety, and a speed advantage have not been established. The new foundation and its acceptance gates are described in the [rebuild plan](docs/REBUILD_PLAN.md).

## What we are building

- The same public functionality and defined behavior as a pinned NumPy release, including dtypes, scalar rules, views, mutation, errors, warnings, and submodules.
- Rust storage and execution designed around checked layouts, shared ownership, controlled mutation, and small reviewed unsafe boundaries.
- Reproducible performance gains on representative Python workloads, including conversion, allocation, and memory costs.

Changing the Python import is acceptable. Reproducing NumPy's binary interface underneath precompiled extensions is not a release requirement. Software requiring an actual `numpy.ndarray` will need explicit interoperability adapters.

Full functionality remains the destination. Early releases will identify their supported subset. Async scheduling, GPU support, JIT compilation, and distributed execution are deferred beyond the compatibility rebuild.

## Current repository

| Location | What exists today |
| --- | --- |
| [raptors-core](raptors-core/) | Legacy Rust array engine, operations, experimental C facade, tests, and benchmarks |
| [raptors-python](raptors-python/) | PyO3 bindings and Python tests for part of that engine |
| [docs](docs/README.md) | Rebuild plan, proposed architecture, development guidance, and evidence requirements |
| [scripts](scripts/) | Legacy test generators; generated stubs are not compatibility evidence |
| [numpy-reference](numpy-reference/) | NumPy reference submodule, which may need initialization |

The presence of a module or a passing test does not establish that its NumPy behavior is complete. Existing code and tests will be audited before reuse.

## Rebuild sequence

1. Establish a reproducible baseline and pin the compatibility contract.
2. Build a differential harness that runs the same cases against NumPy and Raptors.
3. Prove the storage, layout, view, and mutation model.
4. Complete a small numeric path through the Python API.
5. Measure and improve end-to-end performance.
6. Expand to the full declared public functionality.
7. Validate applications, release wheels, and ongoing compatibility.

See the [0.x release roadmap](docs/CONVERSION_ROADMAP.md) for versioned deliverables and exit gates. No release gate has passed yet.

## Working with the legacy prototype

From the repository root:

```bash
cargo build -p raptors-core
cargo test -p raptors-core --tests
```

Use the [Python development guide](raptors-python/DEVELOPMENT.md) for an isolated environment and an explicit extension build. These commands exercise the current implementation; failures are baseline findings, not permission to weaken tests.

The core build script currently generates `raptors-core/target/include/raptors_core.h` inside the source tree, even when Cargo's target directory is redirected. Moving generated output is a v0.1 task.

## Evidence so far

During the initial inspection on 2026-09-28, five core array integration tests passed and the Python crate passed a compile check with warnings. The core library-only test command ran zero tests. The full suites, Python runtime compatibility, and performance against NumPy were not verified.

The [verification record](docs/NUMPY_TEST_VERIFICATION.md) describes these limits and the report required next. Earlier completion percentages and test totals have been withdrawn.

## Documentation

Start with the [documentation index](docs/README.md), [rebuild plan](docs/REBUILD_PLAN.md), and [contribution guide](docs/CONTRIBUTING.md).

Project license declarations need reconciliation before release: the Python metadata declares MIT, but the repository has no top-level license file. Preserve upstream notices for any reused NumPy material; selecting a project license is separate work.
