# Compatibility verification record

Updated: 2026-09-28.

**The 0.1 preview has local evidence, not full NumPy conformance.** The baseline for the legacy implementation, current preview results, and remaining hosted gates are separated below. Earlier claims that broad NumPy suites had been ported and passed are withdrawn.

## Legacy baseline before the preview implementation

Repository revision: `42df6d680b629556e8c1dbccf10e3caea072e3d6`. Local inspection environment: macOS ARM64, Python 3.11.14, NumPy 2.4.3, pytest 9.1.1, and Rust 1.96.0.

| Command | Result | What it establishes |
| --- | --- | --- |
| `cargo test --locked -p raptors-core --lib -- --quiet` | Exit 0 with compiler warnings; 0 tests | The legacy core library test target compiled; it did not test behavior. |
| `cargo check --locked -p raptors-python --quiet` | Exit 0 with 18 warnings | The old binding crate type-checked in this environment; no Python runtime conformance claim. |
| `python3 -m pytest raptors-python/tests --collect-only -q` | Exit 1 before collection | Ambient `pytest_cases` plugin failed against pytest 9 (`IdMaker.__init__` positional-argument mismatch); no test count was obtained. |
| Full legacy Python/Rust suites | Not established | The old CI's totals and `continue-on-error` paths were not reliable evidence. |
| NumPy comparison benchmark | Not run at baseline | No legacy speed result. |

The earlier repository inspection at revision `9fbe407` also recorded five passing `raptors-core` array integration tests and zero library-only tests. Those observations do not repair the missing full baseline. The old implementation and its tests remain audit material, not 0.1 preview coverage.

## Current 0.1 preview results

Local environment: macOS 26.5.2 ARM64, CPython 3.12.13/3.13.11/3.14.3, NumPy 2.5.3, Rust 1.96.0, and uv 0.11.3. The NumPy source submodule is checked out at `dd88c0c19b54ad9ed3533224221285bf0873249a`; the Python oracle/test dependencies are in `raptors-python/uv.lock`.

| Check | Command or artifact | Result |
| --- | --- | --- |
| Storage | `cargo test --locked -p raptors-storage` | 7 passed, 0 failed |
| Legacy core build script | `CARGO_TARGET_DIR=/tmp/raptors-core-out-target cargo check --locked -p raptors-core` | Passed with 4 legacy compiler warnings; generated `raptors_core.h` under Cargo `OUT_DIR`, outside the source tree |
| Binding compile | `cargo check --locked -p raptors-python` | Passed |
| Clippy | `cargo clippy --locked -p raptors-storage -p raptors-python --all-targets -- -D warnings` | Passed |
| Rust formatting | `rustfmt --edition 2021 --check raptors-storage/src/lib.rs raptors-python/src/preview.rs raptors-python/src/preview_lib.rs raptors-python/build.rs` | Passed |
| Unsafe-code boundary | `rg -n '\bunsafe\b' raptors-storage/src` | No matches |
| Miri | `cargo +nightly miri test --locked -p raptors-storage` | 7 passed on `aarch64-apple-darwin` |
| Differential/property suite | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest raptors-python/tests/preview -q` in each locked CPython environment | 48 passed, 0 skipped on 3.12.13, 3.13.11, and 3.14.3 (macOS ARM64) |
| Fault probes | `raptors-python/tests/preview/test_harness.py` | Detects seeded dtype, shape, broadcasting, alias-result, and expired-owner faults |
| Wheel contract | `scripts/check_wheel_contract.py` on the local CPython 3.14 ARM64 wheel | Passed: expected tag, license, extension, and no runtime requirements |
| Package metadata | `python -m twine check <wheel>` | Passed |
| No-NumPy installation | `scripts/check_clean_install.py <wheel>` | Passed: fresh environment had no NumPy, imported Raptors, and mutated a shared view |
| Benchmark | [`docs/benchmarks/raptors-0.1-baseline.json`](benchmarks/raptors-0.1-baseline.json) | 8 measurements recorded; NumPy has lower median latency for all four measured calls in this run; no speed claim |

The Python suite is newly authored against the pinned NumPy oracle; it does not copy upstream NumPy tests. Hypothesis uses deterministic bounded strategies. Preview tests fail on import failure and have no skip or expected-failure markers.

## Release outcome and subsequent workflow changes

The local table above records the evidence assembled before publication. The
tagged [`v0.1.0` release workflow](https://github.com/eddiethedean/raptors/actions/runs/36463556250)
later passed hosted Miri, AddressSanitizer, all twelve version-specific wheel
build and install checks, and published `raptors==0.1.0` to
[PyPI](https://pypi.org/project/raptors/0.1.0/). The earlier build-only
[`workflow_dispatch` run](https://github.com/eddiethedean/raptors/actions/runs/36461927101)
also passed.

The release workflow has since changed for versions after 0.1.0: it builds
eight `cp312-abi3` target wheels and tests each on CPython 3.12, 3.13, and 3.14.
That updated matrix requires its own build-only `workflow_dispatch` run before
the next release tag. It does not alter the already-published 0.1.0 wheel set.

## Reproducibility and interpretation

For the preview environment and test commands, see [`raptors-python/TESTING.md`](../raptors-python/TESTING.md). The 0.1 API boundary and inventory are in [`compat/raptors-0.1.json`](../compat/raptors-0.1.json) and [`compat/numpy-api-2.5.3.json`](../compat/numpy-api-2.5.3.json). The generated inventory is a backlog map with preliminary target assignments, not 13,481 conformance claims.

The benchmark runs equivalent Python calls in separate processes with warmup and repeated samples. `tracemalloc` excludes Rust/native allocations; process peak RSS is coarse and allocator-dependent. This single-host run demonstrates no performance advantage and must not be generalized. See [`PERFORMANCE.md`](PERFORMANCE.md).
