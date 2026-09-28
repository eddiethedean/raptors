# Compatibility verification record

Updated: 2026-09-28.

**The rebuild has no full conformance report yet.** Previous statements that all NumPy core tests had been ported and passed have been withdrawn. The [rebuild plan](REBUILD_PLAN.md) requires a new reproducible baseline.

## Observed checks

During repository familiarization, at legacy revision `9fbe407`:

| Check | Observed result | What it establishes |
| --- | --- | --- |
| `cargo test -p raptors-core --lib --quiet` | Succeeded with warnings; zero tests ran | The library test target built |
| `cargo test -p raptors-core --test array_test --quiet` | Five passed | That small integration suite passed |
| `cargo check -p raptors-python --quiet` | Succeeded with warnings | The Python crate type-checked; no Python runtime conformance claim |
| Full Rust and Python suites | Not run during that inspection | Overall status remains unverified |
| NumPy comparison benchmarks | Not run | No speed advantage established |

These are dated inspection observations, not a clean, locked, multi-platform baseline. Build artifacts were placed outside the repository, but the core build script still recreated its tracked generated header; the original local deletion was restored.

## Source findings requiring regression cases

- Narrow-dtype list construction writes through an `f64` pointer after allocating for the requested dtype.
- Converting all list elements through `f64` can lose integer precision.
- Shared mutable storage lacks a demonstrated concurrency contract.
- Legacy C structures are not NumPy's documented binary layout.
- Arithmetic copies inputs even when their dtypes already match.
- Python test setup may skip collection if the extension cannot import.

These findings motivate tests and design work. They are not a complete audit or a claim that other paths are correct.

## Required v0.1 baseline report

Record the repository revision, exact NumPy/Python/Rust versions, platform, build mode, dependency locks, artifact path, command, exit code, and captured output. Separate:

- Collected, passed, failed, skipped, and expected-failure cases.
- Crashes and aborted runs from ordinary assertion failures.
- Python API conformance from Rust implementation tests.
- Native execution from delegated or fallback execution.
- Actual upstream ports from generated placeholders and newly authored tests.

Every upstream-derived case needs a source revision, path, test identifier, preserved notices, and a review of adapted assertions.

## Reference checkout

The NumPy submodule is already declared. To populate its recorded revision from the repository root:

```bash
git submodule update --init numpy-reference
```

Do not add a second submodule. The recorded checkout is reference material; v0.1 must select and document an exact released NumPy version for the oracle and align the source reference with it.

## Updating this record

Replace observations with new dated evidence when checks are actually run. Keep failures visible until resolved. A rewritten document or a passing compile check must not advance a compatibility milestone. See [test porting](TEST_PORTING.md).
