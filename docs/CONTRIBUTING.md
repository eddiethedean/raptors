# Contributing to Raptors

Raptors targets NumPy's public Python functionality through `import raptors as np`. The published 0.2 package provides a numeric array and dtype foundation; later work follows the [dtype plan](DTYPE_ARCHITECTURE.md) and [roadmap](CONVERSION_ROADMAP.md). The old engine is retained for audit; current changes belong in the checked storage and Python binding path.

## Set up

Use CPython 3.12–3.14 and the locked NumPy oracle from the repository root:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
```

See [Python development](../raptors-python/DEVELOPMENT.md), [building](../raptors-python/BUILD.md), and [testing](../raptors-python/TESTING.md) for full commands. The preview does not require NumPy at runtime.

## Change workflow

1. Name the NumPy 2.5.3 behavior and the exact current release boundary being changed.
2. Add differential and adversarial cases before or with implementation.
3. Identify layout, ownership, aliasing, initialization, dtype, and Python-lifetime invariants affected.
4. Implement the smallest coherent change in the new preview path.
5. Run relevant Rust, Python, safety, package, and benchmark checks.
6. Update the compatibility manifest and verification record with observed results.

Do not weaken expected results, add broad skips, suppress arbitrary errors, or count generated placeholders as coverage. Future inventory entries are preliminary plans, not release-ready semantics; review each against versioned NumPy behavior before implementing it.

## Relevant local checks

```bash
rustfmt --edition 2021 --check raptors-storage/src/lib.rs raptors-python/src/preview.rs raptors-python/src/preview_lib.rs raptors-python/build.rs
cargo test --locked -p raptors-storage
cargo check --locked -p raptors-python
cargo clippy --locked -p raptors-storage -p raptors-python --all-targets -- -D warnings
cargo +nightly miri test --locked -p raptors-storage
uv run --project raptors-python --extra dev --no-sync python -m pytest raptors-python/tests/preview -q
```

The PR CI also runs the storage tests under Miri and AddressSanitizer and tests each supported CPython version. The tag workflow builds/tests every advertised wheel and publishes only after all required jobs pass. `cargo fmt --all` and the legacy test suites are not part of the 0.2 release gate.

## Safety and evidence

The numeric storage crate is intentionally free of `unsafe` code. Keep checked bounds and initialization, shared allocation locking, and owner retention at the storage boundary. `Arc` alone does not justify `Send` or `Sync`. Any future unsafe code needs explicit preconditions and review. Miri and sanitizers supplement review; neither proves the entire package safe.

Performance changes need equivalent Python calls and measurements that include conversion and allocation costs. The 0.2 baseline shows lower median latency for NumPy on all eight measured operations; do not claim acceleration from Rust, Rayon, or SIMD presence.

Preserve license notices and attribution for any reused upstream material. The 0.2 tests are authored against the pinned NumPy oracle, not copied NumPy tests. Keep generated build output and machine-specific environments out of commits.

## Release status

The 0.2.0 release is published after all hosted gates passed. See the [0.2 release record](RELEASE_0_2.md), [0.2 compatibility contract](../compat/raptors-0.2.json), and historical [0.1 record](RELEASE_0_1.md) before changing the advertised API boundary.

Use [docs/README.md](README.md) to find the current project guides. Describe current behavior separately from future plans and observed results.
