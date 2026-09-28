# Contributing to Raptors

Raptors targets NumPy's public Python functionality through `import raptors as np`. The local 0.1 implementation is a deliberately small preview; the complete target and release gates are in the [rebuild plan](REBUILD_PLAN.md) and [roadmap](CONVERSION_ROADMAP.md). The old engine is retained for audit, but new 0.1 work belongs in the checked storage and preview path.

## Set up

Use CPython 3.12–3.14 and the locked NumPy oracle from the repository root:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
```

See [Python development](../raptors-python/DEVELOPMENT.md), [building](../raptors-python/BUILD.md), and [testing](../raptors-python/TESTING.md) for full commands. The preview does not require NumPy at runtime.

## Change workflow

1. Name the NumPy 2.5.3 behavior and the exact 0.1 boundary being changed.
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

The PR CI also runs the storage tests under Miri and AddressSanitizer and tests each supported CPython version. The tag workflow builds/tests every advertised wheel and publishes only after all required jobs pass. `cargo fmt --all` and the legacy test suites are not the 0.1 quality gate.

## Safety and evidence

The 0.1 storage crate is intentionally free of `unsafe` code. Keep checked bounds and initialization, shared allocation locking, and owner retention at the storage boundary. `Arc` alone does not justify `Send` or `Sync`. Any future unsafe code needs explicit preconditions and review. Miri and sanitizers supplement review; neither proves the entire package safe.

Performance changes need equivalent Python calls and measurements that include conversion and allocation costs. The 0.1 benchmark currently shows slower Raptors timings on several measured operations; do not claim acceleration from Rust, Rayon, or SIMD presence.

Preserve license notices and attribution for any reused upstream material. The current 0.1 tests are authored against the pinned NumPy oracle, not copied NumPy tests. Keep generated build output and machine-specific environments out of commits.

## Release status

The 0.1 preview implementation and CPython 3.12–3.14 suites on macOS ARM64 are complete. Its tagged release passed hosted wheel, Linux AddressSanitizer, and publication checks. For later releases, run the current eight-target workflow manually before tagging. See [release evidence](RELEASE_0_1.md).

Use [docs/README.md](README.md) to find the current project guides. Describe current behavior separately from future plans and observed results.
