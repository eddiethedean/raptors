# Python development setup

The supported development path is the 0.1 preview on GIL-enabled CPython 3.12–3.14. The old binding and test files remain for audit; the preview uses `raptors-storage` and the `preview.rs` PyO3 module.

## Create the locked test environment

From the repository root, select one supported interpreter:

```bash
uv sync --project raptors-python --extra dev --locked --python 3.14 --no-install-project
```

This installs the pinned NumPy 2.5.3 oracle and preview test tools into the project environment. NumPy is not a runtime dependency. To place the environment outside the checkout, set `UV_PROJECT_ENVIRONMENT=/tmp/raptors-dev-venv` for both `uv sync` and `uv run` commands.

## Build and test the native preview

```bash
uv run --project raptors-python --extra dev --no-sync maturin develop --manifest-path raptors-python/Cargo.toml --release
uv run --project raptors-python --extra dev --no-sync python -c "import raptors; print(raptors.__file__); print(raptors.__version__)"
uv run --project raptors-python --extra dev --no-sync python -m pytest raptors-python/tests/preview -q
```

Editable installation does not rebuild automatically; rerun `maturin develop` after changing Rust code. The preview test root imports `raptors` directly, so a missing or unloadable extension fails validation.

## Rust and safety checks

From the repository root:

```bash
rustfmt --edition 2021 --check raptors-storage/src/lib.rs raptors-python/src/preview.rs raptors-python/src/preview_lib.rs raptors-python/build.rs
cargo test --locked -p raptors-storage
cargo check --locked -p raptors-python
cargo clippy --locked -p raptors-storage -p raptors-python --all-targets -- -D warnings
cargo +nightly miri test --locked -p raptors-storage
```

CI also runs AddressSanitizer on Linux. See [TESTING.md](TESTING.md) for wheel and clean-install checks.

## Legacy material

The old `raptors-core` engine, Python modules, and `numpy_port` tests are kept for audit and possible validated reuse. They are not the 0.1 implementation or release gate. The pre-preview baseline and known collection failure are in the [verification record](../docs/NUMPY_TEST_VERIFICATION.md).

Choose follow-up work from the [release roadmap](../docs/CONVERSION_ROADMAP.md). Define oracle behavior, add meaningful edge and lifetime cases, review affected invariants, and record the actual result. Preserve unsupported features and failures explicitly.

See [contribution guidance](../docs/CONTRIBUTING.md), [test porting](../docs/TEST_PORTING.md), and the [0.1 release gate](../docs/RELEASE_0_1.md).
