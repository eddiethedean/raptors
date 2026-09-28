# Python development setup

These instructions are for inspecting and testing the **legacy prototype** while the [rebuild](../docs/REBUILD_PLAN.md) is planned. The supported rebuild toolchain and exact NumPy oracle will be pinned for v0.1.

## Isolated environment

From the repository root, on macOS or Linux, use an available Python 3.11 interpreter as an initial inspection environment:

```bash
python3.11 -m venv /tmp/raptors-dev-venv
source /tmp/raptors-dev-venv/bin/activate
python -m pip install maturin numpy pytest pytest-cov
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
maturin develop --manifest-path raptors-python/Cargo.toml
python -c "import raptors; print(raptors.__file__); print(raptors.__version__)"
```

These bootstrap dependencies are not a reproducible lock or the chosen oracle. Record exact versions and pin them before producing baseline evidence. Use a fresh environment instead of the old environment stored under the package directory.

On Windows, create a separate environment with the selected interpreter, activate its `Scripts` environment, and set `PYO3_PYTHON` to that environment's Python executable. A validated Windows setup is part of the release matrix work.

The checked-in [Cargo configuration](../.cargo/config.toml) contains a machine-specific Python path. The explicit environment variable selects your interpreter without rewriting that file.

## Rebuild after Rust changes

Run `maturin develop --manifest-path raptors-python/Cargo.toml` again after changing Rust. Editable installation does not automatically recompile native code. Confirm the imported artifact path to avoid testing an older installation.

For optimized measurements, add `--release`. See [BUILD.md](BUILD.md) for wheel installation tests.

## Existing checks

From the repository root, with the environment active:

```bash
cargo test -p raptors-core --test array_test
cargo check -p raptors-python
python -c "import raptors; print(raptors.__file__)" &&
python -m pytest raptors-python/tests/ -v
```

The explicit import must succeed before Python tests run. Existing collection logic can otherwise skip tests when the module is missing. See [TESTING.md](TESTING.md) for full-suite commands, linking limitations, and expected baseline reporting.

## Repository behavior to account for

- The core build script writes a generated header under `raptors-core/target/include/`, even with an external Cargo target directory.
- The legacy `setup_test_env.sh` rewrites the workspace Cargo configuration. Inspect it before use; it is not the recommended default setup.
- The legacy test runner and Make targets invoke library-only Rust tests. They do not establish full integration-test coverage.
- Current CI and publishing workflows need v0.1 review; their existence is not a support guarantee.

## Rebuild workflow

Choose work from the [current milestone](../docs/CONVERSION_ROADMAP.md). Define reference behavior, add adversarial cases, review affected invariants, implement a small change, and capture actual verification results.

The differential harness, compatibility manifest, and safety automation are planned. Do not describe them as present until implemented. Preserve unresolved failures and unsupported cases explicitly.

See [contribution guidance](../docs/CONTRIBUTING.md) and [test porting](../docs/TEST_PORTING.md).
